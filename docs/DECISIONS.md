# Design decisions (Phases 1 to 3)

Each decision records the choice, the reason and the evidence behind it. Unless stated
otherwise, the Phase 1 numbers (D1-D17) were measured on 2026-10-02 and the Phase 2 numbers
(D18-D25) and the Phase 3 numbers (D26 onward) on 2026-10-03 (UTC), on the test machine:

- Windows 11 on ARM64;
- x64 Python 3.12.13 running under emulation;
- MuJoCo 3.14.0.

## D1. x64 Python under emulation on ARM64 Windows

- **Choice:** a uv-managed x64 CPython 3.12 (`uv venv --python cpython-3.12-windows-x86_64-none`).
- **Why:**
  - On 2026-10-02, MuJoCo 3.14.0 on PyPI had Windows wheels for x64 only.
  - Building MuJoCo for ARM64 from source was not attempted.
- **Cost:** emulation is fast enough for Phase 1.
  - In the first check pass, headless stepping ran at 75x real time in the smoke test and
    79x in `doctor`. In the final pass, while other programs kept the CPU 40-50% busy, it
    ran at 44x and 57x.
  - The viewer held 0.99-1.00x real time.
  - Revisit if training in a later phase is too slow.

## D2. Pinned, locked environment

- **Choice:** `pyproject.toml` pins exact versions, and `uv.lock` locks the whole tree
  (18 packages):
  - Python 3.12 only (`>=3.12,<3.13`);
  - mujoco 3.14.0, numpy 2.5.3 and pillow 12.3.0;
  - pytest 9.1.1 in the dev group.
- **Checks:** `doctor` compares the installed versions with the pins, and
  `uv lock --check --offline` confirms that the lock file matches `pyproject.toml`.
- **`link-mode = "copy"`:** the project sits in a OneDrive-synced folder. Copying keeps uv
  from hard-linking files from its cache into a folder that OneDrive may lock.

## D3. Panda model: pinned Menagerie files, unmodified

- **Choice:** vendor the 72 files of `franka_emika_panda` at MuJoCo Menagerie commit
  `4d038b3feae26ec82b46a4d586379114012a8ac7`.
  - A manifest records the SHA-256 of every file. `doctor`, the tests and the smoke test
    verify them.
  - The scene includes `panda.xml` through `<attach>` (prefix `panda/`) instead of
    editing it.
- **Why:** the robot model is reproducible and auditable, the licence position stays simple,
  and updating the model only means changing the pin.
- **Recovery:** `fetch-assets` downloads missing or changed files from
  raw.githubusercontent.com at the pinned commit and checks each hash.
  - Test: on a temporary copy with 2 files deleted and 1 altered, it restored all three.
  - Afterwards 0 of 72 files were missing and 0 mismatched.

## D4. Frame and scene layout

- **Frame:** the world origin is the Panda base, the tabletop is z = 0 and +x points away
  from the robot. Positions in the robot base frame therefore need no transform.
- **Goal:** visual only (`contype="0"`). Its disc radius equals the 20 mm success tolerance
  on the block centre.
- **Configuration:** the scene parameters, reset regions, hidden-physics ranges and check
  tolerances are in `configs/scene.toml`.

## D5. Block/table friction from one explicit contact pair

- **Problem:** by default MuJoCo combines the friction of two touching geoms, taking the
  larger of the two.
- **Choice:**
  - the explicit `<pair name="block_table">` (condim 3) holds the coefficients;
  - `<exclude body1="table" body2="block">` stops MuJoCo from also creating its default
    contact for these two bodies. Explicit pairs are not affected by excludes.
- **Result:** the hidden friction is exactly the value that the reset writes into
  `pair_friction`.
- **Evidence:** the smoke kick test measures the effective friction as μ = v0²/(2 g d),
  with a 5% tolerance.
  - It kicks the block at 0° for all 9 combinations of mass (0.10, 0.25 and 0.40 kg) and
    friction (0.2, 0.5 and 0.8), and at 45° and 90° with friction 0.5.
  - The measured friction was within -2.29% to +1.58% of the configured value.
  - The normal force was within 0.000% of m·g.
- **Finger exclude:** contact between the two Panda fingers is also excluded. When the
  gripper is closed, the finger pads touch at zero distance, so their contact would only
  flicker; the joint limits already stop the fingers.

## D6. Elliptic friction cone

- **Choice:** `cone="elliptic"` instead of MuJoCo's default pyramidal cone.
- **Evidence:** a kick test with a 0.25 kg block, μ 0.5, a 0.5 m/s start and kicks at 0°,
  22.5°, 45°, 67.5° and 90°.
  - Pyramidal: effective friction was 28.61% low at 45° and 17.9% low at 22.5° and 67.5°.
    The spread across directions was 30.16 percentage points.
  - Elliptic: -2.60% to +1.58%, a spread of 4.18 points.
- **Why it matters:** with the pyramidal cone, the friction the agent experiences would
  depend on the push direction. The agent could mistake that for a property of the block.

## D7. Timing and integrator

- **Choice:**
  - 2 ms physics steps, with 10 steps per 50 Hz control update;
  - `implicitfast`, the integrator the Menagerie Panda model specifies;
  - `impratio` 1.
- **Evidence:** in the smoke hold tests, the final |qvel| was at most 3.63e-9 and the
  block drifted at most 2.9e-9 m.
  - Under gravity, the position servos leave the arm 0.0066 rad from its home targets.
  - The worst actuator tracking error was 0.0073 rad, and the gripper opens to 80.00 mm.

## D8. Seeded reset that rebuilds the state

- **Two random streams:** `SeedSequence(seed).spawn(2)` gives one stream for the layout and
  one for the hidden physics. A seed gives the same layout with or without
  `--randomize-physics`.
- **Reset steps:**
  1. Put back any physics option (`MjModel.opt`) that was edited after loading (D17).
  2. Write the block mass, its solid-cube inertia (m·a²/6) and the pair friction into the
     model.
  3. Run `mj_setConst` so that constants which depend on mass are recomputed.
  4. Restore the authored `<statistic>` centre and extent, which `mj_setConst` recomputes.
     They only frame the free camera, clipping planes and shadows.
  5. Run `mj_resetData`, set the home pose, block pose and goal, then run `mj_forward`.
- **Why:** no state carries over between episodes, so the same seed gives a
  bitwise-identical state (tolerance 0.0):
  - after other episodes;
  - in a fresh runner;
  - in a separate process.
- **Digest:** after 5 s, seed 0 reaches state digest `9ac3950e3c7ab848`. The digest covers
  the full integration state: time, qpos, qvel, act, warm start, ctrl, applied forces and
  mocap.
- **Layouts:** 200 seeds all gave valid, distinct layouts. Resets ran at 1957/s in the first
  check pass and 1277/s in the final, busier one.
- **Scope:** this was tested on one machine with one MuJoCo build only.

## D9. Stability monitor

- **Choice:** after every control update, `ScenarioRunner.step` checks that:
  - qpos, qvel, qacc and ctrl are all finite;
  - no MuJoCo warning counter is set;
  - max |qvel| is at most 50 (`[stability]` in `scene.toml`).
- **On failure:** it raises `SimulationUnstable`, and `step` and `view` exit with code 1.
- **Warnings:** MuJoCo warnings go to stderr instead of a `MUJOCO_LOG.TXT` file.

## D10. Observation boundary

- **Choice:** agent code receives only `Observation`, an allow-list of 18 fields.
  - The contact flags are booleans taken from active contacts (dist ≤ 0), not forces.
  - `pose_source = "simulator_state"` marks that V1 has no perception.
  - The true values are available only from `ScenarioRunner.hidden_truth()`. That method
    and the simulator handles are reserved for the viewer, the renderer, diagnostics and
    the evaluator.
- **Evidence:** with the block at rest, observations for different hidden masses and
  frictions are identical at reset (difference 0). After 5 s they differ by at most
  4.45e-9. The hidden values can therefore be inferred only from interaction.

## D11. Viewer

- **Choice:** MuJoCo's passive viewer (`mujoco.viewer.launch_passive`), stepped by the
  project's own loop in real time.
- **Status panels:** drawn as Pillow images (`Handle.set_images`). MuJoCo's bitmap-font text
  shows garbled letters with the Mesa D3D12 OpenGL driver on this machine.
- **Restore on edit:**
  - The viewer can change the state directly (reset key, keyframes, joint sliders, dragging
    a body while paused, the Equality panel's checkbox). When it does, the scenario is
    restored from the seed, so objects never teleport.
  - Ctrl+right-drag forces act through the physics and are allowed.
  - A change to the actuator targets alone (the Control sliders) is accepted as an operator
    command instead; see D16.
  - Edits in the Physics panel change the model's options, not the state. They are put back
    to the scene's values before the next step, without a restore; see D17.
- **Goal arrow:** added to `user_scn` only, so it never appears in the model, the
  observations or the offscreen renders.

## D12. Camera placement

- **Choice:**
  - `overhead` looks down at 60° from beyond the far edge of the table;
  - `oblique` looks from the far side, to the robot's right.
- **Why:** a straight-down camera was blocked by the arm in its home pose.
  - On seeds 0-49, the block was completely hidden in 29 layouts and the goal in 40. In 49
    layouts the arm hid at least part of the block.
  - On a grid over the reset regions, the block was completely hidden at 144 of 234 poses
    and the goal at 85 of 117 positions.
- **Elevation sweep:** occlusion starts between 65° and 66°.
  - At 65° and below, nothing was hidden.
  - At 66°, at most 4% of the block was hidden.
  - At 68°, up to 53.7% of the block and 19.4% of the goal were hidden.
  - 60° leaves a margin. Nothing is hidden on seeds 0-49 or on the grid, and the smoke test
    measures 0.0% occlusion.
- **Measurement:** these numbers were taken with the single-sample probe (D13). An earlier
  run with multisampling gave the same values.

## D13. Single-sample segmentation for the visibility check

- **Problem:** segmentation renders encode object IDs as colours. With multisampling (MSAA),
  edge pixels are averaged and then decode as other objects.
- **Choice:** `VisibilityProbe` renders with `offsamples = 0`.
- **Checks:** a regression test covers this. Turning multisampling back on makes that test
  fail, which confirms that the test catches the problem.

## D14. Shadow-map coverage

- **Problem:** the key light's shadow map covered too small an area, so the floor in the
  overhead view showed speckle artefacts.
- **Choice:** `shadowclip="2"` and a new key-light position.
- **Evidence:** a speckle metric fell from 3962 to 512. For reference, it is 385 with the
  key light's shadow turned off.

## D15. No git repository created

- The folder is not a git repository. Creating one and choosing a remote are left to the
  owner. `doctor` and the smoke test report this as a warning; see `docs/PROGRESS.md`.
- **Line endings:** `.gitattributes` stops git from converting line endings in
  `assets/third_party/` and `docs/evidence/`.
  - The Panda files are checked byte for byte, and the test machine's git has
    `core.autocrlf=true`.
  - Test: the files were committed in a temporary folder and then cloned.
    - Without the attributes, the clone failed the manifest check: 64 of 72 files had been
      rewritten with CRLF line endings, and `smoke_log.txt` changed too.
    - With them, all 72 files matched and the 8 evidence files were byte-identical.

## D16. Manual joint control in the viewer

- **Problem:** the first viewer gave no way to move the arm. It held its home pose, and
  every viewer edit, including the Control sliders, restored the seed.
- **Choice:** the operator moves the arm one joint at a time from the keyboard. The Control
  sliders in MuJoCo's right-hand panel set the same targets.
  - Up/Down select a joint and Left/Right turn it by one step. `-`/`+` change the step
    (1, 2, 5, 10 or 20 degrees; 5 at the start). Enter opens or closes the gripper, and
    Home returns to the home pose with the gripper closed.
  - Keys and sliders only set actuator targets, through `ScenarioRunner.set_ctrl`, which
    clamps them to the control ranges. The arm moves through its position servos and the
    contacts, and the block moves only when the robot pushes it.
  - An orange disc marks the selected joint. Like the goal arrow, it is in `user_scn` only.
- **Restore rules:** a change to `ctrl` alone is an operator command. A change to time,
  qpos, qvel, the goal's mocap pose or the equality constraint's on/off state still restores
  the seed (D11, D17). A restore also returns the targets to the home pose.
- **Replay:** each command is recorded with the number of control updates since the last
  reset. `view` replays the run headless with these commands and compares state digests.
- **Not Phase 2:** this is joint control by a person, for inspection only. There is no
  inverse kinematics, Cartesian control or pushing controller, and nothing done in the
  viewer is logged as data.
- **Evidence:**
  - 24 new tests: 21 in `tests/test_manual_control.py` and 3 in
    `tests/test_viewer_helpers.py`.
  - The sync code in MuJoCo 3.14.0's `simulate.cc` was read. A Control slider is copied
    into `ctrl` only if it was edited since the last sync; otherwise `ctrl` is copied back
    into the slider. Jog targets therefore show on the sliders and are not overwritten.
  - The panel's Clear button calls `mj_resetCtrl`. A one-off check with the real function
    showed that it changes only `ctrl` in this model, and the session accepted it as one
    command, with J4 clamped to its range.
  - Live key test: 18 key presses were posted to the window with `PostMessageW`, and 11 of
    them changed a target. In one run with MuJoCo's UI panels hidden and one with them
    shown, there were 11 operator commands, no restores and no instability. Both headless
    replays were identical (digests `771c41c789da8b23` and `e3472f883d29c3d8`).
  - With no key pressed, the viewer and smoke-test digests are unchanged
    (`34cde6c9825fa1e4` and `9ac3950e3c7ab848`).
- **Not tested:** nobody has used the controls by hand yet. Slider drags were not
  simulated, because GLFW reads the real cursor position.

## D17. Physics options restored on reset (Phase 1 review)

Found and fixed in the post-phase review (BUILD_PLAN.md section 9). The numbers below were
measured on 2026-10-03 (UTC).

- **Problem:** `reset()` rebuilt the state (D8) but not the model's physics options
  (`MjModel.opt`: timestep, integrator, solver settings, gravity, disable and enable flags,
  contact override and others).
  - The Physics panel of MuJoCo 3.14.0's passive viewer (the left panel, shown with Tab or
    `--show-ui`) can edit all 29 option fields. At the next sync, `simulate.cc` copies the
    edited options into the user's model (line 2181).
  - A panel edit therefore changed the simulation silently, and Backspace or a reset kept
    it. This broke "Preserve contact solver settings across experiments" (BUILD_PLAN.md
    section 4) and same-seed reproducibility (D8).
  - Reproduction: seed 0 after 150 control updates reaches digest `ee9f3f82a6a42cd6`.
    After gravity was turned off and the scene reset, it reached `e724724cf8801f48`; with a
    0.004 s timestep, `6dd0937f2b5cefb6`.
- **Coupled gap:** the Equality panel (right UI) writes `eq_active`, which the viewer did
  not watch. Turning off the finger coupling went undetected.
- **Choice:**
  - `ScenarioRunner` keeps a copy of every option field as loaded. `restore_options()`
    puts back each field that differs, compared bit for bit, and returns their names.
    `reset()` calls it first.
  - The viewer calls it before every step. Edits are put back and logged ("viewer edited
    physics option(s) ...; put back the scene's values"). The state is not touched, so the
    seed is not restored. `view` reports the count as "physics-option revert(s)".
  - The viewer's state check now includes `eq_active`. Toggling the constraint restores
    the seed, like the other state edits.
  - After a revert, the panel can keep showing an edited number until it is redrawn. The
    simulation uses the scene's value.
- **Evidence:**
  - 5 new tests, 133 in total:
    - every one of the 29 fields is put back;
    - a reset after edits reproduces the seed-0 rollout exactly;
    - the viewer session puts option edits back without a restore, and its headless replay
      matches;
    - option and state edits at the same time;
    - the equality toggle restores the seed.
  - With the fix turned off in the test process, all 5 fail.
  - New smoke check "physics options restored", in the reset section and part of the
    Phase 1 gate. After the timestep, the gravity flag and the contact override were edited,
    seed 0 reached the reference digest `9ac3950e3c7ab848` (difference 0).
  - The reproduction above now gives `ee9f3f82a6a42cd6` after all three edits.
  - Live check in the real viewer, with the UI panels shown:
    - The options of the user's model were edited just before a sync, so the viewer copied
      them into its own model, as after a panel edit.
    - The next check put them back, and they kept the scene's values through all 159 later
      syncs, so the viewer did not copy the edit back.
    - There were no restores, and the final digest matched a headless replay
      (`89376baab70c1d47`).
  - With no edits, the smoke and viewer digests are unchanged (`9ac3950e3c7ab848` and
    `34cde6c9825fa1e4`).
- **Not tested:** nobody has clicked the Physics or Equality panel by hand. From Python,
  `Simulate.m` is the user's model, so the viewer's own copy could not be edited directly.
  The step that copies a panel edit into the user's model is taken from the 3.14.0 source.

## Notes for Phase 2

- **Fingertip clearance:**
  - With the hand pointing straight down and the gripper closed, the fingertip collision
    meshes reach 8.86 mm below the TCP.
  - Robot/table contact appears between a TCP height of 9.02 mm (no contact) and 8.50 mm
    (contact), at every position tested.
  - Pushing at the block's mid-height (TCP z 25 mm) leaves about 16 mm of clearance below
    the fingertips.
- **Camera:** when the hand pushes from the far side of the block, it may come between the
  block and the overhead camera. Use the oblique camera as well.
- **Hidden-physics ranges:** the mass and friction ranges are development ranges from
  BUILD_PLAN.md section 4. They are not validated operating ranges.
- **Physics options:** `reset()` and the viewer put back any change to the simulation
  model's options (D17). Controllers should not change them; inverse kinematics should use
  its own model instance.

## D18. Push skill on the controller's own kinematics model (Phase 2)

- **Problem:** BUILD_PLAN.md section 4 asks for a bounded push primitive executed through
  damped least-squares IK and joint-target control, with joint limits, velocity limits and
  retreat trajectories. D10 forbids the controller from using the runner's live model or
  data, and `diagnostics.solve_tcp_down_ik` ran on the runner's model.
- **Choice:**
  - `kinematics.ArmKinematics` loads its own `MjModel` from the scene MJCF and keeps its own
    `MjData`. It only writes arm joint angles and runs forward kinematics: no dynamics, no
    rollouts, no hidden values. `solve` is damped least squares (damping 0.02) on the
    6-D TCP-down pose error. `diagnostics.py` is unchanged.
  - `controller.PushAction` is the primitive: the block face to contact (approach side),
    the push direction in the table plane, the speed (0.04, 0.08 or 0.12 m/s) and the
    duration (0.5, 1.0 or 1.5 s). The stroke is speed x duration (`configs/push.toml`).
  - `PushSkill.execute` runs it in segments: rise to the transit height (TCP z 0.10 m),
    move above the pre-contact point, descend to the push height (TCP z 22 mm, fingertips
    about 13 mm above the table) 12 mm behind the face, stroke with a trapezoidal speed
    profile, dwell 0.1 s, back off 30 mm, and rise again.
  - Every control update solves IK, clamps the joint target 0.05 rad inside each joint's
    range, limits its change to 1.2 rad/s (joints 1-4) or 1.6 rad/s (joints 5-7), well
    below the datasheet limits, and sends it with the gripper closed (ctrl 0). The block
    is never moved directly.
  - `PushSkill.plan` filters every action before use: admissible speed and duration,
    direction within 20 degrees of the face normal, every Cartesian setpoint at least 50 mm
    inside the table outline and at least 15 mm above it, the geometrically predicted block
    end at least 30 mm inside the allowed region, and IK convergence along the whole path.
    The closed fingers are symmetric, so it tries the hand yaw and yaw + 180 degrees; joint
    7's range often allows only one.
  - If a setpoint leaves the bounds or IK fails during execution, the push is aborted and
    `recover` lifts the hand to the transit height.
- **Evidence:**
  - `test_kinematics.py` (9 tests): its own model instance (not the runner's), forward
    kinematics match the observed TCP, TCP-down poses over the push area are reached at
    five hand yaws (-90 to 90 degrees), unreachable targets are reported.
  - `test_push_controller.py` (14 tests): joint-speed limiting, joint-range clamping,
    each filter, a real push that moves the block through finger contact only, and the
    retreat.
  - Gate (before D25): largest IK position error 0.50 mm; largest measured joint speed
    0.709 of the datasheet limit; lowest measured TCP height 13.8 mm, although no command
    went below 22 mm; 120 control updates were rate-limited, none in violation.
  - The Phase 2 review added three tests to `test_kinematics.py` and one to
    `test_push_controller.py` (D25).
- **Observed** (before D25, not measured again after it): at the retreat the measured TCP
  lags the command by about 1 cm, and the block moved less than the geometric prediction;
  the baseline's next measurement corrects for it (D23).
- **Corrected in the Phase 2 review:** this entry first said that the servos "sag by about
  4 mm under contact". The sag was larger and came from gravity, with or without contact:
  up to 10.5 mm in a static hold, and 4.3-8.2 mm below the push height in every gate
  episode. The filters above check only the commanded setpoints, so the 15 mm floor did not
  hold for the hand the simulator moved. D25 fixes this.

## D19. Gate and development seeds fixed before any result

- **Problem:** the gate must not be tuned on, rerun or cherry-picked (BUILD_PLAN.md
  section 9).
- **Choice:** `configs/push.toml` lists the 20 gate seeds (20000-20019) and 30 separate
  development seeds (21000-21029). They were written before any Phase 2 episode ran. The
  gate needs 18 successes. `gate` runs the gate list by default; `gate --seeds development`
  runs the development list.
- **Evidence:** `push.toml` was last saved at 02:46 UTC; the first development run started
  at 02:51 and the gate at 03:05:38. The gate was run once (`docs/evidence/phase2/`).
  The two lists are disjoint (checked when `push.toml` is loaded) and do not overlap the
  seeds used by the Phase 1 checks (0-199).
- **Phase 2 review:** the review ran the gate twice more and kept both results
  (`docs/evidence/phase2/review/`):
  - at 15:27:46 UTC on the unchanged code, to check the reported result. Every record
    field except the trajectory path and the wall-clock time was identical, and so were
    all 20 trajectory files, byte for byte;
  - at 16:05:56 UTC, once, after the D25 fix: 20/20.
  - The D25 issue was found in the original gate's TCP statistics. The fix has no tuned
    parameter, and it was checked on probes and on the development seeds before that run.

## D20. Evaluator monitor hook in `ScenarioRunner.step`

- **Problem:** contacts, joint speeds and the success hold change within one 20 ms control
  update; checking only after each update can miss a short forbidden contact or limit
  violation.
- **Choice:** `step(control_updates, monitor=None)` calls `monitor(model, data)` after every
  physics step. Only the evaluator passes one (`EpisodeJudge`). The controller cannot: it
  sees only `EvaluatedPlant.command`, which returns an `Observation`.
- **Evidence:** with no monitor the stepping code path is unchanged: the Phase 1 smoke test
  still passes (73) with digest `9ac3950e3c7ab848`, and `step --seed 0 --seconds 8` still
  gives `34cde6c9825fa1e4`.

## D21. Episode rules: contacts, limits, success and the push budget

- **Choice** (all in `evaluation.EpisodeJudge`, evaluator side, read from MuJoCo's contact
  list after every physics step):
  - **Contact events:** every robot contact with penetration (`dist <= 0`) opens an event
    with its start, end, push index, phase, peak normal force and penetration. Robot-block
    contacts by the left or right finger are allowed. Any other robot body touching the
    block, any robot-table contact, any robot self-contact or contact with anything else is
    a forbidden collision and fails the episode.
  - **Limits:** a measured joint speed above the Panda datasheet limit, a joint more than
    0.01 rad outside its range, or a TCP below 10 mm fails with `limit_violation`. The
    controller's own limits (D18) are stricter, so these are safety nets. Corrected in the
    Phase 2 review: until D25, that held for the TCP height only on the commands. In the
    gate, the measured hand hung up to 8.2 mm below them, 3.8 mm above this limit.
  - **Success**, stricter than the section 4 wording: block centre within 20 mm and planar
    speed below 0.02 m/s continuously for 0.5 s, **counted only while no robot geom
    touches the block**. Otherwise a block pinned at rest by the fingers could count.
  - **Push budget:** the 9th declared push fails at once with `push_budget_exceeded`.
    Every robot-block contact event outside a declared push also counts as a push; if the
    total exceeds 8, a success is turned into `push_budget_exceeded`. This is conservative:
    a chattering contact can count more than once.
  - **Other failures:** `unstable`, `block_out_of_bounds`, `timeout` (30 s simulated,
    including approach and retreat), `push_budget_exhausted` and `no_feasible_push` (the
    controller stopped without success), `controller_error`. The first outcome is final.
- **Evidence:** `test_evaluation.py` (24 tests, with parameter cases) creates each
  outcome on purpose. Gate: 3315 contact events, all finger-block; 0 forbidden; 0
  unplanned. After D25: 3209, 0 and 0.
- **Observed:** the finger-contact flag switches on and off many times during a stroke,
  which is why there are thousands of events. Each is recorded as it happened.

## D22. CLI command-set test updated for Phase 2

- **Problem:** `test_cli.py` asserted that only the six Phase 1 commands exist, so adding
  commands would fail it.
- **Choice:** the test now allows exactly the Phase 1 commands plus `gate` and `demo`.
  The other commands proposed in BUILD_PLAN.md section 10 (`collect`, `train`, `evaluate`,
  `dashboard`) are still absent, and the test still fails if one appears early.
- **Evidence:** `test_only_phase1_and_phase2_commands_are_registered` passes.

## D23. Fixed-feedback baseline

- **Problem:** BUILD_PLAN.md section 6 requires a competent, fairly tuned feedback baseline
  that uses only the current block position and target error, with no learning.
- **Choice** (`baseline.FeedbackBaseline`):
  - Before each push, wait until the block is at rest (planar speed below 5 mm/s, at most
    1 s), then read the block pose and the target from the `Observation`.
  - Stop and hold the arm still once the block is within 12 mm of the target; stop when the
    8 pushes are used.
  - Candidates: each face whose inward normal is within 90 degrees of the block-to-target
    direction; the direction is that bearing clipped to 20 degrees of the face normal; all
    nine speed and duration pairs.
  - Choose the feasible candidate (D18 filters) whose geometric prediction of the block end
    is closest to the target, preferring slower pushes when equal (0.002 per m/s).
  - It never reads mass or friction and keeps nothing between episodes; the same
    observation always gives the same choice.
- **Tuning:** the parameters were set once from geometry (block size, clearance, tolerance)
  and were not changed after any result. All four development runs succeeded on every seed
  they ran (6/6, then 30/30 three times). Between them, only the evaluation rules were made
  stricter (D21) and code was changed. No development episode failed, so there was nothing
  to tune.
- **Evidence:** gate 20/20 (18 required). Pushes used: mean 1.95, at most 3 of 8. Simulated
  time: mean 8.58 s, at most 13.18 s of 30 s. Final distances 2.8-19.7 mm. All five used
  speed and duration pairs were chosen 3-10 times.
- **After the D25 fix** (no baseline parameter changed): gate 20/20 again. Pushes used: mean
  1.85, at most 3. Simulated time: mean 8.06 s, at most 12.20 s. Final distances
  0.9-18.5 mm, mean 11.3 mm (10.9 mm before); 8 episodes ended more than 15 mm from the
  target (5 before).
- **Limits of this result:** nominal physics only. The final distances of seeds 20002
  (19.7 mm), 20017 (19.2 mm) and 20019 (18.2 mm) are close to the 20 mm tolerance: the
  block came to rest inside 20 mm, but not inside the 12 mm stop distance, after the last
  push chosen. After D25 the largest was 18.5 mm (seed 20010). With other masses and
  frictions the geometric prediction will be worse; that is Phase 3's subject. The Phase 2
  review ran the baseline only as a check of D25 at two opposite corners of the development
  ranges (see D25 and the Notes for Phase 3).

## D24. Viewer demo and evidence frames

- **Choice:**
  - `demo --mode baseline --seed N` runs the baseline through the same `EvaluatedPlant` and
    judge as the gate, in the MuJoCo passive viewer, paced to real time (`--speed`). It
    shows the time, push number, phase, block-target distance, block speed and finger
    contact in the top-left, and the outcome at the end. This text uses MuJoCo's bitmap font,
    which the Mesa D3D12 driver can garble (README.md). It then reruns the seed headless and
    reports whether the outcome and state digest are identical. The viewer is inspection
    only and not part of the gate.
  - `demo --frames DIR` runs headless and renders the overhead and oblique cameras at the
    start, at the middle of each stroke and at the end.
- **Found:** `Handle.set_texts` in MuJoCo 3.14.0 takes the viewer lock itself; called inside
  `handle.lock()` it deadlocked. It is now called outside the lock.
- **Evidence:** frames for seed 20000 were rendered after the gate (03:06:36 UTC). Their
  episode matched the gate's exactly (success, 3 pushes, 12.18 s). In some frames the hand
  hides the block from the overhead camera (see Notes for Phase 2); the oblique view shows
  it.
- **After D25:**
  - The frames were rendered again into `docs/evidence/phase2/review/frames/` (files
    written at 16:08:42-16:08:46 UTC). The episode matched the post-fix gate (success,
    3 pushes, 12.20 s, final distance 6.3 mm). A change of a few millimetres in hand height
    is a pixel or two in these images, so they show the pushes, not the fix.
  - The demo window was run once more (output saved at 16:09:27 UTC). It closed itself
    22.3 s after opening, after the episode. There were 0 physics-option reverts, and the
    headless rerun gave the same outcome and state digest (`d1392b6bccc39de2`).
- **Not tested:** the demo window was run automatically (opened and closed with
  `--duration`) during development and in the review; nobody has watched it by hand.

## D25. Gravity feedforward on the servo targets (Phase 2 review)

- **Problem:**
  - The arm actuators are position servos without gravity compensation. At rest each one
    pushes `kp * (target - q)`, so with `target = q` the arm settles below the commanded
    pose, where the servo force balances the weight of the links.
  - D18's bounds (TCP at least 15 mm above the table, push height 22 mm) were checked only
    on the commanded setpoints. The hand the simulator moved was lower:
    - Static holds at 18 TCP-down poses (x 0.30-0.72 m, y within 0.25 m of the centre line,
      hand yaws 0, 90 and -45 degrees): 1.5-10.5 mm below the command, most at full reach.
    - Gate: in every episode the lowest measured TCP was 4.3-8.2 mm below the 22 mm push
      height (lowest 13.8 mm, seed 20015).
    - First push of development seeds 21000-21009: the fingers touched the block with the
      TCP at 14.5-19.0 mm.
  - The fingertips reach 8.9 mm below the TCP, and robot/table contact starts at a TCP
    height of 8.5-9.0 mm (Notes for Phase 2). At 13.8 mm the fingertips were about 5 mm
    above the table instead of the planned 13 mm, and the TCP was 3.8 mm above the
    evaluator's 10 mm limit (D21). No episode touched the table or failed, but the bounds
    described in D18 did not hold.
- **Choice:**
  - `ArmKinematics.gravity_offset(q)` computes the static joint torques from the robot's own
    links at `q` (MuJoCo's `mj_rne` at zero velocity, on the controller's private model from
    D18) and divides them by each servo's `kp`. The gains are read from the model, and
    `verify_actuators` checks that every arm actuator is a fixed-gain position servo.
  - `PushSkill._send` applies the joint-range and joint-speed limits to the kinematic command
    `q_cmd` as before, then sends `servo_target(q_cmd)`, which is
    `q_cmd + gravity_offset(q_cmd)` clipped to the joint ranges. `hold`, `execute` and
    `recover` all send through `_send`.
  - The controller still plans from `q_cmd`. The reset pose carries no offset, so before the
    first command `obs.arm_qpos_target` is still a kinematic command (`start_state`).
- **Why this is allowed:** it uses only the robot's link masses and servo gains from the
  controller's own copy of the scene model, as a real Franka controller does with its
  dynamics model. The block is not part of the arm's chain, so its hidden mass and friction
  do not enter, and no simulator state is read.
- **Not changed:** the scene, the servo gains, `configs/push.toml`, the baseline (D23), the
  evaluator (D21), and manual control in the viewer (D16), which sets `ctrl` directly.
- **Evidence** (`docs/evidence/phase2/review/`):
  - Static probe, same 18 poses: with the offset, the TCP height error was at most 0.27 mm
    and the position error at most 0.39 mm; without it, 1.5-10.5 mm low. At the home pose
    the offset equals the simulator's own bias force divided by `kp`, with zero difference.
    The largest offset at the probed poses was 0.0113 rad, well inside the 0.05 rad margin
    that `_send` keeps from each joint limit.
  - Contact probe, first push of development seeds 21000-21009: the TCP was at
    21.9-22.3 mm while the fingers touched the block, against 14.5-19.0 mm without the
    offset.
  - Tests:
    - Three new tests in `test_kinematics.py`: the offset holds the TCP where commanded; it
      matches the simulator's bias force and ignores the block; forward kinematics are
      unchanged.
    - One new test in `test_push_controller.py`: the executed TCP holds the push height.
    - Three controller tests were updated to expect `servo_target`.
    - With `servo_target` returning `q_cmd` unchanged, 3 tests fail. With
      `gravity_offset` returning zeros, 3 tests fail. With the fix, the full suite passes
      (184 tests).
  - Stress check, not a benchmark: the 30 development seeds at nominal physics and at two
    opposite corners of the development ranges (0.40 kg with friction 0.8, and 0.10 kg
    with friction 0.2), before and after the fix:
    - 90/90 successes both times;
    - lowest measured TCP 13.6 mm before, 21.8 mm after;
    - 0 robot/table contacts and 0 forbidden contacts both times.
  - Gate, run once after the fix (16:05:56 UTC): 20/20, lowest measured TCP 21.88 mm,
    largest joint-speed ratio 0.709, 0 forbidden contacts (D19, D23).
  - Phase 1 checks unchanged: smoke digest `9ac3950e3c7ab848` and step digest
    `34cde6c9825fa1e4`. Those paths do not use the controller.
- **Side effects:**
  - In the stress check, the first command after a reset moved the servo targets by up to
    0.27 of one rate-limit step (joint 2's home offset, 6.5 mrad). Later changes stayed
    within one step, to four decimal places.
  - The fingers now meet the block at the planned height, 2-8 mm higher than before. At
    0.40 kg and friction 0.8, the block's centre rose by up to 1.3 mm during an episode
    (0.9 mm before). If the block tips about a bottom edge, that is a tilt of about
    3 degrees.
  - The first push's plan was identical on all 20 gate seeds, but its execution was not,
    so later pushes differ. Two seeds (20003 and 20016) needed 2 pushes instead of 3, and
    the largest final distance was 18.5 mm instead of 19.7 mm. Accuracy did not improve
    overall: the mean final distance was 11.3 mm instead of 10.9 mm, and 8 episodes instead
    of 5 ended more than 15 mm from the target.
- **Not tested:** the other two corners of the development ranges (0.10 kg with friction
  0.8, 0.40 kg with 0.2); the retreat lag (D18) after the fix; real hardware.

## Notes for Phase 3

- The gate used nominal physics only (0.25 kg, friction 0.5). The controller never reads
  hidden values, but its geometric prediction assumes the block follows the pusher. With
  low friction the block may slide past; with high mass and friction it may move less.
- Several gate episodes ended 18-20 mm from the target. A Phase 3 comparison should use
  the same success rule and report final distances, not only success counts.
- Contact events chatter during strokes (D21). Use them to measure contact time and force,
  not as a count of pushes.
- Added in the Phase 2 review:
  - **Ceiling risk:** the baseline succeeded in all 20 gate episodes and all 90 stress
    episodes (D25), including two opposite corners of the development ranges, before and
    after the fix. That was 30 development seeds and 2 of the 4 corners, not a benchmark.
    If the baseline is near perfect on the Phase 3 cohorts as well, BUILD_PLAN.md section 6
    requires reporting the ceiling; any revised benchmark must be declared and followed by
    a fresh, untouched test set.
  - **Statistics over all episodes:** the `gate` summary line reports mean pushes and
    simulated time over successful episodes only (`evaluation.py`). Section 6 asks for
    both, across all episodes and conditional on success. Every gate episode succeeded, so
    the two are equal here; Phase 3 reports must give both.
  - **Learner-facing interface:** `EvaluatedPlant` keeps the runner in the attribute
    `_runner`, and Python does not enforce privacy, so code holding the plant can reach
    `hidden_truth()`. The Phase 3 gate (no hidden physical values or scenario labels in the
    learner-facing interface) needs a boundary a learner cannot reach through attributes,
    or a test that shows what the learner receives.
  - **Block tilt:** since D25 the fingers push at the planned height. At 0.40 kg and
    friction 0.8 the block's centre rose by up to 1.3 mm. Effective-contact checks should
    measure tilt and lift rather than assume the block only slides.

## D26. Benchmark manifests, cohorts and splits (Phase 3)

- **Problem:** BUILD_PLAN.md section 6 needs training, validation, final-test and retention
  sets that differ in physical conditions and initial states, not only in seeds, with an
  in-range and a harder out-of-range cohort, and with all budgets, success rules and metrics
  in versioned configuration.
- **Choice:**
  - `configs/benchmark.toml` (`schema_version = 1`, `benchmark_version = "phase3-v1"`)
    defines the rules, cohorts, split sizes, manifest generation, sealing, metrics,
    statistics, replay tolerance and feasibility checks. Its `[rules]` copy of the budgets
    and success rule must equal `push.toml` `[episode]` (schema 1) and `scene.toml`
    `target_tolerance`, and its in-range cohort must equal `scene.toml` `[physics]`; the
    loader refuses to start otherwise (`benchmark_config.py`). Every result records the
    benchmark version, the config SHA-256, the metrics version, the manifest SHA-256, the
    push-config schema version and the episode-schema version.
  - A physical condition is a block mass and a sliding friction. In-range conditions are
    drawn by Latin-hypercube sampling of 0.10-0.40 kg and 0.20-0.80. Out-of-range
    conditions take one parameter from one of four outer bands (mass 0.05-0.08 or
    0.42-0.60 kg, friction 0.10-0.16 or 0.84-1.00), cycled so all four sides appear, and the
    other parameter from the whole outer box (0.05-0.60 kg, 0.10-1.00). Values are rounded
    to 4 decimals.
  - Each condition has 5 initial layouts (block pose and target), with block-to-target
    directions stratified over 8 sectors so one condition's layouts use 5 different sectors.

    | Split | Conditions (in / out) | Scenarios (in / out) | Sealed |
    |---|---|---|---|
    | train | 40 / 0 | 200 / 0 | no |
    | validation | 12 / 8 | 60 / 40 | no |
    | test (final) | 14 / 6 | 70 / 30 | yes |
    | retention | 10 / 0 (train conditions) | 50 / 0 | yes |

  - Train, validation and test share no condition: every condition is at least 0.02 from
    every other in parameter space normalised by the in-range widths (0.30 kg, 0.60).
    Retention reuses 10 train conditions by design (section 6: "earlier training regimes")
    with new layouts. No layout appears in two splits: two layouts count as the same unless
    the block or target centres differ by at least 5 mm or the yaw (modulo 90 degrees) by at
    least 5 degrees.
  - The test split is 100 scenarios over 20 conditions and all 8 goal sectors; it is
    "paired" because the baseline and the frozen candidate will run exactly the same
    scenarios.
  - Scenario and condition ids are random hexadecimal strings, the scenario order inside a
    manifest is shuffled, and the file names are the split names. Manifests are
    evaluator-side files; the agent never receives an id, a file name, an order or a colour
    (there is one block colour for every condition).
  - Generation is deterministic from `generator_seed = 3141593`. Reset seeds start at 31000,
    disjoint from every earlier seed (0-199, 20000-20019, 21000-21029, feasibility
    30000-30002). The layout and physics are explicit in the manifest, so the seed only
    labels the reset. Manifests are never overwritten (`manifests` refuses), and
    `configs/manifests/hashes.json` records each file's SHA-256 with the software versions,
    asset revision and simulation settings.
    `.gitattributes` marks the manifests and `benchmark.toml` as binary, so a future git
    checkout cannot change their bytes (as D15 does for the Panda files): `open_split`
    refuses a manifest whose recorded benchmark-config hash differs from the current file.
  - The out-of-range box was proposed as a candidate, checked with `contact-check`
    (D29-D31 criteria, declared in `[feasibility]` before the run), and fixed unchanged at
    2026-10-03T17:31:59Z. The manifests were generated at 17:38:32Z and the first
    benchmark episode ran at 18:00:03Z. Baseline success was not a criterion.
- **Evidence:**
  - `contact-check` run 1 (started 2026-10-03T17:29:22Z): readback 17/17, kick 17/17,
    hold 9/9, pushes 9/9; feasible box mass 0.05-0.60 kg, friction 0.10-1.00 (the whole
    candidate box).
  - Hashes (`configs/manifests/hashes.json`): train `6eadee60c6ae2a2f…`, validation
    `88dabaa6cb811ea4…`, test `5673ce2e3763c041…`, retention `e8c5857a7d7151a1…` (full
    values in `docs/evidence/phase3/README.md`).
  - `tests/test_manifest.py` (16 tests): sizes, cohorts, labels match physics, disjointness
    in physics and initial states (101,025 layout pairs compared), retention layouts held
    out, test coverage of conditions and sectors, new seeds, ids and order uncorrelated
    with the condition, deterministic regeneration, provenance, tamper and overwrite
    refusal.
- **Limits:** every out-of-range condition still lies inside the box the feasibility check
  accepted. The benchmark is harder only through the physics; layouts, budgets and the
  success rule are unchanged.

## D27. Sealing the final test and retention splits

- **Problem:** section 6 requires final-test outcomes to stay sealed until a candidate is
  frozen (Phase 5). Read conservatively, this includes the baseline, and retention is
  treated the same way.
- **Choice:**
  - `manifest.open_split` is the only way an evaluation loads a manifest. For a sealed split
    it raises `SealedSplitError` unless `freezes/` holds a freeze record that names a
    candidate other than "baseline", whose candidate file still has the recorded SHA-256,
    and that commits to the split's current manifest SHA-256.
  - `manifest.freeze_candidate` writes such a record. It refuses the baseline and never
    overwrites a record. No CLI command calls it; that belongs to Phase 5.
  - `evaluate --split test` and `evaluate --split retention` exit with status 2 and the
    message "REFUSED" before creating any output. `replay` refuses runs of sealed splits.
  - `manifests` without `--verify` refuses (exit 2, "REFUSED") when `hashes.json` already
    exists, so the frozen manifests cannot be regenerated by accident. A first version
    refused with a raw traceback; this was found in the final checks and made a clean
    refusal, with a test.
  - Sealed manifests are read only for structural checks (`read_sealed_for_audit`):
    hashes, counts, disjointness. No episode has run on them.
- **What sealing does not do:** it is a procedural guard in the evaluator, not access
  control. The manifest files are readable, and anyone can write a file into `freezes/`.
  Phase 4 code must not read `configs/manifests/test.json` or `retention.json`; a test for
  that should come with the learner.
- **Evidence:** `tests/test_sealing.py` (17 tests): no freeze record exists in the
  project; the real test and retention splits refuse to open, through the API and the CLI;
  refusal happens before any file is written; the baseline cannot be frozen; a freeze
  opens a scratch sealed split only for that candidate and that manifest hash; a changed
  candidate or a changed manifest invalidates the freeze; replay of a sealed split is
  refused; `manifests` refuses to regenerate and leaves every manifest byte unchanged.
  `evaluate --split test` and `--split retention` were each run once in the final checks
  and refused (exit 2).

## D28. Learner-facing interface in a separate process

- **Problem:** the Phase 2 review found that code holding `EvaluatedPlant` can reach
  `hidden_truth()` through `_runner` (Notes for Phase 3). The Phase 3 gate requires an
  interface that exposes no hidden physical values or scenario labels, with tests.
- **Choice:**
  - The agent runs in its own operating-system process
    (`python -m physical_ai_lab.agent_process --agent baseline`) and exchanges JSON lines
    with the evaluator over standard input and output (`learner_interface.py`, protocol
    version 1). No Python object crosses.
  - To the agent: `episode_start` (episode ordinal within the process, control period,
    budget, first observation), `observation` (the 18 `Observation` fields after the last
    command), `ack`, `episode_over` (no outcome or reason), `shutdown`. From the agent:
    `ready`, `command` (7 joint targets and the gripper input), `note` (phases, decisions,
    push start and end, `probe: true` to declare a probe push), `stopped`, `error`. Every
    message is checked against an allow-list in both directions; unknown types or keys are
    rejected.
  - The evaluator side (`benchmark.RemoteAgentPolicy`) relays commands to
    `EvaluatedPlant`, which stays evaluator-only. `TrafficScan` checks the text of every
    message to the agent for the scenario's ids, split, cohort and band labels and for the
    exact block mass, friction, inertia and reset seed.
  - The fixed baseline runs through this same interface, so the baseline and the Phase 4
    learner get identical observations, command rate and budgets. The baseline code is
    unchanged; `RemotePlant` gives it the same methods it used in Phase 2.
  - The agent process imports no scenario runner, manifest, benchmark, evaluation or
    instrumentation module. `kinematics.py` imports `scenario.verify_actuators`, a static
    check of the agent's own nominal model; it gives no access to the evaluator's
    simulation, which lives in the other process.
- **Evidence:** `tests/test_learner_interface.py` (23 tests): message keys and record keys
  contain no hidden name; extra keys and unknown types are rejected; only plain JSON values
  cross; the agent-side plant holds no simulator, runner, model or data object; the agent
  process loads none of the evaluator modules (checked in a fresh interpreter); agent-side
  source never names a hidden accessor; the traffic scan catches injected values; a real
  episode through the subprocess crosses cleanly; learner views hold no hidden key,
  identifier or value; observations are never later than the last command; transitions
  carry nothing hidden; baseline and learner budgets are the same. In the official runs the
  scan checked every message to the agent with 0 hits (see D32).

## D29. Episode/event schema and contact instrumentation

- **Problem:** Phase 3 needs a versioned episode and event schema (pushes and their
  parameters, contacts, probe actions, limit checks, failures, outcome) that keeps what the
  learner may see apart from evaluator-only data, and contact measures in every episode
  (Notes for Phase 3: tilt and lift, contact chatter).
- **Choice:**
  - `episode_schema.py`, `EPISODE_SCHEMA_VERSION = 1`. An episode record has a `learner`
    part (only what crossed the interface: notes, declared pushes and probes, first and last
    observation, how the agent's side ended), an `evaluator` part (ids, split, cohort,
    seed, requested and read-back physics, layout, outcome, judged pushes, contact events,
    limit checks, contact metrics, event list, counts, compute time, replay digests, traffic
    scan) and a `benchmark` part (versions and hashes). Every record is validated before it
    is written; `learner_view` returns only the learner part and `validate_learner_view`
    rejects hidden keys or evaluator identifiers inside it.
  - Counts per episode: failures, limit violations, forbidden-collision outcomes, forbidden
    contact events, aborted pushes, probe pushes, push-budget overruns.
  - Transitions (observation, action, next observation) are built from interface traffic
    only, for Phase 4.
  - `instrumentation.ContactMetrics` reads the simulation after every physics step and
    never writes it: robot/block contact time, number of contact intervals (chatter, D21;
    never a push count), largest and mean summed normal force; block/table normal force as
    a ratio to the hidden weight (minimum and maximum after a 0.2 s settle, and mean over
    settled steps without robot contact); time with no block/table contact; block lift
    above its placement height; block tilt from lying flat, from the body orientation.
- **Change made before the official runs:** the first version reported the table normal
  ratio "at start", measured in the first physics step after placement. It was a transient
  (0.0 in one scratch record, 0.973 in another), so it was replaced by the settled mean
  ratio without robot contact, and the minimum and maximum now skip the first 0.2 s
  (`SETTLE_SECONDS`). Contact-check run 2 (17:58:49Z) repeated the checks with this
  definition; every other value matched run 1 exactly. Run 1 remains the evidence that
  fixed the cohort.
- **Finding:** the block hops slightly during pushes. In a diagnostic episode (feasibility
  seed 30000, nominal physics), the block had no table contact in 70 physics steps, in
  runs of 1-3 steps (2-6 ms), 64 of them while a finger touched it, with the centre
  0.01-0.10 mm above placement. At rest the soft contact sinks 0.108 mm (the same at every
  tested mass), so a rise of about 0.1 mm clears it. Landings give table-force spikes of up
  to 14 times the weight in the train run. This is simulator behaviour, measured, not a
  logging error; the settled mean ratio without robot contact stayed within 0.98-1.03.
- **Evidence:** `tests/test_episode_schema.py` (24 tests) and `tests/test_benchmark.py`
  (14 tests), including 18 invalid mutations that the validator rejects, every failure
  reason, and plausible contact metrics on a scratch episode.

## D30. CLI command set updated for Phase 3

- **Problem:** `test_cli.py` allowed exactly the Phase 1 and 2 commands (D22).
- **Choice:** Phase 3 adds exactly four commands, and the test now allows exactly these
  plus the earlier eight:
  - `contact-check [--out DIR]`: the effective-contact and feasibility checks;
  - `manifests [--verify]`: generate the manifests once (refuses to overwrite) or verify
    them against `hashes.json`;
  - `evaluate --split {train,validation,test,retention} [--mode baseline] [--limit N]
    [--out DIR]`: the section 10 name; `--mode learner` is rejected until Phase 4, and
    sealed splits are refused (D27);
  - `replay --run DIR [--count N] [--out FILE]`: the determinism check (D31).
  `collect`, `train`, `dashboard` and a freeze command are still absent, and the test
  fails if one appears early.
- **Evidence:** `test_only_phase1_to_phase3_commands_are_registered` and six new
  usage-error cases in `test_invalid_arguments_are_usage_errors` pass (`test_cli.py`, 20
  tests).

## D31. Deterministic replay and its tolerance

- **Problem:** Phase 3 requires that rerunning a manifest scenario, and replaying an
  episode's logged commands, reproduce the trajectory within documented tolerances.
- **Choice:**
  - Every episode stores its actuator inputs after each control update and a 64-bit digest
    of the full integration state (`mj_getState`, `mjSTATE_INTEGRATION`) after each update,
    in `commands/<n>.npz`, plus the initial and final state digests in the record.
  - `replay --run DIR` checks each episode two ways: command replay (reset the scenario
    from the manifest, apply the logged inputs, no agent, no judge, no monitors) and rerun
    (the same scenario through a fresh agent process and the full evaluator). Both must
    match every logged digest; the rerun must also give identical inputs, outcome, pushes,
    final distance, contact metrics, limit checks and learner record.
  - Tolerance (`benchmark.toml` `[replay] state_abs_tolerance = 0.0`): bitwise identity on
    this machine and MuJoCo build. A digest cannot express a non-zero tolerance; if a
    future machine needs one, the replay must compare states, and the change needs a new
    decision.
  - `scenarios_per_split = 2` is the configured minimum; the official checks used more
    (below).
- **Evidence:** replays of the official runs, 2026-10-03: train episodes 1-10 (started
  18:11:11Z) and validation episodes 1-10 (started 18:11:42Z), all identical both ways;
  then all 200 train episodes (18:12:15Z to 18:19:29Z) and all 100 validation episodes
  (18:19:30Z to 18:23:07Z), 300 of 300 identical both ways, at every control update
  (`replay_all.json`; `docs/evidence/phase3/README.md`). `tests/test_benchmark.py` also checks a scratch
  episode and detects a deliberately altered input.
- **Not tested:** other machines, CPU architectures, operating systems, Python or NumPy
  builds and MuJoCo versions. The machine runs x64 Python under emulation on ARM64; native
  ARM64 or x86 runs may differ in the last bits and would fail a bitwise check.

## D32. Baseline statistics and the ceiling

- **Problem:** section 6 asks for success rates with uncertainty that treats episodes
  under the same physical condition as clustered, statistics over all episodes and over
  successes, and an explicit ceiling statement when the baseline is near perfect.
- **Choice:**
  - Success-rate intervals: a percentile cluster bootstrap that resamples physical
    conditions (10,000 resamples, seed 2718281), and a conservative Wilson score interval
    with n equal to the number of conditions. When every episode succeeds, the bootstrap
    is degenerate ([100, 100]); the Wilson interval is then the honest one.
  - Final distance over all episodes (mean, median, 90th percentile, maximum, count above
    15 mm); pushes and simulated time over all episodes and over successes; failures by
    reason; limit, forbidden-contact, aborted-push and probe counts; contact measures;
    wall time split into the evaluator and waiting for the agent.
  - A report prints "CEILING" when every episode succeeded and "NEAR CEILING" at 95% or
    more. Every episode is kept: nothing is hidden, rerun or dropped.
  - Manifests, cohorts and metric definitions were fixed (17:38:32Z) before the first
    benchmark episode (18:00:03Z).
- **Evidence (2026-10-03, fixed baseline, unsealed splits only):**

  | Split / cohort | Successes | 95% CI, Wilson n = conditions | Final distance mean / max (mm) | Pushes mean (max) | Sim time mean (max) |
  |---|---|---|---|---|---|
  | train, in-range (40 conditions) | 200/200 | 91.2-100% | 10.13 / 19.97 | 1.72 (3) | 7.59 s (13.66) |
  | validation, in-range (12) | 60/60 | 75.8-100% | 10.81 / 16.99 | 1.70 (3) | 7.60 s (13.10) |
  | validation, out-of-range (8) | 40/40 | 67.6-100% | 10.22 / 18.45 | 1.82 (3) | 7.92 s (13.12) |

  No failures, no limit violations, no forbidden contacts, no probe pushes; one aborted
  push in train (episode 177: the approach IK did not converge; the episode still
  succeeded). All-episode and success-only means are equal because every episode
  succeeded.
- **Conclusion:** the baseline is at the ceiling on success rate in both cohorts. A
  learner cannot beat it on success rate on these splits; that is a property of the
  benchmark, not a success claim for the baseline or for Phase 3. The benchmark was not
  made harder after these results. Comparisons that can still separate agents are final
  distance, pushes and time.

## Answers to the Notes for Phase 3

- **Nominal physics only:** the train split covers 40 in-range conditions (mass
  0.10-0.40 kg, friction 0.20-0.80) and validation adds 8 out-of-range conditions over
  all four outer bands. The baseline succeeded in all 300 episodes (D32). The friction-high
  band ended furthest from the target (mean 12.86 mm), the mass-low band closest
  (7.97 mm), from 10 episodes each.
- **Final distances:** every report gives final distances over all episodes (D32). In
  train, 40 of 200 episodes ended more than 15 mm from the target and the largest was
  19.97 mm, just inside the 20 mm tolerance.
- **Contact chatter:** contact events are counted as intervals and used for contact time
  and force only (D29). The push count still comes from declared pushes plus contacts
  outside a push (D21).
- **Ceiling risk:** confirmed and reported (D32). No revised benchmark was declared.
- **Statistics over all episodes:** reported over all episodes and over successes (D32).
  The Phase 2 `gate` summary is unchanged.
- **Learner-facing interface:** closed by a separate agent process with JSON messages
  (D28). Nothing that holds the simulator crosses the boundary.
- **Block tilt:** measured from the block orientation in every episode (D29). Largest
  values: train 1.26 mm lift and 2.99 degrees tilt; validation out-of-range 1.49 mm and
  3.40 degrees; contact-check at 0.60 kg and friction 1.0: 1.46 mm and 3.34 degrees. The
  block tips slightly under the push and settles back.

## Notes for Phase 4

- The success rate has no headroom on these splits (D32). Plan the learner's evaluation
  around final distance, pushes and simulated time, with the same clustered statistics,
  and say so before any learner result exists.
- The test and retention splits stay sealed. Phase 4 code must not read
  `configs/manifests/test.json` or `retention.json`, and the learner must use the
  interface of D28 only: `--mode learner` in `evaluate` still needs to be implemented.
- Transitions for learning come from interface traffic (`episode_schema` transitions),
  never from evaluator records.
- If the benchmark is revised to restore headroom, declare it as a new benchmark version
  with a fresh generator seed and new, untouched test and retention splits, before any
  learner result is seen.
