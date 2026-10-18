# Progress

| Phase (BUILD_PLAN.md section 7) | Status |
| --- | --- |
| 1. Reproducible simulation | **Complete.** Gate passed on 2026-10-02, and again on 2026-10-03 after the Phase 1 review |
| 2. Stable pushing and baseline | **Complete.** Gate passed on 2026-10-03: 20/20 nominal episodes (18 required), and 20/20 again after the fix from the Phase 2 review (D25) |
| 3. Benchmark and instrumentation | **Gate passed on 2026-10-03; awaits the post-phase review.** Manifests written and sealed, baseline report on train and validation: 300/300 successes, a ceiling on success rate |
| 4. Learning and persistent memory | Not started |
| 5. Promotion, regression and ablations | Not started |
| 6. Demonstration | Not started |
| 7. Optional high-level language agent | Not started |

To open the scene, run this from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m physical_ai_lab view --seed 0
```

To move the arm, click the 3D view, then use the arrow keys: Up/Down select a joint and
Left/Right turn it. [README.md](../README.md) covers setup and all the viewer controls.
The reasons behind the design choices are in [DECISIONS.md](DECISIONS.md).

To watch the Phase 2 baseline push the block, and to rerun the Phase 2 gate:

```powershell
.\.venv\Scripts\python.exe -m physical_ai_lab demo --mode baseline --seed 20000
.\.venv\Scripts\python.exe -m physical_ai_lab gate
```

To check the Phase 3 benchmark and rerun its baseline report (about 7 min for train and
3.5 min for validation; each run writes a new folder under `runs/phase3/evaluate/`):

```powershell
.\.venv\Scripts\python.exe -m physical_ai_lab manifests --verify
.\.venv\Scripts\python.exe -m physical_ai_lab contact-check
.\.venv\Scripts\python.exe -m physical_ai_lab evaluate --split train --mode baseline
.\.venv\Scripts\python.exe -m physical_ai_lab evaluate --split validation --mode baseline
.\.venv\Scripts\python.exe -m physical_ai_lab replay --run <run folder printed by evaluate> --count 10
```

## Phase 3 gate

The gate in BUILD_PLAN.md: *"Baseline report and manifests exist; the learner-facing
interface exposes no hidden physical values or scenario labels."*
The evidence files are in [docs/evidence/phase3/](evidence/phase3/README.md).

Order of work on 2026-10-03 (UTC): contact checks run 1 at 17:29:23; out-of-range cohort
fixed at 17:31:59; manifests written at 17:38:33; contact checks run 2 (revised table-force
measure, D29) at 17:58:49; first benchmark episode at 18:00:03. Nothing about the
manifests, cohorts, budgets, success rule or metric definitions changed after that.

| Gate item | Result | Evidence |
| --- | --- | --- |
| Manifests exist | **Yes.** Train 200 scenarios (40 conditions), validation 100 (20), final test 100 (20), retention 50 (10 train conditions, new layouts). Each is versioned (`phase3-v1`) and hashed; provenance records the software versions, Panda asset revision, seeds and simulation settings. `manifests --verify` passes | `configs/manifests/`, `hashes.json`; D26 |
| Disjoint in physics and initial states, not only seeds | **Yes.** No condition is shared between train, validation and test; retention reuses train conditions by design, with new layouts. 101,025 layout pairs compared, none closer than 5 mm or 5 degrees. Reset seeds from 31000, disjoint from every earlier seed | `tests/test_manifest.py`; D26 |
| In-range and harder out-of-range cohorts, reported separately | **Yes.** Out-of-range: mass 0.05-0.08 or 0.42-0.60 kg, or friction 0.10-0.16 or 0.84-1.00, fixed from contact and feasibility checks before any benchmark episode | `contact_checks/run1`; D26 |
| Final test sealed | **Yes.** No outcome exists on the test or retention split, for any agent, including the baseline. `evaluate --split test` exits 2 with "REFUSED" before writing anything; only a recorded candidate freeze (Phase 5) can open it | `tests/test_sealing.py`, final checks below; D27 |
| Baseline report exists | **Yes**, on train and validation, per cohort (next table). It is a ceiling: 300/300 successes | `evaluate/train`, `evaluate/validation`; D32 |
| Learner-facing interface exposes no hidden values or labels | **Yes.** The agent runs in a separate process and receives only JSON messages with allow-listed keys: observations, budgets, the control period and acknowledgements. In the two official runs a scan of all 119,560 messages sent to the agent found none of the scenario's ids, split, cohort or band labels, block mass, friction, inertia or reset seed. 23 interface tests and 24 schema tests pass | `tests/test_learner_interface.py`, `tests/test_episode_schema.py`; D28, D29 |
| Hidden randomization drives real physics | **Yes.** Mass and friction come from the manifest through the evaluator only. Read-back in every episode; contact checks for mass alone, friction alone, then both, including the out-of-range corners (next section) | `contact_checks/run1`, `run2`; D29 |
| Deterministic replay | **Yes, on this machine.** Command replay and a full rerun through a new agent process reproduced every logged state digest bitwise in all 200 train and all 100 validation episodes (300/300), checked after every control update, 18:12:15 to 18:23:07 | `replay*.json`; D31 |

**Gate result: PASS.** The gate asks for a report and manifests to exist and for a clean
interface; it does not ask the baseline to do anything. The baseline's perfect score
means the benchmark has no headroom on success rate (see "Baseline report").

### Baseline report (fixed baseline, unsealed splits only)

Run once each; no episode was rerun, dropped or hidden.

| | Train, in-range | Validation, in-range | Validation, out-of-range |
| --- | --- | --- | --- |
| Started (UTC) | 18:00:03 | 18:07:37 (one run for both cohorts) | same run |
| Conditions / episodes | 40 / 200 | 12 / 60 | 8 / 40 |
| Successes | **200/200** | **60/60** | **40/40** |
| 95% CI, cluster bootstrap over conditions | 100-100% | 100-100% | 100-100% |
| 95% CI, Wilson with n = conditions | 91.2-100% | 75.8-100% | 67.6-100% |
| Final distance mean / median / p90 / max (mm) | 10.13 / 10.23 / 17.16 / 19.97 | 10.81 / 11.55 / 16.08 / 16.99 | 10.22 / 10.00 / 16.42 / 18.45 |
| Episodes ending more than 15 mm away | 40 | 11 | 7 |
| Pushes, mean (max), all episodes = successes | 1.72 (3) | 1.70 (3) | 1.82 (3) |
| Simulated time, mean (max), all = successes | 7.59 s (13.66) | 7.60 s (13.10) | 7.92 s (13.12) |
| Failures by reason | none | none | none |
| Limit violations / forbidden contacts / probe pushes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Aborted pushes | 1 (episode 177, IK did not converge on the approach; the episode succeeded) | 0 | 0 |
| Robot/block contact time, mean | 1.76 s | 1.75 s | 1.84 s |
| Largest finger normal force | 41.4 N | 28.0 N | 42.7 N |
| Largest block lift / tilt | 1.26 mm / 2.99 deg | 0.83 mm / 1.28 deg | 1.49 mm / 3.40 deg |
| Wall time per episode (waiting for the agent) | 2.02 s (1.30) | 1.99 s (1.28) | 2.12 s (1.36) |

Out-of-range bands in validation (10 episodes, 2 conditions each, all successes): mean
final distance 12.86 mm with high friction, 9.23 mm with low friction, 10.81 mm with high
mass and 7.97 mm with low mass.

**Ceiling.** The fixed baseline succeeded in all 300 episodes, in both cohorts. On these
splits no learner can beat it on success rate. This is a property of the benchmark, not a
success claim. The benchmark was not made harder after seeing the results. Final distance,
pushes and time still vary and can separate agents (D32, "Notes for Phase 4").

### Effective-contact checks

`contact-check`, run 1 at 17:29:23 (69 s), which fixed the cohort, and run 2 at 17:58:49
(65 s) with the revised table-force measure; every shared value was identical. Groups:
mass alone (0.05, 0.10, 0.25, 0.40, 0.60 kg at friction 0.5), friction alone (0.1, 0.2,
0.5, 0.8, 1.0 at 0.25 kg), then both together at the four in-range and four out-of-range
corners.

| Check | Result |
| --- | --- |
| Read-back of mass, inertia and the block/table pair friction | 17/17. Inertia equals mass x edge^2 / 6 within 2.2e-16 (relative); every block/table contact uses the pair friction |
| Resting normal force against m x g | 17/17, relative error 1.2e-9 (tolerance 0.01). Rest height 0.108 mm below placement at every mass; rest tilt below 1e-6 deg |
| Kick test: measured effective friction against applied | 17/17. Worst relative error 0.016 at friction 0.5, 0.003 at 0.1 and 0.2, 0.010 at 0.8 and 1.0, the same at every mass (tolerance 0.05) |
| 30 s hold at the corners and nominal | 9/9 stable, no block drift |
| Baseline pushes, 3 feasibility seeds x 9 conditions | 27/27 stable successes, 2 pushes each. Largest lift 1.46 mm and tilt 3.34 deg at 0.60 kg and friction 1.0 (limits 5 mm, 10 deg); at low friction at most 0.06 mm and 0.36 deg |

Every episode of the benchmark also records contact time, contact intervals, finger and
table normal forces, time without table contact, lift and tilt (D29). The table force's
instantaneous minimum is 0 and its maximum up to 19 times the weight because the block
hops for 2-6 ms at a time while being pushed; the settled mean without robot contact stays
within 0.98-1.03 of the weight.

## Phase 3 deliverables

| Requirement | Where |
| --- | --- |
| Versioned benchmark configuration: budgets, success rule, cohorts, splits, metrics, statistics, replay tolerance, feasibility criteria | `configs/benchmark.toml` (`schema_version = 1`, `phase3-v1`), loaded and cross-checked against `push.toml` and `scene.toml` by `benchmark_config.py` |
| Training, validation, final-test and retention manifests | `configs/manifests/*.json`, `hashes.json`; `manifest.py`; command `manifests` (`--verify`) |
| Sealing and candidate freeze | `manifest.open_split`, `manifest.freeze_candidate`; no freeze exists (D27) |
| Hidden randomization and effective-contact checks | `contact_checks.py`; command `contact-check` |
| Learner-facing interface | `learner_interface.py` (protocol), `agent_process.py` (agent side); `benchmark.RemoteAgentPolicy` (evaluator side) (D28) |
| Episode/event schema | `episode_schema.py` (`EPISODE_SCHEMA_VERSION = 1`), with learner-only views and transitions (D29) |
| Contact instrumentation | `instrumentation.py` (`ContactMetrics`) |
| Baseline evaluation and report | `benchmark.py`; command `evaluate --split {train,validation} --mode baseline` |
| Deterministic replay | `benchmark.replay_run`; command `replay --run DIR` (D31) |
| CLI | Four new commands, `test_cli.py` updated (D30) |
| Tests | 100 new tests: `test_manifest.py` (16), `test_sealing.py` (17), `test_learner_interface.py` (23), `test_episode_schema.py` (24), `test_benchmark.py` (14) and 6 new cases in `test_cli.py`. The suite has 284 tests in 21 files |
| Evidence | [docs/evidence/phase3/](evidence/phase3/README.md) |
| Unchanged | The scene MJCF, `configs/scene.toml`, `configs/push.toml`, the gate and development seeds, `baseline.py`, the success rule, both Phase 1 reference digests, and the Phase 1 and 2 evidence |

No dependency was added in Phase 3.

## Phase 3 final check run

All on 2026-10-03, after the last code change (18:21 UTC) unless noted. Outputs are in
`docs/evidence/phase3/final_checks/`.

| Check | Command | Started (UTC) | Result |
| --- | --- | --- | --- |
| Environment | `doctor` | 18:26:14 | All required checks passed; 2 warnings (no git repository, OneDrive) |
| Tests | `pytest -q` | 18:23:28 | **284 passed** in 34.5 s (184 from Phases 1 and 2, 100 new) |
| Phase 1 smoke gate | `smoke` | 18:24:09 | **PASS 73**, FAIL 0, WARN 2; reference digest `9ac3950e3c7ab848` unchanged |
| Reference rollout | `step --seed 0 --seconds 8` | 18:26:16 | Digest `34cde6c9825fa1e4` unchanged |
| Phase 2 gate | `gate` | 18:24:20 | **20/20, PASS.** All 20 trajectory files byte-identical to the Phase 2 review's gate; the result file differs only in times, paths and the command line (`gate_compare.txt`) |
| Manifests | `manifests --verify` | 18:26:20 | Hashes match; disjointness passed (101,025 initial-state pairs) |
| No regeneration | `manifests` | 18:26:19 | Exit 2, REFUSED; manifests unchanged. (The first attempt, at 18:17:27, also refused and wrote nothing but printed a traceback; fixed, see problem 6) |
| Sealed final test | `evaluate --split test --mode baseline` | 18:26:18 | Exit 2, REFUSED; nothing written |
| Sealed retention | `evaluate --split retention --mode baseline` | 18:26:18 | Exit 2, REFUSED; nothing written |
| Full replay | `replay --run <train run> --count 200`, then the validation run with `--count 100` | 18:12:14 (before the last code change, which touched only the `manifests` command) | 300/300 identical, both ways, at every control update |
| Lock file | `uv lock --check --offline` | 18:26:21 | Exit 0 |
| Earlier evidence | file times | after the run | No file in `docs/evidence/phase1/` or `phase2/` was written after 16:41:47 UTC, before Phase 3's first file (17:16:38 UTC) |

No `freezes/` folder exists, and no output folder exists for the test or retention split.

## Problems found and fixed in Phase 3

1. **The learner could reach hidden truth through `EvaluatedPlant._runner`** (Phase 2
   review). The agent now runs in its own process and receives JSON only (D28).
2. **The first table-force measure was a transient.** "Ratio at start" was read in the
   first physics step after placement and gave 0.0 or 0.973. It was replaced, before the
   official runs, by a settled measure; the contact checks were rerun (D29).
3. **A docstring claimed the block sank more under a heavier block.** Both contact-check
   runs measured the same 0.108 mm at every mass. The docstring was corrected.
4. **Test expectations about message counts were wrong** in the first draft of
   `test_learner_interface.py`: when the evaluator ends an episode, the agent has received
   as many observations as it sent commands; otherwise one more. The tests now check that.
5. **A future git checkout could have changed the manifests' bytes** (line endings) and
   broken their hashes. `.gitattributes` now keeps `configs/manifests/`,
   `configs/benchmark.toml` and `freezes/` byte-identical.
6. **`manifests` refused to regenerate with a raw traceback** (found in the final checks).
   It refused correctly and wrote nothing, but now prints "REFUSED" and exits 2, like the
   sealed-split refusals, with a test (D27).

Observed but not changed:

- The block hops for 2-6 ms at a time while a finger pushes it (D29). Lift stayed at most
  1.49 mm and tilt 3.40 degrees in every benchmark episode.
- One train episode (177) aborted its second push because the approach IK did not
  converge; the aborted push counted against the budget, and the third push succeeded.
  The message reads "error 0.01 mm", which is the position error only; the position
  tolerance is 0.1 mm, so the orientation tolerance (0.001 rad) must have failed. The
  message comes from the Phase 2 controller and was left unchanged.
- One train episode ended 19.97 mm from the target, inside the 20 mm tolerance by 0.03 mm.

## Phase 2 gate

The gate in BUILD_PLAN.md: *"At least 18/20 nominal smoke-test episodes succeed; all
failures remain visible. This is a controller gate, not a generalization claim."*
The evidence files are in [docs/evidence/phase2/](evidence/phase2/README.md).

The 20 gate seeds (20000-20019) and 30 separate development seeds (21000-21029) were
written into `configs/push.toml` before any Phase 2 episode ran (D19). The gate was run at
03:05:38 UTC, after the development runs, and took 47 s; the tables below are that run.
The Phase 2 review ran it twice more: once on the unchanged code, which reproduced it
exactly, and once after the review's fix (see "Phase 2 review").

| Gate item | Result | Evidence |
| --- | --- | --- |
| At least 18/20 nominal episodes succeed | **20/20 succeeded. Gate PASS** | `gate_log.txt`, `gate_result.json` |
| All failures remain visible | There were no failures. Every episode is printed and saved with its outcome and reason; the gate counts failures by reason and exits with 1 below 18 successes. `test_evaluation.py` creates each failure type on purpose and checks that it is reported | `gate_log.txt`; D21 |
| Pushing through the Panda actuators | Every push is approach, descend, stroke, dwell, back-off and retreat, sent only as joint-servo targets with the gripper closed. 3315 robot-block contact events, all by the two fingers; 0 forbidden contacts; 0 unplanned contacts | `gate_result.json`, contact events; D18, D21 |
| Bounded targets, damped IK, joint and velocity limits, table bounds | Largest IK error 0.50 mm. Largest measured joint speed 0.709 of the datasheet limit. No joint outside its range. Lowest measured TCP height 13.8 mm (fail limit 10 mm), although no command went below 22 mm: the arm sagged under gravity. The review fixed this (D25); in the gate after the fix the lowest was 21.9 mm. 120 control updates were slowed by the rate limit | `gate_result.json`, limit statistics; D18, D25 |
| Competent feedback baseline, no learning | Uses only the measured block position and the target. Mean 1.95 pushes (at most 3 of 8) and 8.58 s simulated (at most 13.18 s of 30 s). Final distances 2.8-19.7 mm | `gate_log.txt`; D23 |
| Diverse starts and directions | Start block-to-target directions span -150 to -7 degrees and 96 to 157 degrees. Block yaw covers the whole circle. Start distances are 84.9-197.2 mm | `gate_result.json`, initial states |
| Records | Per episode: trajectory at 50 Hz (TCP, joints, block pose), contact events from MuJoCo's contact list, every push's parameters, outcome and reason, pushes used and simulated time | `gate_result.json`, `trajectories\seed_*.csv` |
| No shortcuts | The block is moved only by contact; the controller sees only `Observation`; success is stricter than the section 4 wording, because it counts only after the fingers let go | D10, D18, D21 |

| Seed | Pushes | Simulated time (s) | Start distance (mm) | Final distance (mm) |
| --- | --- | --- | --- | --- |
| 20000 | 3 | 12.18 | 173.0 | 10.8 |
| 20001 | 2 | 9.38 | 188.5 | 10.9 |
| 20002 | 2 | 9.38 | 105.7 | 19.7 |
| 20003 | 3 | 12.08 | 125.9 | 5.9 |
| 20004 | 2 | 7.30 | 85.8 | 12.3 |
| 20005 | 2 | 9.36 | 112.1 | 2.9 |
| 20006 | 1 | 4.67 | 154.2 | 17.9 |
| 20007 | 2 | 9.30 | 94.6 | 9.4 |
| 20008 | 1 | 4.63 | 158.1 | 10.9 |
| 20009 | 2 | 8.14 | 142.5 | 10.6 |
| 20010 | 2 | 9.34 | 149.8 | 14.0 |
| 20011 | 2 | 8.00 | 192.9 | 13.3 |
| 20012 | 2 | 8.82 | 98.3 | 8.9 |
| 20013 | 2 | 8.84 | 186.8 | 17.1 |
| 20014 | 2 | 9.42 | 197.2 | 3.7 |
| 20015 | 1 | 4.60 | 109.2 | 3.6 |
| 20016 | 3 | 13.18 | 90.4 | 2.8 |
| 20017 | 1 | 4.66 | 161.9 | 19.2 |
| 20018 | 2 | 8.90 | 84.9 | 4.8 |
| 20019 | 2 | 9.46 | 115.1 | 18.2 |

All 20 episodes succeeded. Three ended close to the 20 mm tolerance: seeds 20002
(19.7 mm), 20017 (19.2 mm) and 20019 (18.2 mm). The block came to rest inside 20 mm after
the last push chosen, but outside the baseline's own 12 mm stop distance. They pass, by a
small margin.

**Development runs** (seeds 21000-21029, all before the gate): 6/6, then 30/30 three times.
No development episode failed, and no parameter in `push.toml` was changed after any result.
Between the runs only the evaluation rules were made stricter and code was changed (D23).

## Phase 2 deliverables

| Requirement | Where |
| --- | --- |
| IK on its own kinematics-only model | `kinematics.py`: `ArmKinematics` loads its own model from the scene and runs damped least squares; it never touches the runner's model or data (D18) |
| Push primitive with approach, contact, push and retreat | `controller.py`: `PushAction` (face, direction, speed, duration) and `PushSkill` (D18) |
| Bounded targets, joint and velocity limits, table-bound and reachability filters | `controller.py`, `PushSkill.plan` and the per-update target limiting; settings in `configs/push.toml` |
| Fixed-feedback baseline | `baseline.py`: `FeedbackBaseline` (D23) |
| Evaluator with contact events, limits, success and failure reasons | `evaluation.py`: `EpisodeJudge` and `EvaluatedPlant`, using the new per-physics-step monitor in `ScenarioRunner.step` (D20, D21) |
| Episode records | `runs\phase2\<set>\<UTC time>\`: `gate_log.txt`, `gate_result.json`, `trajectories\seed_<seed>.csv` |
| Gate command | `.\.venv\Scripts\python.exe -m physical_ai_lab gate` (`--seeds development` for the tuning seeds) |
| Viewer command | `.\.venv\Scripts\python.exe -m physical_ai_lab demo --mode baseline --seed N`; `--frames DIR` saves pictures instead (D24) |
| Settings and seed lists | `configs/push.toml` (`push_config.py` loads and checks it) |
| Tests | 51 new tests: `test_kinematics.py` (12), `test_push_controller.py` (15), `test_evaluation.py` (24), including 4 added in the Phase 2 review. The suite has 184 tests in 16 files |
| Evidence | [docs/evidence/phase2/](evidence/phase2/README.md): gate log and results, 20 trajectories, 15 rendered images of seed 20000, check outputs; the review's evidence is in `review/` |
| Unchanged | The scene MJCF, the physics in `configs/scene.toml`, `diagnostics.py` and both Phase 1 reference digests |

## Phase 2 final check run

These checks ran on 2026-10-03 (UTC), after the last code change of Phase 2. That change
was a corrected comment in `controller.py`, after the gate; it changed no behaviour, so the
gate was not rerun then. A first pass at 03:12:41-03:13:17, before that comment fix, gave
the same results. The Phase 2 review changed the code afterwards; its checks are in
"Phase 2 review" below.

| Check | Command | Started (UTC) | Result |
| --- | --- | --- | --- |
| Environment | `.\.venv\Scripts\python.exe -m physical_ai_lab doctor` | 03:14:08 | Exit 0, all required checks passed. Two warnings: the folder is not a git repository, and it is synced by OneDrive |
| Tests | `.\.venv\Scripts\python.exe -m pytest` | 03:14:10 | 180 passed in 19.52 s |
| Phase 1 smoke test | `.\.venv\Scripts\python.exe -m physical_ai_lab smoke` | 03:14:31 | Exit 0 in 11.8 s; PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; Phase 1 gate PASSED; digest `9ac3950e3c7ab848`, unchanged |
| Phase 1 step digest | `.\.venv\Scripts\python.exe -m physical_ai_lab step --seed 0 --seconds 8` | 03:14:43 | Digest `34cde6c9825fa1e4`, unchanged |
| Phase 2 gate | `.\.venv\Scripts\python.exe -m physical_ai_lab gate` | 03:05:38 | Exit 0; 20/20; PASS (see "Phase 2 gate" above) |
| Lock file | `uv lock --check --offline` | 03:14:45 | Exit 0, "Resolved 18 packages" |

No dependency was added in Phase 2.

## Problems found and fixed in Phase 2

1. **Joint 7 often could not reach the hand yaw.** The closed fingers are symmetric, so IK
   now tries the yaw and the yaw + 180 degrees and keeps the one that converges (D18).
2. **A block held by the fingers could count as a success.** Success now counts only while
   no robot part touches the block. Robot-block contacts outside a declared push now count
   as pushes (D21). Both rules were added during development and make the gate stricter.
3. **The viewer demo froze.** `set_texts` in MuJoCo 3.14.0 takes the viewer lock itself;
   calling it inside the lock deadlocked. It is now called outside (D24).
4. **A test caused a forbidden collision by accident.** A limit test set joint 4 below its
   range, which made the arm touch itself. It now uses joint 7 above its range.
5. **The retreat test was too strict.** The servos lag the command by about 1 cm at the
   retreat. The test now checks that the commanded hand reaches the transit height within
   1 mm, and that the measured hand ends within 2 cm of it, above the block and not
   touching it.
6. **Adding commands broke a Phase 1 test** that allowed only the Phase 1 commands.
   It now allows exactly those plus `gate` and `demo` (D22).
7. **The arm sagged below its commands under gravity.** Found in the Phase 2 review; see
   the next section and D25.

Observed but not changed:

- The block moves less than the geometric prediction, and at the retreat the measured hand
  lags the command by about 1 cm. The baseline corrects this with its next measurement.
  Both were observed before the Phase 2 review and not measured again after it. This
  bullet first blamed a sag of "about 4 mm under contact"; the review found that the sag
  was larger and came from gravity, and fixed it (D25).
- The finger-contact flag switches on and off many times during a stroke. Each event is
  recorded, which gives thousands of events per run.
- When the hand pushes from the far side, it can hide the block from the overhead camera.
  The oblique camera still shows it.

## Phase 2 review

The review followed the prompt in BUILD_PLAN.md section 9: find the most consequential
correctness issue, fix it, rerun the relevant checks and report the evidence, without
expanding the scope. Its evidence is in
[docs/evidence/phase2/review/](evidence/phase2/review/README.md).

- **The reported result was checked first.** The gate was rerun on the unchanged code at
  15:27:46 UTC. Apart from the trajectory paths and wall-clock times, every recorded field
  matched the original run, and all 20 trajectory files were byte-identical. The suite
  passed (180 tests).
- **Issue:** the push skill's height bounds held for its commands but not for the hand.
  - The arm's position servos have no gravity compensation, so the arm settles below each
    commanded pose. In a static hold the TCP was 1.5-10.5 mm low, most at full reach.
  - In the gate no command went below 22 mm, but in every episode the lowest measured TCP
    was 4.3-8.2 mm below that, down to 13.8 mm. The fingertips were then about 5 mm above
    the table instead of the planned 13 mm, and the TCP was 3.8 mm above the evaluator's
    10 mm fail limit.
  - D18 described a 15 mm floor, D21 called the evaluator's limits safety nets behind
    stricter controller limits, and D18 put the sag at "about 4 mm under contact". None of
    this held for the hand the simulator moved. Every gate episode had less margin than
    documented; the sag grows with reach, and no controller bound would have caught it.
- **Fix (D25):** the controller now adds a static gravity offset to its servo targets,
  `q + gravity(q) / kp`, computed from the robot's own links on the controller's private
  model. The joint and speed limits still apply to the kinematic command. Nothing was
  tuned. The scene, servo gains, `push.toml`, baseline and evaluator are unchanged.
- **Evidence:**
  - Static probe at 18 poses: TCP height error at most 0.27 mm with the offset, against
    1.5-10.5 mm without it.
  - Contact probe, first push of development seeds 21000-21009: the fingers met the block
    with the TCP at 21.9-22.3 mm, against 14.5-19.0 mm without the offset.
  - Tests: 4 new and 3 updated. Each of two ways of turning the fix off makes 3 of them
    fail. With the fix the suite passes (184 tests).
  - Stress check, not a benchmark: the 30 development seeds at nominal physics, at
    0.40 kg with friction 0.8, and at 0.10 kg with friction 0.2, before and after the fix.
    90/90 successes both times; lowest measured TCP 13.6 mm before and 21.8 mm after; no
    robot/table or forbidden contacts in either run.
  - The gate, run once after the fix:

| Gate result | Original (03:05:38 UTC) | After the fix (16:05:56 UTC) |
| --- | --- | --- |
| Successes | 20/20, PASS | 20/20, PASS |
| Pushes, mean (max) | 1.95 (3) | 1.85 (3) |
| Simulated time, mean (max) | 8.58 s (13.18 s) | 8.06 s (12.20 s) |
| Final distances | 2.8-19.7 mm | 0.9-18.5 mm |
| Final distance, mean (episodes above 15 mm) | 10.9 mm (5) | 11.3 mm (8) |
| Lowest measured TCP | 13.76 mm | 21.88 mm |
| Lowest commanded TCP | 22.00 mm | 22.00 mm |
| Largest joint speed, fraction of the datasheet limit | 0.7093 | 0.7092 |
| Joints outside their ranges | 0 | 0 |
| Rate-limited control updates | 120 of 8596 | 67 of 8074 |
| Largest IK position error | 0.500 mm | 0.500 mm |
| Contact events (forbidden, unplanned) | 3315 (0, 0) | 3209 (0, 0) |
| Wall-clock time | 47 s | 38 s |

The first push's plan was identical on all 20 seeds. Its execution was not, because the
fingers now push 2-8 mm higher, so later pushes differ: seeds 20003 and 20016 needed 2
pushes instead of 3, and seed 20016's third push had caused 54 of the original 120
rate-limited updates. Per-seed values are in `review/gate_compare.txt`. The issue was found
in the original gate's TCP statistics; the fix was developed on probes and development
seeds, and the gate was run once after it (D19).

Final check run after the review, one command after another on 2026-10-03, after the last
code change (15:58:36 UTC). The outputs are in `review/checks/`.

| Check | Command | Started (UTC) | Result |
| --- | --- | --- | --- |
| Environment | `.\.venv\Scripts\python.exe -m physical_ai_lab doctor` | 16:27:23 | Exit 0, all required checks passed; the same two warnings |
| Tests | `.\.venv\Scripts\python.exe -m pytest` | 16:27:26 | 184 passed in 19.17 s |
| Phase 1 smoke test | `.\.venv\Scripts\python.exe -m physical_ai_lab smoke` | 16:27:47 | Exit 0 in 10.5 s; PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; Phase 1 gate PASSED; digest `9ac3950e3c7ab848`, unchanged |
| Phase 1 step digest | `.\.venv\Scripts\python.exe -m physical_ai_lab step --seed 0 --seconds 8` | 16:27:58 | Exit 0; digest `34cde6c9825fa1e4`, unchanged |
| Phase 2 gate | `.\.venv\Scripts\python.exe -m physical_ai_lab gate --out runs\phase2\review\gate_after_20261003T160555Z` | 16:05:56 | 20/20, PASS (table above); not run again |
| Lock file | `uv lock --check --offline` | 16:28:00 | Exit 0, "Resolved 18 packages" |

No dependency was added in the review.

- **Not tested:** the other two corners of the development ranges (0.10 kg with friction
  0.8, 0.40 kg with 0.2) and the values between them; held-out seeds under other physics;
  the retreat lag after the fix; a person watching the demo window; real hardware.
- **Found but not changed** (see "Notes for Phase 3" in DECISIONS.md):
  - The baseline has not failed an episode yet, so Phase 3 may need to report a ceiling
    (BUILD_PLAN.md section 6).
  - The `gate` summary reports pushes and time over successful episodes only.
  - `EvaluatedPlant` keeps the runner in `_runner`, through which `hidden_truth()` can be
    reached.
  - With the hand at the planned height, a heavy, high-friction block's centre rose by up
    to 1.3 mm, against 0.9 mm before the fix (about 3 degrees of tilt if it tips about an
    edge; inferred, not measured).

## Phase 1 gate

The gate in BUILD_PLAN.md: *"Scene loads locally, resets reproducibly within documented
tolerances, and steps without instability. Save one screenshot and a smoke-test log."*
The evidence files are in [docs/evidence/phase1/](evidence/phase1/README.md).

| Gate item | Result | Evidence |
| --- | --- | --- |
| Scene loads locally | The scene compiled and matched `configs/scene.toml` in 0.54 s | `smoke_log.txt`, section "Scene" |
| Resets reproducibly within documented tolerances | The tolerance is 0, so resets must be bitwise identical. Seed 0 reached state digest `9ac3950e3c7ab848` after 5 s in four cases: the same runner after other episodes, a fresh runner, the same runner after its physics options were edited as the viewer's Physics panel would (D17), and a separate process. All 200 seeds tested gave valid, distinct layouts | `smoke_log.txt`, sections "Seeded reset reproducibility" and "Reset layouts" |
| Steps without instability | 12 hold runs, each with final \|qvel\| ≤ 3.63e-9 and block drift ≤ 2.9e-9 m (note 1). The instability detector stopped an injected NaN velocity and a block thrown at 200 m/s | `smoke_log.txt`, section "Stability" |
| Screenshot | Saved | `scene_composite.png` (offscreen, both cameras), `viewer_seed0.png` (viewer window) |
| Smoke-test log | PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; "Phase 1 gate ...: PASSED" | `smoke_log.txt`, `smoke_result.json` |

1. The hold runs cover seeds 0-4, each with nominal and with randomized physics for 5 s,
   plus two 30 s runs.

## Phase 1 deliverables

| Requirement | Where |
| --- | --- |
| Minimal Python package | `src/physical_ai_lab/` (version 0.1.0), run as `.\.venv\Scripts\python.exe -m physical_ai_lab <command>` |
| Pinned dependencies | `pyproject.toml` (exact pins) and `uv.lock` (18 packages) |
| Panda asset revision and licences | `assets/third_party/mujoco_menagerie/`: 72 unmodified files from Menagerie commit `4d038b3feae26ec82b46a4d586379114012a8ac7`, with their LICENSE and a SHA-256 manifest. See also [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
| Scene with arm, table, free block and goal | `assets/scenes/panda_push_scene.xml`, checked against the values in `configs/scene.toml` when it loads |
| Seeded reset | `ScenarioRunner.reset(seed)` in `scenario.py` |
| Viewer | The `view` command, with manual joint control of the arm (D16) |
| Headless stepping | The `step` command; `render` and `smoke` also run headless |
| Observations kept apart from hidden parameters | `Observation`, an allow-list of 18 fields (`observation.py`). The true values are available only from `ScenarioRunner.hidden_truth()` |
| Real smoke test | The `smoke` command |
| Copilot instructions | `.github/copilot-instructions.md`, identical to the block in BUILD_PLAN.md section 9 |
| Tests | `tests/`: 133 tests in 13 files |
| No learning or dashboard yet | Neither was built, and there is no automatic controller either. The arm moves only when a person jogs it in the viewer |

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26300) on ARM64 |
| GPU and OpenGL | Qualcomm Adreno X1-85; Mesa 26.3.0-devel D3D12 driver, OpenGL 4.6 Compatibility Profile |
| Shell and tools | PowerShell 7.6.6, uv 0.12.5 |
| Python | CPython 3.12.13 x64, running under emulation (see D1) |
| Packages | mujoco 3.14.0, numpy 2.5.3, pillow 12.3.0; pytest 9.1.1 (dev) |
| Panda model | MuJoCo Menagerie `franka_emika_panda` at commit `4d038b3f`, Apache-2.0, unmodified |

## Phase 1 final check run

These checks ran on 2026-10-03 (UTC), after the last Phase 1 code change: the Phase 1
review fix, which restores the physics options (D17). After that, only the documentation
changed until Phase 2 began. The Phase 2 final check run above repeats the doctor, tests,
smoke test and lock check on the Phase 2 code.
Two earlier passes on 2026-10-02 also passed every check: the first, from 16:54 to 16:56
UTC, with 104 tests, and the second, from 19:32 to 19:34 UTC after manual joint control was
added (D16), with 128 tests.

The smoke log records the SHA-256 of three files: `configs/scene.toml`, the scene XML and
the asset manifest. All three still match.

| Check | Command | Started (UTC) | Result |
| --- | --- | --- | --- |
| Environment | `.\.venv\Scripts\python.exe -m physical_ai_lab doctor` | 02:27:06 | Exit 0, all required checks passed. Two warnings: the folder is not a git repository, and it is synced by OneDrive |
| Tests | `.\.venv\Scripts\python.exe -m pytest` | 02:27:09 | 133 passed in 21.40 s |
| Smoke test | `.\.venv\Scripts\python.exe -m physical_ai_lab smoke` | 02:27:36 | Exit 0 in 20.4 s; PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; gate PASSED |
| Viewer | `.\.venv\Scripts\python.exe -m physical_ai_lab view --seed 0 --duration 8 --screenshot runs\viewer\viewer_seed0.png` | 02:28:10 | Exit 0, with 375 control updates at 0.94x real time. There were no operator commands, restores, physics-option reverts or instability, and the headless replay was identical (digest `5121475b4b614a3f`). One stall resync skipped about 0.5 s of updates (note 1) |
| Lock file | `uv lock --check --offline` | 02:28:24 | Exit 0 |

1. The loop once fell more than 0.2 s behind, probably while it captured the window, and
   dropped the backlog instead of running fast to catch up. Other programs were keeping the
   CPU busy. The same command without `--screenshot`, started at 02:28:36, ran 400 updates
   at 1.00x real time with no resyncs. It reached digest `34cde6c9825fa1e4`, the same as in
   both earlier passes, so the viewer physics did not change when no key was pressed.

Other numbers from the smoke log:

- **Layouts:** 942 resets/s. A seed gives the same layout with or without randomized
  physics (checked on seeds 0-19).
- **Hidden physics:**
  - The kick test measures friction from the block's sliding distance. It kicks the block
    at 0° for all 9 combinations of 3 masses and 3 frictions, and at 45° and 90° with
    friction 0.5. Every result was within -2.29% to +1.58% of the configured value.
  - At rest, the observations for different hidden values differed by at most 4.45e-9.
- **Actuators:** after a +0.1 rad step, the worst error was 0.0073 rad after 1 s (the limit
  is 0.02 rad). The gripper opens to 80.00 mm.
- **Reach:** all 50 TCP-down poses over the block region were reachable, with a worst
  position error of 0.289 mm.
- **Speed and cameras:** headless stepping ran at 25x real time. Earlier passes measured
  75x and 44x; the speed varies with how busy the machine is. The arm hid 0.0% of the block
  and goal in both cameras, over 50 seeds.

## Other Phase 1 measurements

These were taken with one-off probe scripts during Phase 1. They are not in the evidence
folder; DECISIONS.md gives the details.

- **Friction cone (D6):** with the pyramidal cone, the effective friction was up to 28.61%
  low for diagonal pushes. With the elliptic cone it was within -2.60% to +1.58%.
- **Camera placement (D12):** occlusion starts between 65° and 66° of camera elevation.
  The overhead camera uses 60°.
- **Shadow speckle (D14):** the speckle metric fell from 3962 to 512.
- **Asset recovery (D3):** `fetch-assets` was run on a temporary copy with 2 files deleted
  and 1 altered, and restored all three.
- **Git line endings (D15):** a cloned copy failed the asset check without `.gitattributes`
  and passed with it.
- **Fingertip clearance:** the closed fingertips reach 8.86 mm below the TCP (see the
  Phase 2 notes in DECISIONS.md).
- **Manual joint control (D16):** 18 key presses were posted to a live viewer window, in
  one run with MuJoCo's panels hidden and one with them shown. Each run applied the 11 that
  changed a target, with no restores, and its headless replay was identical.

## Problems found and fixed in Phase 1

1. **The arm hid the work area from the overhead camera.**
   - The first camera looked straight down and was blocked by the arm in its home pose. On
     seeds 0-49, the block was completely hidden in 29 layouts and the goal in 40.
   - The camera now looks down at 60° from beyond the far edge of the table (D12).
2. **Multisampling corrupted the visibility check.**
   - In segmentation renders, antialiased edge pixels decoded as other objects.
   - The probe now renders single-sample. A regression test fails if multisampling is
     turned back on (D13).
3. **Shadow speckle on the floor.** The key light's shadow map covered too small an area.
   `shadowclip="2"` and a new light position fixed it (D14).
4. **Friction depended on the push direction.** The scene uses the elliptic friction cone
   instead of MuJoCo's default pyramidal cone (D6).
5. **Garbled viewer text.**
   - MuJoCo's bitmap-font text shows garbled letters with this machine's OpenGL driver.
   - The status panels are drawn as images instead (D11). MuJoCo's own F1 help and side
     panels are still affected.
6. **Git would have changed the hashed files.** Git line-ending conversion would have
   broken the asset check on a fresh clone. `.gitattributes` prevents this (D15).
7. **The arm could not be moved by hand.**
   - The viewer held the arm in its home pose, and every edit, including the Control
     sliders, restored the seed.
   - The arrow keys now move one joint at a time, and the Control sliders set the same
     servo targets (D16). Other direct edits still restore the seed.
8. **Physics-panel edits survived a reset.** Found in the Phase 1 review; see the next
   section and D17.

## Phase 1 review

The review followed the prompt in BUILD_PLAN.md section 9: find the most consequential
correctness issue, fix it, rerun the relevant checks and report the evidence, without
expanding the scope.

- **Issue:** `reset()` restored the state but not the model's physics options. In
  MuJoCo 3.14.0's passive viewer, an edit in the Physics panel (the left panel, shown with
  Tab or `--show-ui`) is copied into the simulation's model at the next sync. It silently
  changed the physics (timestep, gravity, solver and contact settings), and Backspace or a
  reset kept it.
  - This broke same-seed reproducibility, which is the Phase 1 gate, and BUILD_PLAN.md
    section 4's "Preserve contact solver settings across experiments".
  - Before the fix, seed 0 after 150 updates reached `e724724cf8801f48` with gravity
    turned off and `6dd0937f2b5cefb6` with a 0.004 s timestep, instead of
    `ee9f3f82a6a42cd6`.
  - A smaller, related gap: the viewer did not watch the Equality panel's checkbox.
- **Fix:** `reset()` now puts back every option that differs from the scene as loaded,
  through `ScenarioRunner.restore_options()`. The viewer puts back option edits before every
  step and reports them; it now treats an equality toggle as a state edit.
- **Evidence:**
  - 5 new tests, which all fail with the fix turned off; the suite has 133 tests, and all
    pass.
  - A new smoke check, "physics options restored", is part of the gate and passes with
    difference 0.
  - The reproduction script now reaches `ee9f3f82a6a42cd6` after each of its three edits.
  - A live check in the real viewer put the edit back once, with no copy-back loop over the
    next 159 syncs and an identical headless replay.
  - Both reference digests are unchanged, and the evidence files were regenerated
    (see "Phase 1 final check run").
- **Not tested:** clicking the Physics or Equality panel by hand; see D17.

## Not run or not verified

- **The final test and retention splits.** Sealed by design (D27): no agent, including
  the baseline, has run on them. The baseline's performance there is unknown until a
  candidate is frozen in Phase 5.
- **Randomized physics for the baseline** was measured in Phase 3 on train (40 in-range
  conditions) and validation (12 in-range and 8 out-of-range conditions): 300/300. Values
  outside the declared cohorts (mass outside 0.05-0.60 kg, friction outside 0.10-1.00)
  were not run.
- **Replay on other machines.** Bitwise replay was checked for all 300 benchmark episodes
  on this machine only (D31).
- **Separate process as a security boundary.** The agent process is a leakage guard, not
  a sandbox: it runs as the same user and could read files such as
  `configs/manifests/test.json` if its code chose to. Phase 4 needs a test that the
  learner does not (D27, "Notes for Phase 4").
- **Watching the demo by hand.** The `demo` viewer was opened and closed automatically
  during development and once in the review. Nobody has watched a baseline episode in the
  window yet. Its rendered frames (`docs/evidence/phase2/frames/` and
  `docs/evidence/phase2/review/frames/`) do show the pushes.
- **Real hardware.** Everything is simulated. Nothing was tried on a real Panda.
- **Hands-on viewer use.** Nobody has tried the manual controls by hand yet; the final
  viewer run closed itself after 8 s. The keys were tested by posting key messages to a live
  window.
  The Control sliders were checked against MuJoCo's source code and with a direct call,
  but no slider drag was simulated. Nobody has clicked the Physics or Equality panel either
  (D17). Open the viewer with the command at the top of this page to try it.
- **Determinism across machines.**
  - Reproducibility was verified on this machine with one MuJoCo build only.
  - Other machines, CPU architectures (including native x64) and MuJoCo builds were not
    tested, and their state digests may differ.
- **Other platforms.**
  - Setup was tested only on Windows 11 on ARM64, with an emulated x64 Python.
  - `uv venv --python 3.12` on x64 Windows was not tried, and neither were Linux or macOS.
- **Native ARM64 MuJoCo.** MuJoCo 3.14.0 has no Windows ARM64 wheel. Building it from
  source was not attempted.
- **Version control.** No git repository exists, so nothing is committed.

## Known issues

- **Small margins to the 20 mm tolerance.** In the original gate, seeds 20002, 20017 and
  20019 finished 18-20 mm from the target. After the review's fix the largest was 18.5 mm
  (seed 20010), but 8 episodes instead of 5 finished more than 15 mm away, and some
  development episodes in the review's stress check finished at 19.9 mm. They pass, but
  under other physics some similar episodes may fail.
- **Many contact events.** Contact chatter during strokes makes the event lists long
  (3315 in the gate, 3209 after the fix). Contacts outside a declared push count as pushes,
  so chatter there could use up the budget. None occurred in either gate run.
- **OneDrive.** The project folder is synced by OneDrive, which can lock files while `uv`
  works. `doctor` and `smoke` warn about it, and README.md explains what to do.
- **Garbled MuJoCo text.** MuJoCo's own text in the viewer is garbled on this machine's
  OpenGL driver, as described above.
- **No project licence.** The project has no licence of its own yet. Third-party licences
  are listed in THIRD_PARTY_NOTICES.md.
- **Hidden-physics ranges.** `contact-check` measured that mass and friction take effect
  (read-back, resting normal force, kick-test friction, holds and pushes) from 0.05 to
  0.60 kg and friction 0.10 to 1.00 (D29). The in-range cohort keeps the development
  ranges of BUILD_PLAN.md section 4.
- **Benchmark ceiling.** The fixed baseline succeeded in all 300 benchmark episodes it may
  run, in both cohorts. On success rate a learner cannot beat it on these splits; final
  distance, pushes and time remain (D32). The benchmark was not made harder after the
  results; doing so needs a declared revised benchmark and a fresh test set.
- **The block hops while pushed.** It loses table contact for 2-6 ms at a time during
  strokes, lifting at most 1.49 mm and tilting at most 3.40 degrees in the benchmark (D29).

## Version control

Creating the repository is left to the owner. BUILD_PLAN.md section 10 suggests the name
`physical-ai-learning-lab`. From the project folder:

```powershell
git init
git add .
git commit -m "Phases 1-3: MuJoCo Panda push scene, baseline, benchmark and instrumentation"
```

- `.gitignore` keeps out `.venv/`, `runs/` and the caches. The Phase 2 evidence that
  matters is copied to `docs/evidence/phase2/`.
- `.gitattributes` keeps the Panda files, the evidence, `configs/benchmark.toml`, the
  manifests and future freeze records byte-identical (D15, D26).
- The Panda meshes add about 34 MB; the Phase 3 evidence adds about 21 MB, mostly
  episode logs and logged commands.

## Next

1. **Phase 2 review.** Done; see "Phase 2 review" above and D25.
2. **Phase 3 (benchmark and instrumentation)** is built and its gate passed; see "Phase 3
   gate" above and D26-D32.
3. **Phase 3 review** (BUILD_PLAN.md section 9), to be requested by the owner.
4. **Phase 4 (learning and persistent memory)** has not started. Before starting, read
   "Notes for Phase 4" at the end of DECISIONS.md: the ceiling, which metrics can still
   separate a learner from the baseline, the learner interface, and keeping the sealed
   splits unread.
