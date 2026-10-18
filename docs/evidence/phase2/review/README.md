# Phase 2 review evidence

These files are the evidence of the Phase 2 review. The review followed BUILD_PLAN.md
section 9: find the most consequential correctness issue, fix it, rerun the relevant checks
and report the evidence. Everything here ran on 2026-10-03 (UTC) on the same test machine
as the Phase 2 gate:

- Windows 11 on ARM64;
- x64 Python 3.12.13 running under emulation;
- MuJoCo 3.14.0.

The issue and the fix are described in docs/PROGRESS.md ("Phase 2 review") and
docs/DECISIONS.md (D25). In short: the arm's position servos have no gravity compensation,
so in the original gate the hand ran 4.3-8.2 mm below the controller's commands, down to a
TCP height of 13.8 mm. The controller now adds a static gravity offset to its servo
targets.

## Code state

- **Before the fix:** the code as Phase 2 left it, which produced the original gate in
  [the folder above](../README.md).
- **The fix:** `src\physical_ai_lab\kinematics.py` (saved 15:53:27 UTC),
  `src\physical_ai_lab\controller.py` (15:53:55), `tests\test_kinematics.py` (15:56:09) and
  `tests\test_push_controller.py` (15:58:36). No other file under `src`, `tests`, `configs`
  or `assets`, nor `pyproject.toml` or `uv.lock`, changed after the Phase 2 final checks.
- Runs marked before the fix started before 15:53:27. Runs marked after the fix, and the
  runs with the fix turned off, started after 15:58:36. The two probes ran in between,
  after both source files were saved, while the tests were being written. The comparison
  scripts and the hash listing only read result files.

Each file is an unchanged copy of the output of the run that produced it, unless a note says
otherwise; every copy was checked against its source with SHA-256. Paths inside the files
are in the project folder. Most point to `runs/`, the local output folder that is not kept
in the repository.

The commands below are run in PowerShell from the project folder, after:

    $py = '.\.venv\Scripts\python.exe'
    $s = 'docs\evidence\phase2\review\scripts'

The scripts in `scripts\` are review checks, not project features. They ran from the
review's scratch folder and were copied here unchanged.

## Check of the reported result (before the fix)

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `gate_rerun\console_output.txt` | `& $py -m physical_ai_lab gate` | started 15:27:46, finished 15:28:26 | 20/20 succeeded, gate PASS; run folder `runs\phase2\gate\20261003T152746Z` |
| `gate_rerun\compare.txt` | `& $py $s\review2_rerun_compare.py docs\evidence\phase2 runs\phase2\gate\20261003T152746Z` | saved 16:15:16 | Only `started_utc`, `finished_utc` and, in each of the 20 records, `trajectory_file` and `wall_seconds` differ; 20 of 20 trajectory files byte-identical |
| `gate_rerun\trajectories_sha256.txt` | PowerShell `Get-FileHash` on both trajectory folders | computed 16:27:10 | 20 of 20 hashes identical |
| `checks\pytest_rerun_unchanged_code.txt` | `& $py -m pytest -q` (note 1) | saved 15:28:52 | 180 passed in 23.04 s |

## Probes (after the source change)

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `probes\ff_probe.txt` | `& $py $s\review2_ff_probe.py` | saved 15:54:20 | Static hold for 1 s at 18 poses (6 hand positions with x from 0.30 to 0.72 m, 3 yaws each), in a bare MuJoCo model built from the scene file, without the project's controller. With the offset, the TCP settled within 0.27 mm of the commanded height (0.39 mm in position); without it ("raw" columns), 1.54-10.54 mm low. At the home pose the offset equals MuJoCo's own bias force divided by the servo gain (difference 0). |
| `probes\contact_z_probe.txt` | `& $py $s\review2_contact_z.py` | saved 15:58:21 | One baseline push on each of the development seeds 21000-21009. Measured TCP height while the gripper touched the block: 21.92-22.30 mm with the fix, 14.54-18.99 mm with the offset replaced by zeros (push height 22.0 mm). |

## Tests with the fix turned off

`review2_nofix_plugin.py` is a pytest plugin that turns the fix off in one of two ways,
chosen by `REVIEW2_NOFIX`.

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `probes\nofix_servo.txt` (excerpt, note 2) | `$env:PYTHONPATH = (Resolve-Path $s).Path; $env:REVIEW2_NOFIX = 'servo'; & $py -m pytest -q -p review2_nofix_plugin tests\test_kinematics.py tests\test_push_controller.py` | saved 15:59:25 | 3 failed, 24 passed. `servo_target` returns the IK solution unchanged, as the controller did before the review. |
| `probes\nofix_offset.txt` (excerpt, note 2) | The same, with `$env:REVIEW2_NOFIX = 'offset'` | saved 15:59:31 | 3 failed, 24 passed. `gravity_offset` returns zeros. |
| `probes\nofix_servo_rerun_full.txt`, `probes\nofix_offset_rerun_full.txt` | The same two commands, run again with the plugin in this folder | started 16:32:05 and 16:32:12 | Full output, with the same 3 failures each. Without the plugin, the same two test files pass (27 passed). |

## Checks after the fix

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `checks\pytest_after.txt` | `& $py -m pytest -q` (note 1) | saved 16:00:07 | 184 passed in 21.09 s. From its timing, it ran alongside the next two commands, each in its own process. |
| `checks\smoke_after.txt` | `& $py -m physical_ai_lab smoke` | started 15:59:46 | Finished in 11.6 s; PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; Phase 1 gate PASSED; digest `9ac3950e3c7ab848`, unchanged; run folder `runs\smoke\20261003T155946Z` |
| `checks\step_after.txt` | `& $py -m physical_ai_lab step --seed 0 --seconds 8 --json` | saved 15:59:58 | 400 control updates, stable, largest `qvel` magnitude 2.7e-09; digest `34cde6c9825fa1e4`, unchanged |

## Stress check (development seeds; not a benchmark)

The 30 development seeds (21000-21029) ran under three fixed settings of the hidden block
physics: nominal (0.25 kg, friction 0.5), heavy and grippy (0.40 kg, 0.8), and light and
slippery (0.10 kg, 0.2). The scripts pass these values to the scenario reset in place of the
seeded draw. The baseline, its configuration and its observations are unchanged. The other
two corners of the development ranges (0.10 kg with 0.8, 0.40 kg with 0.2) were not run.

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `stress\stress_before.txt`, `stress\stress_before_summary.json` (note 3) | `& $py $s\review2_stress.py runs\phase2\review\stress_before_20261003T154116Z` | started 15:41:18, finished 15:44:04 | Before the fix. 90/90 successes; lowest measured TCP 13.6 mm, up to 8.4 mm below the push height; no robot/table or forbidden contacts |
| `stress\stress_after.txt`, `stress\stress_after_summary.json` (note 3) | `& $py $s\review2_stress2.py run runs\phase2\review\stress_after_20261003T160235Z` | started 16:02:36, finished 16:05:26 | After the fix. 90/90 successes; lowest measured TCP 21.82 mm, at most 0.18 mm below the push height; no robot/table or forbidden contacts |
| `stress\stress_compare.txt`, `stress\compare_to_before.json` | `& $py $s\review2_stress2.py compare runs\phase2\review\stress_before_20261003T154116Z runs\phase2\review\stress_after_20261003T160235Z` | saved 16:05:34 | Side by side per setting. No success lost or gained; push counts changed on 4 nominal and 3 light, slippery seeds; the heavy block's centre rose by up to 0.947 mm before and 1.302 mm after |

## Gate after the fix

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `gate_after\console_output.txt`, `gate_after\gate_log.txt`, `gate_after\gate_result.json`, `gate_after\trajectories\seed_*.csv` | `& $py -m physical_ai_lab gate --out runs\phase2\review\gate_after_20261003T160555Z` | started 16:05:56, finished 16:06:34 | 20/20 succeeded, gate PASS |
| `gate_compare.txt` | `& $py $s\review2_gate_compare.py docs\evidence\phase2\gate_result.json runs\phase2\review\gate_after_20261003T160555Z\gate_result.json` | saved 16:12:16 | Summary of both runs and a per-seed table |

`--out` only sets the output folder; the seeds, configuration and code path are those of
the plain `gate` command. The summary table is in docs/PROGRESS.md ("Phase 2 review").

- The first push's plan was identical on all 20 seeds. Seeds 20003 and 20016 used 2 pushes
  instead of 3.
- Final distances, computed from the two `gate_result.json` files: mean 10.85 mm before and
  11.26 mm after; above 15 mm, 5 episodes before and 8 after; above 18 mm, 3 before and 1
  after. The fix put the hand at the planned height; it did not make the baseline more
  accurate.

## Frames and viewer demo after the fix

| File | Command | Time (UTC) | Result |
| --- | --- | --- | --- |
| `frames\seed20000_*.png` (15 files), `checks\frames_after.txt` | `& $py -m physical_ai_lab demo --mode baseline --seed 20000 --frames docs\evidence\phase2\review\frames` | files written 16:08:42-16:08:46 | Success, 3 pushes, 12.20 s simulated, final distance 6.3 mm, as for seed 20000 in the gate after the fix. The same three views and five moments as the Phase 2 frames. |
| `checks\demo_after.txt` | `& $py -m physical_ai_lab demo --mode baseline --seed 20000` (note 4) | saved 16:09:27 | The viewer opened, ran the episode and closed itself 22.3 s after opening; 0 physics-option reverts; a headless rerun of the seed gave the identical outcome and state (digest `d1392b6bccc39de2`) |

## Final check run

One command after another, after the last code change. `checks\final_run_times.txt` holds
the start and end times and exit codes recorded by the PowerShell wrapper.

| File | Command | Started (UTC) | Result |
| --- | --- | --- | --- |
| `checks\final_doctor.txt` | `& $py -m physical_ai_lab doctor` | 16:27:23 | Exit 0, all required checks passed; mujoco 3.14.0, numpy 2.5.3, pillow 12.3.0, pytest 9.1.1; Panda asset: 72 files match the manifest (Menagerie commit `4d038b3feae2`); warnings: not a git repository, OneDrive-synced folder |
| `checks\final_pytest.txt` | `& $py -m pytest` | 16:27:26 | Exit 0; 184 passed in 19.17 s |
| `checks\final_smoke.txt` | `& $py -m physical_ai_lab smoke` | 16:27:47 | Exit 0 in 10.5 s; PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0; Phase 1 gate PASSED; digest `9ac3950e3c7ab848`; run folder `runs\smoke\20261003T162747Z` |
| `checks\final_step_seed0.txt` | `& $py -m physical_ai_lab step --seed 0 --seconds 8` | 16:27:58 | Exit 0; 400 control updates, stable; digest `34cde6c9825fa1e4` |
| `checks\final_uv_lock_check.txt` | `uv lock --check --offline` | 16:28:00 | Exit 0; "Resolved 18 packages in 1ms" |

The gate was not run again; no code changed after the gate run at 16:05:56.

## Reproduced from this folder

At 16:31-16:32 UTC, after the copies were made, `review2_ff_probe.py`,
`review2_contact_z.py`, `review2_gate_compare.py` and `review2_rerun_compare.py` were run
again from `scripts\` with the commands above. Their output was identical to the saved
files. The two runs with the fix turned off gave the same failures (the
`*_rerun_full.txt` files). The stress scripts and the gates were not run again.

## Not copied

These stay in `runs\` on the test machine only:

- the 180 trajectory files of the two stress runs (about 17 MB), in
  `runs\phase2\review\stress_before_20261003T154116Z\` and
  `runs\phase2\review\stress_after_20261003T160235Z\`;
- the rerun gate's `gate_result.json`, `gate_log.txt` and trajectories, in
  `runs\phase2\gate\20261003T152746Z\`. `gate_rerun\compare.txt` and
  `gate_rerun\trajectories_sha256.txt` record how they compare with the original;
- the smoke-test run folders `runs\smoke\20261003T155946Z\` and
  `runs\smoke\20261003T162747Z\` (log, results and screenshots).

## Notes

1. The flags of these two pytest runs were not saved. Their output is in pytest's quiet
   format (`-q`).
2. The excerpts keep a header line written by the review's wrapper, the plugin's message,
   the FAILED lines and the summary line; the rest of the output was not saved. The full
   reruns were added later, on the same code.
3. Each stress script writes `stress_summary.json` into its run folder; the copies here were
   renamed. `review2_stress.py` ran on the code before the fix, so running it now runs the
   fixed code and does not reproduce the "before" numbers. `review2_stress2.py run` uses the
   same seeds and settings and also records trajectory metrics; `compare` reads both runs'
   trajectory files, which were not copied.
4. Any other flags of this command were not saved. The window was not checked by eye in the
   review; the result above is what the program reported.
