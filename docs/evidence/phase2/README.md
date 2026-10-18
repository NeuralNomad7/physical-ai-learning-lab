# Phase 2 evidence

These files are the outputs of the Phase 2 gate and of the final checks. They were run on
2026-10-03 (UTC) on the test machine:

- Windows 11 on ARM64;
- x64 Python 3.12.13 running under emulation;
- MuJoCo 3.14.0.

The gate seeds (20000-20019) and the development seeds (21000-21029) were written into
`configs/push.toml` before any Phase 2 episode was run (DECISIONS.md, D19). The gate was run
once, after the development runs, and its files here are unchanged. The Phase 2 review later
reran it on the unchanged code, with identical results, and once after its fix (D25). Those
runs are in [review/](review/README.md).

Each file is an unchanged copy of the output of the run that produced it. Paths inside the
files point to `runs/`, a local output folder that is not kept in the repository.

| File | Command, run from the project folder | Started (UTC) | Result |
| --- | --- | --- | --- |
| `gate_log.txt`, `gate_result.json`, `trajectories\seed_*.csv` | `.\.venv\Scripts\python.exe -m physical_ai_lab gate` | 03:05:38 | **20/20 succeeded, gate PASS** (note 1) |
| `frames\seed20000_*.png` | `.\.venv\Scripts\python.exe -m physical_ai_lab demo --mode baseline --seed 20000 --frames docs\evidence\phase2\frames` | 03:06:36 | Success, 3 pushes, 12.18 s (note 2) |
| `doctor_output.txt` | `.\.venv\Scripts\python.exe -m physical_ai_lab doctor` | 03:14:08 | See note 3 |
| `pytest_output.txt` | `.\.venv\Scripts\python.exe -m pytest` | 03:14:10 | 180 passed in 19.52 s |
| `smoke_output.txt` | `.\.venv\Scripts\python.exe -m physical_ai_lab smoke` | 03:14:31 | See note 4 |
| `step_seed0_output.txt` | `.\.venv\Scripts\python.exe -m physical_ai_lab step --seed 0 --seconds 8` | 03:14:43 | State digest `34cde6c9825fa1e4`, same as Phase 1 |
| `uv_lock_check_output.txt` | `uv lock --check --offline` | 03:14:45 | Exit code 0, "Resolved 18 packages" |

1. **Gate:**
   - exit code 0; finished at 03:06:25 UTC; run folder `runs\phase2\gate\20261003T030537Z`;
   - 20 nominal-physics episodes, 20 successes (18 required), failures by reason: none;
   - successful episodes used 1.95 pushes on average (at most 3 of the 8 allowed) and
     8.58 s of simulated time on average (at most 13.18 s of the 30 s allowed);
   - 3315 robot-block contact events, all by the two fingers; 0 forbidden contacts, 0
     unplanned contacts;
   - largest measured joint speed: 0.709 of the datasheet limit; lowest measured TCP height:
     13.8 mm (limit 10 mm), although no command went below 22 mm, because the arm sagged
     under gravity (fixed in the Phase 2 review, D25); largest IK position error: 0.50 mm;
     no joint outside its range;
   - start positions, directions and block yaw: the block-to-target direction at the start
     spans -150 to -7 degrees and 96 to 157 degrees, and the block yaw spans the whole circle.
   - `gate_result.json` holds, per episode: the seed, the outcome and its reason, pushes
     used, simulated time, the initial and final state, every push's parameters and the
     candidates considered, every contact event read from MuJoCo's contacts, and limit
     statistics.
   - `trajectories\seed_<seed>.csv` holds the trajectory at every control update (50 Hz):
     joint positions and targets, TCP position, block pose, block speed, distance to the
     target, contact flags, push index and phase.
2. **Frames:** rendered offscreen from the same episode as the first gate seed, after the
   gate. Its outcome matched the gate exactly. There are three views (overhead, oblique and
   both side by side) at five moments: `start`, `push1_mid_stroke`, `push2_mid_stroke`,
   `push3_mid_stroke` and `end`. In some frames the hand hides the block from the overhead
   camera; the oblique view shows it.
3. **Doctor:** exit code 0 and `doctor: all required checks passed`. There are two
   warnings: the folder is not a git repository, and it is synced by OneDrive.
4. **Smoke test:**
   - exit code 0, finished in 11.8 s; run folder `runs\smoke\20261003T031431Z`;
   - PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0, and the Phase 1 gate PASSED;
   - the two warnings are the same as the doctor's;
   - reference digest `9ac3950e3c7ab848`, unchanged from Phase 1.

## Results per gate episode

| Seed | Result | Pushes | Simulated time (s) | Start distance (mm) | Final distance (mm) |
| --- | --- | --- | --- | --- | --- |
| 20000 | Success | 3 | 12.18 | 173.0 | 10.8 |
| 20001 | Success | 2 | 9.38 | 188.5 | 10.9 |
| 20002 | Success | 2 | 9.38 | 105.7 | 19.7 |
| 20003 | Success | 3 | 12.08 | 125.9 | 5.9 |
| 20004 | Success | 2 | 7.30 | 85.8 | 12.3 |
| 20005 | Success | 2 | 9.36 | 112.1 | 2.9 |
| 20006 | Success | 1 | 4.67 | 154.2 | 17.9 |
| 20007 | Success | 2 | 9.30 | 94.6 | 9.4 |
| 20008 | Success | 1 | 4.63 | 158.1 | 10.9 |
| 20009 | Success | 2 | 8.14 | 142.5 | 10.6 |
| 20010 | Success | 2 | 9.34 | 149.8 | 14.0 |
| 20011 | Success | 2 | 8.00 | 192.9 | 13.3 |
| 20012 | Success | 2 | 8.82 | 98.3 | 8.9 |
| 20013 | Success | 2 | 8.84 | 186.8 | 17.1 |
| 20014 | Success | 2 | 9.42 | 197.2 | 3.7 |
| 20015 | Success | 1 | 4.60 | 109.2 | 3.6 |
| 20016 | Success | 3 | 13.18 | 90.4 | 2.8 |
| 20017 | Success | 1 | 4.66 | 161.9 | 19.2 |
| 20018 | Success | 2 | 8.90 | 84.9 | 4.8 |
| 20019 | Success | 2 | 9.46 | 115.1 | 18.2 |

Several final distances are close to the 20 mm tolerance (19.7, 19.2 and 18.2 mm). The
baseline stops pushing at 12 mm or closer; these episodes ended because the block came to
rest inside 20 mm after the last push it chose. They meet the criterion, but by a small
margin.

## Order of the runs

1. Development runs on seeds 21000-21029 (`runs\phase2\dev_scratch\`, not copied here):
   `try1` (02:51, seeds 21000-21005, 6/6), `try2_all30` (02:52, 30/30),
   `try3_release_rule` (02:53, 30/30) and `try4_final_code` (03:04, 30/30, the final
   controller and evaluator). No development episode failed.
   - Between the runs, the evaluation rules were made stricter (success counts only after
     the robot has let go of the block; unplanned robot-block contacts count as pushes;
     DECISIONS.md, D21), and the baseline, evaluator, CLI and viewer demo code was changed.
   - `configs/push.toml` was last saved at 02:46, before the first development run, so no
     parameter was changed after any result.
2. The gate, once (03:05:38).
3. The frames (03:06:36).
4. New tests were added (no controller change), then a first pass of the final checks
   (03:12:41-03:13:17).
5. A comment in `controller.py` was corrected (it named the filter method `check` instead of
   `plan`; no behaviour change), and the final checks above were run again (03:14:08-03:14:45),
   replacing the first pass's files. The results were the same: 180 passed, smoke PASS 73,
   both digests unchanged.

The gate ran on the controller code before that comment change. The change touched only a
docstring, so the gate was not rerun then. The Phase 2 review's rerun on the code after the
comment change (15:27:46 UTC) reproduced it: every recorded value except paths and
wall-clock times, and all 20 trajectory files byte for byte (`review/gate_rerun/`).
