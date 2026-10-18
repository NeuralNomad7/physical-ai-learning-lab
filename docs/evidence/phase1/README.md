# Phase 1 evidence

These files are the outputs of the final Phase 1 checks. They were run on 2026-10-03 (UTC)
after the last code change, on the test machine:

- Windows 11 on ARM64;
- x64 Python 3.12.13 running under emulation;
- MuJoCo 3.14.0.

The last code change was the Phase 1 review fix: a reset, and the viewer before every step,
now put back any edited physics option (DECISIONS.md, D17). All the checks were run again
after it, and these files replaced those of the second pass, which ran from 19:32 to 19:34
UTC on 2026-10-02 after manual joint control was added (D16). The first pass ran from
16:54 to 16:56 UTC the same day.

Each file is an unchanged copy of the output of the run that produced it. Paths inside the
files point to `runs/`, a local output folder that is not kept in the repository.

| File | Command, run from the project folder | Started (UTC) | Result |
| --- | --- | --- | --- |
| `doctor_output.txt` | `.\.venv\Scripts\python.exe -m physical_ai_lab doctor` | 02:27:06 | See note 1 |
| `pytest_output.txt` | `.\.venv\Scripts\python.exe -m pytest` | 02:27:09 | 133 passed in 21.40 s |
| `smoke_log.txt`, `smoke_result.json`, `scene_composite.png` | `.\.venv\Scripts\python.exe -m physical_ai_lab smoke` | 02:27:36 | See note 2 |
| `viewer_output.txt`, `viewer_seed0.png` | `.\.venv\Scripts\python.exe -m physical_ai_lab view --seed 0 --duration 8 --screenshot runs\viewer\viewer_seed0.png` | 02:28:10 | See note 3 |

1. **Doctor:** exit code 0 and `doctor: all required checks passed`. There are two
   warnings: the folder is not a git repository, and it is synced by OneDrive.
2. **Smoke test:**
   - exit code 0, finished in 20.4 s;
   - PASS 73, FAIL 0, WARN 2, INFO 24, SKIP 0, and the Phase 1 gate PASSED;
   - the two warnings are the same as the doctor's;
   - the run folder was `runs\smoke\20261003T022737Z`.
   - The new check is "physics options restored" (D17). After the timestep, the gravity
     flag and the contact override were edited, seed 0 reached the reference digest with
     difference 0.
   - The reference digest (`9ac3950e3c7ab848`) and every measured physics value match the
     earlier passes, and `scene_composite.png` is byte-identical to the second pass's.
   - Only the speeds are lower. Other programs kept the CPU busy, so headless stepping ran
     at 25x real time; the earlier passes measured 75x and 44x.
3. **Viewer:**
   - exit code 0;
   - 375 control updates at 0.94x real time;
   - no operator commands (no keys were pressed), scenario restores, physics-option
     reverts or instability events;
   - 1 stall resync (note 4);
   - the headless replay matched exactly (digest `5121475b4b614a3f`);
   - the 2560x1440 screenshot was taken 3.0 s after the start.
4. **Stall resync:**
   - Once, the real-time loop fell more than 0.2 s behind. The viewer then dropped the
     backlog, about 0.5 s of control updates, instead of running fast to catch up.
   - The likely cause is the window capture, which runs inside the loop, on a busy CPU.
   - The same command without `--screenshot`, started at 02:28:36, had 400 updates at
     1.00x real time and no resyncs. It reached digest `34cde6c9825fa1e4`, the same as both
     earlier passes' viewer runs, so the review fix did not alter the viewer physics when
     nothing is edited.

`uv lock --check --offline` also passed (exit code 0, started 02:28:24) during the same check
pass. It does not have an output file here.

The interactive viewer run closed itself after 8 s. Nobody inspected or used the window by
hand during that run. The live tests of the manual joint control and of the option revert
are not in this folder; their results are in DECISIONS.md, D16 and D17.