# Phase 3 evidence: benchmark and instrumentation

Copied from `runs/phase3/` on 2026-10-03 after the final checks. Every file is as the run
wrote it, except where noted. All times are UTC on 2026-10-03. Machine: Windows 11 on
ARM64, CPython 3.12.13 x64 under emulation, mujoco 3.14.0, numpy 2.5.3, Panda asset
MuJoCo Menagerie commit `4d038b3feae2`. Commands run from the project folder as
`.\.venv\Scripts\python.exe -m physical_ai_lab <command>`.

Results and their discussion are in [docs/PROGRESS.md](../../PROGRESS.md) ("Phase 3
gate") and [docs/DECISIONS.md](../../DECISIONS.md) (D26-D32).

## Order of work

| Time | Step | Command | Result |
| --- | --- | --- | --- |
| 17:29:23 | Contact checks, run 1 (69 s) | `contact-check --out runs\phase3\contact_checks\run1` | All checks passed; feasible box mass 0.05-0.60 kg, friction 0.10-1.00 |
| 17:31:59 | Out-of-range cohort fixed in `configs/benchmark.toml` | (edit) | Mass 0.05-0.08 or 0.42-0.60 kg, or friction 0.10-0.16 or 0.84-1.00 |
| 17:38:33 | Manifests written | `manifests` | Four manifests and `hashes.json` |
| 17:58:49 | Contact checks, run 2 (65 s), with the revised table-force measure (D29) | `contact-check --out runs\phase3\contact_checks\run2` | All checks passed; every value shared with run 1 identical |
| 18:00:03-18:06:51 | Baseline on train | `evaluate --split train --mode baseline` | 200/200, exit 0 |
| 18:07:36-18:11:04 | Baseline on validation | `evaluate --split validation --mode baseline` | 100/100, exit 0 |
| 18:11:11 | Replay, train episodes 1-10 | `replay --run runs\phase3\evaluate\train\20261003T180003Z --count 10` | All identical |
| 18:11:42 | Replay, validation episodes 1-10 | `replay --run runs\phase3\evaluate\validation\20261003T180737Z --count 10` | All identical |
| 18:12:14-18:23:07 | Full replay, all 300 episodes | `replay --run runs\phase3\evaluate\train\20261003T180003Z --count 200`, then `replay --run runs\phase3\evaluate\validation\20261003T180737Z --count 100`; results saved as `replay_all.json` in each run folder | 300/300 identical |
| 18:17-18:26 | Final checks | see `final_checks/` | See below |

Nothing about the manifests, cohorts, budgets, success rule or metric definitions changed
after 17:38:33. No episode has run on the test or retention split.

## Manifests (`manifests/`)

Byte copies of `configs/manifests/` and `configs/benchmark.toml`. The originals in
`configs/` are the ones the code checks.

| File | SHA-256 | Contents |
| --- | --- | --- |
| `benchmark.toml` | `140e4f7d70340cc3f90e547385426f9a30ec68f8003b2ac54ba7fd5e0fcf2f25` | Benchmark `phase3-v1`, schema 1, metrics v1, episode schema v1 |
| `train.json` | `6eadee60c6ae2a2fa51696ceeac7faafe3441ebbf1e1f956b635b63777f62e06` | 40 in-range conditions, 200 scenarios; open |
| `validation.json` | `88dabaa6cb811ea48c8d0c88f0d74d3a09f3328e88006079d2391269f4557c15` | 20 conditions (12 in-range, 8 out-of-range), 100 scenarios (60 and 40); open |
| `test.json` | `5673ce2e3763c0417674a7102eab3386cfe1defd9a3f6178142c0faa2e1d948f` | 20 conditions (14 and 6), 100 scenarios (70 and 30); **sealed** |
| `retention.json` | `e8c5857a7d7151a103f0d6fce2362d21ad6a177ac08b55d675fedf4485c9f2d4` | 10 train conditions with 50 new initial states; **sealed** |
| `hashes.json` | (holds the four hashes above) | Written 17:38:33 |

## Contact checks (`contact_checks/`)

`run1/contact_checks.json` and `run2/contact_checks.json`: every check, criterion and
measured value. Their console output was not saved; the JSON files are the record.

## Baseline report (`baseline_train/`, `baseline_validation/`)

Each folder is a full copy of one `evaluate` run:

- `summary.txt`, `summary.json`: the report, per cohort and per out-of-range band;
- `episodes.jsonl`: one record per episode, all episodes, in manifest order, with the
  evaluator record and the learner-visible record kept apart (D29);
- `commands/<n>.npz`: the actuator inputs and state digest after every control update;
- `replay.json` (first 10 episodes) and `replay_all.json` (all episodes);
- `agent_stderr.log`: the agent process's error output (empty in both runs).

| | Train | Validation, in-range | Validation, out-of-range |
| --- | --- | --- | --- |
| Successes | 200/200 (40 conditions) | 60/60 (12) | 40/40 (8) |
| 95% CI, Wilson with n = conditions | 91.2-100% | 75.8-100% | 67.6-100% |
| Final distance mean / max (mm) | 10.13 / 19.97 | 10.81 / 16.99 | 10.22 / 18.45 |
| Pushes mean, all episodes | 1.72 | 1.70 | 1.82 |
| Failures, limit violations, forbidden contacts, probe pushes | 0 | 0 | 0 |
| Messages to the agent scanned, hidden-value hits | 79,210, 0 | 23,792, 0 | 16,558, 0 |

The baseline succeeded in every episode: a ceiling on success rate, not a success claim.

## Logs (`logs/`)

- `evaluate_train_console.txt`: the console of the train run, captured as it ran.
- `evaluate_validation_console.txt`: **not a verbatim capture.** The validation run's
  console was not saved to a file; this file records the observed start, end and exit
  code and copies that run's `summary.txt` unchanged.
- `replay_train_console.txt`, `replay_validation_console.txt`,
  `replay_all_console.txt`: the replay consoles, captured as they ran.

## Final checks (`final_checks/`)

Each `.txt` file holds the command's output between its `START_UTC` and `END_UTC` lines and
its exit code.

| File | Command | Started | Result |
| --- | --- | --- | --- |
| `doctor.txt` | `doctor` | 18:26:14 | All required checks passed (2 warnings: no git, OneDrive) |
| `pytest.txt` | `python -m pytest -q` | 18:23:28 | 284 passed in 34.5 s |
| `smoke.txt`, `smoke_log.txt` | `smoke` | 18:24:09 | PASS 73, FAIL 0; digest `9ac3950e3c7ab848` |
| `step_seed0_8s.txt` | `step --seed 0 --seconds 8` | 18:26:16 | Digest `34cde6c9825fa1e4` |
| `gate.txt`, `gate_log.txt`, `gate_compare.txt` | `gate` | 18:24:20 | 20/20, PASS; trajectories byte-identical to the Phase 2 review's gate |
| `manifests_verify.txt` | `manifests --verify` | 18:26:20 | Hashes match; 101,025 initial-state pairs disjoint |
| `manifests_regenerate_refused.txt` | `manifests` | 18:26:19 | Exit 2, REFUSED |
| `manifests_regenerate_before_fix.txt` | `manifests` | 18:17:27 | Exit 1 with a traceback; refused and wrote nothing. Led to the clean refusal (PROGRESS.md, problem 6) |
| `evaluate_test_refused.txt` | `evaluate --split test --mode baseline` | 18:26:18 | Exit 2, REFUSED |
| `evaluate_retention_refused.txt` | `evaluate --split retention --mode baseline` | 18:26:18 | Exit 2, REFUSED |
| `uv_lock_check.txt` | `uv lock --check --offline` | 18:26:21 | Exit 0 |

## Not covered

- Other machines, CPU architectures and MuJoCo builds: replay is verified bitwise on this
  machine only.
- The test and retention splits: sealed; no outcome exists.
- Real hardware.
