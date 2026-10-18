# Physical AI Learning Agent: MuJoCo + GitHub Copilot

Build brief for Vivan | October 2, 2026 | Proposed implementation, not measured results

## 1. The proof to build

Build a simulated Franka Panda arm that pushes a block into a target zone. Change the block's mass and surface friction without revealing those values to the agent. The agent observes the result of a bounded push, updates its prediction of how the block moves, chooses another action, and retains useful experience across restarts.

The demonstration must answer: **Does experience improve performance on unseen conditions under the same action and time budget, without materially degrading previously successful conditions?**

Use MuJoCo throughout version 1. A reasonable planning estimate is 8–12 focused development sessions; controller and contact debugging may extend this. Deliver a working scene and baseline in the first two sessions. Treat the later learning targets as hypotheses to test.

The eventual claim is narrow: “This agent learns a task-specific action-to-motion model and reuses validated experience in simulation.” Do not claim general physical understanding, real-world transfer, foundation-model training, or autonomous recursive code improvement.

## 2. What recursive and self-learning mean here

| Capability | Implementation | Observable evidence |
| --- | --- | --- |
| Geometry | Target-relative block pose, dimensions, contact approach, table bounds, collision-aware retreat | Valid approach poses; blocked unsafe paths |
| Physics | Learn how push direction, speed, duration, and observed context affect translation and rotation | Predicted versus observed motion error |
| Reasoning | Select a useful push, inspect error, revise the hypothesis, retry within a budget | Timestamped action and diagnostic events |
| Recursion | Repeat predict → act → measure → revise until success or budget exhaustion | Bounded loop with explicit stop reasons |
| Self-learning | Update a numerical dynamics model and persistent experience memory | Changed checkpoint, improved held-out results |
| Retention | Evaluate old scenarios before promoting a candidate | Regression scorecard and rollback |

An LLM's explanation is not evidence of learning. Version 1 can run without any runtime LLM. Copilot builds the software; the numerical learner runs inside the simulation application.

## 3. Minimal architecture

Use one Python project and a small local dashboard. Keep the numerical control loop independent of the UI.

| Module | Responsibility | Boundary |
| --- | --- | --- |
| Scenario runner | Own MuJoCo model, reset state, randomize conditions, record truth | Never hand the live MjModel/MjData objects to the learner |
| Observation adapter | Return block pose/velocity, robot state, target and sensor-like contact observations | V1 uses simulator-derived pose; label this explicitly |
| Skill controller | Approach, contact, push, retreat using joint actuation and bounded Cartesian targets | Never teleport the block or drag the arm with a mocap weld |
| Predictor | Predict block displacement and yaw from a proposed push and observable context | Cannot read hidden mass, friction, scenario identity or future observations |
| Planner | Score candidate pushes by goal progress, uncertainty, time and constraint checks | Fixed candidate and action budgets |
| Learner + memory | Fit updated predictor from observed transitions; persist checkpoints | Version data, model, normalization and provenance together |
| Evaluator | Compare baseline, learner and ablations on fixed scenario manifests | Own hidden truth and final scoring |
| Dashboard | Show replay, predictions, outcomes, learning curve and provenance | Render logged events, not invented reasoning |

Run a separate process boundary for the learner if practical. At minimum pass serialized observations and actions through narrow interfaces and test for hidden-state leakage. Planning must use the learned predictor, not rollouts against a copied ground-truth simulator.

## 4. Scene, controls and scope

- Robot: pinned Franka Emika Panda asset from MuJoCo Menagerie; preserve the asset's license and attribution. Use a closed gripper as the initial pushing surface.
- Scene: flat table, one approximately 5 cm block, target marker, overhead and oblique cameras. Set the tabletop and block comfortably inside the validated reach envelope.
- Action: a bounded push primitive with approach side, direction, speed and duration. Use three admissible speeds and three durations initially; filter every candidate for reachability and table boundaries.
- Controller: damped least-squares inverse kinematics with joint-target control, joint limits, velocity limits and retreat trajectories. Verify the model's actuator definitions before setting controls.
- Timing starting point: 0.002 s physics steps and a 50 Hz Cartesian control update. Advance exactly ten physics steps per control update; measure stability before changing these defaults. Planning happens between pushes.
- Episode budget: at most 8 pushes and 30 seconds of simulated time, including approach and retreat. Count exploratory pushes against the budget.
- Success: block center within 20 mm of target center and planar speed below 0.02 m/s continuously for 0.5 s. Make the displayed target consistent with this center-based criterion.
- Failure: timeout, NaN/unstable simulation, block leaving the allowed table region, forbidden collision, or a configured limit violation. Count failures in the results.

Initial development ranges: block mass 0.10–0.40 kg and effective block/table sliding friction 0.20–0.80. These are starting design choices, not validated operating ranges. Check effective contact friction and MuJoCo's contact-pair rules; changing one geom property does not by itself prove the contact condition changed. Maintain physically consistent inertia when changing mass. Preserve contact solver settings across experiments.

First validate mass and friction changes separately, then jointly. Freeze the benchmark before the final comparison. Defer obstacles, vision inference, bimanual manipulation and robot swaps.

## 5. The learning method

### First predictor

Start with a regularized regression model mapping bounded push features and current observed state to block displacement and yaw change. Include action duration, speed, approach geometry, initial block velocity and relevant interactions. Fit on an initial calibration dataset collected only from training scenarios. Add a small bootstrap ensemble if prediction uncertainty is needed; treat its spread as an estimate and check its calibration.

A simple implementation is preferable to starting with reinforcement learning. If regression underfits, first inspect contact failures and residuals. Only then try a small MLP or another nonlinear predictor with the same evaluation protocol.

### Within each episode

1. Observe the block and robot through the observation adapter.
2. Initialize from the global learned prior. Retrieve compatible prior transitions using observable geometry and motion response, never hidden scenario labels.
3. Predict outcomes for feasible candidate pushes.
4. Choose a push expected to reduce target error. If uncertainty is high, allow one bounded diagnostic push selected for useful observations.
5. Execute through the controller; record the actual trajectory, result and prediction residual.
6. Update the local response model using the new transition and a regularized global prior.
7. Replan until success, constraint failure or budget exhaustion.

Do not infer exact mass and friction from a single push. They can be confounded, especially in slow position-controlled pushing. V1 learns effective motion response. Explicit physical parameter identification is a later experiment requiring appropriate sensors, excitation and identifiability checks.

### Between episodes

Store observed transitions with model version, observable features, actions, outcomes and uncertainty. Train a candidate using training experience plus replay from previous training regimes. Evaluate it on a separate validation suite. Promote only if the predeclared validation gates pass; otherwise retain the current model. Save a rollback checkpoint.

Persist numerical model weights and normalization statistics as well as experience. A restart that loses the learned predictor fails the retention requirement. Avoid giving block colors or filenames a stable relationship to hidden physical conditions.

## 6. Fair evaluation and acceptance gates

Use three non-overlapping manifests: training, validation and sealed final test. Separate physical parameter combinations and initial states, not merely random-number seeds. Keep an in-range generalization cohort and a harder out-of-range cohort separate in the report. Choose the harder ranges after scene feasibility checks and before evaluating learned performance.

For a manageable first benchmark, use 100 paired final-test scenarios spanning multiple parameter combinations and goal directions, plus 50 retention scenarios with held-out initial states from previously learned regimes. Evaluate 3 independently trained checkpoints when feasible. Record software versions, seeds, split hashes and simulation settings. Treat repeated samples from the same physical condition as clustered when estimating uncertainty.

| Comparison | What it establishes |
| --- | --- |
| Tuned fixed feedback baseline | Fair reference: observes position and corrects errors but retains no learned state |
| Online adaptation with memory reset each episode | Benefit of within-episode learning |
| Full adaptation with persisted training memory | Added benefit of experience across episodes |
| Frozen pretrained predictor with online updates disabled | Whether additional online adaptation contributes |

Give every system identical initial states, observations, action choices and budgets. Tune the fixed baseline on training/validation data; do not deliberately cripple it. Reset the full learner to the same trained checkpoint before each final-test episode. Allow within-episode updates, but discard them afterward so test results cannot contaminate subsequent tests. Evaluate continuing lifelong accumulation only in a separately labeled stream experiment.

Proposed primary target: **at least 15 percentage points higher success than the tuned fixed baseline on the in-range held-out cohort**, with no more than a 5 percentage point decline on retention scenarios and no increase in constraint violations. Report paired confidence intervals, counts and absolute rates; an uncertain result remains inconclusive even if its point estimate hits the target. These thresholds are intended engineering gates, not achieved outcomes.

Also report final target error, pushes to success, prediction error, simulated duration, compute time, constraint violations and failures. Report pushes and time both across all episodes and conditional on success to avoid hiding failed runs. Compare prediction error on the same reserved transitions, rather than different trajectories chosen by different agents.

Promotion uses validation only; final test is opened after freezing the candidate. If the fixed baseline is already near perfect, report the ceiling. Any revised benchmark must be declared and followed by a fresh untouched test set. Do not cherry-pick successful videos or retroactively weaken success criteria.

## 7. Copilot implementation sequence

Complete these in order. Each phase ends with a runnable artifact, results or an explicit blocker. The prompts are designed as separate Copilot tasks.

### Phase 1: Reproducible simulation

**Prompt:**

> Read this build brief. Inspect the repository and local Python/MuJoCo environment. Implement a minimal Python package with a pinned dependency configuration, a documented Panda asset revision and licenses, and a MuJoCo scene containing the arm, table, free block and goal. Provide seeded reset, viewer and headless stepping commands. Keep agent observations separate from hidden simulator parameters. Run a real smoke test and report the commands and results. If graphics or asset access is unavailable, report the exact blocker. Do not build learning or a dashboard yet.

**Gate:** Scene loads locally, resets reproducibly within documented tolerances, and steps without instability. Save one screenshot and a smoke-test log.

### Phase 2: Stable pushing and baseline

**Prompt:**

> Implement approach, contact, push and retreat through the Panda actuators. Add bounded Cartesian targets, damped IK, joint/velocity limits and table-bound checks. Build a competent feedback baseline using current block position and target error, with no learning. Test diverse nominal starting positions and directions. Record trajectories, actual contact events, success and failure reasons. Do not move the block directly, bypass robot dynamics or simplify the success condition to make the test pass.

**Gate:** At least 18/20 nominal smoke-test episodes succeed; all failures remain visible. This is a controller gate, not a generalization claim.

### Phase 3: Benchmark and instrumentation

**Prompt:**

> Implement training, validation, final-test and retention manifests, hidden mass/friction randomization, effective-contact checks, and deterministic scenario replay within documented tolerances. Add observation-boundary tests and a structured episode/event schema. Record baseline performance before implementing a learner. Define all budgets, success conditions and metrics in versioned configuration. Keep final-test outcomes sealed until candidate freeze.

**Gate:** Baseline report and manifests exist; the learner-facing interface exposes no hidden physical values or scenario labels.

### Phase 4: Learning and persistent memory

**Prompt:**

> Implement the regression predictor, bounded candidate-action planner, online local updates and persistent training memory described in the brief. Start with the simplest numerical model and show prediction residuals. Save weights, normalizers, training provenance and schema version in checkpoints. Restart the process and prove the learned state reloads. Add adaptation and memory-reset modes. Do not add a runtime LLM or RL trainer yet.

**Gate:** New observations change the numerical predictions; checkpoint reload reproduces predictions; adaptation and memory can be independently disabled.

### Phase 5: Promotion, regression and ablations

**Prompt:**

> Implement candidate evaluation on validation scenarios, replay of earlier training regimes, explicit promotion gates and rollback. Compare all four systems under matched budgets. Freeze the selected checkpoint before the final test; reset it between final-test episodes and discard episode updates afterward. Produce machine-readable results and an honest report with paired uncertainty estimates and failure cases. Do not change benchmark definitions in response to test outcomes.

**Gate:** Learning claim is supported, inconclusive or rejected with evidence. A deliberately corrupted candidate fails validation and leaves the promoted checkpoint intact.

### Phase 6: Demonstration

**Prompt:**

> Build a compact local dashboard backed by real episode logs and MuJoCo-rendered video. Show baseline and learned replay, predicted versus actual motion, target error, push count, checkpoint identity and aggregate evaluation. Clearly distinguish live execution from replay. Include Start, Pause, Reset episode, Load checkpoint, Clear memory and Replay controls. Replays must show their run ID and scenario ID. Export a 60–90 second demonstration and a results report; label limitations and measured outcomes accurately.

**Gate:** The demonstration reproduces from a documented command and the displayed metrics agree with the underlying result files.

### Phase 7: Optional high-level language agent

Only after the learning proof works, add a replaceable runtime LLM interface. It may request a probe, summarize residuals or select among validated skills through a schema. Numeric observations must support its claims. The deterministic planner remains available; an invalid or timed-out model response must not block safe termination. The LLM never emits joint torques or modifies validation gates.

Treat Copilot access as the development tool. Do not assume its subscription provides credentials or credits for arbitrary runtime inference; verify any chosen provider separately.

## 8. Suggested repository contents

| Path | Contents |
| --- | --- |
| `BUILD_PLAN.md` | This brief |
| `.github/copilot-instructions.md` | Short permanent project rules below |
| `pyproject.toml` and lockfile | Package, entry points and pinned dependencies |
| `assets/` | Scene MJCF and pinned robot asset/license reference |
| `src/physical_ai_lab/` | Environment, observations, skills, controller, predictor, planner, memory, evaluator and CLI |
| `configs/` | Task, action budgets and benchmark definitions |
| `tests/` | Control limits, observation boundary, reset/replay, checkpoint and promotion tests |
| `dashboard/` | Small local evidence viewer |
| `docs/` | Architecture, decisions, progress and demo instructions |
| `runs/` | Local trajectories, metrics and videos; exclude large outputs from Git |
| `checkpoints/` | Versioned model artifacts and provenance |

Keep the first stack to Python, MuJoCo, NumPy, a small regression library if useful, pytest and plotting/video export. Add a simple dashboard only after logged results work. No ROS dependency, cloud backend, vector database or multi-agent framework is needed for V1.

## 9. Permanent Copilot instructions

Copy this block into `.github/copilot-instructions.md`:

```text
Build the MuJoCo physical learning experiment defined in BUILD_PLAN.md.
Use small runnable increments and update docs/PROGRESS.md after each phase.
Document dependency versions, robot asset revision, commands and actual results.
Never fabricate execution results, metrics, screenshots or learning curves.
Keep simulator truth separate from agent observations and learned predictors.
Never expose hidden mass, friction, scenario identity or future state to the agent.
Use real robot actuation and contact dynamics; never teleport task objects.
Use a competent fixed-feedback baseline with the same action/time budget.
Count probe actions, failures and limit violations in evaluation.
Persist numeric learned state; a textual explanation is not a learning update.
Use validation for candidate selection and reserve final test for frozen candidates.
Keep the control loop independent of any optional LLM or dashboard.
Do not weaken tests, success criteria or benchmark difficulty to claim success.
Report blockers precisely and preserve the most recent working implementation.
```

Use your strongest available reasoning model for architecture, the controller, leakage review and interpretation of experiments. Use a faster capable model for routine UI or documentation. Spend additional tokens on reviewing failing trajectories and alternative explanations for results. Keep one phase per task so errors do not compound inside a giant generation.

After each phase, request: “Review this phase against BUILD_PLAN.md. Identify the most consequential correctness issue, fix it, rerun the relevant check, and report the evidence. Do not expand scope.”

## 10. Proposed CLI and your first action

The commands below are the interface Copilot should implement; they do not exist yet:

```powershell
python -m physical_ai_lab doctor
python -m physical_ai_lab demo --mode baseline
python -m physical_ai_lab collect --split train
python -m physical_ai_lab train
python -m physical_ai_lab evaluate --split validation --all-modes
python -m physical_ai_lab evaluate --split test --checkpoint promoted --all-modes
python -m physical_ai_lab dashboard
```

Create or open a dedicated repository named `physical-ai-learning-lab`, save this file as `BUILD_PLAN.md`, and paste:

> Read BUILD_PLAN.md and implement Phase 1 now. First inspect my environment and repository. Create the permanent Copilot instructions from section 9, then implement the reproducible MuJoCo scene and smoke test. Run everything available locally. End with the exact command I can use to open the scene, evidence of what passed, and any remaining blockers. Do not begin Phase 2 until the scene and reset checks pass.

Use the local MuJoCo installation for interactive simulation. If working through a remote Copilot task, have it commit the implementation and provide local run instructions; do not assume its environment has your desktop GPU or a working display. Headless stepping and rendered video are separate checks, since rendering still needs a supported graphics setup.

## 11. A credible 90-second demo

| Time | What the audience sees | What you say |
| --- | --- | --- |
| 0–15 s | Arm, block, target, task and budget | “The task stays the same. The block's physical response changes.” |
| 15–30 s | Matched baseline attempt with actual result | “This controller corrects its position errors but does not retain experience.” |
| 30–50 s | Learned agent's predicted and observed motion; bounded retry | “It compares the result with its prediction and updates how it expects a push to move the block.” |
| 50–65 s | Fresh process loading a checkpoint; unseen test case | “The numerical model survives a restart, and this condition was held out from training.” |
| 65–80 s | Aggregate comparison and retention report | State the actual rates, sample counts and uncertainty, including failures. |
| 80–90 s | Checkpoint provenance and limits | “This demonstrates task-specific learning in simulation. Vision and hardware transfer are the next experiments.” |

Choose the illustrated case by a documented rule, such as the first test scenario or a median-error scenario, and retain the aggregate scorecard. If learning does not outperform the baseline, present the observed limitation and next hypothesis rather than scripting a success.

## 12. Expansion after the proof

1. Add an RGB-D observation pipeline and quantify the loss relative to simulator-derived pose.
2. Test unseen block sizes/shapes while retaining one robot and one task family.
3. If justified by prediction errors or planning cost, train a compact residual model or policy on collected transitions and repeat the same comparisons.
4. Transfer the task to Isaac Sim as an independent simulator check; do not interpret simulator agreement as hardware validation.
5. Introduce real hardware only through a separate calibration and controls validation project.

This experiment can later expose a reusable skill interface for broader orchestration. Keep that integration outside the first demonstration.

## Official implementation references

Checked October 2, 2026. These document the tooling; the architecture, numerical ranges and performance targets above are proposed design choices.

- MuJoCo Python bindings, stepping, viewer and rendering: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo model/contact configuration reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- MuJoCo Menagerie robot assets: https://github.com/google-deepmind/mujoco_menagerie
- Panda model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
- GitHub Copilot repository instructions: https://docs.github.com/en/copilot/how-tos/configure-custom-instructions-in-your-ide/add-repository-instructions-in-your-ide
