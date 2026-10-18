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
