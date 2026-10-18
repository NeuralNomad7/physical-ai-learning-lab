"""Agent process entry point: ``python -m physical_ai_lab.agent_process --agent baseline``.

See learner_interface.py for the protocol. This process never creates a ScenarioRunner, never
loads the manifests, benchmark, evaluation or instrumentation modules, and holds no evaluator
state; tests/test_learner_interface.py checks the modules it loads and what their source names.
(kinematics.py imports scenario.verify_actuators, a static check of the agent's own nominal
model; it gives no access to the evaluator's simulation, which lives in another process.)
"""

from __future__ import annotations

import sys

from .learner_interface import main

if __name__ == "__main__":
    sys.exit(main())
