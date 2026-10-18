"""Pytest plugin for the Phase 2 review: disable the gravity feedforward to show the new tests catch it.

REVIEW2_NOFIX=servo  -> PushSkill.servo_target(q) returns q.copy() (the pre-review _send command path)
REVIEW2_NOFIX=offset -> ArmKinematics.gravity_offset(q) returns zeros
"""
import os

import numpy as np


def pytest_configure(config):
    from physical_ai_lab.controller import PushSkill
    from physical_ai_lab.kinematics import ArmKinematics

    mode = os.environ.get("REVIEW2_NOFIX", "servo")
    if mode == "servo":
        PushSkill.servo_target = lambda self, q: q.copy()
    elif mode == "offset":
        ArmKinematics.gravity_offset = lambda self, q: np.zeros(len(self.arm_dadr))
    else:
        raise ValueError(mode)
    print(f"[review2_nofix] gravity feedforward disabled (mode={mode})")
