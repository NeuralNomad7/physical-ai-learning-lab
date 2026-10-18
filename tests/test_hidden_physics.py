"""Hidden mass and friction are really applied and measurably change the contact dynamics.

The kick test gives the resting block a horizontal speed v0 and measures the slide
distance d; for Coulomb friction mu_eff = v0^2 / (2 g d).
"""

from __future__ import annotations

import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.diagnostics import kick_test
from physical_ai_lab.scenario import HiddenPhysics, ScenarioRunner


@pytest.mark.parametrize("mass_end", ["low", "high"])
@pytest.mark.parametrize("friction_end", ["low", "high"])
def test_kick_measures_the_configured_friction(
    runner: ScenarioRunner, cfg: SceneConfig, mass_end: str, friction_end: str
) -> None:
    c, p = cfg.checks, cfg.physics
    mass = getattr(p.mass_range, mass_end)
    mu = getattr(p.sliding_friction_range, friction_end)
    result = kick_test(runner, HiddenPhysics(mass, mu), v0=c.kick_speed)
    assert abs(result.relative_error) <= c.kick_mu_rel_tolerance, result
    normal_error = abs(result.normal_force - result.expected_normal_force) / result.expected_normal_force
    assert normal_error <= c.normal_force_rel_tolerance, result
    assert result.contact_frictions and all(f == pytest.approx(mu, rel=1e-12) for f in result.contact_frictions)
    assert result.inertia == pytest.approx((mass * cfg.block.edge**2 / 6.0,) * 3, rel=1e-12)  # solid cube
    assert result.final_speed < 1e-3
    assert runner.applied_physics() == HiddenPhysics(mass, mu)


def test_friction_changes_the_slide_and_mass_changes_the_load(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c, p = cfg.checks, cfg.physics
    slippery = kick_test(runner, HiddenPhysics(p.nominal_mass, p.sliding_friction_range.low), v0=c.kick_speed)
    grippy = kick_test(runner, HiddenPhysics(p.nominal_mass, p.sliding_friction_range.high), v0=c.kick_speed)
    # d is proportional to 1/mu: a 4x friction ratio must give roughly a 4x distance ratio.
    assert slippery.distance / grippy.distance == pytest.approx(
        p.sliding_friction_range.high / p.sliding_friction_range.low, rel=0.1
    )
    light = kick_test(runner, HiddenPhysics(p.mass_range.low, p.nominal_sliding_friction), v0=c.kick_speed)
    heavy = kick_test(runner, HiddenPhysics(p.mass_range.high, p.nominal_sliding_friction), v0=c.kick_speed)
    assert heavy.normal_force / light.normal_force == pytest.approx(p.mass_range.high / p.mass_range.low, rel=0.02)
    # Coulomb sliding: the same friction gives the same slide whatever the mass.
    assert heavy.distance == pytest.approx(light.distance, rel=2 * c.kick_mu_rel_tolerance)
