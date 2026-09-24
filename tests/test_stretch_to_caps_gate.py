"""``blend_stretch_to_caps`` gates the slow-into-the-caps fallbacks, and is off by default.

Without the switch, a stroke no clock in [d_lo, d_hi] can fit raises in ``vae_retime_group`` and
``blend_cutamp_plan`` keeps that operation's original segments exactly as cuRobo emitted them -- the
behaviour every config had before the fallbacks existed. With it, the stroke is slowed past d_hi
until it fits, and a run whose blending failed is slowed into the same caps (see
tests/test_trajectory_retiming.py for what the two fallbacks themselves guarantee).

The VAE scorer and optimizer are stubbed: which candidate the optimizer proposes is not under test,
only what happens once none of them fits.
"""

import numpy as np
import pytest
import torch

from tiptop import trajectory_blending, vae_retiming
from tiptop.trajectory_blending import BlendConfig, blend_cutamp_plan, resolve_blend_config
from tiptop.vae_retiming import _KNOTS, _time_knots, vae_retime_group

DOF = 7
DT = 0.02


class TestResolveBlendConfig:
    def test_off_by_default(self):
        assert resolve_blend_config({"blend_trajectory": True, "blend_mode": "vae"}).stretch_to_caps is False
        assert BlendConfig().stretch_to_caps is False

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_sets_it(self, value):
        cfg = resolve_blend_config({"blend_trajectory": True, "blend_stretch_to_caps": value})
        assert cfg.stretch_to_caps is value

    @pytest.mark.parametrize("bad", ["false", "true", 0.5, 2])
    def test_only_a_real_boolean_is_accepted(self, bad):
        """bool("false") is True: a quoted value must fail, not switch the fallback on."""
        with pytest.raises(ValueError, match="must be true or false"):
            resolve_blend_config({"blend_trajectory": True, "blend_stretch_to_caps": bad})


class _Scorer:
    device = torch.device("cpu")

    def score_emitted(self, pos, duration, target):
        return 0.0


def _stub_vae(monkeypatch, duration):
    """One candidate: the even time-warp at ``duration`` seconds."""
    tau = _time_knots(torch.zeros(1, _KNOTS - 1))
    monkeypatch.setattr(vae_retiming, "_scorer", lambda *_: _Scorer())
    monkeypatch.setattr(vae_retiming, "_optimize", lambda *_args, **_kw: [(duration, tau)])


def _sweep(n=16, sweep=1.5):
    s = np.linspace(0.0, 1.0, n)[:, None]
    q0 = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.8])
    return q0 + (3 * s**2 - 2 * s**3) * sweep


def _retime(stretch_to_caps, caps):
    positions = _sweep()
    orig = (len(positions) - 1) * DT
    vel_cap, acc_cap = np.full(DOF, caps[0]), np.full(DOF, caps[1])
    smoothing, lead_speed, trail_speed, max_duration_mult = 3e-4, 0.0, 0.0, 2.0
    out = vae_retime_group(
        positions,
        DT,
        orig,
        vel_cap,
        acc_cap,
        smoothing,
        lead_speed,
        trail_speed,
        max_duration_mult,
        stretch_to_caps=stretch_to_caps,
    )
    return out, orig


class TestVaeRetimeGroup:
    def test_without_the_switch_a_stroke_no_clock_fits_raises(self, monkeypatch):
        _stub_vae(monkeypatch, duration=0.6)
        with pytest.raises(RuntimeError, match=r"no duration in \[.*\]s that meets the velocity"):
            _retime(False, caps=(0.5, 2.0))

    def test_with_the_switch_it_is_slowed_into_the_caps(self, monkeypatch):
        _stub_vae(monkeypatch, duration=0.6)
        (pos, vel, acc, dt_out), orig = _retime(True, caps=(0.5, 2.0))
        assert dt_out * (len(pos) - 1) > 2.0 * orig  # past d_hi
        assert np.abs(vel).max() <= 0.5
        assert np.abs(acc).max() <= 2.0

    def test_an_admissible_stroke_is_the_same_either_way(self, monkeypatch):
        _stub_vae(monkeypatch, duration=0.6)
        (off, _), (on, _) = _retime(False, caps=(1e3, 1e4)), _retime(True, caps=(1e3, 1e4))
        for a, b in zip(off, on):
            np.testing.assert_array_equal(a, b)


class _Plan:
    """The fields blend_cutamp_plan and _slow_to_caps read off a cuRobo JointState."""

    def __init__(self, velocity):
        self.velocity = velocity
        self.position = torch.cumsum(velocity * DT, dim=0)
        self.acceleration = torch.zeros_like(velocity)
        self.jerk = torch.zeros_like(velocity)
        self.joint_names = [f"j{i}" for i in range(DOF)]


def _plan_with_one_fast_operation():
    t = torch.linspace(0.0, 1.0, 50)[:, None].repeat(1, DOF)
    fast = {"type": "trajectory", "plan": _Plan(4.0 * torch.sin(torch.pi * t)), "dt": DT, "label": "Place(x)"}
    return [fast, {"type": "gripper", "label": "open"}]


class TestBlendCutampPlanFallback:
    @pytest.fixture(autouse=True)
    def _blending_fails(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise RuntimeError("no clock fits")

        monkeypatch.setattr(trajectory_blending, "_blend_trajectory_steps", fail)

    def _run(self, **config):
        plan = _plan_with_one_fast_operation()
        out = blend_cutamp_plan(plan, BlendConfig(enabled=True, **config), np.full(DOF, 1.0), np.full(DOF, 1e3))
        return plan, out

    def test_without_the_switch_the_original_segments_pass_through(self):
        plan, out = self._run()
        assert len(out) == len(plan)
        assert all(a is b for a, b in zip(out, plan))

    def test_with_the_switch_they_are_slowed_into_the_caps(self):
        plan, out = self._run(stretch_to_caps=True)
        assert out[0] is not plan[0]
        assert out[0]["dt"] > plan[0]["dt"]
        assert float(out[0]["plan"].velocity.abs().max()) <= 1.0
        assert out[1] is plan[1]
