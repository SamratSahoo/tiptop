"""``resolve_solver_effort``: cuTAMP's particle count and optimization steps, for tiptop-run and tiptop-server.

Both entrypoints resolve through it, so a config's ``num_particles`` / ``opt_steps_per_skeleton`` means the same
thing whichever one runs it.
"""

import pytest

from tiptop.motion_planning import resolve_solver_effort


def test_absent_keys_keep_the_passed_values():
    assert resolve_solver_effort(None, 256, 500) == (256, 500)
    assert resolve_solver_effort({}, 256, 500) == (256, 500)
    assert resolve_solver_effort({"num_particles": None, "opt_steps_per_skeleton": None}, 256, 500) == (256, 500)
    assert resolve_solver_effort({"encoder_weight": 25000}, 256, 500) == (256, 500)


def test_overrides_win_over_the_passed_values():
    assert resolve_solver_effort({"num_particles": 512, "opt_steps_per_skeleton": 600}, 256, 500) == (512, 600)
    assert resolve_solver_effort({"num_particles": 512}, 256, 500) == (512, 500)
    assert resolve_solver_effort({"opt_steps_per_skeleton": 600}, 256, 500) == (256, 600)


def test_values_are_read_as_ints():
    """JSON can hand back 512.0 for 512."""
    resolved = resolve_solver_effort({"num_particles": 512.0, "opt_steps_per_skeleton": 600.0}, 256, 500)
    assert resolved == (512, 600)
    assert all(type(value) is int for value in resolved)


@pytest.mark.parametrize(
    ("overrides", "num_particles", "opt_steps_per_skeleton"),
    [
        ({"num_particles": 0}, 256, 500),
        ({"opt_steps_per_skeleton": -1}, 256, 500),
        ({}, 0, 500),
        ({}, 256, 0),
    ],
)
def test_non_positive_values_are_rejected(overrides, num_particles, opt_steps_per_skeleton):
    with pytest.raises(ValueError, match="num_particles and opt_steps_per_skeleton must be positive"):
        resolve_solver_effort(overrides, num_particles, opt_steps_per_skeleton)
