"""``grasp_rank_conf_weight`` must reach cuTAMP, and must distinguish absent from zero.

cuTAMP ranks the satisfying particles for motion refinement and executes the first one cuRobo can
plan. Its default ranking is summed M2T2 grasp confidence alone, so ``grasp_center_weight`` and
``grasp_pose_change_weight`` shape only the logged breakdown, never the grasp that runs. This knob
is what puts the soft costs back into that ranking, which makes two things worth pinning: that an
absent key leaves every existing config on the old ranking, and that ``0.0`` reaches cuTAMP as 0.0
rather than being swallowed as falsy the way ``grasp_center_weight`` is.
"""

import pytest

from tiptop.motion_planning import resolve_grasp_center_cost, resolve_grasp_rank_conf_weight


def test_absent_key_keeps_the_confidence_only_ranking():
    assert resolve_grasp_rank_conf_weight(None) is None
    assert resolve_grasp_rank_conf_weight({}) is None
    assert resolve_grasp_rank_conf_weight({"grasp_center_weight": 30}) is None


def test_zero_survives_as_a_setting():
    """0.0 means rank on soft cost alone -- the opposite of "not set", so truthiness will not do."""
    assert resolve_grasp_rank_conf_weight({"grasp_rank_conf_weight": 0.0}) == 0.0
    assert resolve_grasp_rank_conf_weight({"grasp_rank_conf_weight": 0}) == 0.0


@pytest.mark.parametrize("value", [5, 5.0, "5.0"])
def test_weight_is_read_as_a_float(value):
    """YAML can hand back an int, and tamp_overrides round-trips through JSON in some paths."""
    assert resolve_grasp_rank_conf_weight({"grasp_rank_conf_weight": value}) == 5.0


def test_gate_and_ranking_knobs_are_independent():
    """Ranking on the soft costs is only useful if a grasp soft cost is enabled, but the two keys
    are separate: one config may want the cost logged without changing which plan executes."""
    overrides = {"grasp_center_weight": 30, "grasp_rank_conf_weight": 5.0}
    assert resolve_grasp_center_cost(overrides) is True
    assert resolve_grasp_rank_conf_weight(overrides) == 5.0
    assert resolve_grasp_center_cost({"grasp_rank_conf_weight": 5.0}) is False


def test_build_tamp_config_forwards_the_weight():
    """The knob is inert unless build_tamp_config actually puts it on the TAMPConfiguration."""
    from tiptop.planning import build_tamp_config

    config = build_tamp_config(
        num_particles=32,
        max_planning_time=5.0,
        opt_steps=10,
        robot_type="fr3_robotiq",
        time_dilation_factor=1.0,
        grasp_center_cost=True,
        grasp_rank_conf_weight=5.0,
    )
    assert config.grasp_rank_conf_weight == 5.0
    assert build_tamp_config(
        num_particles=32,
        max_planning_time=5.0,
        opt_steps=10,
        robot_type="fr3_robotiq",
        time_dilation_factor=1.0,
    ).grasp_rank_conf_weight is None
