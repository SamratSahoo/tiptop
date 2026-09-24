"""Tests for build_tamp_config -> TAMPConfiguration plumbing of the placement-region knobs.

Needs cuTAMP with support-fitted placement (cutamp.utils.support). No GPU or robot.

Run with: pytest tests/test_tamp_config.py -v
"""

import pytest

from cutamp.config import validate_tamp_config
from tiptop.motion_planning import resolve_placement_support
from tiptop.planning import build_tamp_config

BASE = dict(
    num_particles=256,
    max_planning_time=30.0,
    opt_steps=500,
    robot_type="panda",
    time_dilation_factor=0.2,
)

# The placement block of the two task configs that set placement_* keys, inlined because this
# repository does not carry the task configs: "Store Bread in Closed Box" (4_bread_box.yml, and its
# _v2/_v3/_diffusion copies, which set the same seven keys) and "Solve Constrained Puzzle"
# (1_toy_puzzle_v3.yml, which leaves the margin and flatness tolerance at their defaults).
BREAD_BOX = {
    "placement_support": True,
    "placement_support_margin": 0.005,
    "placement_support_required": True,
    "placement_into_surface": True,
    "placement_fill_occluded": True,
    "placement_min_seen_frac": 0.25,
    "placement_flatness_tol": 0.012,
}
TOY_PUZZLE_V3 = {
    "placement_support": True,
    "placement_support_required": True,
    "placement_into_surface": True,
}


class TestPlacementSupport:
    """resolve_placement_support -> build_tamp_config, the cfg/tamp `placement_support` knob."""

    def test_absent_keeps_the_bounding_box_region(self):
        for overrides in (None, {}, {"placement_support": False}, {"grasp_center_weight": 30}):
            config = build_tamp_config(**BASE, placement=resolve_placement_support(overrides))
            assert config.placement_check == "obb"
            assert config.placement_shrink_dist == 0.01
            assert config.placement_ignores_target_surface is False
            validate_tamp_config(config)

    def test_absent_builds_the_config_it_always_has(self):
        """Opt-in: with no placement_* key the config is the one build_tamp_config made before."""
        assert resolve_placement_support({}) == {}
        assert build_tamp_config(**BASE, placement=resolve_placement_support({})) == build_tamp_config(**BASE)

    def test_sub_keys_do_nothing_without_the_switch(self):
        """Every other placement_* key is read only under `placement_support: true`."""
        switched_off = {**BREAD_BOX, "placement_support": False}
        assert resolve_placement_support(switched_off) == {}
        assert build_tamp_config(**BASE, placement=resolve_placement_support(switched_off)) == build_tamp_config(
            **BASE
        )

    def test_enabled_switches_the_region_and_lets_objects_into_containers(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support({"placement_support": True}))
        assert config.placement_check == "support"
        # The support region applies its own clearance; cuTAMP rejects the two together.
        assert config.placement_shrink_dist is None
        assert config.support_margin == 0.01
        assert config.support_flatness_tol == 0.008
        assert config.placement_support_required is True
        assert config.placement_ignores_target_surface is True
        validate_tamp_config(config)

    def test_occluded_fill_is_off_unless_asked_for(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support({"placement_support": True}))
        assert config.support_fill_occluded is False
        assert config.support_min_seen_frac == 0.25
        on = build_tamp_config(
            **BASE,
            placement=resolve_placement_support(
                {"placement_support": True, "placement_fill_occluded": True, "placement_min_seen_frac": 0.4}
            ),
        )
        assert on.support_fill_occluded is True
        assert on.support_min_seen_frac == 0.4
        validate_tamp_config(on)

    def test_knobs_are_read_from_the_overrides(self):
        config = build_tamp_config(
            **BASE,
            placement=resolve_placement_support(
                {
                    "placement_support": True,
                    "placement_support_margin": 0.025,
                    "placement_support_required": False,
                    "placement_into_surface": False,
                }
            ),
        )
        assert config.support_margin == 0.025
        assert config.placement_support_required is False
        assert config.placement_ignores_target_surface is False
        validate_tamp_config(config)

    def test_the_bread_box_config_opts_in(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support(BREAD_BOX))
        assert config.placement_check == "support"
        # This task's box is only placeable with its occluded floor counted; see the yml.
        assert config.support_fill_occluded is True
        assert config.support_min_seen_frac == 0.25
        assert config.support_margin == 0.005
        assert config.support_flatness_tol == 0.012
        assert config.placement_ignores_target_surface is True
        validate_tamp_config(config)

    def test_the_toy_puzzle_config_opts_in_with_default_margins(self):
        config = build_tamp_config(**BASE, placement=resolve_placement_support(TOY_PUZZLE_V3))
        assert config.placement_check == "support"
        assert config.placement_support_required is True
        assert config.placement_ignores_target_surface is True
        assert (config.support_margin, config.support_flatness_tol) == (0.01, 0.008)
        assert config.support_fill_occluded is False
        validate_tamp_config(config)

    def test_an_out_of_range_value_is_refused_by_cutamp(self):
        """Nothing in tiptop range-checks these; cuTAMP's validate_tamp_config (run by run_cutamp) does."""
        config = build_tamp_config(
            **BASE, placement=resolve_placement_support({"placement_support": True, "placement_min_seen_frac": 1.5})
        )
        with pytest.raises(ValueError, match="support_min_seen_frac"):
            validate_tamp_config(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
