"""The table-plane vote is opt-in: `table_plane_support_vote` switches it on, nothing else does.

tests/test_table_plane.py pins the signed score itself. These pin the switch around it: with the key
absent or false, segment_table_with_ransac picks the plane the absolute-distance score always
picked -- including the impostor it picked on eval/2026-09-07_21-31-41 -- and process_scene_geometry
reads the key from the live tiptop config, which is where a cfg/tamp override puts it.

Run with: pytest tests/test_table_plane_gate.py -v
"""

import numpy as np
import pytest

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import apply_perception_overrides
from tiptop.perception.segmentation import TABLE_BOX_CLEARANCE, segment_table_with_ransac

SPACING = 0.006  # grid pitch, above the 5 mm voxel the fit downsamples to


def _scene():
    """A tabletop at z=0 with an undetected board 2.5 cm up on it, two objects on the table and one on
    the board -- the shape of the 2026-09-07_21-31-41 scene. By absolute distance the board is within
    3 cm of all three contact points and wins 3/3; by support only the one object on it counts."""
    n = 100
    rows, cols = np.mgrid[0:n, 0:n]
    xyz = np.stack([rows * SPACING, cols * SPACING, np.zeros((n, n))], axis=-1)
    xyz[10:40, 10:40, 2] = 0.025  # the board: a plane, but no detection
    xyz[..., 2] += 0.001 * ((rows + cols) % 3 - 1)  # a millimetre of relief, so no plane is 0 thick
    ramp = np.linspace(0.0, 0.025, 16).reshape(4, 4)
    objects = [((60, 60), 0.005), ((60, 80), 0.005), ((20, 20), 0.035)]  # (corner, bottom z)
    masks = np.zeros((len(objects), 1, n, n), dtype=np.uint8)
    for i, ((r, c), bottom) in enumerate(objects):
        xyz[r : r + 4, c : c + 4, 2] = bottom + ramp
        masks[i, 0, r : r + 4, c : c + 4] = 1
    return xyz, np.full((n, n, 3), 0.5), masks


def _table_top(support_vote):
    # open3d's RANSAC is not deterministic even under a fixed seed; these two planes are clean
    # enough that it finds both every time (0 misses in 200 runs of each vote).
    xyz, rgb, masks = _scene()
    box = segment_table_with_ransac(xyz, rgb, masks, support_vote=support_vote)
    return float(box.bounds[1, 2]) + TABLE_BOX_CLEARANCE  # undo the deliberate sink


def test_the_default_keeps_the_absolute_distance_vote():
    """Off, the fit is exactly the original -- which here crowns the board, as it did on that run."""
    assert _table_top(support_vote=False) == pytest.approx(0.025, abs=1e-3)


def test_the_support_vote_picks_the_tabletop():
    assert _table_top(support_vote=True) == pytest.approx(0.0, abs=1e-3)


@pytest.fixture
def cfg():
    c = tiptop_cfg()
    original = c.perception.get("table_plane_support_vote")
    yield c
    c.perception.table_plane_support_vote = original


class TestSwitch:
    def test_tiptop_yml_ships_it_off(self, cfg):
        assert cfg.perception.table_plane_support_vote is False

    def test_a_cfg_tamp_override_turns_it_on(self, cfg):
        assert apply_perception_overrides(cfg, {"table_plane_support_vote": True}) == {
            "table_plane_support_vote": (False, True)
        }
        assert tiptop_cfg().perception.table_plane_support_vote is True

    def test_false_is_a_value_not_a_non_positive_magnitude(self, cfg):
        cfg.perception.table_plane_support_vote = True
        assert apply_perception_overrides(cfg, {"table_plane_support_vote": False}) == {
            "table_plane_support_vote": (True, False)
        }

    @pytest.mark.parametrize("bad", ["false", "true", 0.5, 2])
    def test_only_a_real_boolean_is_accepted(self, cfg, bad):
        """bool("false") is True: a quoted value must fail, not switch the vote on."""
        with pytest.raises(ValueError, match="must be true or false"):
            apply_perception_overrides(cfg, {"table_plane_support_vote": bad})
        assert cfg.perception.table_plane_support_vote is False

    @pytest.mark.parametrize("setting, expected", [(None, False), (False, False), (True, True)])
    def test_process_scene_geometry_reads_it_from_the_live_config(self, cfg, monkeypatch, setting, expected):
        from tiptop import tiptop_run

        seen = {}

        class Stop(Exception):
            pass

        def capture(*_args, **kwargs):
            seen.update(kwargs)
            raise Stop

        monkeypatch.setattr(tiptop_run, "segment_table_with_ransac", capture)
        if setting is None:
            del cfg.perception["table_plane_support_vote"]  # a tiptop.yml that predates the key
        else:
            cfg.perception.table_plane_support_vote = setting
        xyz, rgb, masks = _scene()
        with pytest.raises(Stop):
            tiptop_run.process_scene_geometry(xyz, rgb, masks, bboxes=[], grasps={})
        assert seen["support_vote"] is expected
