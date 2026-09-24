"""Where a surface's observed points go, and what a support-region failure turns into.

`placement_support: true` (see tiptop.motion_planning.resolve_placement_support) fits each placement
region to the surface's OBSERVED points rather than its bounding box. These tests pin the plumbing
around that fit, which is the same whether the fit is on or off:

- segment_pointcloud_by_masks keeps each object's raw points, including the ones below `max_z` that
  its point cloud drops -- the floor of a shallow container is exactly those points;
- create_tamp_environment hangs them on the environment for the detected SURFACES only;
- run_planning reports NoSupportRegion as a planning failure, and owns its tolerance dicts rather
  than writing into cuTAMP's module-level defaults.

Needs cuRobo and cuTAMP importable (tiptop.tiptop_run imports both). No GPU or robot.

Run with: pytest tests/test_placement_support_plumbing.py -v
"""

import copy

import numpy as np
import pytest
from curobo.geom.types import Cuboid, Mesh

from tiptop import planning, tiptop_run
from tiptop.perception.segmentation import segment_pointcloud_by_masks

UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]
TABLE = Cuboid(name="table", dims=[1.2, 1.6, 0.1], pose=[0.5, 0.0, -0.07, *UNIT_QUAT])


def _mesh(name, center, half=0.03) -> Mesh:
    corners = [[sx * half, sy * half, sz * half] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return Mesh(name=name, vertices=corners, faces=[[0, 1, 2]], pose=[*center, *UNIT_QUAT])


MESHES = {
    "box": _mesh("box", (0.6, 0.2, 0.05)),
    "plate": _mesh("plate", (0.4, 0.2, 0.01)),
    "bread": _mesh("bread", (0.5, -0.1, 0.03)),
}
GOAL = [{"predicate": "on", "args": ["bread", "box"]}]
POINTS = {label: np.full((50, 3), i, dtype=float) for i, label in enumerate(["box", "plate", "bread", "table"])}


class TestEnvironment:
    def test_support_points_reach_the_detected_surfaces_only(self):
        env, _ = tiptop_run.create_tamp_environment(
            MESHES, TABLE, GOAL, include_workspace=False, support_points=POINTS
        )
        # Not the movable bread, and not the table: its cuboid is a RANSAC slab deliberately offset
        # from its observed top face, which a support fit would fight.
        assert set(env.support_points) == {"box"}
        assert env.support_points["box"] is POINTS["box"]

    def test_a_surface_named_by_the_caller_gets_its_points_too(self):
        """The reset path's extra_surface_labels make a surface of the plate; it is fitted like any other."""
        env, _ = tiptop_run.create_tamp_environment(
            MESHES, TABLE, GOAL, include_workspace=False, extra_surface_labels={"plate"}, support_points=POINTS
        )
        assert set(env.support_points) == {"box", "plate"}

    def test_no_support_points_is_an_empty_map(self):
        env, _ = tiptop_run.create_tamp_environment(MESHES, TABLE, GOAL, include_workspace=False)
        assert env.support_points == {}

    def test_a_surface_without_points_is_left_out(self):
        env, _ = tiptop_run.create_tamp_environment(
            MESHES, TABLE, GOAL, include_workspace=False, support_points={"plate": POINTS["plate"]}
        )
        assert env.support_points == {}

    def test_processed_scene_defaults_to_no_points(self):
        scene = tiptop_run.ProcessedScene(table_cuboid=TABLE, object_meshes={}, object_pcds={}, grasps={})
        assert scene.object_support_points == {}


class TestRunPlanning:
    @staticmethod
    def _plan(monkeypatch, run_cutamp):
        monkeypatch.setattr(planning, "run_cutamp", run_cutamp)
        env, surfaces = tiptop_run.create_tamp_environment(MESHES, TABLE, GOAL, include_workspace=False)
        config = planning.build_tamp_config(
            num_particles=8, max_planning_time=1.0, opt_steps=1, robot_type="panda", time_dilation_factor=0.2
        )
        return planning.run_planning(
            env, config, q_init=np.zeros(7), ik_solver=None, grasps={}, motion_gen=None, all_surfaces=surfaces
        )

    def test_no_support_region_is_a_planning_failure(self, monkeypatch):
        from cutamp.utils.support import NoSupportRegion

        def raise_it(*_a, **_k):
            raise NoSupportRegion("No level patch of 'box' is large enough to support an object")

        plan, _, reason = self._plan(monkeypatch, raise_it)
        assert plan is None
        assert reason.startswith("No level patch of 'box'")

    def test_no_grasps_is_still_a_planning_failure(self, monkeypatch):
        from cutamp.particle_initialization import NoGraspsError

        def raise_it(*_a, **_k):
            raise NoGraspsError("no M2T2 grasps for 'bread'")

        plan, _, reason = self._plan(monkeypatch, raise_it)
        assert plan is None and "bread" in reason

    def test_cutamps_default_tolerances_are_not_written_into(self, monkeypatch):
        """The per-surface tolerances go into this call's own copy, not the process-wide default.

        Checked by identity as well as by value: a shallow copy shares cuTAMP's inner per-type dicts,
        and once any earlier call in the process has written through one, a before/after snapshot
        taken here would compare the already-mutated default with itself.
        """
        from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

        before = copy.deepcopy((default_constraint_to_tol, default_constraint_to_mult))
        seen = {}

        class RecordingChecker(planning.ConstraintChecker):
            def __init__(self, constraint_to_tol):
                seen["tol"] = constraint_to_tol
                super().__init__(constraint_to_tol)

        class RecordingReducer(planning.CostReducer):
            def __init__(self, constraint_to_mult):
                seen["mult"] = constraint_to_mult
                super().__init__(constraint_to_mult)

        monkeypatch.setattr(planning, "ConstraintChecker", RecordingChecker)
        monkeypatch.setattr(planning, "CostReducer", RecordingReducer)
        plan, _, reason = self._plan(monkeypatch, lambda *_a, **_k: ([], None, None))
        assert plan == [] and reason is None
        for handed, default in ((seen["tol"], default_constraint_to_tol), (seen["mult"], default_constraint_to_mult)):
            assert handed is not default
            for con_type, inner in handed.items():
                assert inner is not default.get(con_type), f"{con_type} shares cuTAMP's default dict"
        assert (default_constraint_to_tol, default_constraint_to_mult) == before
        # The tolerances did reach the checker cuTAMP was handed, the yaw one included (inert unless
        # a support region pins a placement's yaw).
        placement_tol = seen["tol"][planning.StablePlacement.type]
        for surface in ("table", "box"):
            assert placement_tol[f"{surface}_in_xy"] == 1e-2
            assert placement_tol[f"{surface}_yaw"] == 7e-2


def _container_scene(n=40, floor_z=0.01, wall_z=0.08):
    """A (n, n) structured cloud of an open box: a low floor ringed by tall walls, on a table at z=0."""
    rows, cols = np.mgrid[0:n, 0:n]
    xyz = np.stack([cols * 0.005, rows * 0.005, np.zeros((n, n))], axis=-1)
    box = np.zeros((n, n), dtype=bool)
    box[5:35, 5:35] = True
    floor = np.zeros((n, n), dtype=bool)
    floor[8:32, 8:32] = True
    xyz[box, 2] = wall_z
    xyz[floor, 2] = floor_z
    # A little relief on the walls so their hull is not degenerate.
    xyz[box & ~floor, 2] += 0.001 * (rows[box & ~floor] % 3)
    rgb = np.full((n, n, 3), 0.5)
    masks = box[None, None].astype(np.uint8)
    return xyz, rgb, masks, [{"label": "box", "box_2d": [0, 0, 1000, 1000]}], floor


class TestSegmentation:
    def test_support_points_keep_the_container_floor_the_point_cloud_drops(self):
        xyz, rgb, masks, bboxes, floor = _container_scene()
        max_z = 0.03  # between the floor (0.01) and the walls (0.08), as a near-table filter would be
        meshes, pcds, support = segment_pointcloud_by_masks(xyz, rgb, masks, bboxes, max_z=max_z, return_pcd=True)
        assert set(meshes) == set(pcds) == set(support) == {"box"}
        # The point cloud (and so the mesh) is built from the walls alone...
        assert np.asarray(pcds["box"].points)[:, 2].min() > max_z
        # ...while the support points are everything observed under the mask, floor included.
        assert len(support["box"]) == int(masks[0, 0].sum())
        assert np.isclose(support["box"][:, 2].min(), 0.01)
        assert int(np.isclose(support["box"][:, 2], 0.01).sum()) == int(floor.sum())

    def test_without_return_pcd_it_is_still_just_the_meshes(self):
        xyz, rgb, masks, bboxes, _ = _container_scene()
        meshes = segment_pointcloud_by_masks(xyz, rgb, masks, bboxes, max_z=0.03)
        assert isinstance(meshes, dict) and set(meshes) == {"box"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
