"""Work out what a scene reset has to achieve, so the next rollout can start from a clean table.

A finished TAMP rollout leaves the scene in its GOAL state -- for "place the toys on the plate" the
toys end up on the plate -- so the next episode cannot start until someone puts them back. The
data-collection UI's "Reset scene" button runs one extra, deliberately UNRECORDED TAMP plan that
does exactly that against the already-warm session.

This module is the pure half of that: given the perceived objects, decide which ones are stacked on
another object and emit the ``on(obj, table)`` atoms that put them back, plus clip the table down to
the sub-box those placements are allowed to land in (``reset_placement_region``,
``clip_surface_to_region``). It touches no robot, no cuRobo solver and no GPU, so it is unit-testable
(``tests/test_scene_reset.py``); the orchestration lives in ``tiptop_run._run_scene_reset``.

**The rule is purely geometric, and deliberately so.** An object needs resetting iff some OTHER
perceived object is supporting it, where "A supports B" means B's centroid lies inside A's XY
footprint and A's footprint is the larger of the two. Two things fall out of that one relation: what
to move (the supported objects) and what must be protected from being moved (the supporting ones --
see ``build_reset_goal``).

Two alternatives were tried and rejected against the archived runs under
``data-collection/runs/*/tamp/*/``:

* *Matching the previous rollout's goal by label.* Perception renames the same physical object
  between runs of the SAME instruction often enough to matter -- in ``tdf0.5_jd_blend`` one plate is
  ``plate`` in 33 runs and ``white_plate`` in 17. Anything keyed on the label silently loses both the
  "what to move" and the "don't move the plate" halves in a third of resets.
* *Height above the table.* There is no usable datum. The table collision cuboid's top is
  ``surface_z - 0.02`` by construction (``perception/segmentation.py``), and measured against the
  true plane a flat plate's reconstructed underside still reads 2-3 cm high -- overlapping the
  standing objects -- because a container reconstructs as a shell of its camera-visible top face.

Neither the previous goal nor any table height is read here as a result.
"""

import logging
from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)

# An object counts as sitting on a surface when its centroid falls inside that surface's XY
# footprint, grown by this margin -- so a toy perched half over a plate's rim still counts.
DEFAULT_XY_MARGIN = 0.02

# Smallest placement zone worth planning against, per axis (metres). A zone narrower than a couple
# of object diameters has nowhere to put a second toy, so an empty/absurd config is refused loudly
# rather than turned into "no reset plan" twenty seconds later.
MIN_REGION_EXTENT = 0.10


def reset_placement_region(overrides: dict | None = None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The configured ``((x_lo, x_hi), (y_lo, y_hi))`` reset placements are confined to, or None.

    Robot base frame, metres. Two layers, most specific first: ``reset_placement_region`` in a
    ``cfg/tamp/*.yml``'s ``tamp_overrides`` (per data-collection config, same JSON the other knobs
    ride -- see :func:`goal_clearing.resolve_clear_goal_surfaces`), then
    ``scene_reset.placement_region`` in the tiptop config. None at both -- the key absent, or
    explicitly null in either -- means "anywhere on the perceived table", which is what every config
    did before this existed.

    Absolute rather than an inset from the perceived table because what this keeps the arm away from
    -- the third-person camera on its tripod, the table edges -- does not move when RANSAC's table
    fit wobbles by a few cm from run to run.
    """
    if overrides is not None and "reset_placement_region" in overrides:
        # Present-but-null/false is a deliberate "no region for this config", not a fall-through.
        region = overrides["reset_placement_region"] or None
        source = "tamp_overrides"
    else:
        region = (tiptop_cfg().get("scene_reset") or {}).get("placement_region")
        source = "tiptop config scene_reset.placement_region"
    if region is None:
        return None
    try:
        x_lo, x_hi = (float(v) for v in region["x"])
        y_lo, y_hi = (float(v) for v in region["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"placement region from {source} must look like {{x: [lo, hi], y: [lo, hi]}}, got {region!r}"
        ) from exc
    for axis, (lo, hi) in (("x", (x_lo, x_hi)), ("y", (y_lo, y_hi))):
        if hi - lo < MIN_REGION_EXTENT:
            raise ValueError(
                f"placement region from {source}: {axis} spans {hi - lo:.3f} m, under the "
                f"{MIN_REGION_EXTENT} m floor -- too narrow to place into"
            )
    return (x_lo, x_hi), (y_lo, y_hi)


def clip_surface_to_region(surface, region: tuple[tuple[float, float], tuple[float, float]] | None):
    """``surface`` with its XY footprint intersected with ``region``, or ``surface`` unchanged.

    Cuboid in, Cuboid out: only ``dims[:2]`` and ``pose[:2]`` move, so the z extent -- which is what
    placement height is read off (``place_4dof_sampler`` takes the OBB's ``surface_z``) -- is
    untouched, and the box stays axis-aligned as perception produced it.

    Returns ``surface`` itself when ``region`` is None, and logs and returns it unchanged when the
    intersection would be degenerate: a mis-set region must not silently become an unplaceable
    sliver, and the un-clipped behaviour is the one every run before this had.
    """
    if region is None:
        return surface
    (x_lo, x_hi), (y_lo, y_hi) = region
    center = np.asarray(surface.pose, dtype=float)[:2]
    half = np.asarray(surface.dims, dtype=float)[:2] / 2.0
    lo = np.maximum(center - half, [x_lo, y_lo])
    hi = np.minimum(center + half, [x_hi, y_hi])
    extent = hi - lo
    if np.any(extent < MIN_REGION_EXTENT):
        _log.warning(
            f"Placement region x={[x_lo, x_hi]} y={[y_lo, y_hi]} meets the perceived "
            f"'{surface.name}' (x={(center[0] - half[0]):.3f}..{(center[0] + half[0]):.3f}, "
            f"y={(center[1] - half[1]):.3f}..{(center[1] + half[1]):.3f}) in only "
            f"{extent[0]:.3f} x {extent[1]:.3f} m -- ignoring the region and placing on the whole "
            "surface. Check the placement region against where the table actually is."
        )
        return surface
    clipped_center = (lo + hi) / 2.0
    _log.info(
        f"Confining placements to x={lo[0]:.3f}..{hi[0]:.3f}, y={lo[1]:.3f}..{hi[1]:.3f} "
        f"({extent[0]:.3f} x {extent[1]:.3f} m of '{surface.name}')"
    )
    return replace(
        surface,
        dims=[float(extent[0]), float(extent[1]), float(surface.dims[2])],
        pose=[float(clipped_center[0]), float(clipped_center[1]), *surface.pose[2:]],
    )


def world_aabb(obj) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounds ``(lo, hi)`` of a perceived cuRobo Cuboid or Mesh.

    Both come out of the perception pipeline with an identity rotation and their centre in
    ``pose[:3]`` (``convert_trimesh_to_curobo_mesh`` re-centres the vertices, ``Cuboid.dims`` is the
    full extent), so the local bounds translate straight into the world.
    """
    center = np.asarray(obj.pose, dtype=float)[:3]
    vertices = getattr(obj, "vertices", None)
    if vertices is not None:
        verts = np.asarray(vertices, dtype=float)
        return center + verts.min(axis=0), center + verts.max(axis=0)
    dims = getattr(obj, "dims", None)
    if dims is not None:
        half = np.asarray(dims, dtype=float) / 2.0
        return center - half, center + half
    raise TypeError(f"Cannot compute world bounds for {type(obj).__name__} (no vertices, no dims)")


def _footprint_hull(obj) -> ConvexHull | None:
    """Convex hull of ``obj``'s vertices projected into the world XY plane, or None.

    None for a Cuboid (it has no vertices, and its axis-aligned bounds are already exact) and for a
    degenerate outline Qhull cannot triangulate; both fall back to the bounds.
    """
    vertices = getattr(obj, "vertices", None)
    if vertices is None:
        return None
    points = np.asarray(vertices, dtype=float)[:, :2] + np.asarray(obj.pose, dtype=float)[:2]
    try:
        return ConvexHull(points)
    except (QhullError, ValueError):
        return None


def footprint_area(obj) -> float:
    """Area of ``obj``'s XY footprint. Used to decide which of two overlapping objects is on top."""
    hull = _footprint_hull(obj)
    if hull is not None:
        return float(hull.volume)  # for a 2-D hull scipy's `volume` IS the enclosed area
    lo, hi = world_aabb(obj)
    return float((hi[0] - lo[0]) * (hi[1] - lo[1]))


def _within_footprint(obj, surface, xy_margin: float) -> bool:
    """Whether ``obj``'s centroid sits inside ``surface``'s XY footprint, grown by ``xy_margin``.

    Against the real outline rather than the bounding box: a round plate's AABB is its circumscribing
    square, so an object standing on the table diagonally beside one would otherwise be read as
    sitting on it and pointlessly picked up.
    """
    centroid = np.asarray(obj.pose, dtype=float)[:2]
    hull = _footprint_hull(surface)
    if hull is not None:
        # Qhull's equations are unit-normal half-spaces `n·x + d <= 0`, so the left-hand side is a
        # signed distance and comparing it to the margin grows the hull outwards.
        return bool(np.all(hull.equations[:, :2] @ centroid + hull.equations[:, 2] <= xy_margin))
    lo, hi = world_aabb(surface)
    return bool(np.all(centroid >= lo[:2] - xy_margin) and np.all(centroid <= hi[:2] + xy_margin))


def supporting_surfaces(object_meshes: dict, xy_margin: float = DEFAULT_XY_MARGIN) -> dict[str, str]:
    """Map each stacked object to the object holding it up: ``{supported_label: surface_label}``.

    "A supports B" is B's centroid inside A's margin-grown XY footprint with A's footprint the larger
    of the two. Area is what settles which one is on top, and it needs no height datum: a plate or a
    bowl is several times wider than whatever is sitting in it -- including a toy sunk BELOW a deep
    bowl's rim, which no height comparison would catch -- whereas two objects standing side by side
    on the table do not overlap in XY at all.

    An object enclosed by several candidates is attributed to the SMALLEST, i.e. the innermost thing
    it is directly resting on: a toy on a plate on a pad is on the plate, not the pad.
    """
    areas = {label: footprint_area(obj) for label, obj in object_meshes.items()}
    supported_by: dict[str, str] = {}
    for label, obj in object_meshes.items():
        candidates = [
            other
            for other, surface in object_meshes.items()
            if other != label and areas[other] > areas[label] and _within_footprint(obj, surface, xy_margin)
        ]
        if candidates:
            supported_by[label] = min(candidates, key=lambda other: areas[other])
    return supported_by


def build_reset_goal(
    object_meshes: dict, table_cuboid, xy_margin: float = DEFAULT_XY_MARGIN
) -> tuple[list[dict], set[str]]:
    """Grounded atoms + designated surfaces for a scene reset.

    Returns ``on(obj, <table>)`` for the objects to put back, and the set of labels that must be
    treated as SURFACES even though nothing is placed on them. That second half is load-bearing:
    surfaces are otherwise inferred from the second argument of the goal's ``on`` atoms, and a reset
    goal names nothing but the table -- so the plate the objects are coming off would be classified
    movable and the planner would be free to pick the plate up instead of the toys.

    Only the LEAVES of the support tree move: an object that is itself holding something up is never
    picked, so the arm can never carry a loaded plate across the table. Clearing the leaves is what
    unloads it, and the next reset can move it if it genuinely needs moving.

    An empty atom list means nothing is stacked, and the caller skips planning entirely.
    """
    supported_by = supporting_surfaces(object_meshes, xy_margin)
    surfaces = set(supported_by.values())
    atoms = [
        {"predicate": "on", "args": [label, table_cuboid.name]}
        for label in sorted(supported_by)
        if label not in surfaces
    ]
    return atoms, surfaces


def reset_goal_builder(xy_margin: float = DEFAULT_XY_MARGIN):
    """The ``goal_builder`` ``tiptop_run.run_perception`` calls to plan a reset instead of a task.

    Takes ``(processed_scene, detected_atoms)`` and returns ``(atoms, surface_labels)``. The atoms
    Gemini grounded from the instruction are discarded: the instruction is there to steer DETECTION,
    and the goal is the geometric one above.
    """

    def build(processed_scene, _detected_atoms):
        atoms, surfaces = build_reset_goal(processed_scene.object_meshes, processed_scene.table_cuboid, xy_margin)
        _log.info(
            f"Reset goal: {[a['args'] for a in atoms] or '(nothing is stacked)'}; "
            f"supporting surfaces {sorted(surfaces) or '(none)'}"
        )
        return atoms, surfaces

    return build
