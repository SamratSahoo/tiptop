"""Can a container's mask still carry the object resting on it into its SUPPORT points?

SAM2 is seeded from Gemini's boxes, and a box around a book contains the marker lying on it -- the
book's mask came back covering 81% of the marker's pixels (3_pen_open_book/failure/2026-09-06_23-06-41).
On this branch the disjoint masks shape only the per-object support points cuTAMP fits placement
regions to (`placement_support: true`); the meshes and point clouds keep SAM2's masks, so default
perception is unchanged. The last two tests pin that split.

Ported from LJ1356/tiptop@37b9678 (the resolve_mask_overlaps tests are LJ's, unchanged).

    pytest tests/test_mask_overlaps.py -q
"""

from __future__ import annotations

import cv2
import numpy as np

from tiptop.perception.segmentation import resolve_mask_overlaps, segment_pointcloud_by_masks


def _masks(*specs, shape=(40, 40)):
    """Build (n, H, W) boolean masks from (y0, y1, x0, x1) slices."""
    out = np.zeros((len(specs), *shape), dtype=bool)
    for i, (y0, y1, x0, x1) in enumerate(specs):
        out[i, y0:y1, x0:x1] = True
    return out


def test_small_object_keeps_every_pixel_the_container_also_claimed():
    """The failing case: a marker fully contained in the book's mask must not lose a single pixel."""
    marker, book = 0, 1
    masks = _masks((10, 14, 10, 14), (0, 30, 0, 30))
    before = masks.sum(axis=(1, 2))

    out = resolve_mask_overlaps(masks, ["marker", "book"])

    assert out[marker].sum() == before[marker]
    assert (out[marker] == masks[marker]).all()
    # The book keeps everything else and gives up exactly the contested region.
    assert out[book].sum() == before[book] - before[marker]
    assert not (out[book] & masks[marker]).any()


def test_no_pixel_is_claimed_twice_and_the_union_is_unchanged():
    """Disjointness is the postcondition; no pixel may be invented or dropped from the scene."""
    masks = _masks((0, 30, 0, 30), (10, 14, 10, 14), (5, 25, 20, 35), (36, 39, 36, 39))

    out = resolve_mask_overlaps(masks, ["book", "marker", "tray", "cube"])

    assert out.sum(axis=0).max() <= 1
    assert (out.any(axis=0) == masks.any(axis=0)).all()


def test_three_way_nesting_resolves_to_the_innermost_claimant():
    """A marker on a book on a tray: each contested pixel goes to the smallest mask claiming it."""
    tray, book, marker = 0, 1, 2
    masks = _masks((0, 30, 0, 30), (5, 20, 5, 20), (10, 14, 10, 14))

    out = resolve_mask_overlaps(masks, ["tray", "book", "marker"])

    assert (out[marker] == masks[marker]).all()
    assert (out[book] == (masks[book] & ~masks[marker])).all()
    assert (out[tray] == (masks[tray] & ~masks[book])).all()


def test_disjoint_masks_are_returned_untouched():
    masks = _masks((0, 10, 0, 10), (20, 30, 20, 30))
    assert (resolve_mask_overlaps(masks, ["a", "b"]) == masks).all()


def test_single_mask_is_a_no_op():
    masks = _masks((0, 10, 0, 10))
    assert (resolve_mask_overlaps(masks, ["a"]) == masks).all()


def test_empty_mask_survives_without_claiming_anything():
    """A detection SAM2 found nothing for has area 0, so it wins every tie -- it must still stay empty."""
    empty, other = 0, 1
    masks = _masks((0, 0, 0, 0), (0, 20, 0, 20))

    out = resolve_mask_overlaps(masks, ["ghost", "book"])

    assert out[empty].sum() == 0
    assert (out[other] == masks[other]).all()


def test_more_masks_than_labels_does_not_raise():
    """The caller trims labels to len(bboxes); logging must not index past it."""
    masks = _masks((0, 30, 0, 30), (10, 14, 10, 14))
    out = resolve_mask_overlaps(masks, ["book"])
    assert out.sum(axis=0).max() <= 1


def _cloth_and_board(n=40):
    """A cloth at z=-0.015 with a puzzle board (top z=0.002) on it; the cloth's mask covers the board."""
    rows, cols = np.mgrid[0:n, 0:n]
    xyz = np.stack([cols * 0.005, rows * 0.005, np.full((n, n), -0.015)], axis=-1)
    board = np.zeros((n, n), dtype=bool)
    board[10:30, 10:30] = True
    xyz[board, 2] = 0.002
    xyz[..., 2] += 0.0005 * ((rows + cols) % 2)  # some relief, so no hull is degenerate
    rgb = np.full((n, n, 3), 0.5)
    cloth = np.ones((n, n), dtype=bool)
    masks = np.stack([cloth, board])[:, None].astype(np.uint8)
    bboxes = [{"label": "cloth", "box_2d": [0, 0, 1000, 1000]}, {"label": "board", "box_2d": [250, 250, 750, 750]}]
    return xyz, rgb, masks, bboxes, board


def test_support_points_stop_at_what_rests_on_the_surface():
    xyz, rgb, masks, bboxes, board = _cloth_and_board()
    _, _, support = segment_pointcloud_by_masks(xyz, rgb, masks, bboxes, max_z=-1.0, return_pcd=True)
    # The board's pixels are the board's: none of the cloth's support points is up at board height...
    assert support["cloth"][:, 2].max() < -0.01
    assert len(support["cloth"]) == int((~board).sum())
    # ...and the board keeps every one of its own.
    assert len(support["board"]) == int(board.sum())


def test_meshes_and_point_clouds_keep_the_masks_sam2_returned():
    """Default perception is untouched: the cloth's point cloud still includes the board's points."""
    xyz, rgb, masks, bboxes, board = _cloth_and_board()
    _, pcds, _ = segment_pointcloud_by_masks(xyz, rgb, masks, bboxes, max_z=-1.0, return_pcd=True)
    assert np.asarray(pcds["cloth"].points)[:, 2].max() > 0.0


def test_erosion_applies_to_the_disjoint_mask():
    """The hole the board opens in the cloth gets the same edge clearance as the cloth's outline."""
    xyz, rgb, masks, bboxes, board = _cloth_and_board()
    _, _, support = segment_pointcloud_by_masks(
        xyz, rgb, masks, bboxes, max_z=-1.0, return_pcd=True, erode_pixels=2
    )
    kernel = np.ones((5, 5), np.uint8)
    expected = cv2.erode((~board).astype(np.uint8), kernel, iterations=1).astype(bool)
    assert len(support["cloth"]) == int(expected.sum())
