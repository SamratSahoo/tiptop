"""Alias names for the trajectory-encoder overrides (``tiptop.override_aliases``).

A cfg/tamp ``tamp_overrides`` block may say ``encoder_path`` / ``encoder_weight`` / ``blend_mode: encoder``
where tiptop reads ``vae_path`` / ``vae_manifold_weight`` / ``blend_mode: vae``. The rename happens once, in
``tiptop_websocket_server._load_curobo_overrides``, which both tiptop-run and tiptop-server load with.
"""

import json

import pytest

from tiptop.override_aliases import resolve_override_aliases

CKPT = "/ckpts/encoder.pt"
NEW_NAMES = {"encoder_weight": 25000, "encoder_path": CKPT, "blend_trajectory": True, "blend_mode": "encoder"}
OLD_NAMES = {"vae_manifold_weight": 25000, "vae_path": CKPT, "blend_trajectory": True, "blend_mode": "vae"}


def test_new_names_become_the_canonical_ones():
    resolved = resolve_override_aliases(NEW_NAMES)
    assert resolved == OLD_NAMES
    assert list(resolved) == list(OLD_NAMES)


def test_old_names_pass_through_unchanged():
    resolved = resolve_override_aliases(OLD_NAMES)
    assert resolved == OLD_NAMES
    assert list(resolved) == list(OLD_NAMES)


def test_alias_and_canonical_with_equal_values_keep_one():
    resolved = resolve_override_aliases({"encoder_path": CKPT, "grasp_threshold": 0.02, "vae_path": CKPT})
    assert resolved == {"vae_path": CKPT, "grasp_threshold": 0.02}
    assert list(resolved) == ["vae_path", "grasp_threshold"]


@pytest.mark.parametrize(
    ("alias", "canonical", "alias_value", "canonical_value"),
    [("encoder_path", "vae_path", CKPT, "/ckpts/other.pt"), ("encoder_weight", "vae_manifold_weight", 25000, 5000)],
)
def test_alias_and_canonical_that_disagree_are_rejected(alias, canonical, alias_value, canonical_value):
    with pytest.raises(ValueError, match=f"both {alias}=.* and {canonical}="):
        resolve_override_aliases({canonical: canonical_value, alias: alias_value})


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("encoder", "vae"), (" Encoder ", "vae"), ("vae", "vae"), ("spline", "spline"), ("flow", "flow")],
)
def test_blend_mode_encoder_means_vae(mode, expected):
    """Matched the way resolve_blend_config reads blend_mode: case- and whitespace-insensitive."""
    assert resolve_override_aliases({"blend_mode": mode}) == {"blend_mode": expected}


def test_renamed_keys_keep_their_position_and_the_input_is_untouched():
    overrides = {
        "grasp_threshold": 0.02,
        "encoder_weight": 25000,
        "encoder_path": CKPT,
        "blend_trajectory": True,
        "blend_mode": "encoder",
        "blend_smoothing": 3.0e-4,
    }
    before = dict(overrides)
    resolved = resolve_override_aliases(overrides)
    assert list(resolved) == [
        "grasp_threshold",
        "vae_manifold_weight",
        "vae_path",
        "blend_trajectory",
        "blend_mode",
        "blend_smoothing",
    ]
    assert overrides == before


@pytest.fixture
def load_curobo_overrides():
    """The real loader; importing the websocket server needs the full pixi environment (cuRobo, cuTAMP)."""
    from tiptop.tiptop_websocket_server import _load_curobo_overrides

    return _load_curobo_overrides


@pytest.mark.parametrize("overrides", [NEW_NAMES, OLD_NAMES], ids=["new-names", "old-names"])
def test_loading_a_json_file(load_curobo_overrides, tmp_path, overrides):
    """``tiptop-run --curobo-overrides <file>.json``."""
    path = tmp_path / "tamp_overrides.json"
    path.write_text(json.dumps(overrides, indent=2))
    resolved = load_curobo_overrides(str(path))
    assert resolved == OLD_NAMES
    assert list(resolved) == list(OLD_NAMES)


@pytest.mark.parametrize("overrides", [NEW_NAMES, OLD_NAMES], ids=["new-names", "old-names"])
def test_loading_inline_json(load_curobo_overrides, overrides):
    resolved = load_curobo_overrides(json.dumps(overrides))
    assert resolved == OLD_NAMES
    assert list(resolved) == list(OLD_NAMES)


def test_loading_a_conflicting_file_is_rejected(load_curobo_overrides, tmp_path):
    path = tmp_path / "tamp_overrides.json"
    path.write_text(json.dumps({"vae_manifold_weight": 5000, "encoder_weight": 25000}))
    with pytest.raises(ValueError, match="encoder_weight"):
        load_curobo_overrides(str(path))
