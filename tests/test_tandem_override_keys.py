"""The TANDEM branch's own tamp_overrides keys pass the strict check in override_keys.

TANDEM adds surface-fitted placement, two perception switches and the stretch-to-caps fallback on top
of main; each is read from tamp_overrides, so each must be in SUPPORTED_OVERRIDE_KEYS or every config
that sets it fails at startup. The pre-rename spelling of the fallback is refused like the others.
"""

import pytest

from tiptop.override_keys import SUPPORTED_OVERRIDE_KEYS, check_override_keys

TANDEM_KEYS = {
    "placement_support": True,
    "placement_support_margin": 0.005,
    "placement_flatness_tol": 0.012,
    "placement_support_required": True,
    "placement_into_surface": True,
    "placement_fill_occluded": True,
    "placement_min_seen_frac": 0.25,
    "table_plane_support_vote": True,
    "disjoint_object_masks": True,
    "retime_stretch_to_caps": True,
}


def test_every_tandem_key_is_supported():
    assert set(TANDEM_KEYS) <= SUPPORTED_OVERRIDE_KEYS
    check_override_keys(TANDEM_KEYS)


def test_the_old_name_of_the_fallback_is_refused():
    with pytest.raises(ValueError, match=r"\['blend_stretch_to_caps'\]"):
        check_override_keys({"blend_stretch_to_caps": True})
