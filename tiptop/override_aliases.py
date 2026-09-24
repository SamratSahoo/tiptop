"""Alternative names accepted for cfg/tamp ``tamp_overrides`` keys and values.

The trajectory-encoder knobs are read under their original names -- ``vae_path``, ``vae_manifold_weight``
and ``blend_mode: vae`` -- by several resolvers. A config may use ``encoder_path``, ``encoder_weight`` and
``blend_mode: encoder`` instead: :func:`resolve_override_aliases` rewrites them to the canonical names when
the overrides are loaded, so the resolvers only ever see those.
"""

# Alias key -> the canonical key tiptop reads.
OVERRIDE_KEY_ALIASES = {
    "encoder_path": "vae_path",
    "encoder_weight": "vae_manifold_weight",
}

# Alias ``blend_mode`` value -> the canonical value trajectory_blending reads.
BLEND_MODE_ALIASES = {"encoder": "vae"}


def resolve_override_aliases(overrides: dict) -> dict:
    """``overrides`` with alias keys renamed in place (key order kept) and alias ``blend_mode`` values mapped.

    An alias given together with its canonical key is fine when the two values are equal (one is kept),
    and a ValueError otherwise.
    """
    for alias, canonical in OVERRIDE_KEY_ALIASES.items():
        if alias in overrides and canonical in overrides and overrides[alias] != overrides[canonical]:
            raise ValueError(
                f"cuRobo overrides set both {alias}={overrides[alias]!r} and {canonical}={overrides[canonical]!r}; "
                f"{alias} is another name for {canonical}, so give only one of them"
            )
    resolved = {}
    for key, value in overrides.items():
        key = OVERRIDE_KEY_ALIASES.get(key, key)
        if key in resolved:
            continue
        if key == "blend_mode" and isinstance(value, str):
            value = BLEND_MODE_ALIASES.get(value.strip().lower(), value)
        resolved[key] = value
    return resolved
