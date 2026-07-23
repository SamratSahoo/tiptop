"""Learned components for tiptop.

Currently: the (geometry-only) trajectory-*timing* network (``timing_net``) that ``neural_blending`` uses
to replace the hand-tuned time law with a profile learned from DROID human teleop. Training + DROID
extraction + eval scripts live alongside it; checkpoints default to ``tiptop/tiptop/checkpoints``.
"""

from tiptop.networks.timing_net import (
    DEFAULT_CKPT,
    N_CURV,
    N_FEAT,
    N_KNOTS,
    TimingModel,
    TimingNet,
    normalize_profile,
    stroke_features,
    stroke_speed_profile,
    tau_grid,
)

__all__ = [
    "DEFAULT_CKPT",
    "N_CURV",
    "N_FEAT",
    "N_KNOTS",
    "TimingModel",
    "TimingNet",
    "normalize_profile",
    "stroke_features",
    "stroke_speed_profile",
    "tau_grid",
]
