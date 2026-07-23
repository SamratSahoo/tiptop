"""Learned components for tiptop.

Currently: the trajectory-*timing* network (``timing_net``) that ``neural_blending`` uses to replace the
hand-tuned time law with a profile learned from DROID human teleop. Training + DROID extraction scripts
live alongside it (``train_timing``, ``extract_stroke_timing``, ``eval_timing``); checkpoints default to
``tiptop/tiptop/checkpoints``.
"""

from tiptop.networks.timing_net import (
    DEFAULT_CKPT,
    DEFAULT_LANG_MODEL,
    EVENT_KINDS,
    KIND_TO_IDX,
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
    "DEFAULT_LANG_MODEL",
    "EVENT_KINDS",
    "KIND_TO_IDX",
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
