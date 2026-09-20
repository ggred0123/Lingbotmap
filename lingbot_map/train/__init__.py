"""Training pipeline for the streaming self-distillation (docs/phase1-plan.md).

The released package is inference only; everything under ``train/`` is Phase 1.

    losses      T3  the §3.3 pair -- gauge-invariant, unlabeled
    label_bank  T4  offline fresh-teacher label generation and sampling
"""

from lingbot_map.train.losses import (
    SelfDistillLoss, rel_pose_loss, depth_si_loss,
    closed_form_scale, closed_form_scale_and_shift,
)

__all__ = [
    "SelfDistillLoss", "rel_pose_loss", "depth_si_loss",
    "closed_form_scale", "closed_form_scale_and_shift",
]
