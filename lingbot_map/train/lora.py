"""LoRA adapters for the self-distillation trainer (no peft dependency).

WHY.  Every correction arm so far (docs/gtabs-plan.md §12-14) improves the
scenes it rolled on and damages every scene it did not, and the damage grows
monotonically with steps.  The leading reading is co-adaptation of ~850 M
free parameters to the caches the model has seen (ledger N15).  LoRA turns the
update into a rank-r subspace of the frozen released weights, so the rank is a
capacity dial: if the hold-out damage scales with r while the in-domain gain
does not, the damage is spare capacity being spent on seen caches; if r=4
damages as much as full fine-tuning, the gradient direction itself is the
problem and the loss / credit path (ledger O4) is where to look next.

WHAT.  ``LoRALinear`` replaces an ``nn.Linear`` in place and KEEPS ITS
STATE-DICT KEYS: ``<name>.weight`` / ``<name>.bias`` stay the frozen released
tensors, and ``<name>.lora_A`` / ``<name>.lora_B`` are the only new keys.  So

  * ``model.state_dict()`` round-trips (the theta_0 / probe snapshots in
    trainer.py load it back with strict=True) and
  * ``merged_state_dict(model)`` returns the released key set with
    W + (alpha/r) B A folded in, which is what goes into ``ckpt["model"]``.
    Every downstream reader (experiments/_strip_optim.py, the benchmark's
    strict=False load, mcd_distance_ladder.py) therefore sees a plain
    checkpoint and needs no change.  The adapter itself is saved beside it
    under ``ckpt["lora"]`` for --resume.

★ THE ROLLOUT SEES THE ADAPTER.  ``forward`` adds the low-rank delta whether
or not grad is enabled, so the on-policy pool rolls with the adapted weights,
exactly as full fine-tuning did.  ``merged`` only exists to make a merged
module cheap to run at eval; the trainer never sets it.

Targets are named by parameter-group prefix (PARAM_GROUPS in trainer.py):
``global`` (aggregator.global_blocks -- the cache-reading path, the default
and the reason the encoder is frozen in the first place), ``frame``,
``camera`` (camera_head.trunk), ``depth`` (depth_head).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_PREFIX = {
    "global": "aggregator.global_blocks.",
    "frame": "aggregator.frame_blocks.",
    "camera": "camera_head.trunk.",
    "depth": "depth_head.",
}
DEFAULT_MODULES = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B (A x), with W, b frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Re-register the released tensors under their original names so the
        # state-dict keys do not move (see module docstring).
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.r = r
        self.alpha = float(alpha)
        self.scale = self.alpha / r
        dev, dt = self.weight.device, self.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dt))
        # Same init as the reference implementation: A random, B zero, so the
        # adapted model IS theta_0 at step 0 (the identity branch's premise).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.merged:
            return y
        d = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return y + d * self.scale

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold the delta into W (eval convenience; not used by the trainer)."""
        if not self.merged:
            self.weight.add_(self.delta().to(self.weight.dtype))
            self.merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        if self.merged:
            self.weight.sub_(self.delta().to(self.weight.dtype))
            self.merged = False

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"alpha={self.alpha}, frozen_base=True")


def _parse_targets(targets: Iterable[str]) -> List[str]:
    out = []
    for t in targets:
        t = t.strip()
        if not t:
            continue
        if t not in TARGET_PREFIX:
            raise ValueError(f"unknown LoRA target '{t}'; choose from {sorted(TARGET_PREFIX)}")
        out.append(TARGET_PREFIX[t])
    if not out:
        raise ValueError("no LoRA targets given")
    return out


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float = 0.0,
                targets: Iterable[str] = ("global",),
                modules: Iterable[str] = DEFAULT_MODULES) -> List[str]:
    """Replace the matching nn.Linear leaves in place and freeze everything else.

    Returns the qualified names of the wrapped layers.  Only ``lora_A`` /
    ``lora_B`` are left with requires_grad, so build_optimizer / clip /
    L2SP -- which all filter on requires_grad -- pick up exactly the adapter.
    """
    prefixes = _parse_targets(targets)
    modules = tuple(modules)
    wrapped: List[str] = []
    # Collect first, then mutate: replacing while iterating named_modules
    # invalidates the walk.
    todo: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
            continue
        if not any(name.startswith(p) for p in prefixes):
            continue
        if name.rsplit(".", 1)[-1] not in modules:
            continue
        todo.append((name, mod))
    for name, mod in todo:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(mod, r, alpha, dropout))
        wrapped.append(name)
    if not wrapped:
        raise RuntimeError(f"LoRA matched no nn.Linear under {prefixes} named {modules}")
    for n, p in model.named_parameters():
        p.requires_grad_(is_lora_param(n))
    return wrapped


def is_lora_param(name: str) -> bool:
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def lora_modules(model: nn.Module) -> List[Tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """The adapter alone (a few tens of MB), for --resume."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if is_lora_param(k)}


def load_lora_state(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """Strict on the adapter keys: a rank / target mismatch is an error, not a
    silent partial load."""
    want = {k for k in model.state_dict() if is_lora_param(k)}
    have = set(sd)
    if want != have:
        raise RuntimeError(
            f"LoRA state mismatch: checkpoint has {len(have)} adapter tensors, "
            f"model expects {len(want)} (missing {sorted(want - have)[:3]}..., "
            f"unexpected {sorted(have - want)[:3]}...)")
    model.load_state_dict(sd, strict=False)


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Released key set, CPU, with every adapter folded into its base weight.

    This is what ckpt["model"] holds for a LoRA run, so every consumer that
    reads ckpt["model"] into the plain GCTStream keeps working unchanged.
    """
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if not is_lora_param(k)}
    for name, m in lora_modules(model):
        key = f"{name}.weight"
        w = sd[key]
        if m.merged:
            continue          # already folded on the card
        sd[key] = (w.float() + m.delta().cpu()).to(w.dtype)
    return sd


def lora_summary(model: nn.Module) -> str:
    mods = lora_modules(model)
    n_ad = sum(p.numel() for n, p in model.named_parameters() if is_lora_param(n))
    n_all = sum(p.numel() for p in model.parameters())
    by_group: Dict[str, int] = {}
    for name, _ in mods:
        for g, pre in TARGET_PREFIX.items():
            if name.startswith(pre):
                by_group[g] = by_group.get(g, 0) + 1
    r = mods[0][1].r if mods else 0
    alpha = mods[0][1].alpha if mods else 0
    return (f"[lora] r={r} alpha={alpha} layers={len(mods)} {by_group}  "
            f"adapter {n_ad / 1e6:.2f} M of {n_all / 1e6:.0f} M ({100 * n_ad / max(n_all, 1):.2f}%)")
