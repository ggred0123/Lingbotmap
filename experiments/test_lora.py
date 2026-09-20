"""CPU checks for lingbot_map/train/lora.py (no GPU, no data).

    python experiments/test_lora.py                 # toy module, seconds
    python experiments/test_lora.py --real          # + the released GCTStream on CPU

What is checked, in the order the trainer relies on it:
  1. B = 0 at init  ->  adapted output == base output (the model IS theta_0).
  2. state_dict keys: released keys unchanged, only *.lora_A / *.lora_B added,
     and a strict=True round trip through model.state_dict() works (the theta_0
     / probe snapshots in trainer.py do exactly that).
  3. only the adapter requires grad; the base receives no gradient.
  4. after updates, merged_state_dict() loaded into a PLAIN copy reproduces the
     adapted output (what every downstream checkpoint reader sees).
  5. lora_state_dict / load_lora_state round trip on a fresh injection.
  6. --real: injection on GCTStream matches the expected layer count and the
     merged key set equals the released checkpoint's key set.
"""
import argparse
import copy
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from lingbot_map.train import lora as LORA  # noqa: E402


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        return self.proj(torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, -1) @ v)


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn, self.mlp = Attn(d), Mlp(d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Toy(nn.Module):
    """Mirrors the trainer's PARAM_GROUPS prefixes."""

    def __init__(self, d=32, n=2):
        super().__init__()
        self.aggregator = nn.Module()
        self.aggregator.patch_embed = nn.Linear(d, d)
        self.aggregator.frame_blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.aggregator.global_blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.camera_head = nn.Module()
        self.camera_head.trunk = nn.ModuleList([Block(d) for _ in range(n)])
        self.depth_head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.aggregator.patch_embed(x)
        for b in self.aggregator.frame_blocks:
            x = b(x)
        for b in self.aggregator.global_blocks:
            x = b(x)
        for b in self.camera_head.trunk:
            x = b(x)
        return self.depth_head(x)


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def toy_tests():
    torch.manual_seed(0)
    base = Toy()
    ref = copy.deepcopy(base)
    x = torch.randn(2, 5, 32)
    y0 = ref(x)
    keys0 = set(base.state_dict())

    wrapped = LORA.inject_lora(base, r=4, alpha=8, targets=("global", "camera"))
    print(LORA.lora_summary(base))
    check(len(wrapped) == 2 * 2 * 4, f"wrapped {len(wrapped)} layers (2 groups x 2 blocks x 4 linears)")
    check(torch.allclose(base(x), y0), "1. B=0 at init -> output identical to theta_0")

    keys1 = set(base.state_dict())
    added = keys1 - keys0
    check(keys0 <= keys1, "2. released keys unchanged")
    check(all(LORA.is_lora_param(k) for k in added) and len(added) == 2 * len(wrapped),
          f"2. only lora_A/lora_B added ({len(added)})")
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    base.load_state_dict(sd, strict=True)
    check(True, "2. strict=True state_dict round trip")

    train = [n for n, p in base.named_parameters() if p.requires_grad]
    check(all(LORA.is_lora_param(n) for n in train) and len(train) == 2 * len(wrapped),
          f"3. only the adapter requires grad ({len(train)} tensors)")
    base(x).sum().backward()
    check(all(p.grad is None for n, p in base.named_parameters() if not LORA.is_lora_param(n)),
          "3. base weights receive no gradient")
    base.zero_grad(set_to_none=True)

    opt = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        (base(x) ** 2).mean().backward()
        opt.step()
    y1 = base(x)
    check(not torch.allclose(y1, y0), "4. adapter actually moved the output")
    plain = Toy()
    plain.load_state_dict(LORA.merged_state_dict(base), strict=True)
    check(torch.allclose(plain(x), y1, atol=1e-5), "4. merged_state_dict into a plain model reproduces the adapted output")
    check(set(plain.state_dict()) == keys0, "4. merged key set == released key set")

    fresh = Toy()
    fresh.load_state_dict(ref.state_dict())
    LORA.inject_lora(fresh, r=4, alpha=8, targets=("global", "camera"))
    LORA.load_lora_state(fresh, LORA.lora_state_dict(base))
    check(torch.allclose(fresh(x), y1, atol=1e-6), "5. lora_state_dict -> load_lora_state round trip")
    bad = Toy()
    LORA.inject_lora(bad, r=8, alpha=8, targets=("global",))
    try:
        LORA.load_lora_state(bad, LORA.lora_state_dict(base))
        check(False, "5. rank/target mismatch must raise")
    except RuntimeError:
        check(True, "5. rank/target mismatch raises")

    m = next(mod for _, mod in LORA.lora_modules(base))
    y_a = base(x)
    m.merge_()
    check(torch.allclose(base(x), y_a, atol=1e-5), "merge_() keeps the function")
    m.unmerge_()
    check(torch.allclose(base(x), y_a, atol=1e-5), "unmerge_() restores it")


def real_test():
    from phase0_density_sweep import build_model  # noqa: E402  (experiments/ on sys.path below)
    ckpt = "/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/ckpt/lingbot-map.pt"
    model = build_model(ckpt, "cpu", 518, 14, 8192, -1, 8)
    released = set(torch.load(ckpt, map_location="cpu", weights_only=False).get("model", {}))
    wrapped = LORA.inject_lora(model, r=16, alpha=32, targets=("global",))
    print(LORA.lora_summary(model))
    check(len(wrapped) == 24 * 4, f"6. global_blocks: {len(wrapped)} wrapped (24 blocks x qkv/proj/fc1/fc2)")
    n_ad = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    check(6e6 < n_ad < 7e6, f"6. adapter size {n_ad / 1e6:.2f} M at r=16")
    merged = LORA.merged_state_dict(model)
    check(set(merged) == set(k for k in model.state_dict() if not LORA.is_lora_param(k)),
          "6. merged key set == model's released key set")
    missing = released - set(merged)
    check(not missing, f"6. every released checkpoint key present in merged ({len(missing)} missing)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    toy_tests()
    if a.real:
        sys.path.insert(0, "experiments")
        real_test()
    print("ALL OK")
