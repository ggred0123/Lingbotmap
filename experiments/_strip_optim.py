"""Strip optimizer state from a training checkpoint for benchmarking.

setup_local/README.md:77 -- ckpt_train/*.pt is 11.5 GB (weights + optimizer +
stream snapshots); the bench only needs the weights, which stay byte-identical.

    python experiments/_strip_optim.py ckpt_train/a3.step200.pt bench_ckpt/sd_a3_step200.pt
"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
torch.save({"model": ck["model"], "step": ck.get("step")}, dst)
print(f"{src} -> {dst}  (step {ck.get('step')})")
