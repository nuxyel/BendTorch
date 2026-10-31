#!/usr/bin/env python3
"""Conv2d (forward and all three gradients) against torch.

  ../bend-lab/reference/.venv-lucidrains/bin/python reference/conv2d_check.py

`demo/conv2d_probe.bend` runs n=1, h=w=4, c=2, co=3, k=3, p=1, stride 1 and prints
y, dX, dW and db as raw words. This builds the same tensors, runs F.conv2d and its
autograd, and compares all four.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def words(line):
    return np.array([int(w) for w in line.split()[1:]], dtype=np.uint32).view(np.float32)


def main():
    binary = os.path.join(ROOT, ".scratch", "conv2d_probe")
    if not os.path.exists(binary):
        subprocess.run([BEND, "demo/conv2d_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    blocks = {}
    for line in out.stdout.strip().splitlines():
        blocks[line.split()[0]] = words(line)

    got_y = blocks["y"].reshape(16, 3)
    got_dx = blocks["dx"].reshape(16, 2)
    got_dw = blocks["dw"].reshape(3, 18)
    got_db = blocks["db"].reshape(1, 3)

    x = torch.from_numpy(values(32)).reshape(1, 4, 4, 2).permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    w = torch.from_numpy(values(54, 5)).reshape(3, 2, 3, 3).requires_grad_(True)
    b = torch.from_numpy(values(3, 11)).reshape(1, 3).requires_grad_(True)
    dy = torch.from_numpy(values(48, 17)).reshape(16, 3)

    y = F.conv2d(x, w, b.reshape(3), stride=1, padding=1)                 # (1, 3, 4, 4)
    want_y = y[0].permute(1, 2, 0).reshape(16, 3).detach().numpy()
    y.backward(dy.reshape(1, 4, 4, 3).permute(0, 3, 1, 2))
    want_dx = x.grad[0].permute(1, 2, 0).reshape(16, 2).numpy()
    want_dw = w.grad.reshape(3, 18).numpy()
    want_db = b.grad.reshape(1, 3).numpy()

    gaps = {
        "y": np.abs(got_y - want_y).max(),
        "dx": np.abs(got_dx - want_dx).max(),
        "dW": np.abs(got_dw - want_dw).max(),
        "db": np.abs(got_db - want_db).max(),
    }
    for k, v in gaps.items():
        print(f"conv2d {k}: max |Bend - torch| {v:.3e}")
    ok = all(v < 1e-6 for v in gaps.values())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
