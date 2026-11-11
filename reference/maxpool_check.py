#!/usr/bin/env python3
"""MaxPool2d (forward and the input gradient) against torch, on three shapes.

  reference/.venv/bin/python reference/maxpool_check.py

`demo/maxpool_probe.bend` prints y and dX as raw words and this builds the same
tensors, runs F.max_pool2d(..., return_indices=True) and its autograd, and
compares both:

  c0  1 x 2 x 4 x 4, k 2, s 2   — windows tile the image
  c1  1 x 2 x 5 x 5, k 3, s 2   — windows overlap by one row and column
  c2  2 x 1 x 4 x 3, k 3, s 1   — every output overlaps its neighbours; two
      outputs can send gradient to the same input cell, and the backward adds

The tie case is c2's value sequence plus the check's own: where two cells of a
window are equal, PyTorch's index is the first in row-major order and the
gradient goes to that cell. `make check` runs this.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")

# (tag, n, c, h, w, k, s, x_start, dy_start)
CASES = (
    ("c0", 1, 2, 4, 4, 2, 2, 0, 11),
    ("c1", 1, 2, 5, 5, 3, 2, 3, 17),
    ("c2", 2, 1, 4, 3, 3, 1, 5, 23),
)


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def words(line):
    return np.array([int(w) for w in line.split()[1:]], dtype=np.uint32).view(np.float32)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "maxpool_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/maxpool_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    blocks = {}
    for line in out.stdout.strip().splitlines():
        blocks[line.split()[0]] = words(line)

    ok = True
    for tag, n, c, h, w, k, s, xs, ds in CASES:
        oh, ow = (h - k) // s + 1, (w - k) // s + 1
        rows, orows = h * w, oh * ow
        got = {name: blocks[f"{tag}.{name}"] for name in ("y", "dx")}
        x = torch.from_numpy(values(n * rows * c, xs)).reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        x.requires_grad_(True)
        dy = torch.from_numpy(values(n * orows * c, ds)).reshape(n, oh, ow, c).permute(0, 3, 1, 2)
        y, idx = F.max_pool2d(x, kernel_size=k, stride=s, return_indices=True)
        want_y = y.permute(0, 2, 3, 1).reshape(n * orows, c).detach().numpy()
        y.backward(dy)
        gaps = {
            "y": np.abs(got["y"].reshape(n * orows, c) - want_y).max(),
            "dx": np.abs(got["dx"].reshape(n * rows, c) - x.grad.permute(0, 2, 3, 1).reshape(n * rows, c).numpy()).max(),
        }
        line = "  ".join(f"{key} {v:.3e}" for key, v in gaps.items())
        bad = [key for key, v in gaps.items() if not v < 1e-6]
        print(f"{tag}  {n}x{c}x{h}x{w} k{k} s{s}  {line}  {'FAIL ' + ','.join(bad) if bad else 'ok'}")
        ok &= not bad

    # the probe's tie case: 1x1x2x2, k 2, s 1, [[1, 1], [0, 0]] -> max at index 0, gradient there
    tie_y, tie_dx = blocks["c3.y"][0], blocks["c3.dx"]
    probe_tie = float(tie_y) == 1.0 and tie_dx.tolist() == [1.0, 0.0, 0.0, 0.0]
    print(f"c3  tie in the window  y {float(tie_y)}, dx {tie_dx.tolist()}  {'ok' if probe_tie else 'FAIL'}")
    ok &= probe_tie

    x = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], requires_grad=True)
    y, idx = F.max_pool2d(x, kernel_size=2, stride=1, return_indices=True)
    y.backward(torch.ones_like(y))
    tie_ok = float(y.item()) == 1.0 and int(idx.item()) == 0 and x.grad[0, 0].flatten().tolist() == [1.0, 0.0, 0.0, 0.0]
    print(f"tie  torch index {int(idx.item())}, grad {x.grad[0, 0].flatten().tolist()}  {'ok' if tie_ok else 'FAIL'}")
    ok &= tie_ok

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
