#!/usr/bin/env python3
"""Tile against PyTorch.

  reference/.venv/bin/python reference/tile_check.py [probe]

`demo/tile_probe.bend` prints, as raw 32-bit words, the forward of a 2 x 4 input broadcast over t = 3 (6 x 4) and the
backward of a 6 x 4 dY (the 2 x 4 gradient). This builds the same tensors and compares:

  y   x.repeat_interleave(3, dim=0)
  dx  dY.view(2, 3, 4).sum(dim=1)      (each image's gradient is the sum of its three rows, not the last one)

The second is the law nuxyel asked for, measured: a backward that kept the last row would pass a shape check and fail
this one.
"""
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "tile_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/tile_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    lines = out.stdout.strip().splitlines()
    got_y = np.array([int(w) for w in lines[0].split()[1:]], dtype=np.uint32).view(np.float32).reshape(6, 4)
    got_dx = np.array([int(w) for w in lines[1].split()[1:]], dtype=np.uint32).view(np.float32).reshape(2, 4)

    x = torch.tensor(values(8).reshape(2, 4))
    dy = torch.tensor(values(24, 3).reshape(6, 4))
    want_y = x.repeat_interleave(3, dim=0).numpy()
    want_dx = dy.view(2, 3, 4).sum(dim=1).numpy()

    ey = np.abs(got_y - want_y).max()
    ed = np.abs(got_dx - want_dx).max()
    print(f"Tile: max |Bend - torch| forward {ey:.3e}  backward {ed:.3e}  (tolerance 1e-6)")
    # the discriminating case: a backward that kept the last row would differ
    last_row = np.abs(got_dx - dy.view(2, 3, 4)[:, -1, :].numpy()).max()
    print(f"a last-row backward would be off by {last_row:.3e}")
    ok = ey <= 1e-6 and ed <= 1e-6 and last_row > 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
