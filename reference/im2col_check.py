#!/usr/bin/env python3
"""im2col against torch's unfold.

  ../bend-lab/reference/.venv-lucidrains/bin/python reference/im2col_check.py

`demo/im2col_probe.bend` prints the gathered matrix for n=1, h=w=4, c=2, k=3,
p=1, s=1 as raw 32-bit words (Mat.print). This builds the same x and compares
against F.unfold, whose layout is (learned, unfold writes (c, kh, kw) per
position in exactly the order the gather uses).
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")


def values(n):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(n)], dtype=np.float32)


def main():
    binary = os.path.join(ROOT, ".scratch", "im2col_probe")
    if not os.path.exists(binary):
        subprocess.run([BEND, "demo/im2col_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    words = [int(w) for w in out.stdout.split()[1:]]
    got = np.array(words, dtype=np.uint32).view(np.float32).reshape(16, 18)

    # torch: (1, 2, 4, 4) in the same flat order the probe builds it (row-major, channel-minor)
    flat = torch.from_numpy(values(32))
    x = flat.reshape(1, 4, 4, 2).permute(0, 3, 1, 2)  # (1, 2, 4, 4)
    want = F.unfold(x, kernel_size=3, padding=1, stride=1)[0].t().contiguous().numpy()
    want = want.reshape(16, 18)

    gap = np.abs(got - want).max()
    print(f"im2col: max |Bend - torch| {gap:.3e}  shape {got.shape}")
    print("PASS" if gap == 0 else "FAIL")
    sys.exit(0 if gap == 0 else 1)


if __name__ == "__main__":
    main()
