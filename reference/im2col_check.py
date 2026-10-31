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
    lines = out.stdout.strip().splitlines()
    words = [int(w) for w in lines[0].split()[1:]]
    got = np.array(words, dtype=np.uint32).view(np.float32).reshape(16, 18)
    back = np.array([int(w) for w in lines[1].split()[1:]], dtype=np.uint32).view(np.float32).reshape(16, 2)
    back2 = np.array([int(w) for w in lines[2].split()[1:]], dtype=np.uint32).view(np.float32).reshape(16, 2)

    # torch: (1, 2, 4, 4) in the same flat order the probe builds it (row-major, channel-minor)
    flat = torch.from_numpy(values(32))
    x = flat.reshape(1, 4, 4, 2).permute(0, 3, 1, 2)  # (1, 2, 4, 4)
    want = F.unfold(x, kernel_size=3, padding=1, stride=1)[0].t().contiguous().numpy()
    want = want.reshape(16, 18)

    gap = np.abs(got - want).max()
    ident = np.abs(back - values(32).reshape(16, 2)).max()
    ident2 = np.abs(back2 - values(32).reshape(16, 2)).max()
    print(f"im2col: max |Bend - torch| {gap:.3e}  shape {got.shape}")
    print(f"col2im(im2col(x)) at k=1: max |Bend - x| {ident:.3e}")
    print(f"col2im(im2col(x)) at k=s=2, p=0: max |Bend - x| {ident2:.3e}")
    ok = gap == 0 and ident == 0 and ident2 == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
