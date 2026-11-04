#!/usr/bin/env python3
"""RoPE2d against a hand-built 2D rotation.

  reference/.venv/bin/python reference/rope2d_check.py

`demo/rope2d_probe.bend` prints, as raw 32-bit words, the forward of a 2x2 image (th = tw = 2, 4 tokens, one head
of 4 numbers split 2 for y and 2 for x) and the backward applied to that forward's own output. This builds the
same input, rotates the y pair by `y` and the x pair by `x` (base 10000, so the first inverse frequency is 1 on
each axis), and compares:

  y     the forward, against the hand rotation
  back  the backward of y, against x itself (at (y,x) then (-y,-x) is the identity)

The two axes have different angles at the same distance (rows 1 and 2), which is the case a single 1D rotation
cannot pass.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")


def values(n):
    return np.array([((i * 5) % 11 - 5) / 4 for i in range(n)], dtype=np.float32)


def rot(a, b, ang):
    c = np.float32(np.cos(np.float32(ang)))
    s = np.float32(np.sin(np.float32(ang)))
    return (np.float32(a * c) - np.float32(b * s), np.float32(b * c) + np.float32(a * s))


def main():
    binary = os.path.join(ROOT, ".scratch", "rope2d_probe")
    if not os.path.exists(binary) or os.path.getmtime(binary) < os.path.getmtime(
            os.path.join(ROOT, "demo", "rope2d_probe.bend")):
        subprocess.run([BEND, "demo/rope2d_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    lines = out.stdout.strip().splitlines()
    got_y = np.array([int(w) for w in lines[0].split()[1:]], dtype=np.uint32).view(np.float32).reshape(4, 4)
    got_b = np.array([int(w) for w in lines[1].split()[1:]], dtype=np.uint32).view(np.float32).reshape(4, 4)

    x = values(16).reshape(4, 4)
    want = x.copy()
    for r in range(4):
        y, xp = r // 2, r % 2          # th = tw = 2; position p = r, y = p // tw, x = p % tw
        a, b = x[r, 0], x[r, 1]        # the y part: pair (0, 1), angle y * 1
        want[r, 0], want[r, 1] = rot(a, b, float(y))
        a, b = x[r, 2], x[r, 3]        # the x part: pair (2, 3), angle x * 1
        want[r, 2], want[r, 3] = rot(a, b, float(xp))

    dy = np.abs(got_y - want).max()
    db = np.abs(got_b - x).max()
    print(f"RoPE2d: max |Bend - hand rotation| {dy:.3e}  (tolerance 1e-6)")
    print(f"RoPE2d: max |back(fwd(x)) - x|    {db:.3e}  (tolerance 1e-6)")
    ok = dy <= 1e-6 and db <= 1e-6
    # the case a 1D rotation cannot pass: rows 1 and 2 are the same distance from row 0 along different axes
    print(f"the two axes differ: y[1] != y[2] -> {not np.allclose(got_y[1], got_y[2])}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
