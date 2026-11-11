#!/usr/bin/env python3
"""Attention's causal products over the triangle against the full sums, bit for bit.

  reference/.venv/bin/python reference/triangle_check.py [.scratch/triangle_probe]

`demo/triangle_probe.bend` runs O = P·V, dQ = G·K, dK += Gᵀ·Q and dV += Pᵀ·dO two ways on the P and G the causal
path builds: summing only the triangle (tmul, tacc, what Attention runs) and summing every key and row (pmul, pacc,
what FullAttention runs), at four shapes, and prints how many cells differ in their bits. Every product must have no
differing cell, and the control (a P with numbers past the diagonal) must have some, or the comparison could not see
a difference. Standard library only.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")

# shape: (t, w, hd); o and dq have t·w cells, dk and dv t·hd
SHAPES = {"gqa": (7, 12, 3), "vit": (16, 64, 64), "one": (1, 4, 4), "smol": (128, 576, 64)}


def main():
    # the probe to test: the argument (make check builds it with the demos, mutate.py builds each mutant's), else
    # .scratch/triangle_probe, built when missing (an existing one is not rebuilt, so pass the binary after a change)
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "triangle_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/triangle_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    seen = {}
    for ln in out.stdout.strip().splitlines():
        m = re.fullmatch(r"(\w+) (\w+) cells (\d+) differing (\d+)", ln.strip())
        if not m:
            raise SystemExit(f"unexpected line: {ln!r}")
        seen[(m[1], m[2])] = (int(m[3]), int(m[4]))
    ok = True
    for name, (t, w, hd) in SHAPES.items():
        for what in ("o", "dq", "dk", "dv"):
            cells, diff = seen.get((name, what), (None, None))
            want = t * w if what in ("o", "dq") else t * hd
            good = cells == want and diff == 0
            ok &= good
            print(f"{name:5} {what:2}: {diff} of {cells} cells differ (want 0 of {want})" + ("" if good else "  <-"))
    cells, diff = seen.get(("control", "o"), (None, None))
    good = cells == 112 and diff is not None and diff > 0
    ok &= good
    print(f"control: a P with numbers past the diagonal differs in {diff} of {cells} cells (want some)" + ("" if good else "  <-"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
