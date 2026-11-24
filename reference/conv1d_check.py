#!/usr/bin/env python3
"""Conv1d (forward and all three gradients) against torch.nn.functional.conv1d on five shapes:
standard, odd-surplus stride, dilated, grouped+strided+dilated, and depthwise (k=7, one channel per group).
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")

# (tag, n, t, g, cg, cog, k, p, stride, dilation, x_start, w_start, b_start, dy_start)
CASES = (
    ("c0", 2, 5, 1, 2, 3, 3, 1, 1, 1, 0, 5, 11, 17),
    ("c1", 1, 6, 1, 2, 3, 3, 1, 2, 1, 3, 7, 13, 19),
    ("c2", 1, 7, 1, 2, 3, 3, 2, 1, 2, 9, 21, 25, 29),
    ("c3", 2, 8, 2, 2, 3, 3, 1, 2, 2, 2, 11, 23, 31),
    ("c4", 1, 9, 4, 1, 1, 7, 3, 1, 1, 6, 15, 37, 41),
)


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def words(line):
    return np.array([int(w) for w in line.split()[1:]], dtype=np.uint32).view(np.float32)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "conv1d_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/conv1d_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    blocks = {}
    for line in out.stdout.strip().splitlines():
        blocks[line.split()[0]] = words(line)

    ok = True
    for tag, n, t, g, cg, cog, k, p, stride, dilation, xs, ws, bs, ds in CASES:
        eff_k = 1 + dilation * (k - 1)
        ot = (t + 2 * p - eff_k) // stride + 1
        ci, co, cgk = g * cg, g * cog, cg * k
        rows, orows = n * t, n * ot

        got = {name: blocks[f"{tag}.{name}"] for name in ("y", "dx", "dw", "db")}
        x = torch.from_numpy(values(rows * ci, xs)).reshape(n, t, ci).permute(0, 2, 1).contiguous()
        x.requires_grad_(True)
        wt = torch.from_numpy(values(co * cgk, ws)).reshape(co, cg, k).requires_grad_(True)
        bt = torch.from_numpy(values(co, bs)).reshape(co).requires_grad_(True)
        dy = torch.from_numpy(values(orows * co, ds)).reshape(n, ot, co).permute(0, 2, 1)

        y = F.conv1d(x, wt, bt, stride=stride, padding=p, dilation=dilation, groups=g)
        want_y = y.permute(0, 2, 1).reshape(orows, co).detach().numpy()
        y.backward(dy)
        gaps = {
            "y": np.abs(got["y"].reshape(orows, co) - want_y).max(),
            "dx": np.abs(got["dx"].reshape(rows, ci) - x.grad.permute(0, 2, 1).reshape(rows, ci).numpy()).max(),
            "dW": np.abs(got["dw"].reshape(co, cgk) - wt.grad.reshape(co, cgk).numpy()).max(),
            "db": np.abs(got["db"].reshape(co) - bt.grad.reshape(co).numpy()).max(),
        }
        line = "  ".join(f"{name} {v:.3e}" for name, v in gaps.items())
        bad = [name for name, v in gaps.items() if not v < 1e-6]
        print(f"{tag}  n={n} t={t} g={g} cg={cg} cog={cog} k={k} p={p} s={stride} d={dilation}  {line}  {'FAIL ' + ','.join(bad) if bad else 'ok'}")
        ok &= not bad
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
