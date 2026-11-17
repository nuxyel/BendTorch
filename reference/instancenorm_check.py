#!/usr/bin/env python3
"""InstanceNorm as GroupNorm with one channel per group, against PyTorch (BendTorch#43).

  reference/.venv/bin/python reference/instancenorm_check.py [probe]

`demo/instancenorm_probe.bend` runs `GroupNorm(2n, 5n, 3n, 1n, eps)` over 2 sequences of 5 steps and 3 channels
(a 10 x 3 matrix, channels last) and prints y, dx and the gain's and bias's gradients as raw words; dY is fixed. This
builds the same tensors in InstanceNorm1d's layout, (N, C, L):

  off   F.instance_norm(x, eps=eps)                      nn.InstanceNorm1d(3), its default: gain 1 and bias 0
  on    F.instance_norm(x, weight=g, bias=b, eps=eps)    nn.InstanceNorm1d(3, affine=True)

Every tensor is held to oracle.py's tolerance. Two controls say what the probe is not: one group of all three channels
(`F.group_norm(x, 1)`) and the statistics of the whole batch (`F.batch_norm` in training mode) must both be far from
its numbers, so a wrong grouping or statistics shared between the sequences would fail.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import bits_equal, close, parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")
EPS = 1e-5
N, L, C = 2, 5, 3


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def values2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def ncl(x):
    """BendTorch's rows (sequence, step) and columns (channel) as PyTorch's (N, C, L)."""
    return x.reshape(N, L, C).permute(0, 2, 1)


def rows(y):
    return y.permute(0, 2, 1).reshape(N * L, C)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "instancenorm_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/instancenorm_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    got = parse(out.stdout)
    failures, worst, same, total = [], 0.0, 0, 0

    def cmp(name, ref):
        nonlocal worst, same, total
        if name not in got:
            failures.append(f"{name}: not printed")
            return
        worst = max(worst, close(name, got[name], ref, failures))
        s, n = bits_equal(got[name], ref)
        same, total = same + s, total + n

    dy = torch.tensor(values(N * L * C, 8).reshape(N * L, C))
    for tag in ("off", "on"):
        x = torch.tensor(values2(N * L * C).reshape(N * L, C)).requires_grad_(True)
        g = torch.tensor(values(C, 4) if tag == "on" else np.ones(C, dtype=np.float32)).requires_grad_(True)
        b = torch.tensor(values(C, 9) if tag == "on" else np.zeros(C, dtype=np.float32)).requires_grad_(True)
        if tag == "on":
            y = rows(F.instance_norm(ncl(x), weight=g, bias=b, eps=EPS))
        else:
            y = rows(F.instance_norm(ncl(x), eps=EPS))
            # the same function with the gain and the bias written out, for their gradients
            rows(F.instance_norm(ncl(x.detach()), weight=g, bias=b, eps=EPS)).backward(dy)
        y.backward(dy)
        for name, ref in [("y", y), ("dx", x.grad), ("dg", g.grad), ("db", b.grad)]:
            cmp(f"{tag}.{name}", ref)

    # the controls: what a wrong grouping, or statistics over the whole batch, would print
    x = torch.tensor(values2(N * L * C).reshape(N * L, C))
    one_group = float(np.abs(got["off.y"].reshape(N * L, C) - rows(F.group_norm(ncl(x), 1, eps=EPS)).numpy()).max())
    batch = float(np.abs(got["off.y"].reshape(N * L, C) - rows(F.batch_norm(ncl(x), None, None, training=True, eps=EPS)).numpy()).max())
    if not (one_group > 1e-3 and batch > 1e-3):
        failures.append(f"a control is too close: one group {one_group:.3e}, batch statistics {batch:.3e}")

    for f in failures:
        print("  " + f)
    print(f"instancenorm: {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance; "
          f"one group of 3 channels would be off by {one_group:.3e}, batch statistics by {batch:.3e}")
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
