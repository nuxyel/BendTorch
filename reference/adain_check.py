#!/usr/bin/env python3
"""AdaIN, built from BendTorch's layers, against PyTorch (BendTorch#43).

  reference/.venv/bin/python reference/adain_check.py [probe]

`demo/adain_probe.bend` builds adaptive instance normalization from `Tile`, `GroupNorm` (one channel per group, which
is InstanceNorm), `Fork`, `Both`, `Seq`, `Id`, `Mul2`, `Add2` and `Linear`, with no new layer in torch/, and prints y
and every gradient as raw words. Two sequences of five steps and four channels (10 rows, channels last), a style of 3
numbers per sequence; dY is fixed. This builds the same tensors, InstanceNorm in PyTorch's (N, C, L) layout:

  mod    y = InstanceNorm(x) * m.repeat_interleave(5) + sh.repeat_interleave(5)
  id     mod with m = 1 and sh = 0 must give InstanceNorm's numbers (y, dx, dgain, dbias), and the same as torch's
  adain  m, sh = Lm(style), Lsh(style); y = InstanceNorm(x) * m_t + sh_t

m stands for AdaIN1d's 1 + gamma, with the 1 folded into Lm's bias, and Lm and Lsh are the two halves of its one
Linear. The last check writes StyleTTS2's AdaIN1d as it is, `(1 + gamma) * norm(x) + beta` with an
`nn.InstanceNorm1d(4, affine=False)` and one `nn.Linear(3, 8)` cut by `chunk`, from the same weights, and holds the
probe's y to it too. Every tensor is held to oracle.py's tolerance.
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
N, T, C, SD = 2, 5, 4, 3


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def values2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def t(a, r, c):
    return torch.tensor(a.reshape(r, c)).requires_grad_(True)


def lin(i, o, s):
    return t(values(o * i, s), o, i), t(values(o, s + 40), 1, o)


def tile(x):
    return x.repeat_interleave(T, dim=0)


def ncl(x):
    """BendTorch's rows (sequence, step) and columns (channel) as PyTorch's (N, C, L)."""
    return x.reshape(N, T, C).permute(0, 2, 1)


def rows(y):
    return y.permute(0, 2, 1).reshape(N * T, C)


def gain_bias():
    return torch.ones(C, requires_grad=True), torch.zeros(C, requires_grad=True)


def inorm(x, g, b):
    return rows(F.instance_norm(ncl(x), weight=g, bias=b, eps=EPS))


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "adain_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/adain_probe.bend", "-o", binary], cwd=ROOT, check=True)
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

    # mod
    m, sh, x = t(values(8, 21), 2, C), t(values(8, 25), 2, C), t(values2(N * T * C, 31), N * T, C)
    g, b = gain_bias()
    y = inorm(x, g, b) * tile(m) + tile(sh)
    y.backward(torch.tensor(values(N * T * C, 95).reshape(N * T, C)))
    for name, ref in [("mod.y", y), ("mod.dm", m.grad), ("mod.dsh", sh.grad), ("mod.dx", x.grad),
                      ("mod.in.dg", g.grad), ("mod.in.db", b.grad)]:
        cmp(name, ref)

    # id: a multiplier of 1 and a shift of 0 is InstanceNorm, in Bend (the same numbers) and against torch
    x = t(values2(N * T * C, 31), N * T, C)
    g, b = gain_bias()
    y = inorm(x, g, b)
    y.backward(torch.tensor(values(N * T * C, 95).reshape(N * T, C)))
    id_same = True
    for part, ref in [("y", y), ("dx", x.grad), ("in.dg", g.grad), ("in.db", b.grad)]:
        cmp(f"id.{part}", ref)
        cmp(f"in.{part}", ref)
        if f"id.{part}" in got and f"in.{part}" in got and not np.array_equal(got[f"id.{part}"], got[f"in.{part}"]):
            failures.append(f"id.{part}: mod with m = 1 and sh = 0 differs from InstanceNorm alone")
            id_same = False

    # adain
    style, x = t(values2(N * SD, 23), N, SD), t(values2(N * T * C, 33), N * T, C)
    (wm, bm), (ws, bs) = lin(SD, C, 9), lin(SD, C, 11)
    g, b = gain_bias()
    y = inorm(x, g, b) * tile(F.linear(style, wm, bm[0])) + tile(F.linear(style, ws, bs[0]))
    y.backward(torch.tensor(values(N * T * C, 97).reshape(N * T, C)))
    for name, ref in [("adain.y", y), ("adain.ds", style.grad), ("adain.dx", x.grad), ("adain.m.dw", wm.grad),
                      ("adain.m.db", bm.grad), ("adain.sh.dw", ws.grad), ("adain.sh.db", bs.grad),
                      ("adain.in.dg", g.grad), ("adain.in.db", b.grad)]:
        cmp(name, ref)

    # StyleTTS2's AdaIN1d as written: one Linear of width 2c cut in two, (1 + gamma) * norm(x) + beta
    fc, norm = torch.nn.Linear(SD, 2 * C), torch.nn.InstanceNorm1d(C, affine=False)
    with torch.no_grad():
        fc.weight.copy_(torch.cat((wm, ws)))
        fc.bias.copy_(torch.cat((bm[0] - 1, bs[0])))
        gamma, beta = torch.chunk(fc(style).view(N, 2 * C, 1), chunks=2, dim=1)
        y1d = rows((1 + gamma) * norm(ncl(x)) + beta)
    own = close("adain.y against AdaIN1d as written", got.get("adain.y", np.zeros(N * T * C)), y1d, failures)

    for f in failures:
        print("  " + f)
    print(f"adain: {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance; against "
          f"AdaIN1d as written {own:.3f}; mod at m = 1, sh = 0 {'is' if id_same else 'is not'} InstanceNorm")
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
