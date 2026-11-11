#!/usr/bin/env python3
"""adaLN, built from BendTorch's layers, against PyTorch (BendTorch#19).

  reference/.venv/bin/python reference/adaln_check.py [probe]

`demo/adaln_probe.bend` builds a conditioned block from `Tile`, `Fork`, `Both`, `Seq`, `Id`, `Mul2`, `Add2`,
`LayerNorm`, `SiLU` and `Linear`, with no new layer in torch/, and prints y and every gradient as raw words. Two images
of three tokens, width 4 (6 rows), a condition of 3 numbers per image; dY is fixed. This builds the same tensors:

  gate   y = h + g.repeat_interleave(3) * f(h)                      f = Linear(4, 4)
  mod    y = LayerNorm(h) * m.repeat_interleave(3) + sh.repeat_interleave(3)
  id     mod with m = 1 and sh = 0 must give LayerNorm's numbers (y, dh, dgain, dbias), and the same as torch's
  block  c = silu(cond); g, m, sh = Lg(c), Lm(c), Lsh(c); y = h + g_t * f(LayerNorm(h) * m_t + sh_t)

m stands for DiT's 1 + scale, with the 1 folded into Lm's bias. Every tensor is held to oracle.py's tolerance.
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


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def values2(n, start=0):
    return np.array([((i * i * 5 + i * 3) % 17 - 8) / 4 for i in range(start, start + n)], dtype=np.float32)


def t(a, r, c):
    return torch.tensor(a.reshape(r, c)).requires_grad_(True)


def lin(i, o, s):
    return t(values(o * i, s), o, i), t(values(o, s + 40), 1, o)


def tile(x):
    return x.repeat_interleave(3, dim=0)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "adaln_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/adaln_probe.bend", "-o", binary], cwd=ROOT, check=True)
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

    # gate
    g, h = t(values(8, 20), 2, 4), t(values2(24, 30), 6, 4)
    w, b = lin(4, 4, 5)
    y = h + tile(g) * F.linear(h, w, b[0])
    y.backward(torch.tensor(values(24, 90).reshape(6, 4)))
    for name, ref in [("gate.y", y), ("gate.dg", g.grad), ("gate.dh", h.grad), ("gate.f.dw", w.grad), ("gate.f.db", b.grad)]:
        cmp(name, ref)

    # mod
    m, sh, h = t(values(8, 21), 2, 4), t(values(8, 25), 2, 4), t(values2(24, 31), 6, 4)
    lg, lb = t(values(4, 61), 1, 4), t(values(4, 67), 1, 4)
    y = F.layer_norm(h, (4,), lg[0], lb[0], EPS) * tile(m) + tile(sh)
    y.backward(torch.tensor(values(24, 95).reshape(6, 4)))
    for name, ref in [("mod.y", y), ("mod.dm", m.grad), ("mod.dsh", sh.grad), ("mod.dh", h.grad),
                      ("mod.ln.dg", lg.grad), ("mod.ln.db", lb.grad)]:
        cmp(name, ref)

    # id: zero scale and shift is LayerNorm, in Bend (the same numbers) and against torch
    h = t(values2(24, 31), 6, 4)
    lg, lb = t(values(4, 61), 1, 4), t(values(4, 67), 1, 4)
    y = F.layer_norm(h, (4,), lg[0], lb[0], EPS)
    y.backward(torch.tensor(values(24, 95).reshape(6, 4)))
    id_same = True
    for part, ref in [("y", y), ("dh", h.grad), ("ln.dg", lg.grad), ("ln.db", lb.grad)]:
        cmp(f"id.{part}", ref)
        cmp(f"ln.{part}", ref)
        if f"id.{part}" in got and f"ln.{part}" in got and not np.array_equal(got[f"id.{part}"], got[f"ln.{part}"]):
            failures.append(f"id.{part}: mod with m = 1 and sh = 0 differs from LayerNorm alone")
            id_same = False

    # block
    cond, h = t(values2(6, 23), 2, 3), t(values2(24, 33), 6, 4)
    (wg, bg), (wm, bm), (ws, bs) = lin(3, 4, 7), lin(3, 4, 9), lin(3, 4, 11)
    lg, lb = t(values(4, 61), 1, 4), t(values(4, 67), 1, 4)
    wf, bf = lin(4, 4, 5)
    c = F.silu(cond)
    gg, mm, ss = F.linear(c, wg, bg[0]), F.linear(c, wm, bm[0]), F.linear(c, ws, bs[0])
    mod = F.layer_norm(h, (4,), lg[0], lb[0], EPS) * tile(mm) + tile(ss)
    y = h + tile(gg) * F.linear(mod, wf, bf[0])
    y.backward(torch.tensor(values(24, 97).reshape(6, 4)))
    for name, ref in [("block.y", y), ("block.dcond", cond.grad), ("block.dh", h.grad),
                      ("block.g.dw", wg.grad), ("block.g.db", bg.grad), ("block.m.dw", wm.grad), ("block.m.db", bm.grad),
                      ("block.sh.dw", ws.grad), ("block.sh.db", bs.grad), ("block.ln.dg", lg.grad),
                      ("block.ln.db", lb.grad), ("block.f.dw", wf.grad), ("block.f.db", bf.grad)]:
        cmp(name, ref)

    for f in failures:
        print("  " + f)
    print(f"adaln: {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance; "
          f"mod at m = 1, sh = 0 {'is' if id_same else 'is not'} LayerNorm")
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
