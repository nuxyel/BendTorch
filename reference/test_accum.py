"""demo/accum_test.bend against PyTorch: gradient accumulation (Train.accumulate, Train.flush; BendTorch #36).

The MLP of test_mlp.py, Linear(7, 6) -> ReLU -> Linear(6, 3) with cross-entropy, on 20 rows split into k = 4
micro-batches of 5, three optimizer steps under SGD, momentum, Adam, AdamW and AdamW with BF16 moments. For each seed:

1. k = 1: one accumulate and a flush a step, through the template forms and the _at forms, give Train.step's bits.
2. k = 4 against PyTorch's accumulated step: (loss / 4).backward() on each micro-batch, then one optimizer.step():
   every micro-batch's loss, the gradient summed over one step's four, the parameters, the moments and Adam's
   1 - b1^t and 1 - b2^t after three steps (t counts flushes: 3, not 12), all within oracle.py's tolerance; the
   gradient after the last flush is zero, and the _at forms and torch/fast.bend's linear layers give the template
   forms' bits.
3. k = 4 against one step on the 20 rows at once (Bend's Train.step): the same up to the order of the sums, within the
   same tolerance; the worst distance is printed.

    reference/.venv/bin/python reference/test_accum.py .scratch/accum_test
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oracle import bits_equal, close, run, write_f32, write_u8
from test_mlp import AdamW16, NEAR_TIE, model

K, B = 4, 5

OPTS = {
    "sgd": lambda ps: torch.optim.SGD(ps, lr=0.1),
    "momentum": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9),
    "adam": lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8),
    "adamw": lambda ps: torch.optim.AdamW(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    "adamw16mv": lambda ps: AdamW16(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, v16=True),
}
NAMES = ["w1", "b1", "w2", "b2"]


def bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1).view(np.uint32)


def state_ref(tag, m, opt):
    """Train.dump's layout: the parameters, then the moments in use, then 1 - b1^t and 1 - b2^t for Adam."""
    ps = list(m.parameters())
    flat = lambda ts: torch.cat([t.detach().reshape(-1) for t in ts])
    out = [flat(ps)]
    if tag == "momentum":
        out.append(flat([opt.state[p]["momentum_buffer"] for p in ps]))
    elif tag in ("adam", "adamw"):
        st = [opt.state[p] for p in ps]
        out += [flat([s["exp_avg"] for s in st]), flat([s["exp_avg_sq"] for s in st])]
        t = int(st[0]["step"])
        out.append(torch.tensor([1 - 0.9 ** t, 1 - 0.999 ** t], dtype=torch.float64))
    return torch.cat([o.to(torch.float64) for o in out])


def case(binary: str, seed: int, failures: list[str], worst: dict) -> None:
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(K * B, 7, generator=g)
    y = torch.randint(0, 3, (K * B,), generator=g)
    init = [torch.randn(6, 7, generator=g) * 0.5, torch.randn(6, generator=g) * 0.1,
            torch.randn(3, 6, generator=g) * 0.5, torch.randn(3, generator=g) * 0.1]
    if seed % 3 == 0:  # a row of zeros and negative inputs: relu's dead units and its zero gradient
        x[0] = 0.0
        x[1] = -x[1].abs()
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for name, t in zip(NAMES, init):
            write_f32(dp / f"{name}.bin", t)
        write_f32(dp / "x.bin", x)
        write_u8(dp / "y.bin", y.tolist())
        got = run(binary, d)

    def check(kind, name, bend, ref):
        worst[kind] = max(worst[kind], close(f"seed {seed} {name}", bend, ref, failures))
        if kind == "torch":
            s, t = bits_equal(bend, ref)
            worst["same"], worst["total"] = worst["same"] + s, worst["total"] + t

    def same(name, a, b):
        if not np.array_equal(bits(a), bits(b)):
            failures.append(f"seed {seed} {name}: not the same bits")

    mb = [(x[j * B:(j + 1) * B], y[j * B:(j + 1) * B]) for j in range(K)]

    # the gradient summed over one step's four micro-batches, each loss / 4
    m = model(init)
    for xj, yj in mb:
        (F.cross_entropy(m(xj), yj) / K).backward()
    for name, p in zip(NAMES, m.parameters()):
        check("torch", f"gsum.{name}", got[f"gsum.{name}"], p.grad)

    for tag, make in OPTS.items():
        # 1. k = 1: Train.step's bits, through both forms
        for other in ("one", "oneat"):
            same(f"{other}.{tag}.loss", got[f"{other}.{tag}.loss"], got[f"step.{tag}.loss"])
            same(f"{other}.{tag}.state", got[f"{other}.{tag}.state"], got[f"step.{tag}.state"])
            for name in NAMES:
                same(f"{other}.{tag}.g.{name}", got[f"{other}.{tag}.g.{name}"], got[f"step.{tag}.g.{name}"])
        # the _at forms, and torch/fast.bend's linear layers, give the template forms' bits at k = 4 too
        for other in ("at", "fast"):
            same(f"{other}.{tag}.loss", got[f"{other}.{tag}.loss"], got[f"acc.{tag}.loss"])
            same(f"{other}.{tag}.state", got[f"{other}.{tag}.state"], got[f"acc.{tag}.state"])

        # 2. k = 4 against PyTorch's accumulated step
        m = model(init)
        opt = make(m.parameters())
        losses = []
        for _ in range(3):
            opt.zero_grad()
            for xj, yj in mb:
                loss = F.cross_entropy(m(xj), yj) / K
                loss.backward()
                losses.append(loss.item())
            opt.step()
        check("torch", f"acc.{tag}.loss", got[f"acc.{tag}.loss"].reshape(-1)[1:], torch.tensor(losses))
        for name in NAMES:
            if np.any(bits(got[f"acc.{tag}.g.{name}"]) != 0):
                failures.append(f"seed {seed} acc.{tag}.g.{name}: the gradient is not zero after the flush")
        if tag == "adamw16mv":  # packed moments: compare the parameters, as test_mlp.py does, leaving out the ties
            n = sum(p.numel() for p in m.parameters())
            keep = ~torch.cat([opt.ties.get(p, torch.zeros_like(p, dtype=torch.bool)).reshape(-1) for p in m.parameters()])
            ref = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
            check("torch", f"acc.{tag}.params", got[f"acc.{tag}.state"].reshape(-1)[:n][keep.numpy()], ref[keep])
            worst["ties"] += int((~keep).sum())
        else:
            check("torch", f"acc.{tag}.state", got[f"acc.{tag}.state"], state_ref(tag, m, opt))

        # 3. k = 4 against Bend's one step on all 20 rows
        n = 81  # the parameters: 6·7 + 6 + 3·6 + 3
        a, b = got[f"acc.{tag}.state"].reshape(-1)[:n], got[f"big.{tag}.state"].reshape(-1)[:n]
        check("big", f"acc.{tag} against big.{tag}", a, torch.tensor(b.astype(np.float64)))
        worst["big_abs"] = max(worst["big_abs"], float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
        # the 20-row loss is the mean of the four quarter losses of the same parameters: the first step's
        l4 = got[f"acc.{tag}.loss"].reshape(-1)[1:5].astype(np.float64).sum()
        check("big", f"big.{tag}.loss[0] against acc's first four", np.array([l4]),
              torch.tensor(got[f"big.{tag}.loss"].reshape(-1)[1:2].astype(np.float64)))


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    worst = {"torch": 0.0, "big": 0.0, "big_abs": 0.0, "ties": 0, "same": 0, "total": 0}
    seeds = 8
    for seed in range(seeds):
        case(binary, seed, failures, worst)
    print(f"accum: {seeds} seeds, k = {K} micro-batches of {B} rows, {len(OPTS)} optimizers; k = 1, the _at forms and fast.bend "
          f"bit-identical; against PyTorch's accumulated step {worst['total']} numbers, {worst['same']} bit-identical, "
          f"worst {worst['torch']:.3f} of the tolerance; against one "
          f"step on all {K * B} rows worst {worst['big']:.3f} of the tolerance (largest parameter difference "
          f"{worst['big_abs']:.3e}); {worst['ties']} BF16-state weights left out, their moment within {NEAR_TIE} F32 "
          f"ulps of a rounding midpoint")
    if failures:
        print("\n".join(failures[:20]))
        sys.exit(1)
    print("accum: PASS")


if __name__ == "__main__":
    main()
