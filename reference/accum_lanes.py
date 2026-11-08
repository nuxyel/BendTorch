"""demo/accum_lanes.bend: gradient accumulation over the layers of torch/fast.bend and torch/gpu.bend (BendTorch #36).

The demo's five models (lin, mlp, attn, gn, emb), each run four ways (acc: k = 4 micro-batches a step; big: one step on
the four at once; one: k = 1; step: Train.step), print their state, gradient and loss as 32-bit words. Given the output
of the copy built with torch/main.bend's layers and the output of a lane (torch/fast.bend's, or torch/gpu.bend's, on
the CPU threads or on the device), this checks, in both outputs:

1. k = 1 is Train.step: one's numbers are step's, bit for bit;
2. k = 4 is one step on the four micro-batches: acc's state (parameters, moments, 1 - b1^t and 1 - b2^t) and its
   gradient (zero after the flush) are big's within oracle.py's tolerance, the sums being added in another order;

and prints one line: for each model, the largest distance in ulps between the lane's numbers and main.bend's over its
four runs (the five models in order), then the largest between the lane's one and step runs. With --zero every number
of that line must be 0 (the lanes on the CPU threads); with --gpu, lin's and the k = 1 number must be (the device takes
CUDA's float exp in SiLU, softmax and cross-entropy, so the other models are printed without a bound).

    reference/.venv/bin/python reference/accum_lanes.py [--zero | --gpu] <main-lane output> <lane output>
"""

import sys

import numpy as np
import torch

from oracle import close

MODELS = ["lin", "mlp", "attn", "gn", "emb"]
RUNS = ["acc", "big", "one", "step"]
PARTS = ["state", "g", "loss"]


def read(path: str) -> dict[str, np.ndarray]:
    out = {}
    for line in open(path):
        tag, *words = line.split()
        out[tag] = np.array([int(w) for w in words], dtype=np.uint32)
    want = {f"{m}.{r}.{p}" for m in MODELS for r in RUNS for p in PARTS}
    missing = sorted(want - out.keys())
    if missing:
        sys.exit(f"accum_lanes: {path} lacks {', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}")
    return out


def ulps(a: np.ndarray, b: np.ndarray) -> int:
    """demo/gpu_layers_test.bend's distance: 0 for equal bits or two NaNs, across zero when the signs differ."""
    if a.shape != b.shape:
        return 2**32 - 1
    fa, fb = a.view(np.float32), b.view(np.float32)
    ma, mb = (a & 0x7FFFFFFF).astype(np.int64), (b & 0x7FFFFFFF).astype(np.int64)
    d = np.where((a >> 31) == (b >> 31), np.abs(ma - mb), ma + mb)
    d = np.where((a == b) | (np.isnan(fa) & np.isnan(fb)), 0, d)
    return int(d.max()) if d.size else 0


def checks(name: str, out: dict[str, np.ndarray], failures: list[str]) -> float:
    worst = 0.0
    for m in MODELS:
        for p in PARTS:
            if not np.array_equal(out[f"{m}.one.{p}"], out[f"{m}.step.{p}"]):
                failures.append(f"{name}: {m}: k = 1 differs from Train.step ({p})")
        for p in ["state", "g"]:
            acc = out[f"{m}.acc.{p}"].view(np.float32)
            big = out[f"{m}.big.{p}"].view(np.float32).astype(np.float64)
            worst = max(worst, close(f"{name}: {m}: acc against big ({p})", acc, torch.from_numpy(big), failures))
    return worst


def main(argv: list[str]) -> int:
    mode = argv[1] if argv[1:2] in (["--zero"], ["--gpu"]) else None
    paths = argv[2:] if mode else argv[1:]
    if len(paths) != 2:
        print(__doc__)
        return 2
    ref, lane = read(paths[0]), read(paths[1])
    failures: list[str] = []
    w_ref = checks("main.bend", ref, failures)
    w_lane = checks("lane", lane, failures)
    dist = [max(ulps(lane[f"{m}.{r}.{p}"], ref[f"{m}.{r}.{p}"]) for r in RUNS for p in PARTS) for m in MODELS]
    k1 = max(ulps(lane[f"{m}.one.{p}"], lane[f"{m}.step.{p}"]) for m in MODELS for p in PARTS)
    print(" ".join(str(d) for d in dist + [k1]))
    print(f"accum_lanes: acc against big at worst {max(w_ref, w_lane):.3f} of the tolerance", file=sys.stderr)
    if mode == "--zero" and any(dist + [k1]):
        failures.append("the lane's numbers are not main.bend's bits")
    if mode == "--gpu" and (dist[0] or k1):
        failures.append("lin (no exp) or k = 1 is not exact on the device")
    for f in failures[:20]:
        print(f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
