"""BendTorch #24: the optimizer configuration lives in the Train. demo/cfg_test.bend's Linear(4 -> 2) under MSE against
torch.optim.AdamW on the same weights, input and target: a learning-rate sweep given once to Train.new, a warmup and a
mid-run change through Train.with_cfg (PyTorch: param_groups[0]["lr"] before each step), a checkpoint read back into a
Train of the same or another learning rate (bit for bit against the uninterrupted runs), and the refused steps (another
optimizer kind, or a configuration OptCfg.ok rejects): finite before the first update, NaN after.

    reference/.venv/bin/python reference/test_cfg.py .scratch/cfg_test
"""

import sys

import numpy as np
import torch

from oracle import close, run

W0 = [[0.1, 0.2, 0.3, 0.4], [0.05, 0.15, 0.25, 0.35]]
B0 = [0.0, 0.1]
X = [[0.5, 0.25, 1.0, 0.75]]
T = [[1.0, 0.0]]


def f32(x: float) -> float:
    return float(np.float32(x))


def train(lrs: list[float]) -> torch.Tensor:
    """AdamW steps at the given learning rates, one per step; the parameters after them (w row by row, then b)."""
    lin = torch.nn.Linear(4, 2)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(W0))
        lin.bias.copy_(torch.tensor(B0))
    opt = torch.optim.AdamW(lin.parameters(), lr=lrs[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    x, t = torch.tensor(X), torch.tensor(T)
    for lr in lrs:
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad()
        torch.nn.functional.mse_loss(lin(x), t).backward()
        opt.step()
    return torch.cat([lin.weight.detach().reshape(-1), lin.bias.detach().reshape(-1)])


def main():
    got = run(sys.argv[1])
    failures: list[str] = []
    worst = 0.0
    # the learning rates as Bend holds them: F32 literals, and the warmup's 0.002 (k + 1) taken in F32
    warm = [f32(np.float32(0.002) * np.float32(k + 1)) if k < 5 else f32(0.01) for k in range(20)]
    want = {
        "sweep1": train([f32(0.1)] * 20),
        "sweep2": train([f32(0.01)] * 20),
        "sweep3": train([f32(0.001)] * 20),
        "warmup": train(warm),
        "change": train([f32(0.01)] * 10 + [f32(0.001)] * 10),
        "straight10": train([f32(0.01)] * 10),
        "change5": train([f32(0.01)] * 5 + [f32(0.001)] * 5),
    }
    for name, ref in want.items():
        worst = max(worst, close(name, got[name], ref, failures))
    # three learning rates given once give three runs (#24: before, the step's configuration won and they were equal)
    sweeps = {got[f"sweep{i}"].tobytes() for i in (1, 2, 3)}
    if len(sweeps) != 3:
        failures.append("sweep: the three learning rates given to Train.new did not give three runs")
    # a checkpoint read into a Train resumes bit for bit, under that Train's configuration
    for name, same in [("resume", "straight10"), ("resume_lr", "change5")]:
        if got[name].view(np.uint32).tolist() != got[same].view(np.uint32).tolist():
            failures.append(f"{name}: not {same}'s bits")
    refused = ["sgd_adamw", "adamw_sgd", "momentum_adamw", "adamw_bf16", "bad_c1"]
    for name in refused:
        l1, l2, l3, p = got[f"{name}.l1"], got[f"{name}.l2"], got[f"{name}.l3"], got[f"{name}.p"]
        if not (np.isfinite(l1).all() and np.isnan(l2).all() and np.isnan(l3).all() and np.isnan(p).all()):
            failures.append(f"{name}: a refused step must leave the first loss finite and every later number NaN")
    same = got["same_kind.p"]
    if not np.isfinite(same).all():
        failures.append("same_kind: a learning-rate change within AdamW was refused")
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print(f"cfg: the sweep, the warmup and a mid-run change match torch.optim.AdamW (worst {worst:.3f} of the "
          f"tolerance); a checkpoint resumes bit for bit under the Train it is read into; {len(refused)} refused steps "
          f"(another kind, or OptCfg.ok's rejection) are NaN from the next step on")


if __name__ == "__main__":
    main()
