"""The PyTorch twin of demo/tiny_bf16opt.bend: tiny_ref.py's Llama trained by AdamW with its moments stored in BF16.

    reference/.venv/bin/python reference/tiny_bf16opt_ref.py train <steps> <m|mv>    # float32: the loss of each step
    reference/.venv/bin/python reference/tiny_bf16opt_ref.py train64 <steps> <m|mv>  # the same in float64 (the reference)

AdamW is written out, in torch.optim.adam._single_tensor_adam's order (decoupled weight decay), because
torch.optim.AdamW keeps its state in the parameter's dtype and updates the weights with the moments before any rounding
to a storage type. Here each moment is rounded to BF16 right after its update (bf(x) = x.to(torch.bfloat16).to(x.dtype):
to nearest, ties to even) and the weight update reads the stored, rounded value, as demo/tiny_bf16opt.bend does.
m: only the first moment is stored in BF16; mv: both. The weights and the gradient stay float32 (float64 in train64,
where only the moments' storage is BF16).
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

import tiny_ref as R


def bf(x):
    return x.to(torch.bfloat16).to(x.dtype)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    mode, steps, which = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert which in ("m", "mv"), which
    p = R.load()
    if mode == "train64":
        p = {k: v.detach().double().requires_grad_(True) for k, v in p.items()}
    lr, (b1, b2), eps, wd = R.LR, R.BETAS, R.ADAM_EPS, R.WD
    m = {k: torch.zeros_like(v) for k, v in p.items()}
    v2 = {k: torch.zeros_like(v) for k, v in p.items()}
    for step in range(steps):
        for t in p.values():
            t.grad = None
        x, y = R.batch(step)
        loss = R.forward(p, x, y)
        loss.backward()
        t1 = step + 1
        step_size = lr / (1 - b1 ** t1)
        bc2 = math.sqrt(1 - b2 ** t1)
        with torch.no_grad():
            for k, w in p.items():
                g = w.grad
                w.mul_(1 - lr * wd)
                m[k] = bf(m[k].lerp(g, 1 - b1))
                nv = v2[k].mul(b2).addcmul(g, g, value=1 - b2)
                v2[k] = bf(nv) if which == "mv" else nv
                w.addcdiv_(m[k], v2[k].sqrt().div(bc2).add(eps), value=-step_size)
        print("loss", R.bits(loss) if loss.dtype == torch.float32 else repr(loss.item()))
    for name, _ in R.shapes():
        print(f"final.{name}", R.bits(p[name]) if p[name].dtype == torch.float32
              else " ".join(repr(float(x)) for x in p[name].detach().reshape(-1)))


if __name__ == "__main__":
    main()
