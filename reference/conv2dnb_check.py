#!/usr/bin/env python3
"""Conv2dNB (forward and both gradients) against torch, on three shapes.

  reference/.venv/bin/python reference/conv2dnb_check.py

`demo/conv2dnb_probe.bend` prints y, dX and dW as raw words for each of three
cases and this builds the same tensors, runs F.conv2d(..., bias=None) and its
autograd, and compares all three:

  c0  1 x 4 x 4, 2 channels, 3 out, 3 x 3, padding 1, stride 1  — a padded border
  c1  1 x 7 x 5, the same kernel, padding 1, stride 2           — h + 2p - k = 6, even
  c2  1 x 7 x 6, the same kernel, padding 1, stride 2           — h + 2p - k = 5, odd: the
      last window floors, and col2im's guards skip the candidate that runs off the end

The no-bias layer has the weight alone: P is co x c*k*k, the parameter count is co*c*k*k, and
the forward, backward and optimizer step are LinearNB's. `make check` runs this.
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")

# (tag, h, w, c, co, k, p, stride, x_start, w_start, dy_start)
CASES = (
    ("c0", 4, 4, 2, 3, 3, 1, 1, 0, 5, 17),
    ("c1", 7, 5, 2, 3, 3, 1, 2, 3, 7, 19),
    ("c2", 7, 6, 2, 3, 3, 1, 2, 9, 21, 29),
)


def values(n, start=0):
    return np.array([((i * 7) % 13 - 6) / 4 for i in range(start, start + n)], dtype=np.float32)


def words(line):
    return np.array([int(w) for w in line.split()[1:]], dtype=np.uint32).view(np.float32)


def main():
    if len(sys.argv) > 1:
        binary = os.path.abspath(sys.argv[1])
    else:
        binary = os.path.join(ROOT, ".scratch", "conv2dnb_probe")
        if not os.path.exists(binary):
            subprocess.run([BEND, "demo/conv2dnb_probe.bend", "-o", binary], cwd=ROOT, check=True)
    out = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    blocks = {}
    for line in out.stdout.strip().splitlines():
        blocks[line.split()[0]] = words(line)

    ok = True
    for tag, h, w, c, co, k, p, stride, xs, ws, ds in CASES:
        oh = (h + 2 * p - k) // stride + 1
        ow = (w + 2 * p - k) // stride + 1
        rows, orows, ckk = h * w, oh * ow, c * k * k

        got = {name: blocks[f"{tag}.{name}"] for name in ("y", "dx", "dw")}
        x = torch.from_numpy(values(rows * c, xs)).reshape(1, h, w, c).permute(0, 3, 1, 2).contiguous()
        x.requires_grad_(True)
        wt = torch.from_numpy(values(co * ckk, ws)).reshape(co, c, k, k).requires_grad_(True)
        dy = torch.from_numpy(values(orows * co, ds)).reshape(1, oh, ow, co).permute(0, 3, 1, 2)

        y = F.conv2d(x, wt, None, stride=stride, padding=p)
        want_y = y[0].permute(1, 2, 0).reshape(orows, co).detach().numpy()
        y.backward(dy)
        gaps = {
            "y": np.abs(got["y"].reshape(orows, co) - want_y).max(),
            "dx": np.abs(got["dx"].reshape(rows, c) - x.grad[0].permute(1, 2, 0).reshape(rows, c).numpy()).max(),
            "dW": np.abs(got["dw"].reshape(co, ckk) - wt.grad.reshape(co, ckk).numpy()).max(),
        }
        line = "  ".join(f"{k} {v:.3e}" for k, v in gaps.items())
        bad = [k for k, v in gaps.items() if not v < 1e-6]
        print(f"{tag}  {h}x{w} k{k} p{p} s{stride}  {line}  {'FAIL ' + ','.join(bad) if bad else 'ok'}")
        ok &= not bad
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
