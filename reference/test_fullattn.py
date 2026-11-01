#!/usr/bin/env python3
"""The causal and full attention probabilities for the 3x3 toy, from PyTorch.

  ../bend-lab/reference/.venv-lucidrains/bin/python reference/test_fullattn.py [bend_output.txt]

`demo/fullattn_test.bend` prints the same two 3x3 matrices from Bend's own
`probs` (causal) and `fprobs` (full). This builds q and k with the same numbers,
computes softmax(scale * q k^T) with and without the causal mask, and compares
the printed values within float32 summation-order tolerance.

It also checks the invariant the design rests on: the last row of P sees every
key in both modes, so causal and full must agree there, and in Bend they agree
bit for bit.
"""
import os
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.path.expanduser("~/.bend/bin/bend")

Q = np.array([[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]], dtype=np.float32)
K = np.array([[-0.2, 0.4], [0.7, 0.1], [-0.3, -0.5]], dtype=np.float32)
SCALE = 0.5
TOL = 2e-6


def reference(causal):
    q = torch.from_numpy(Q)[None, None]
    k = torch.from_numpy(K)[None, None]
    s = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    if causal:
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1)[0, 0].numpy()


def bend_output():
    if len(sys.argv) > 1:
        return open(sys.argv[1]).read()
    out = subprocess.run([BEND, "demo/fullattn_test.bend"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    return out.stdout


def parse(text):
    blocks = {}
    for name in ("causal", "full"):
        m = re.search(rf"^{name}\n([0-9.eE+ -]+)$", text, re.M)
        if not m:
            raise SystemExit(f"no {name} block in the Bend output")
        blocks[name] = np.array([float(v) for v in m.group(1).split()],
                                dtype=np.float32).reshape(3, 3)
    return blocks


def main():
    got = parse(bend_output())
    ok = True
    for name, causal in (("causal", True), ("full", False)):
        want = reference(causal)
        gap = np.abs(got[name] - want).max()
        rows = np.abs(got[name].sum(1) - 1).max()
        print(f"{name}: max |Bend - torch| {gap:.3e}  max |row sum - 1| {rows:.3e}")
        ok = ok and gap <= TOL and rows <= TOL
    same = np.array_equal(got["causal"][2], got["full"][2])
    print(f"last row identical in both modes: {same}")
    ok = ok and same
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
