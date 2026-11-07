"""Does `make smol`'s test catch real bugs? Each mutant breaks the code a little, demo/smol.bend is rebuilt and
reference/test_smol.py must fail. Unlike scripts/mutate.py, the mutants go into a git worktree of HEAD under .scratch/,
so the working tree is never edited. Needs data/smol135/ (scripts/smol_data.sh). About 5 minutes a mutant.

    reference/.venv/bin/python scripts/mutate_smol.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WT = ROOT / ".scratch/mutate-smol"
ENV = dict(os.environ, BEND_NO_TELEMETRY="1", PATH=f"{Path.home()}/.bend/bin:{os.environ['PATH']}")

# (name, file, before, after, expected[, times]): expected is "caught", or why the mutant cannot be seen at this scale (its
# effect is below float32's resolution over 3 steps; measured separately), with the test that catches it; times is how
# many times `before` is in the file (1 unless given), every one of them mutated. AdamW's decay is in its three loops
# (adam_l, and adamh_l and adamhv_l for the BF16 moments since 860d12f): "no weight decay" drops it from all three.
CE_LAST = "survives: the class it drops has probability <= 1.8e-6 here; scripts/mutate.py catches it (MNIST)"
NO_DECAY = "survives: lr x wd moves each weight by 1e-6 a step; scripts/mutate.py catches it ('adamw: no decoupled weight decay')"
EPS = "survives: the final norm's mean of squares is >= 255, so eps moves 1/rms by <= 1.8e-7; a constant of the demo"
MUTANTS = [
    ("cross-entropy: the row's tree sum one class short", "torch/main.bend",
     "tree_sum(U32.from_nat(1n+k), sc)", "tree_sum(U32.from_nat(k), sc)", CE_LAST),
    ("RMSNorm backward: the last number of each row drops its mean term", "torch/main.bend",
     "R5{x, dy, g, Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32))",
     "R5{x, dy, g, Array.set(F32, dx, (off + j : U32), ((b * c : F32) * r : F32))", "caught"),
    ("embedding: a repeated id keeps only its last gradient", "torch/main.bend",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, (a + b : F32))",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, b)", "caught"),
    ("AdamW: no weight decay in main.bend's step (the norms and the embedding)", "torch/main.bend",
     "adam_decay(dec, lr, wd, x)", "x", NO_DECAY, 3),
    ("attention: scale 0.124 instead of 1/8", "demo/smol.bend",
     "64n, 0.125, {==}, {==}, {==})", "64n, 0.124, {==}, {==}, {==})", "caught"),
    ("final RMSNorm: eps 1e-4 instead of 1e-5", "demo/smol.bend",
     "~Stack(~b), ~F.RMSNorm(R(b), D(), 0.00001))", "~Stack(~b), ~F.RMSNorm(R(b), D(), 0.0001))", EPS),
]


def sh(*args, cwd=WT, check=True):
    return subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, check=check)


def main():
    if not WT.exists():
        sh("git", "worktree", "add", "--detach", str(WT), "HEAD", cwd=ROOT)
        (WT / "data").symlink_to(ROOT / "data")
    sh("git", "checkout", "--detach", subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                     text=True).stdout.strip())
    (WT / ".scratch").mkdir(exist_ok=True)  # ignored by git, so a new worktree has none
    sh("git", "checkout", "--", ".")
    sh("bend", "demo/smol.bend", "-o", ".scratch/smol")  # the control: the unmutated program builds and passes
    control = sh(sys.executable, "reference/test_smol.py", ".scratch/smol", check=False)
    assert control.returncode == 0, "the unmutated program fails test_smol.py:\n" + control.stdout[-2000:]
    print("control: the unmutated program passes", flush=True)
    results = []
    for name, path, before, after, expected, *times in MUTANTS:
        sh("git", "checkout", "--", ".")
        p = WT / path
        src = p.read_text()
        n = times[0] if times else 1
        assert src.count(before) == n, f"{name}: the text to mutate is not {n} times in {path}"
        p.write_text(src.replace(before, after))
        build = sh("bend", "demo/smol.bend", "-o", ".scratch/smol", check=False)
        if build.returncode != 0 or not (WT / ".scratch/smol").exists():
            results.append((name, expected, "caught by the build"))
        else:
            t = sh(sys.executable, "reference/test_smol.py", ".scratch/smol", check=False)
            lines = [line for line in t.stdout.splitlines() if line.startswith(("smol grads: error", "smol train 3, loss", "smol loss"))]
            results.append((name, expected, ("caught" if t.returncode != 0 else "survived") + " | " + " | ".join(lines)))
        (WT / ".scratch/smol").unlink(missing_ok=True)
        got = results[-1][2]
        ok = got.startswith("caught") == (expected == "caught")
        print(("ok  " if ok else "BAD ") + name, "->", got if expected == "caught" else got.split(" |")[0] + " (" + expected + ")", flush=True)
    sh("git", "checkout", "--", ".")
    bad = [n for n, e, g in results if g.startswith("caught") != (e == "caught")]
    print(f"mutate_smol: {len(results)} mutants, {len(results) - len(bad)} as expected" + (f"; NOT: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
