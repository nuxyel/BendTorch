"""demo/tiny_bf16opt.bend (AdamW with packed BF16 moments) against its twin, reference/tiny_bf16opt_ref.py.

For each storage choice (m: the first moment in BF16, the second in F32; mv: both in BF16):

1. `steps` AdamW steps (default 20) in Bend and in the float32 twin: the first 20 losses within the shared tolerance
   (oracle.close). The final weights against the twin's float64 run with the same BF16 storage are reported, not
   judged: a moment that lands on the other side of a BF16 rounding boundary moves its weight by up to 2^-8 of the
   update, so per-tensor errors after 20 steps are dominated by which roundings flip (test_tiny.py's rule, rms at most
   2x the float32 twin's per tensor, is printed with the tensors past it; a criterion for weights is Renan's call).
2. The criterion of M11 for BF16 storage (Renan, 2026-10-18; test_tiny_mixed.py's RATIO and WINDOW), against the float64
   twin with the same storage: over the first 20 losses Bend's largest and rms deviation at most 8x the float32 twin's;
   every mean of 100 consecutive losses within 0.05 of the reference's (from 100 steps; the long run is 2000).
3. Information: each storage choice's 100-step means against F32 AdamW (demo/tiny_llama.bend, and tiny_ref.py in
   float64), what the BF16 moments cost the training.
4. Checkpoints (when steps <= 100): 30 steps, a checkpoint, 20 more in a new process give the bits of 50 steps.

    reference/.venv/bin/python reference/test_tiny_bf16opt.py .scratch/tiny_bf16opt .scratch/tiny_llama [steps]
"""

import subprocess
import sys
import tempfile

import numpy as np
import torch

from oracle import bits_equal, close, parse

RATIO = 8.0
WINDOW = 0.05


def out_of(*cmd):
    return subprocess.run(list(cmd), check=True, capture_output=True, text=True).stdout


def floats(out):
    d = {}
    for line in out.splitlines():
        d.setdefault(line.split()[0], []).extend(float(w) for w in line.split()[1:])
    return {k: np.array(v) for k, v in d.items()}


def means(a, w=100):
    k = a.size // w
    return a[:k * w].reshape(k, w).mean(1) if k else np.array([])


def main():
    binary, f32bin = sys.argv[1], sys.argv[2]
    steps = sys.argv[3] if len(sys.argv) > 3 else "20"
    py = sys.executable
    failures: list[str] = []
    bf = parse(out_of(f32bin, "train", steps))["loss"].astype(np.float64).reshape(-1)[1:]
    f64 = floats(out_of(py, "reference/tiny_ref.py", "train64", steps))["loss"]
    for which in ("m", "mv"):
        print(f"== {which}: {'both moments' if which == 'mv' else 'the first moment'} in BF16, {steps} steps")
        b = parse(out_of(binary, "train", steps, which))
        r = parse(out_of(py, "reference/tiny_bf16opt_ref.py", "train", steps, which))
        r64 = floats(out_of(py, "reference/tiny_bf16opt_ref.py", "train64", steps, which))
        bl = b["loss"].astype(np.float64).reshape(-1)[1:]
        rl = r["loss"].astype(np.float64).reshape(-1)
        n = min(20, bl.size)
        worst = close(f"{which} losses (first {n})", bl[:n], torch.from_numpy(rl[:n].copy()), failures)
        same, total = bits_equal(b["loss"].reshape(-1)[1:n + 1], torch.from_numpy(r["loss"].reshape(-1)[:n].copy()))
        print(f"  first {n} losses against the float32 twin: worst {worst:.3f} of the tolerance, {same} of {total} "
              "bit-identical")
        wworst, wsame, wtotal, wfail, sb, st = 0.0, 0, 0, [], 0.0, 0.0
        for name, rw in r.items():
            if not name.startswith("final."):
                continue
            bw = b[name].astype(np.float64).reshape(-1)
            ref = r64[name]
            err = np.abs(bw - ref) / (1e-6 + 1e-5 * np.abs(ref))
            wworst = max(wworst, float(err.max()))
            s, t = bits_equal(b[name], torch.from_numpy(np.asarray(rw).copy()))
            wsame, wtotal = wsame + s, wtotal + t
            sb += float(np.sum((bw - ref) ** 2))
            st += float(np.sum((np.asarray(rw, dtype=np.float64).reshape(-1) - ref) ** 2))
            if err.max() > 1.0:
                eb = np.sqrt(np.mean((bw - ref) ** 2))
                et = np.sqrt(np.mean((np.asarray(rw, dtype=np.float64).reshape(-1) - ref) ** 2))
                wfail.append((eb / et, name))
        big = [f"{nm[6:]} {x:.2f}x" for x, nm in sorted(wfail, reverse=True) if x > 2.0]
        print(f"  final weights against the float64 twin (information): worst element {wworst:.3f} of the tolerance; "
              f"{len(wfail)} tensors past it, Bend's rms error {max(wfail, default=(0, ''))[0]:.2f}x the float32 twin's "
              f"at most ({', '.join(big) if big else 'none'} past 2x); over all weights {np.sqrt(sb / st):.2f}x; "
              f"{wsame} of {wtotal} numbers bit-identical to the float32 twin")
        m64 = r64["loss"]
        da, db = np.abs(bl[:n] - m64[:n]), np.abs(rl[:n] - m64[:n])
        r1 = da.max() / db.max() if db.max() > 0 else (0.0 if da.max() == 0 else np.inf)
        r2 = np.sqrt((da ** 2).mean()) / np.sqrt((db ** 2).mean()) if db.max() > 0 else r1
        print(f"  first {n} losses: Bend's deviation from the float64 twin {r1:.2f}x (max) and {r2:.2f}x (rms) the "
              f"float32 twin's (at most {RATIO:g}x)")
        if max(r1, r2) > RATIO:
            failures.append(f"{which} first {n} steps: Bend's deviation {max(r1, r2):.2f}x the twin's, over {RATIO:g}x")
        mb, mr, mm, mf, m6 = means(bl), means(rl), means(m64), means(bf), means(f64)
        if mb.size:
            w, wt = np.abs(mb - mm).max(), np.abs(mr - mm).max()
            print(f"  100-step means: Bend at most {w:.4f} from the float64 twin (the float32 twin {wt:.4f}; at most "
                  f"{WINDOW:g})")
            if w > WINDOW:
                failures.append(f"{which} 100-step means: Bend {w:.4f} from the float64 twin, over {WINDOW:g}")
            print("  means, Bend:          " + " ".join(f"{x:.4f}" for x in mb))
            print("  means, float64 twin:  " + " ".join(f"{x:.4f}" for x in mm))
            print("  means, Bend F32 AdamW:" + " ".join(f"{x:.4f}" for x in mf))
            print("  means, float64 F32:   " + " ".join(f"{x:.4f}" for x in m6))
            print(f"  BF16 moments against F32 AdamW (information): Bend's means at most {np.abs(mb - mf).max():.4f} "
                  f"from Bend F32's, the float64 twin's {np.abs(mm - m6).max():.4f} from float64 F32's; last window "
                  f"{mb[-1]:.4f} against {mf[-1]:.4f} (float64: {mm[-1]:.4f} against {m6[-1]:.4f})")
        if int(steps) <= 100:
            with tempfile.TemporaryDirectory() as d:
                ck = f"{d}/ck.bin"
                out_of(binary, "save", "30", ck, which)
                resumed = out_of(binary, "resume", ck, "30", "20", which)
            whole = out_of(binary, "train", "50", which)
            finals = [[line for line in o.splitlines() if line.startswith("final.")] for o in (resumed, whole)]
            ok = bool(finals[0]) and finals[0] == finals[1]
            if not ok:
                failures.append(f"{which} checkpoint: 30 steps + resume 20 differs from 50 steps")
            print(f"  checkpoint: 30 steps, saved, resumed for 20: the final weights {'are' if ok else 'are NOT'} the "
                  "same bits as 50 steps without a stop")
    if failures:
        print("FAIL")
        for f in failures[:20]:
            print("  " + f)
        sys.exit(1)
    print("tiny bf16opt: PASS")


if __name__ == "__main__":
    main()
