"""Mutation testing: change the code or the specification on purpose and check that a gate catches the change.

Each mutant replaces one exact piece of text in one file, runs the gate that should see it, and restores the file.
A mutant is caught when the gate fails. Some mutants are expected to survive: they compute the same function (an
equivalent mutant) or they sit outside what any gate checks; those are listed with the reason, so the report shows
the limits of the checks as well as their reach.

    reference/.venv/bin/python scripts/mutate.py        # or: make mutate
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV = dict(os.environ, BEND_NO_TELEMETRY="1",
           PATH=f"{Path.home()}/.bend/bin:{Path.home()}/.elan/bin:{os.environ['PATH']}")
PY = "reference/.venv/bin/python"


def laws():
    out = subprocess.run(["bend", "PROOF.bend", "--check-only"], capture_output=True, text=True, env=ENV).stdout
    return "ALL PROOFS CHECK" in out


def numeric(demo, test):
    def gate():
        b = subprocess.run(["bend", f"demo/{demo}.bend", "-o", f".scratch/mut_{demo}"], capture_output=True, env=ENV)
        if b.returncode != 0:
            return False
        return subprocess.run([PY, f"reference/{test}", f".scratch/mut_{demo}"], capture_output=True).returncode == 0
    return gate


OPS, MLP = numeric("ops_test", "test_ops.py"), numeric("mlp_test", "test_mlp.py")
TEN = numeric("tensor_test", "test_tensor.py")
IMAP = numeric("imap_check", "check_imap.py")
ST = numeric("safetensors_test", "test_safetensors.py")


ST135 = Path(os.environ.get("LLAMA_BEND", "../llama.bend")) / "data/smollm2-135m/hf/model.safetensors"


def smol_st():
    # demo/smol.bend's loss with the weights read piece by piece from SmolLM2-135M's model.safetensors (llama.bend's
    # checkout, as make smol finds it) must be bench/bits.txt's digest; skipped where the file is not (CI)
    st = ST135
    want = next(l.split(": ")[1].strip() for l in open("bench/bits.txt") if l.startswith("smol loss, threads 16: "))
    b = subprocess.run(["bend", "demo/smol.bend", "-o", ".scratch/mut_smol"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    out = subprocess.run([".scratch/mut_smol", "+RTS", "--threads", "16", "-RTS", "loss", str(st)], capture_output=True, text=True).stdout
    kept = "".join(l + "\n" for l in out.splitlines() if not l.startswith("train_ms "))
    import hashlib
    return hashlib.sha256(kept.encode()).hexdigest()[:16] == want


SMOLST = smol_st


def fast():
    # torch/fast.bend's layers and optimizer step must keep main.bend's bits (demo/fast_test.bend prints 12 zeros)
    b = subprocess.run(["bend", "demo/fast_test.bend", "-o", ".scratch/mut_fast_test"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_fast_test", "+RTS", "--threads", "16", "-RTS"], capture_output=True, text=True).stdout.split() == ["0"] * 13


FAST = fast


def gpu():
    # torch/gpu.bend's layers must keep main.bend's bits: demo/gpu_test.bend
    # prints 8 zeros; built without CUDA, its bangs run on the CPU threads, the same code
    b = subprocess.run(["bend", "demo/gpu_test.bend", "-o", ".scratch/mut_gpu_test"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_gpu_test", "+RTS", "--threads", "16", "-RTS"], capture_output=True, text=True).stdout.split() == ["0"] * 8


GPU = gpu


def gpu_layers():
    # torch/gpu.bend's other layers, optimizer step, losses and embedding must keep main.bend's bits with +RTS --gpu off -RTS:
    # demo/gpu_layers_test.bend prints 13 zeros
    b = subprocess.run(["bend", "demo/gpu_layers_test.bend", "-o", ".scratch/mut_gpu_layers"], capture_output=True, env=ENV)
    if b.returncode != 0:
        return False
    return subprocess.run([".scratch/mut_gpu_layers", "+RTS", "--gpu", "off", "--threads", "16", "-RTS"], capture_output=True, text=True).stdout.split() == ["0"] * 13


GPUL = gpu_layers


def gpu_expcr():
    # make gpu's exactness check: with exp_cr in place of F32.exp, every GPU layer case gives main.bend's bits on the
    # device (scripts/gpu_expcr.sh prints 13 zeros); needs CUDA, and make mutate runs under the heavy-run lock
    r = subprocess.run(["scripts/gpu_expcr.sh"], capture_output=True, text=True, env=ENV)
    return r.returncode == 0 and r.stdout.strip().splitlines()[-1].split() == ["0"] * 13


EXPCR = gpu_expcr


def has_cuda():
    # gpu_expcr needs a CUDA toolkit and a device; on a machine without them (CI's runners) its mutant is skipped and
    # said so, since there the check would build for the CPU, where the host's exp is the device's
    home = os.environ.get("CUDA_HOME", os.path.expanduser("~/.local/cuda12"))
    if not Path(home, "include", "cuda.h").exists() or shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
INFRA = numeric("infra_test", "test_infra.py")
DTYPE = numeric("dtype_test", "test_dtype.py")
MNIST = numeric("mnist", "test_mnist.py")


def tiny16():
    # AdamW with packed BF16 moments on the tiny Llama (fast.bend's layers) against its twin, with a checkpoint
    b = subprocess.run(["bend", "demo/tiny_bf16opt.bend", "-o", ".scratch/mut_tiny_bf16opt"], capture_output=True, env=ENV)
    if b.returncode != 0 or not Path(".scratch/tiny_llama").exists():
        return False
    return subprocess.run([PY, "reference/test_tiny_bf16opt.py", ".scratch/mut_tiny_bf16opt", ".scratch/tiny_llama", "20"],
                          capture_output=True).returncode == 0


TINY16 = tiny16

# (name, file, before, after, gate, expected): expected is "caught", or the reason it is allowed to survive
MUTANTS = [
    ("SGD: allocate two moment buffers", "torch/main.bend",
     "case CSGD{lr}: NoMom{}", "case CSGD{lr}: TwoMom{Layer.zeros(P, X, Y, S, m), Layer.zeros(P, X, Y, S, m)}", INFRA, "caught"),
    ("momentum: initialize no buffer", "torch/main.bend",
     "case CMomentum{lr, mu}: OneMom{Layer.zeros(P, X, Y, S, m)}", "case CMomentum{lr, mu}: NoMom{}", MLP, "caught"),
    ("load: advance the cursor one element short", "torch/main.bend",
     "(off + U32.from_nat(Nat.mul(r, c)) : U32), load_cells", "(off + U32.from_nat(Nat.sub(Nat.mul(r, c), 1n)) : U32), load_cells", INFRA, "caught"),
    ("load: every matrix reads from source offset zero", "torch/main.bend",
     "load_cells(Nat.mul(r, c), 0, off, Array.get(F32, xs, off), dst)", "load_cells(Nat.mul(r, c), 0, 0, Array.get(F32, xs, 0), dst)", INFRA, "caught"),
    ("store: each matrix writes one cell short", "torch/main.bend",
     "store_cells(Nat.mul(r, c), 0, off,", "store_cells(Nat.sub(Nat.mul(r, c), 1n), 0, off,", INFRA, "caught"),
    ("checkpoint from arrays: Adam's powers swapped", "torch/main.bend",
     "Array.set(F32, Array.set(F32, xs, off, p1), (off + 1 : U32), p2)", "Array.set(F32, Array.set(F32, xs, off, p2), (off + 1 : U32), p1)", INFRA, "caught"),
    ("cross-entropy: the row's tree sum one class short", "torch/main.bend",
     "tree_sum(U32.from_nat(1n+k), sc)", "tree_sum(U32.from_nat(k), sc)", MNIST, "caught"),
    ("checkpoint: SGD writes unused scalar powers", "torch/main.bend",
     "TrainPL{Train{p, g, NoMom{}, p1, p2, l}, xp}", "TrainPL{Train{p, g, NoMom{}, p1, p2, l}, List.append(&2, F32, xp, [p1, p2])}", INFRA, "caught"),
    ("checkpoint: momentum omits its buffer", "torch/main.bend",
     "TrainPL{Train{p, g, OneMom{mm}, p1, p2, l}, List.append(&2, F32, xp, xm)}", "TrainPL{Train{p, g, OneMom{mm}, p1, p2, l}, xp}", INFRA, "caught"),
    ("checkpoint: restore swaps Adam's powers", "torch/main.bend",
     "case Tuple{xs, p2}: Train{p, g, TwoMom{mm, vv}, p1, p2, 0.0}", "case Tuple{xs, p2}: Train{p, g, TwoMom{mm, vv}, p2, p1, 0.0}", INFRA, "caught"),
    ("binary file: accept extra trailing bytes", "torch/load.bend",
     "require(W64.eq(size, W64.mul_small(W64.of(n), 4)),", "require(True{},", INFRA, "caught"),
    ("safetensors: the data counted from the header's start, not its end", "torch/load.bend",
     "W64.add(W64.of(8), W64.of(len))", "W64.of(len)", ST, "caught"),
    ("safetensors: an F32 tensor's shape not checked", "torch/load.bend",
     "case True{} _: St.plan_shape(St.shape_eq(sh, want), name, True{},", "case True{} _: St.plan_shape(True{}, name, True{},", ST, "caught"),
    ("safetensors: BF16 read as F32", "torch/load.bend",
     "case _ True{}: St.plan_shape(St.shape_eq(sh, want), name, False{},", "case _ True{}: St.plan_shape(St.shape_eq(sh, want), name, True{},", ST, "caught"),
    ("safetensors: each tensor one cell after the last one's end", "torch/load.bend",
     "St.fill(t, path, start, recs, f2, a2, (at + n : U32))", "St.fill(t, path, start, recs, f2, a2, ((at + n : U32) + 1 : U32))", ST, "caught"),
    ("safetensors: the data_offsets' span not checked", "torch/load.bend",
     "St.plan_span(W64.eq(W64.add(b, W64.mul_small(W64.of(n), width)), e),", "St.plan_span(True{},", ST, "caught"),
    ("safetensors: a backslash in a string not an escape", "torch/load.bend",
     "St.str_go(U32.is_eq(b, 34), U32.is_eq(b, 92), b, acc)", "St.str_go(U32.is_eq(b, 34), False{}, b, acc)", ST, "caught"),
    ("safetensors: an unclosed header accepted", "torch/load.bend",
     "require(Bool.and(U32.is_zero(d), U32.is_zero(bad)),", "require(U32.is_zero(bad),", ST, "caught"),
    ("piece by piece: the second half of the stack reads the first half's blocks", "demo/smol.bend",
     "LStack15(path, h, Nat.add(i, 15n), y)", "LStack15(path, h, i, y)", SMOLST, "caught"),
    ("piece by piece: a piece's numbers start one cell in", "torch/load.bend",
     "St.fill(layout, path, start, recs, f, Array.new(F32, T.depth(U32.to_nat(n)), 0.0), 0)", "St.fill(layout, path, start, recs, f, Array.new(F32, T.depth(U32.to_nat(n)), 0.0), 1)", SMOLST, "caught"),
    ("piece by piece: the final norm read from another tensor of its shape", "demo/smol.bend",
     '[L.Entry{"model.norm.weight", [U32.from_nat(D())]}], nm)', '[L.Entry{"model.layers.0.input_layernorm.weight", [U32.from_nat(D())]}], nm)', SMOLST, "caught"),
    ("run-time training: report a zero loss", "torch/main.bend",
     "sf, cfg, st, p1, p2, l, bf(p, s, dy, g))", "sf, cfg, st, p1, p2, 0.0, bf(p, s, dy, g))", MNIST, "caught"),
    ("fast SGD: each block starts at zero", "torch/fast.bend",
     "p2_b2(T.sgd_l(cnt, i0, lr,", "p2_b2(T.sgd_l(cnt, 0, lr,", FAST, "caught"),
    ("fast momentum: each block starts at zero", "torch/fast.bend",
     "mom_b(T.mom_l(cnt, i0, lr,", "mom_b(T.mom_l(cnt, 0, lr,", FAST, "caught"),

    ("fast optimizer: every block's AdamW starts at number 0", "torch/fast.bend",
     "q4_b(T.adam_l(cnt, i0, lr,", "q4_b(T.adam_l(cnt, 0, lr,", FAST, "caught"),
    ("fast copies: each block writes its copy one place late", "torch/fast.bend",
     "a2_b2(T.copy_n(k, i0, i0, b,", "a2_b2(T.copy_n(k, i0, (i0 + 1 : U32), b,", FAST, "caught"),
    ("MSE: the tree sum drops an odd last number", "torch/main.bend",
     "ts_carry(U32.is_eq(U32.mod(m, 2), 1), m,", "ts_carry(False{}, m,", OPS, "caught"),
    ("fast linear: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.lin_b3(n, i, o, w, b, dx, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear first: the weight gradient is not accumulated", "torch/fast.bend",
     "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn_acc(o, n, i, par, dy, x, gw))", "T.linf_b3(n, i, o, w, b, gb, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("fast linear without bias: the weight gradient is not accumulated", "torch/fast.bend",
     "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn_acc(o, n, i, par, dy, x, g))", "T.lnb_b3(n, i, o, w, dx, TF.matmul_tn(o, n, i, par, dy, x))", FAST, "caught"),
    ("gpu linear: the weight gradient is not accumulated", "torch/gpu.bend",
     "T.lin_b3(n, i, o, w, b, dx, gb, TG.matmul_tn_acc(o, n, i, 0n, dy, x, gw))", "T.lin_b3(n, i, o, w, b, dx, gb, TG.matmul_tn(o, n, i, 0n, dy, x))", GPU, "caught"),
    ("gpu element-wise: each leaf skips its last cell", "torch/gpu.bend",
     "case EN{+n}: ew1_loop(~f, TG.share(t, n, l), t, l, o, Array.get(F32, x, t))", "case EN{+n}: ew1_loop(~f, Nat.sub(TG.share(t, n, l), 1n), t, l, o, Array.get(F32, x, t))", GPUL, "caught"),
    ("gpu RMSNorm: the gain's gradient overwrites the one it is given", "torch/gpu.bend",
     "Array.set(F32, ga, j, (v + s : F32))", "Array.set(F32, ga, j, s)", GPUL, "caught"),
    ("gpu RoPE: the backward pass rotates forward", "torch/gpu.bend",
     "T.Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}", "T.Bwd{p, rope_go(n, w, True{}, t, h, half, base, dy), g}", GPUL, "caught"),
    ("gpu attention: dK and dV leave out the group's last head", "torch/gpu.bend",
     "tsum_h(Nat.sub(gn, 1n), ia, ib, j,", "tsum_h(Nat.sub(gn, 2n), ia, ib, j,", GPUL, "caught"),
    ("gpu attention: the second row of a pair of P's rows reads its keys one place late", "torch/gpu.bend",
     "Bool.pick(U32, lo, c, ((c - p : U32) - 1 : U32))", "Bool.pick(U32, lo, c, (c - p : U32))", GPUL, "caught"),
    ("gpu attention: the softmax leaves out a row's last position", "torch/gpu.bend",
     "sm_loop(q, (r + l : U32), l, t32, tr, T.pc_a(T.psmc(U32.to_nat((U32.mod(r, t32) + 1 : U32))", "sm_loop(q, (r + l : U32), l, t32, tr, T.pc_a(T.psmc(U32.to_nat(U32.mod(r, t32))", GPUL, "caught"),
    ("gpu attention: a column of P's or G's packed triangle steps one row short", "torch/gpu.bend",
     "tdot(p, (ia + (i + 1 : U32) : U32), (ib + sb : U32)", "tdot(p, (ia + i : U32), (ib + sb : U32)", GPUL, "caught"),
    ("gpu attention: o and dQ stop one position before the query's", "torch/gpu.bend",
     "T.dot0(U32.to_nat((i + 1 : U32)), toff(", "T.dot0(U32.to_nat(i), toff(", GPUL, "caught"),
    ("gpu attention: kᵀ takes every column from the head's first", "torch/gpu.bend",
     "((U32.mod(bj, kv) * hd32 : U32) + U32.mod(r, hd32) : U32)", "((U32.mod(bj, kv) * hd32 : U32) + 0 : U32)", GPUL, "caught"),
    ("gpu attention: dQ's cells read K's column of the head's first key/value head", "torch/gpu.bend",
     "(U32.div(hi, g) * hd32 : U32) : U32) + U32.mod(col, hd32) : U32), 1, kvw, a, b)", "(0 * hd32 : U32) : U32) + U32.mod(col, hd32) : U32), 1, kvw, a, b)", GPUL, "caught"),
    ("gpu MSE: the gradient has the wrong sign", "torch/gpu.bend",
     "MS{Array.set(F32, y, i, ((a - b : F32) * kk : F32))", "MS{Array.set(F32, y, i, ((b - a : F32) * kk : F32))", GPUL, "caught"),
    ("gpu Adam: the gradient is not zeroed", "torch/gpu.bend",
     "Array.get(F32, Array.set(F32, g, i, 0.0), u), Array.get(F32, Array.set(F32, m, i, nm), u)", "Array.get(F32, g, u), Array.get(F32, Array.set(F32, m, i, nm), u)", GPUL, "caught"),
    ("gpu embedding: the backward pass overwrites the gradient's row instead of adding to it", "torch/gpu.bend",
     "ES3{ids, dy, Array.set(F32, g, gi, (a + b : F32))}", "ES3{ids, dy, Array.set(F32, g, gi, b)}", GPUL, "caught"),
    ("gpu embedding: the forward pass copies every row's first column", "torch/gpu.bend",
     "ef_3(i, j, d, ids, o, Array.get(F32, e, ((f2u(v) * d : U32) + j : U32)))", "ef_3(i, j, d, ids, o, Array.get(F32, e, (f2u(v) * d : U32)))", GPUL, "caught"),
    ("gpu cross-entropy: the label's cell gets no -1", "torch/gpu.bend",
     "(((x / s : F32) - T.hit(U32.is_eq(j, y)) : F32) / nf : F32)", "((x / s : F32) / nf : F32)", GPUL, "caught"),
    ("gpu cross-entropy: an odd block sum left over is dropped instead of carried", "torch/gpu.bend",
     "case 1n+q False{}: tnode(q, U32.is_lt(U32.shr(h), len), lo, len, U32.shr(h), a)", "case 1n+q False{}: TS{a, 0.0}", GPUL, "caught"),
    ("gpu cross-entropy: blocks of one cell instead of up to 512", "torch/gpu.bend",
     "cx_2(n, c, labels, k, ms_p(9n, U32.from_nat(c))", "cx_2(n, c, labels, k, ms_p(0n, U32.from_nat(c))", GPUL, "survives: an equivalent mutant, tree_sum's passes give the same sum from any block size that divides the row (the law block_sum)"),
    ("gpu exactness check: main.bend keeps CUDA's F32.exp (only gpu.bend's is swapped)", "scripts/gpu_expcr.sh",
     "for f in main gpu; do", "for f in gpu; do", EXPCR, "caught"),
    ("dtype: BF16 rounds half away from even (the kept bit not added)", "torch/dtype.bend",
     "U32.shrn(((u + 32767 : U32) + (U32.shrn(u, 16n) .&. 1 : U32) : U32), 16n)", "U32.shrn((u + 32768 : U32), 16n)", DTYPE, "caught"),
    ("dtype: int8 clamps at 126", "torch/dtype.bend",
     "F32.clamp(rne_f((w / s : F32)), F32.neg(127.0), 127.0)", "F32.clamp(rne_f((w / s : F32)), F32.neg(126.0), 126.0)", DTYPE, "caught"),
    ("dtype: a NaN keeps BF16 rounding instead of the canonical NaN", "torch/dtype.bend",
     "pick_u(U32.is_gt((u .&. 2147483647 : U32), 2139095040), 65535, bf16_rne(u))", "bf16_rne(u)", DTYPE, "caught"),
    ("spec: Mᵀ·g drops the rows after the first", "torch/spec.bend",
     "vadd(vscale(gi, r), mtv(rt, gt))", "vadd(vscale(gi, r), mtv(rt, Nil{}))", laws, "caught"),
    ("spec: g ⊗ v uses only the first entry of g", "torch/spec.bend",
     "vscale(gi, v) <> outer(gt, v)", "vscale(gi, v) <> outer(Nil{}, v)", laws, "caught"),
    ("attention: a query also sees the next position", "torch/main.bend",
     "prow3(off, q, k, pw(row, off, sc, 0n, p))", "prow3(off, q, k, pw(row, off, sc, 1n, p))", laws, "caught"),
    ("attention: a query also sees the next position (numbers)", "torch/main.bend",
     "prow3(off, q, k, pw(row, off, sc, 0n, p))", "prow3(off, q, k, pw(row, off, sc, 1n, p))", OPS, "caught"),
    ("attention: the softmax leaves out the last visible position", "torch/main.bend",
     "pr_of(q, k, psmc(n, off, p))", "pr_of(q, k, psmc(Nat.sub(n, 1n), off, p))", laws, "caught"),
    ("attention: the softmax leaves out the last visible position (numbers)", "torch/main.bend",
     "pr_of(q, k, psmc(n, off, p))", "pr_of(q, k, psmc(Nat.sub(n, 1n), off, p))", OPS, "caught"),
    ("attention: query head hi reads key/value head hi mod g", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", laws, "caught"),
    ("attention: query head hi reads key/value head hi mod g (numbers)", "torch/main.bend",
     "  Nat.div(hi, g)\n", "  Nat.mod(hi, g)\n", OPS, "caught"),
    ("attention: head_cols gives query head hi the columns of key/value head hi mod g", "torch/main.bend",
     "U32.from_nat(Nat.mul(kv_of(hi, g), hd))}", "U32.from_nat(Nat.mul(Nat.mod(hi, g), hd))}", laws, "caught"),
    ("attention: head_cols gives query head hi the columns of key/value head hi mod g (numbers)", "torch/main.bend",
     "U32.from_nat(Nat.mul(kv_of(hi, g), hd))}", "U32.from_nat(Nat.mul(Nat.mod(hi, g), hd))}", OPS, "caught"),
    ("attention: the backward pass's dP row reads one value past the query's position", "torch/main.bend",
     "  gr2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g, drow(i, hd,", "  gr2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g, drow(1n+i, hd,", laws, "caught"),
    ("attention: the tree of sequences puts the right half's outputs first", "torch/main.bend",
     "      a b = fleaf(l, h, dm, Nil{}) fleaf(r, h, dm, Nil{})\n      List.append(&1, Array<F32>, a, b)", "      a b = fleaf(l, h, dm, Nil{}) fleaf(r, h, dm, Nil{})\n      List.append(&1, Array<F32>, b, a)", laws, "caught"),
    ("attention: the backward tree puts the right half's gradients first", "torch/main.bend",
     "      a b = bleaf(l, kv, dm, Nil{}) bleaf(r, kv, dm, Nil{})\n      List.append(&1, DB, a, b)", "      a b = bleaf(l, kv, dm, Nil{}) bleaf(r, kv, dm, Nil{})\n      List.append(&1, DB, b, a)", laws, "caught"),
    ("Fork: the backward pass drops the second layer's input gradient", "torch/main.bend",
     "      Bwd{Two{p1, p2}, add(dx1, dx2), Two{g1, g2}}", "      Bwd{Two{p1, p2}, dx1, Two{g1, g2}}", laws, "caught"),
    ("attention: the backward pass forgets the scale", "torch/main.bend",
     "gfix(Nat.sub(t, 1n), off, s, sc,", "gfix(Nat.sub(t, 1n), off, s, 1.0,", OPS, "caught"),
    ("attention: a query reads one key past its position", "torch/main.bend",
     "  drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", "  drow(1n+i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", laws, "caught"),
    ("attention: a query reads one key past its position (numbers)", "torch/main.bend",
     "  drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", "  drow(1n+i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks,", OPS, "caught"),
    ("cross-entropy: the gradient forgets the label", "torch/main.bend",
     "((x - hit(U32.is_eq(j, y)) : F32) / nf : F32)", "(x / nf : F32)", OPS, "caught"),
    ("scratch arrays: half the capacity they need", "torch/main.bend",
     "  depth_go(32n, U32.is_le(U32.from_nat(n), 1), U32.from_nat(n), 1, 0n)", "  depth_go(32n, U32.is_le(U32.from_nat(n), 2), U32.from_nat(n), 2, 0n)", OPS, "caught"),
    ("relu: the gradient passes at exactly 0", "torch/main.bend",
     "pick(F32.is_le(y, 0.0), 0.0, dy)", "pick(F32.is_lt(y, 0.0), 0.0, dy)", OPS, "caught"),
    ("silu: the derivative misses its 1 +", "torch/main.bend",
     "((dy * s : F32) * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32)", "((dy * s : F32) * (x * (1.0 - s : F32) : F32) : F32)",
     OPS, "caught"),
    ("erf: the sign is not put back for |x| >= 0.84375", "torch/main.bend",
     "pick(F32.is_lt(x, 0.0), F32.neg(e), e)", "e", OPS, "caught"),
    ("erf: rounds to 1 from 3.5 on", "torch/main.bend",
     "erf_top(F32.is_ge(a, 3.92), x, a)", "erf_top(F32.is_ge(a, 3.5), x, a)", OPS, "caught"),
    ("erf: NaN taken as large (erf(NaN) = 1)", "torch/main.bend",
     "erf_top(F32.is_ge(a, 3.92), x, a)", "erf_top(Bool.not(F32.is_lt(a, 3.92)), x, a)", OPS, "caught"),
    ("erf: the small piece drops its x^12 term", "torch/main.bend",
     "erf_hm(0.000823960464913398, z, 8.829222497297451e-05)", "F32.neg(0.000823960464913398)", OPS, "caught"),
    ("erf: a negative coefficient's Horner step subtracts the wrong way", "torch/main.bend",
     "  ((t * s : F32) - c : F32)", "  (c - (t * s : F32) : F32)", OPS, "caught"),
    ("erf: the piece around 2 evaluated around 1.5", "torch/main.bend",
     "erf_sign(x, erf_p2((a - 2.0 : F32)))", "erf_sign(x, erf_p2((a - 1.5 : F32)))", OPS, "caught"),
    ("erf: the small piece used up to 1.25", "torch/main.bend",
     "erf_i0(F32.is_ge(a, 0.84375), x, a)", "erf_i0(F32.is_ge(a, 1.25), x, a)", OPS, "caught"),
    ("erf: the first piece's constant term (1.35e-8, a fifth of an ulp) dropped", "torch/main.bend",
     "erf_hp(1.3508899776581984e-08, t, ", "erf_hp(0.0, t, ", OPS, "caught"),
    ("gelu_erf: the forward pass takes erf(x), not erf(x / sqrt 2)", "torch/main.bend",
     "((x * 0.5 : F32) * (1.0 + erf((x * 0.7071067811865476 : F32))", "((x * 0.5 : F32) * (1.0 + erf(x)", OPS, "caught"),
    ("gelu_erf: the derivative misses the x of x pdf", "torch/main.bend",
     "(dy * (gelu_erf_cdf(x) + (x * pdf : F32) : F32) : F32)", "(dy * (gelu_erf_cdf(x) + pdf : F32) : F32)", OPS, "caught"),
    ("gelu_erf: the backward pass is the tanh form's", "torch/main.bend",
     "ActIn(~gelu_erf_v, ~gelu_erf_d, n, c)", "ActIn(~gelu_erf_v, ~gelu_d, n, c)", OPS, "caught"),
    ("softmax: the backward pass adds instead of subtracting", "torch/main.bend",
     "smb_fix(Nat.sub(c, 1n), off, s,", "smb_fix(Nat.sub(c, 1n), off, F32.neg(s),", OPS, "caught"),
    ("rmsnorm: the backward pass drops the mean term", "torch/main.bend",
     "rmsb_w(Nat.sub(d, 1n), 0, off, r, (s / df : F32), dx, dg,", "rmsb_w(Nat.sub(d, 1n), 0, off, r, 0.0, dx, dg,", OPS, "caught"),
    ("rope: the backward pass rotates forward", "torch/main.bend",
     "Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}",
     "Bwd{p, rope_go(n, w, True{}, t, h, half, base, dy), g}", OPS, "caught"),
    ("embedding: a repeated id keeps only its last gradient", "torch/main.bend",
     "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, (a + b : F32))", "emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, b)", OPS, "caught"),
    ("linear: the bias gradient is not accumulated", "torch/main.bend",
     "TA.Mat.col_sums_acc(n, o, dy, gb))", "TA.Mat.col_sums(n, o, dy))", MLP, "caught"),
    ("dropout: the optimizer step does not move the mask's counter", "torch/main.bend",
     "Step{DropP{(t + 1 : U32)}, g, st}", "Step{DropP{t}, g, st}", OPS, "caught"),
    ("dropout: the mask ignores the step", "torch/main.bend",
     "mix32(U32.xor(seed, mix32(t)))", "mix32(seed)", OPS, "caught"),
    ("tie: the first layer's gradient forgets the last layer's", "torch/main.bend",
     "tie_b4(P, Q, X, q, gq, Layer.bwd(P, X, Y, S1, la, p, s1, dy, gp))",
     "tie_b4(P, Q, X, q, gq, Layer.bwd(P, X, Y, S1, la, p, s1, dy, Layer.zeros(P, X, Y, S1, la)))", OPS, "caught"),
    ("adam config: the check accepts any c within 1e-5 of 1 - b", "torch/main.bend",
     "((d * d : F32) <= 0.0000000000000142108547152020037174224853515625 : F32)", "((d * d : F32) <= 0.0000000001 : F32)", INFRA, "caught"),
    ("adam config: CAdam's check ignores c2", "torch/main.bend",
     "case CAdam{lr, b1, b2, eps, wd, c1, c2}:\n      Bool.and(beta_ok(b1, c1), beta_ok(b2, c2))",
     "case CAdam{lr, b1, b2, eps, wd, c1, c2}:\n      beta_ok(b1, c1)", INFRA, "caught"),
    ("adam config: CAdamW's check ignores c2", "torch/main.bend",
     "case CAdamW{lr, b1, b2, eps, wd, c1, c2}:\n      Bool.and(beta_ok(b1, c1), beta_ok(b2, c2))",
     "case CAdamW{lr, b1, b2, eps, wd, c1, c2}:\n      beta_ok(b1, c1)", INFRA, "caught"),
    ("adam config: the check accepts a negative c", "torch/main.bend",
     "(c > 0.0 : F32)", "(c > -1.0 : F32)", INFRA, "caught"),
    ("adam config: a rejected configuration starts at 0, not NaN", "torch/main.bend",
     "    case False{}:\n      (0.0 / 0.0 : F32)", "    case False{}:\n      0.0", INFRA, "caught"),
    ("momentum: the update ignores the buffer", "torch/main.bend",
     "(x + (F32.neg(lr) * nb : F32) : F32)", "(x + (F32.neg(lr) * y : F32) : F32)", MLP, "caught"),
    ("adamw: no decoupled weight decay", "torch/main.bend",
     "(w * (1.0 - (lr * wd : F32) : F32) : F32)", "w", MLP, "caught"),
    ("adam: no bias correction of the second moment", "torch/main.bend",
     "((F32.sqrt(v) / bc2 : F32) + eps : F32)", "(F32.sqrt(v) + eps : F32)", MLP, "caught"),
    ("linear: the weight gradient is not accumulated", "torch/main.bend",
     "TA.Mat.matmul_tn_acc(o, n, i, par, dy, x, gw))", "TA.Mat.matmul_tn(o, n, i, par, dy, x))", MLP, "caught"),
    ("adamw16: the moments' BF16 rounding rounds half up (the kept bit not added)", "torch/main.bend",
     "U32.shrn(((u + 32767 : U32) + (U32.shrn(u, 16n) .&. 1 : U32) : U32), 16n))\n\ndef ph_of",
     "U32.shrn((u + 32768 : U32), 16n))\n\ndef ph_of", DTYPE, "caught"),
    ("adamw16: a NaN moment is rounded instead of made 0xFFFF", "torch/main.bend",
     "ph_pick(U32.is_gt((u .&. 2147483647 : U32), 2139095040), 65535,", "ph_pick(False{}, 65535,", DTYPE, "caught"),
    ("adamw16: an odd element reads the low half of its word", "torch/main.bend",
     "    case True{}:\n      U32.shrn(w, 16n)\n    case False{}:\n      (w .&. 65535 : U32)",
     "    case True{}:\n      (w .&. 65535 : U32)\n    case False{}:\n      (w .&. 65535 : U32)", MLP, "caught"),
    ("adamw16: writing a high half clears the low one", "torch/main.bend",
     "      ((w .&. 65535 : U32) .|. U32.shln(h, 16n) : U32)", "      U32.shln(h, 16n)", MLP, "caught"),
    ("adamw16: the weight update reads the first moment before its rounding", "torch/main.bend",
     "adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), ph_to(hm), nv)",
     "adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), adam_m(c1, ph_to(ph_half(odd, a)), gg), nv)", MLP, "caught"),
    ("adamw16: v16 ignored, v kept in F32", "torch/main.bend",
     "      PLeaf{U32.from_nat(n), ph_words(n), ph_words(n)}", "      PLeafV{U32.from_nat(n), ph_words(n), Array.new(F32, depth(n), 0.0)}", MLP, "caught"),
    ("adamw16: the packed tree joins its parts swapped", "torch/main.bend",
     "case Packed{x} Packed{y}: Packed{PNode{x, y}}", "case Packed{x} Packed{y}: Packed{PNode{y, x}}", MLP, "caught"),
    ("adamw16: a checkpoint does not write a leaf's v", "torch/main.bend",
     "pts_v(n, m, (off + ph_hw(n) : U32), words_out(U32.to_nat(ph_hw(n)),", "pts_v(n, m, (off + ph_hw(n) : U32), words_out(0n,", TINY16, "caught"),
    ("adamw16: fast.bend leaves a packed state without its step", "torch/fast.bend",
     "    case o w g T.Packed{t}: T.Opt.mat(r, c, o, w, g, T.Packed{t})\n", "", TINY16, "caught"),
    ("tensor: div's gradient for the divisor loses its sign", "torch/tensor.bend",
     "(0.0 - (d * ((p / q : F32) / q : F32) : F32) : F32)", "(d * ((p / q : F32) / q : F32) : F32)", TEN, "caught"),
    ("tensor: narrow starts its window on the wrong axis", "torch/tensor.bend",
     "U32.from_nat(Nat.mul(st, inner(s, d))))", "U32.from_nat(Nat.mul(st, inner(s, 0n))))", TEN, "caught"),
    ("tensor: bmm reads its second operand transposed", "torch/tensor.bend",
     "u(Nat.mul(n, k)), u(k), 1, u(Nat.mul(k, m)), u(m), 1}, dx, dy", "u(Nat.mul(n, k)), u(k), 1, u(Nat.mul(k, m)), 1, u(m)}, dx, dy", TEN, "caught"),
    ("tensor: broadcasting allows a size-1 axis on one side only", "torch/tensor.bend",
     "Bool.or(bc_pair(x, y, z), bc_pair(y, x, z))", "bc_pair(x, y, z)", laws, "caught"),
    ("tensor: the whole-tensor sum maps each element to its own position (right only because a 1-cell array wraps)", "torch/tensor.bend",
     "Drop{1, U32.from_nat(numel(s))}", "Drop{U32.from_nat(numel(s)), 1}", TEN, "survives the numbers: the 1-cell result array wraps every index to 0, so the sum is right by accident; the bounds check catches it (below)"),
    ("tensor: the whole-tensor sum maps each element to its own position (bounds)", "torch/tensor.bend",
     "Drop{1, U32.from_nat(numel(s))}", "Drop{U32.from_nat(numel(s)), 1}", IMAP, "caught"),
    ("tensor: cat's shape adds the first operand's axis twice", "torch/tensor.bend",
     "setd(a, d, Nat.add(at1(a, d), at1(b, d)))", "setd(a, d, Nat.add(at1(a, d), at1(a, d)))", TEN, "caught"),
]


def main():
    escaped, skipped = [], []
    cuda = has_cuda()
    for name, path, before, after, gate, expected in MUTANTS:
        if gate is EXPCR and not cuda:
            print(f"skip {'':8} {name}  [needs a CUDA device]")
            skipped.append(name)
            continue
        if gate is SMOLST and not ST135.exists():
            print(f"skip {'':8} {name}  [needs {ST135}]")
            skipped.append(name)
            continue
        p = Path(path)
        src = p.read_text()
        assert src.count(before) >= 1, f"{name}: the text to mutate is not in {path}"
        try:
            p.write_text(src.replace(before, after, 1))
            caught = not gate()
        finally:
            p.write_text(src)
        verdict = "caught" if caught else "survived"
        ok = (verdict == "caught") == (expected == "caught")
        print(f"{'ok ' if ok else 'BAD'} {verdict:8} {name}" + ("" if expected == "caught" else f"  [{expected}]"))
        if not ok:
            escaped.append(name)
    print(f"mutate: {len(MUTANTS)} mutants, {len(MUTANTS) - len(escaped) - len(skipped)} as expected"
          + (f", {len(skipped)} skipped (no CUDA device or no model.safetensors)" if skipped else ""))
    sys.exit(1 if escaped else 0)


if __name__ == "__main__":
    main()
