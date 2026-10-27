# Plan

Each milestone has a pass criterion that a command checks. Estimates are planning guesses, revised as milestones close;
they are not commitments. Decisions and measurements go to [`NOTES.md`](../NOTES.md), the architecture to
[`design.md`](design.md).

## Goal and success bar

A machine learning framework in Bend 2 whose pieces are general (a training step that does not know the model), whose
shapes are in the types, whose structural guarantees are laws checked by Bend's kernel, and whose numbers agree with
PyTorch. Success: real models (an MLP, a Llama) trained in Bend with the same results as their PyTorch twins, every claim
reproducible with one command, and the speed measured and reported as it is.

## Done (2026-10-15)

| # | Milestone | Pass criterion | Status |
|---|---|---|---|
| M1 | The layer protocol, `Linear`, `ReLU`, `Seq`, cross-entropy, SGD/momentum/Adam/AdamW, `Train.step`; MNIST on them | gradients and 3 steps of each optimizer equal PyTorch's on 12 seeds (`test_mlp.py`); 50 MNIST batches of each optimizer: the same loss within the tolerance and the same hits (`test_mnist.py`) | done |
| M2 | The layers of a Llama and the combinators | every layer's output and gradients within the tolerance on 10 seeds, the activations also on IEEE edge cases (`test_ops.py`) | done |
| M3 | A two-layer character-level Llama trained in Bend | all gradients and 20 AdamW steps within the tolerance (`test_tiny.py`); the loss falls (4.23 to 2.63 in 500 steps) | done |
| M4 | Laws | `adjoint`, `weight_adjoint`, `adjoint_seq`, `causal`, `kv_of_group`, `gqa_head` pass `--verdict`; `tests/bad_*.bend` rejected with their expected errors (`make laws`) | done |
| M5 | Hardening | 20 mutants behave as listed (`make mutate`); IEEE edge cases through the activations; benchmark script and first numbers (`make bench`) | done |

| v1 | What a first release needs | `LinearFirst` (MNIST 30% faster, same bits); attention's sequences in parallel; generation from the trained Llama equal to PyTorch's from the same weights; checkpoints (a resumed run gives the same bits); `Dropout` and `Tie` against PyTorch, bit-exact where possible; the exact tier of the numerical policy; 23 mutants | done |

| fast | The linear layers over bend-ml-tensor-fork (2026-10-17) | `torch/fast.bend`, the demos on it: `make bits` identical, `demo/fast_test.bend` 0 differing words (1 and 16 threads, JavaScript), 33 mutants as expected; MNIST's epoch 3.6 → 2.9 s on one thread, 3.3 → 2.4 s on 16 | done |

## Next

bend-ml's products come first, because they are the bottleneck; the milestones below are this repository's part. Each closes with `make check`, `make bits` (or an explained
change of bits), `make mutate` and a NOTES entry.

### M6 · Row-wise layers on arrays, in parallel (~4 days)

Phase 1 done 2026-10-17: attention over arrays, the same bits, the tiny Llama's step 12.2 → 4.1 ms on one thread, law
`causal_keys` (NOTES). Phase 2 done 2026-10-17: every other row-wise layer over arrays, the same bits, 4.0 → 1.1 ms on one
thread, law `causal` rewritten over `prow` (NOTES). Phase 3 done 2026-10-17: parallelism where it pays, measured on one
decoder block at SmolLM2's sizes (`demo/smol_block.bend`): 16 threads 1.6-1.7x faster at 135M's, 360M's and 1.7B's
sizes, one thread unchanged, the same bits (NOTES). **M6 done**, except one clause of its criterion: the tiny Llama is
not faster on 16 threads than on one, and cannot be at 32 wide; the gain is shown on the block.

Needs M13 first (my call, 2026-10-16): M6 touches every layer, so CI runs on every push before it starts.

RMSNorm, LayerNorm, softmax, log-softmax, RoPE, attention and the losses move from lists to `Array` kernels (a row at a
time, in place), parallel over rows and heads with balanced parallel lets.

- Pass: `test_ops.py` and `test_tiny.py` within the tolerance; the exact tier unchanged; `make bits` identical where the
  order of the sums is kept (and every moved bit explained); the tiny Llama's step faster on 1 thread and again on 16
  (`make bench`).
- Risk: sharing an input across a parallel let still copies it (bendlang/bend#1376).

### M7 · General tensors (done 2026-10-16)

`Tensor<shape: List<Nat>>` over one `Array<F32>`, row-major. Operations, each with its backward pass, its test against
PyTorch and its shape law:

- element-wise with broadcasting, the broadcast proved by an argument (`Broadcast<a, b, out>`), and its gradient reduced
  back to each operand's shape;
- sum and mean over an axis, with and without `keepdim`; the whole-tensor sum;
- reshape and flatten (a product proof), permute and transpose (copies), slice/narrow (a bounds proof), concat (the
  sizes add up);
- batched matmul over a leading batch axis, and its backward;
- ranks 0 to 4 at least; `Mat<r, c>` becomes `Tensor<[r, c]>` without a copy.

First an afternoon prototype (broadcast add and a sum over an axis at rank 3, with their laws), measuring the checker's
time and how long a call's types get. Decision point, written in NOTES: adopt and port the layers, or keep `Mat` and add
the operations on it.

- Pass: every operation within the tolerance against PyTorch on random shapes (including size-1 axes and empty
  tensors), its shape law by `--verdict`, rejected programs for a wrong broadcast, slice and concat.
- Done: `torch/tensor.bend`, decision in NOTES and design.md (adopted next to `Mat`, the layers not ported). 16
  operations and their gradients within the tolerance on 204 random cases (ranks 0-4, size-1 and empty axes; worst 0.070
  of it), natively and in JavaScript; 7 shape laws by `--verdict`; 4 rejected programs; 5 new mutants, all caught.

### M8 · Laws about the code that runs (done 2026-10-17)

Done: `kv_cols`, `causal_grad`, `attn_par_fwd`, `attn_par_bwd`, `seq_chain`, `fork_sum`, and `causal` rewritten over the
count the softmax's own loop reports, all by `--verdict`; 46 mutants as expected, the two that used to slip past the
laws now caught by them (NOTES). Two items changed on the way: bend-ml-tensor-array's parallel product cannot be stated
as a law (each block reads its own packed copy of its rows, and equal values would need the contents of arrays written
by index), so it stays tested; and the chain rule is a law about `Seq` and `Fork`'s own code on the model's linear
layers, since a law cannot quantify over layers (they are templates). I accepted both changes as completing M8
(2026-10-17). M7's shape laws, which the first text of M8 listed, were proved in M7.

Before it, today's laws are about structure and about the exact `Nat` model of the linear layers (`spec.bend`); the claim this
repository repeats most, the same bits at every thread count, is tested, not proved. M8 moves laws onto the code the
demos execute, where that code is inside the proofs:

- **Parallel = sequential, where the tree is safe.** Attention's tree over sequences (`fpar`, `bpar`) returns, at every
  depth, the outputs of running the sequences one after another (`fleaf`, `bleaf`), in order; bend-ml-tensor-array's
  parallel product (blocks of rows or columns appended as lists) gives its sequential rows at every depth. Not for
  bend-ml-tensor-fork or `torch/fast.bend`: in Base's model `Array.fork` gives two copies and `Array.join` keeps the
  first, so no law about the shared-array tree can be stated; those stay tested bit for bit (`fast_test`, tensor-fork's
  `tests/bits.bend`).
- **Attention reads key/value head `kv_of`** in the array code both passes execute (the head loops of `fheads` and
  `bgroup`), so the mutant that bypasses `kv_of` and survives the laws today is caught by one.
- **Causal gradients**: the backward pass sends nothing to a key or value at a position after the query's (a law about
  the count of the G row `g_set` builds, as `causal` is about `prow`'s).
- **`Seq`'s backward is the chain rule for any two layers** of the linear model, not only for two linear layers.
- Pass: the laws by `--verdict`; the two mutants that survive the laws today (`kv_of` bypassed, a softmax over one
  position too few) and new mutants of the parallel trees are caught by laws; every new law explained in
  `torch/README.md`.

### M9 · Training infrastructure, the rest (done 2026-10-17)

- Sizes read at run time (bend-ml's `examples/runtime_batch.bend` pattern): a batch size from the command line.
- An optimizer state per optimizer: SGD keeps no buffers, momentum one, Adam two (the step's signature changes in every
  layer and combinator, not its arithmetic).
- Weights and checkpoints from one binary file, read straight into the arrays: `Layer.load` reads from an `Array<F32>`
  at an offset instead of a list (a list of SmolLM2-135M's 135M numbers would take several GB), the file read by
  `bend-ml-binary-read` (a temporary exception, loading code only, until Bend reads files natively; Renan, 2026-10-17).
- A shorter way to write a model's types (helpers that name the composite types).
- Pass: the run-time-sized MNIST trains like the literal one (same bits); SGD's checkpoint holds only the parameters;
  the tiny Llama and MNIST loaded from one file give the same bits, and a resumed checkpoint too; peak memory of the 360M
  block under SGD and AdamW, and the load time of a SmolLM2-135M-sized file, measured. Writing checkpoints straight from
  the arrays waits for M10, where the size matters.

Implementation checks: command-line batch sizes 100, 37 and 8191 (including partial training/evaluation batches);
optimizer-specific buffers in both implementations; array cursors and exact-size binary files; shorter pair/residual
types. Checkpoints resume exactly under all four optimizers with tied weights and Dropout. The original output bits
and the tiny Llama's AdamW checkpoint layout are retained. Memory and load-time measurements remain open until the
machine is idle; M9 is not closed before those results are recorded. Done: 360M-sized block 234 MiB (SGD) and 354 MiB
(AdamW) peak on 1 thread; the 538 MB SmolLM2-135M-sized file read in 0.26 s (warm cache, 1.0 GiB peak: the array's
capacity is the next power of two); an eager `Layer.count` that slowed the tiny Llama 6% found and made lazy (NOTES).

### M10 · SmolLM2-135M and 360M fine-tuning (done 2026-10-17)

Load SmolLM2-135M with `torch/load.bend` into BendTorch's layers (tied embeddings through `Tie`) and fine-tune it for a
few steps against PyTorch; then 360M, whose F32 training state (weights, gradient and AdamW's two moments, 16 bytes a
parameter, about 5.8 GB) fits this machine's 32 GB. 1.7B (about 27 GB of state) waits for M11's BF16 storage or for
fine-tuning a part of the weights (proposed 2026-10-17, my call to confirm).

- Pass: one step's gradients against `torch.autograd` on the real weights; loss and memory per step measured.
- State (2026-10-17, NOTES): both models load, train and give the same bits on 1 and 16 threads; the loss is within the
  shared tolerance of PyTorch's and memory per step is measured. The gradients are not within it, and PyTorch's own
  float32 gradients are not either, against float64 (17 of 272 tensors at 135M). New gradient criterion (2026-10-17):
  per tensor, Bend's rms error against float64 at most 8x PyTorch float32's, at most 2x in the median. Training losses
  after the first: within the tolerance of a float64 run, or at most 8x PyTorch float32's error against it. Both sizes
  pass (`make smol`). Fine-tuning (2026-10-18): 400 steps on Tiny Shakespeare, the held-out loss 3.52 -> 3.24 in
  Bend, within the check's bound against float64 (at most 8x PyTorch float32's deviation; over one run it is 0.4-1.6x,
  set by chaos: one weight 1 ulp away moves it that much, NOTES 2026-10-20) (`make finetune`); `make mutate-smol` shows which
  bugs the full-size test catches and which are below float32's resolution (NOTES). Step time measured: 5.0-7.0x PyTorch's (135M 5.3 s, 360M 11.7 s per AdamW step on 16 threads).
- Before it (done 2026-10-17, NOTES): AdamW's memory beyond its two moments is the arrays' power-of-two capacity. Exact
  loading was dropped (2026-10-17): the loader's doubled source array is released before training allocates its
  state, so it never sets the peak; it comes back if loading does (M11, 1.7B).
- Changed 2026-10-17: this milestone no longer waits for bend-ml's transposed band product (`Wᵀ·dy` over `Bands`).
  Since M6 phase 3 the backward products are BendTorch's own, over bend-ml-tensor-fork (`dy·W` and `dyᵀ·x`, parallel,
  no copies), and the block trains at SmolLM2's sizes against PyTorch; since M9 the weights load through `load.bend`,
  not llama.bend's `Bands`. The band product stays a bend-ml item, for training in llama.bend's layout.

### M11 · Dtypes (~1-2 weeks)

- BF16 and F16 stored two per `U32`, converted to F32 for compute with PyTorch's rounding (round to nearest even), the
  conversions tested bit for bit against `tensor.to(torch.bfloat16)`.
- Mixed precision: BF16 weights and activations in storage, F32 arithmetic and F32 master weights in the optimizer.
- int8 with a scale per row, for inference.
- F64: not in Bend 2.0.38; a two-F32 ("double-word") accumulator as an experiment for long sums, and an upstream request
  draft.
- Pass: conversions bit-exact; a mixed-precision tiny Llama within a stated tolerance of the F32 one, measured. **Done 2026-10-18** (`torch/dtype.bend`, `make check`, `make mixed-long`; NOTES): conversions bit-exact, int8 within the tolerance, mixed precision within the adopted criterion; SmolLM2-1.7B's fit left for after v1.

### M12 · Backends (~1 week, GPU open-ended)

- A JavaScript lane: the test programs built with `bend -o x.js` and run with Bun (not node: the file reads use
  `bun:ffi`), compared with the same oracles. **Done 2026-10-16** (`make check-js`): every test passes and the training
  and generation outputs are the native build's bits; `exp`/`tanh` happened to match libm's on every input tested.
- WASM: not a Bend 2.0.38 target; only JS inside a bundled page (`bend page.html -o dir`). Recorded, not built.
- GPU: element-wise and product kernels with `!` on the RTX 4050, measured against the CPU on the same work; adopted
  only where faster.
- Pass: `make check-js` green; a GPU measurement in NOTES whatever its result. **GPU products done 2026-10-18** (`torch/gpu.bend` over
  bend-ml-tensor-gpu, `make gpu`): the same bits as the CPU; slower than the CPU at MNIST's, the tiny Llama's and a
  SmolLM2-sized block's sizes, so the demos stay on `fast.bend` (NOTES).

### M13 · CI (done 2026-10-16; before M6)

The workflow exists and has never run (Actions is off on the private remote). The free plan gives 2,000 minutes a month
to private repositories, and a full run (setup, `check`, `bits`, `mutate`, `check-js`) should take 10-15 minutes, so not
everything on every push: `make check` and `make bits` on each push to `devel`, cancelling a run still in progress
when a newer push arrives, and the full run on pull requests to `main` and nightly. Measure the first runs and fix
whatever differs on a clean Ubuntu runner. Done: quick 6.4 min (cold caches), full 10.4 min, both green (NOTES).

### M14 · A frozen, held-out numerical corpus (~2 days)

Inputs drawn with a seed never used in development, their PyTorch results computed once and stored with a checksum
(`bench/corpus/`), the policy versioned; the corpus is run at release time, not while developing, so it stays held out.
IEEE edge cases extended to the norms, softmax and the losses. **Done 2026-10-18** (`bench/corpus/` corpus-2, `make corpus`; NOTES).

### M15 · A benchmark against other frameworks (~1 week, after M10)

As bend-ml and llama.bend were measured against PyTorch, NumPy, llama.cpp, candle and llm.c: the same models trained
in other frameworks and languages, from the same initial weights on the same data.

- Workloads: the MNIST MLP (an epoch), the tiny Llama (training steps, then sampling), M10's SmolLM2-135M fine-tuning
  step, and the single products and layers (forward and backward) on their shapes.
- Systems (versions pinned, each built and run as its docs say): PyTorch (eager, and `torch.compile` where it works on
  CPU), JAX, tinygrad, a NumPy reference with hand-written backward, Rust's candle and burn, llm.c for the GPT-style step,
  Julia's Flux. Which ones can run each workload is decided by trying, and written down.
- Measured with my benchmark harness: an idle machine, alternated runs, one thread pinned and 16 threads; time per step,
  start-up, peak memory; every system's loss curve against PyTorch's within the stated tolerance.
- Pass: a `BENCHMARK.md` with the machine, the versions, every number and the losses shown; the method and raw results
  in my private notes. Its numbers go into the README before the repository opens.

## The order to v1 (agreed 2026-10-18)

1. Idle-machine run 1: the native target and FMA (below), and the GPU probes (matrix products on the GPU against the
   CPU and against the same algorithm handwritten in CUDA).
2. Where the GPU's remaining gap to handwritten CUDA goes; upstream reports with the numbers.
3. **M12's GPU part, products only**: a GPU products package in bend-ml (coalesced layouts, tiled products, forward
   and backward), `torch/gpu.bend` here (isolated like `fast.bend`, my decision 2026-10-18) training MNIST and the tiny
   Llama, llama.bend generating with it. SmolLM2's training stays on the CPU for v1.
4. **M11 in full** and **M14**, alongside step 3.
5. **M15** with PyTorch (eager and `torch.compile`), JAX, tinygrad, TensorFlow, candle, burn, dfdx, llm.c, Flux.jl and
   Lux.jl, on the CPU and on the GPU (my decision, 2026-10-18; BendTorch's GPU layers against each system's CUDA
   build); a system that cannot run a workload on one of them is recorded as such.
6. Idle-machine run 2: every final timing.
7. Going public (below).

## For the next idle-machine run (added 2026-10-18)

Measurements that need an idle machine, to take in the next run made for something else: none at the moment (the native
target and FMA were timed on 2026-10-18, `NOTES.md`).

## To try: training on 16 threads (added 2026-10-25)

Status 2026-10-26: AdamW's two paths are in blocks (NOTES), RMSNorm stays (0.5% of the step), the cross entropy
over the array is in bend-ml, not published. Timing waits for an idle window.

Found by timing BendTorch v1.1.1's pieces alone at 1 and 16 threads (the same bits on both):

- **AdamW**: `Opt.mat` updates all r x c weights in one sequential loop (`adam_l`), 1.0x on 16 threads over a 49152 x
  576 matrix. With F32 moments `fast.bend`'s layers already split the update (`opt_mat`, `par4`); two paths still take
  the sequential loop: the embedding (`fast.bend`'s `Embedding` is `T.Embedding`, whose step is `T.mat_step`), which in
  SmolLM2 is that 49152 x 576 table, tied to the output head; and every matrix with packed BF16 moments
  (`CAdamWBF16`, the 1.7B's), which `opt_mat` hands to `T.Opt.mat` in one block. Split both into blocks; each weight's
  arithmetic is unchanged, so no bit can move. Pass: the same `make bits` and `bits-smol`, the update faster on 16
  threads, `train 3` measured.
- **RMSNorm**: a 512-row step (forward, backward, AdamW) is 1.0x on 16 threads. Profile where its time goes (the
  layer's split, the optimizer or the setup) before changing anything.
- **Cross entropy and the MNIST step**: bend-ml's `Mat.softmax_ce` walks its rows as lists, one after another; over
  an `Array` the same rows give the same bits in 3.4x fewer cycles. That change belongs to bend-ml; time it here
  inside the MNIST demo and `train 3` before it is published.

## Going public

The repository is private until a v1 I am satisfied with; then it opens. Before that:

- an audit of every file and of the whole history that will be published: nothing private, no analysis or method
  beyond a one-line result in `NOTES.md` and `docs/` (the detailed material belongs in my private repositories), no link
  to private repositories;
- a README with the measured numbers (M15's benchmark among them), `CONTRIBUTING.md` and the GitHub templates on `main`, CI green;
- a decision about the BendHub package name and a check of the name's conflicts.
