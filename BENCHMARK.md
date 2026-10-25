# Benchmark

BendTorch trains the same four models as eleven other systems: PyTorch (eager and `torch.compile`), JAX, tinygrad,
TensorFlow, candle, burn, dfdx, Flux.jl, Lux.jl and llm.c. Every system starts from the same initial weights, reads the
same batches in the same order and prints its loss before each update, so that I can check that it trains the same
before I compare its speed. This file has every number of that comparison, with the losses next to the times, and one
step of SmolLM2-1.7B against PyTorch alone (W6).

## The machine and the run

- Intel Core Ultra 7 155H (6 performance cores, 22 threads), 32 GB, NVIDIA RTX 4050 Laptop GPU 6 GB (driver 615.78.08),
  Linux 7.2.8, on AC power with the performance profile.
- Bend 2.0.39 (BendTorch at commit 92f0486), clang 23.1.3, Python 3.14.8, PyTorch 2.14.1.
- One idle round on 2026-10-24 (3 h 13 min, nothing else running): every case 5 times for BendTorch against PyTorch (SmolLM2-135M:
  3) and 3 times for the eleven systems, in alternated rounds (even rounds run each group in reverse order). The SmolLM2-135M
  step on one CPU thread ran once per system (marked ¹), because it alone took 44 minutes a round. The first full round,
  on 2026-10-18 with Bend 2.0.37, gave the same verdicts and, for the other systems, the same times within a few percent.
- "1 thread" is pinned to one performance core (`taskset -c 3`) with every library's thread count set to 1; "16
  threads" is unpinned with every count set to 16. "GPU" runs the model on the RTX 4050.
- Each cell is the median of the rounds, in ms per batch (MNIST) or per training step, from the program's own clock
  around the training loop: loading, compiling and warm-up steps are outside it.

## What "trains the same" means here

PyTorch eager on one CPU thread, in F32, is the reference, and the rule is BendTorch's one tolerance:
`|x - torch| <= 1e-6 + 1e-5·|torch|`, step by step.

- **The block and SmolLM2** (3 to 5 steps): every loss within that tolerance of PyTorch's ("worst" is the largest
  error, as a share of the tolerance; at most 1 passes).
- **MNIST and the tiny Llama** (600 and 200 steps): no F32 system stays within the tolerance of another for that long,
  PyTorch included, so each run is compared with PyTorch's float64 run of the same model. It passes when its distance
  from float64 (the RMS over the steps) is at most 8 times PyTorch F32's own ("RMS" is that ratio). MNIST must also give
  PyTorch's 9,108 test hits.

## W1 · MNIST MLP (784-128-10, SGD at lr 0.1, batches of 100), one epoch: ms per batch

| system | 1 thread | 16 threads | GPU | losses |
|---|---:|---:|---:|---|
| **BendTorch** | 2.83 | 2.17 | 14.3 | pass: loss 0.5148658 as PyTorch's, 9,108 hits as PyTorch's |
| PyTorch 2.14.1, eager | 0.61 | 0.83 | 0.40 | pass: RMS 0.01-1.00x |
| torch.compile (PyTorch 2.14.1) | 0.67 | 0.52 | 0.50 | pass: RMS 0.01-1.00x |
| JAX 0.11.2 + optax 0.2.8 | 0.39 | 0.30 | 0.05 | pass: RMS 0.01-1.00x |
| tinygrad 0.14.0 | 3.91 | 3.79 | 2.16 | pass: RMS 0.02-1.00x |
| TensorFlow 2.21.0 / Keras 3.15.1 | 0.60 | 0.67 | 0.30 | pass: RMS 0.01-1.00x |
| candle 0.11.0 | 0.79 | 0.61 | 0.40 | pass: RMS 0.01-1.00x |
| burn 0.22.0 | 0.66 | 0.65 | 3.63 | CPU pass 1.00x; GPU fail 82x (its F32 matmuls run as TF32) |
| burn 0.22.0, TF32 off (patched) | - | - | 3.65 | pass: RMS 1.00x |
| dfdx 0.13.0 | 0.77 | 0.59 | 0.22 | pass: RMS 0.01-0.02x |
| Flux.jl 0.16.12 | 0.57 | 2.21 | 0.29 | pass: RMS 0.01-1.00x |
| Lux.jl 1.31.4 | 0.75 | 2.62 | 0.31 | pass: RMS 0.01-1.00x |

llm.c has no MLP path. BendTorch's clock covers all 600 batches at a resolution of 0.1 s (the others: batches
51-600). The losses of the other systems were judged on the last 189-190 steps their logs kept, and in full in the
preparation runs, with the same verdicts.

## W2 · the two-layer Llama (65 characters, width 32, 4 query and 2 key/value heads), 200 AdamW steps: ms per step

| system | 1 thread | 16 threads | GPU | losses |
|---|---:|---:|---:|---|
| **BendTorch** | ≤ 0.80 | ≤ 1.00 | ≤ 84.3 | pass: RMS 0.49x, the same bits on 1 and 16 threads and on the GPU |
| PyTorch 2.14.1, eager | 2.74 | 7.43 | 5.14 | pass: RMS 0.74-1.00x |
| torch.compile (PyTorch 2.14.1) | 0.78 | 0.86 | 1.80 | CPU fail 10.15x; GPU 7.13-8.51x, a fail in one of six rounds over two nights |
| JAX 0.11.2 + optax 0.2.8 | 0.32 | 0.22 | 0.32 | fail: RMS 10.12-10.16x |
| tinygrad 0.14.0 | 8.06 | 8.29 | 3.04 | fail: RMS 10.58-10.64x |
| TensorFlow 2.21.0 / Keras 3.15.1 | 0.88 | 1.24 | 3.16 | fail: RMS 18.52-18.54x |
| TensorFlow, AdamW in PyTorch's order (mine) | 0.85 | 1.21 | 2.35 | pass: RMS 0.51-2.91x |
| candle 0.11.0 | 0.54 | 0.54 | 4.49 | pass: RMS 0.75-1.28x |
| burn 0.22.0 | 0.69 | 0.68 | 65.5 | fail: RMS 9.09-14.43x |
| burn 0.22.0, TF32 off (patched) | - | - | 68.2 | fail: RMS 9.10x |
| dfdx 0.13.0 (wrong gradients) | 0.34 | 0.34 | 1.91 | fail: RMS 19.24-23.80x |
| dfdx 0.13.0 with workarounds (mine) | 0.34 | 0.35 | 1.87 | CPU fail 9.05x; GPU pass 6.66x |
| Flux.jl 0.16.12 | 0.93 | 1.17 | 4.62 | fail: RMS 9.46-9.57x |
| Flux.jl, AdamW in PyTorch's order (mine) | 1.40 | 0.97 | 4.73 | pass: RMS 0.94-2.46x |
| Lux.jl 1.31.4 | 1.11 | 1.27 | 4.31 | fail: RMS 9.46-9.48x |
| Lux.jl, AdamW in PyTorch's order (mine) | 1.06 | 1.10 | 4.41 | pass: RMS 0.72-2.46x |
| llm.c, llama3 branch + my driver | does not run | does not run | 0.31 | fail: RMS 8.86x |

BendTorch's cells are the wall time of the whole program over 200 steps, start-up and loading included (10 ms
resolution), so they are upper bounds; the others are steps 2-200 on the program's clock. PyTorch eager on 16 threads
swung between 3.1 and 7.6 ms a step from round to round. BendTorch's own PyTorch twin, timed with its start-up taken
out, gives 1.9 ms a step on one thread, so on one thread BendTorch is 2.4x to 3.5x faster than PyTorch eager here.

## W3 · one decoder block at SmolLM2-135M's sizes (8 x 64 rows, width 576), AdamW, MSE: ms per step

| system | 1 thread | 16 threads | GPU | losses |
|---|---:|---:|---:|---|
| **BendTorch** | 659 | 140 | 205 | pass: worst 0.016x |
| PyTorch 2.14.1, eager | 103 | 39.9 | 4.60 | pass: worst 0.00-0.02x |
| torch.compile (PyTorch 2.14.1) | 102 | 32.5 | 3.47 | pass: worst 0.01-0.09x |
| JAX 0.11.2 + optax 0.2.8 | 101 | 24.4 | 3.30 | pass: worst 0.01-0.02x |
| tinygrad 0.14.0 | 12,359 | 12,098 | 75.7 | pass: worst 0.01-0.05x |
| TensorFlow 2.21.0 / Keras 3.15.1 | 105 | 27.7 | 5.40 | fail: worst 1.94-1.96x |
| TensorFlow, AdamW in PyTorch's order (mine) | 105 | 29.9 | 5.33 | pass: worst 0.01-0.02x |
| candle 0.11.0 | 171 | 124 | 11.0 | CPU fail 8.82x; GPU pass 0.01x |
| burn 0.22.0 | 135 | 136 | 31.3 | CPU fail 8.83x; GPU pass 0.03x |
| burn 0.22.0, TF32 off (patched) | - | - | 32.3 | pass: worst 0.02x |
| dfdx 0.13.0 (wrong gradients) | 134 | 77.0 | 4.33 | fail: worst 7.96-8.43x |
| dfdx 0.13.0 with workarounds (mine) | 135 | 81.7 | 4.67 | CPU fail 8.82x; GPU pass 0.08x |
| Flux.jl 0.16.12 | 204 | 181 | 16.7 | pass: worst 0.01-0.02x |
| Flux.jl, AdamW in PyTorch's order (mine) | 210 | 190 | 34.7 | pass: worst 0.01-0.02x |
| Lux.jl 1.31.4 | 206 | 175 | 21.7 | pass: worst 0.02-0.07x |
| Lux.jl, AdamW in PyTorch's order (mine) | 197 | 173 | 25.0 | pass: worst 0.02-0.07x |
| llm.c, llama3 branch + my driver | does not run | does not run | 4.80 | pass: worst 0.01x |

BendTorch times 3 steps after one warm-up step, the others 3 after two.

## W4 · SmolLM2-135M, one AdamW fine-tuning step on 2 x 128 tokens: ms per step

| system | 1 thread | 16 threads | GPU | losses |
|---|---:|---:|---:|---|
| **BendTorch** | 14,244 | 4,335 | no GPU build | pass: worst 0.414x |
| PyTorch 2.14.1, eager | 2,354 ¹ | 924 | 125 | pass: worst 0.00-0.56x |
| torch.compile (PyTorch 2.14.1) | 2,109 ¹ | 625 | 77.5 | pass: worst 0.01-0.55x |
| JAX 0.11.2 + optax 0.2.8 | 2,099 ¹ | 544 | 68.4 | pass: worst 0.47-0.52x |
| tinygrad 0.14.0 | 122,570 ¹ | 120,442 | 511 | pass: worst 0.54x |
| TensorFlow 2.21.0 / Keras 3.15.1 | 2,445 ¹ | 650 | 145 | fail: worst 4.75-4.76x |
| TensorFlow, AdamW in PyTorch's order (mine) | 2,427 ¹ | 643 | 139 | pass: worst 0.51-0.57x |
| candle 0.11.0 | 4,365 ¹ | 3,256 | 267 | CPU fail 3.78x; GPU pass 0.57x |
| burn 0.22.0 | 3,451 ¹ | 3,462 | 854 | fail: worst 4.01-53.64x |
| burn 0.22.0, TF32 off (patched) | - | - | 871 | fail: worst 1.23x |
| dfdx 0.13.0 (wrong gradients) | 2,806 ¹ | 1,763 | 140 | fail: worst 275.54-408.44x |
| dfdx 0.13.0 with workarounds (mine) | 2,808 ¹ | 1,759 | 142 | CPU fail 4.04x; GPU pass 0.59x |
| Flux.jl 0.16.12 | 3,253 ¹ | 2,740 | 207 | pass: worst 0.52-0.54x |
| Flux.jl, AdamW in PyTorch's order (mine) | 3,092 ¹ | 2,852 | 382 | pass: worst 0.52-0.54x |
| Lux.jl 1.31.4 | 3,328 ¹ | 2,780 | 309 | pass: worst 0.51-0.54x |
| Lux.jl, AdamW in PyTorch's order (mine) | 3,332 ¹ | 2,788 | 302 | pass: worst 0.52-0.54x |
| llm.c, llama3 branch + my driver | does not run | does not run | 78.2 | pass: worst 0.57x |

¹ one run. BendTorch times its 2 steps with no warm-up, so its first step includes making AdamW's moments; the others
time 3 steps after two warm-up steps. BendTorch has no GPU build of the full model yet; on the CPU it trains in 4.3 GiB
(one thread) to 4.6 GiB (16) of peak memory.

## W6 · SmolLM2-1.7B, two AdamW steps on 2 x 128 tokens from the real weights: s per step

Measured apart from the round above, and only against PyTorch: Bend 2.0.39, 2026-10-24, idle machine, 3 alternated
rounds, 16 threads, each run in a 25 GB memory limit with no swap. At this size AdamW's moments in F32 do not fit in
32 GB next to the weights, the gradient and the activations, so BendTorch keeps m and v packed in BF16 (`T.CAdamWBF16`,
an opt-in; `torch/README.md` has its measured cost), and the PyTorch twin does the same AdamW in the same order with its
moments stored as `bfloat16`. Both load the weights from the BF16 `model.safetensors` and time the 2 steps with no
warm-up.

| system | 16 threads | peak memory | losses |
|---|---:|---:|---|
| **BendTorch** | 42.0 | 24.1 GB | pass: 0.66x, 0.90x |
| PyTorch 2.14.1, eager, same BF16 moments | 13.3 | 23.6 GB | reference |

BendTorch is 3.2x slower here, against 4.7x on SmolLM2-135M in W4. Its losses are 2.6379600 and 2.5548434, PyTorch's
2.6379418 and 2.5548196, the same bits in every round. The first loss is 1.0e-6 from PyTorch's float64 one (PyTorch F32:
1.7e-5); the second step has no float64 reference, because a float64 run does not fit in memory. I did not time it on
one thread.

## Why the failures fail

None of the failing systems is tuned to pass. I checked each twin for a real difference instead, and these are the
causes I found:

- **AdamW's scalars in float32** (the ~9-10x cluster in W2: torch.compile on the CPU, JAX, tinygrad, burn,
  dfdx-fix on the CPU, Flux, Lux). PyTorch eager computes the bias corrections `1 - β^t` in double precision; these
  compute them, or keep `β^t` as a running product, in F32. The same model with PyTorch's order in F32 and the
  corrections in float64 passes: that is what the "AdamW in PyTorch's order (mine)" rows are, written by me, not by
  the frameworks. BendTorch had the same kind of drift until 2026-10-18 (38x PyTorch's distance from float64 over 100
  steps); it now carries `1 - β^t` without a power and lands at 0.49x.
- **Keras' AdamW adds epsilon after the bias correction** (Kingma and Ba's "epsilon hat" form), so its effective
  epsilon differs from PyTorch's: TensorFlow fails W2-W4. With PyTorch's order it passes everywhere.
- **An F32 running mean of the loss** on the CPU (candle, burn and dfdx-fix in W3 and W4): the block's output agrees
  to 8 digits and the gradients are unaffected, but the mean of 294,912 terms is summed one by one. The GPU paths sum
  in trees and pass.
- **TF32 matmuls in burn's GPU path**: with the TF32 registration removed from cubecl-cuda, W1 and W3 pass and its
  matmuls are 1.9-3.0x slower.
- **Two gradient bugs in dfdx 0.13.0**: on the CPU, a slice's backward pass assigns instead of adding (RoPE loses a
  term); on the GPU, the reshape of a broadcast tensor runs its backward over the storage, not the logical elements.
  The "with workarounds" row avoids both.
- **llm.c** passes on the first batch's gradients (every tensor) and on the block and SmolLM2, and drifts on the
  tiny Llama from step 60 on, deterministically. It is not fast math. The likely cause is AdamW magnifying the rounding
  of another summation order.

Several GPU paths (dfdx, TensorFlow, torch.compile, JAX, Flux) and TensorFlow on one CPU thread give different losses
from run to run, by up to 2% at the end of W2 and 1e-7 to 6e-7 in W3 and W4; every verdict above was the same in every
round but one: torch.compile on the GPU in W2 sits near the line, at 7.13-7.46x in five rounds over two nights and
8.51x in one. BendTorch gives the same bits run after run and on 1 and 16 threads; on the GPU too, except where CUDA's float
`exp` rounds differently (the block's first loss is 1 ulp off the CPU's).

## What it says about BendTorch

- **Correct everywhere**: BendTorch passes all four workloads, on the CPU and on the GPU. Of the eleven systems as they
  ship, only PyTorch eager passes every workload on every device; on the tiny Llama only PyTorch eager, candle and
  torch.compile on the GPU (in five rounds of six) pass without a rewrite of their AdamW.
- **Slower where products dominate**: 4.6x slower than PyTorch eager on MNIST (5.2x over the whole epoch), 6.4x (one thread) and 3.5x (16
  threads) on the block, 6.1x and 4.7x on SmolLM2-135M. Its products are chains of scalar F32 operations in Bend,
  with no BLAS: the math is Bend.
- **Faster where PyTorch's per-operation overhead dominates**: on the tiny Llama, 2.4x to 3.5x faster than PyTorch
  eager on one thread and 3x to 7x on 16 (PyTorch's own time swings there), about even with torch.compile and burn, slower than candle, dfdx and JAX (which
  fail W2 as they ship, except candle).
- **The GPU path is slower than 16 CPU threads** at every size here: most of a GPU step is Bend's runtime (filling
  new arrays on one lane, a fixed cost between kernels), not the arithmetic ([`docs/limitations.md`](docs/limitations.md)).

## Not in this table

- W5, the single operations (a 512 x 576 x 1536 product and its two backward products, RMSNorm, attention): the
  other systems were timed on it, but BendTorch was not timed on the same shapes in this round.
- Generation speed: llama.bend's own README has it for SmolLM2.
