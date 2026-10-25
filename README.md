<p align="center">
  <img src="docs/media/banner.svg" alt="BendTorch: training neural networks in Bend 2. Gradients in the types, laws checked by the kernel, checked against PyTorch step by step." width="100%">
</p>

<p align="center">
  <a href="https://github.com/nuxyel/BendTorch/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nuxyel/BendTorch/ci.yml?branch=main&style=flat-square&label=ci&labelColor=4d4a44&color=7e9a5e" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-8b83b5?style=flat-square&labelColor=4d4a44" alt="MIT license"></a>
  <a href="https://bend-lang.com"><img src="https://img.shields.io/badge/Bend-2.0.39-8b83b5?style=flat-square&labelColor=4d4a44" alt="Bend 2.0.39"></a>
  <img src="https://img.shields.io/badge/laws%20proved-22-7e9a5e?style=flat-square&labelColor=4d4a44" alt="22 laws proved">
  <img src="https://img.shields.io/badge/planted%20bugs-98%2F98%20as%20expected-7e9a5e?style=flat-square&labelColor=4d4a44" alt="98 planted bugs, 98 as expected">
</p>

**BendTorch** trains neural networks in [Bend 2](https://bend-lang.com) with general pieces: layers with typed forward
and backward passes, combinators that compose them, losses, four optimizers and one training step that works for any
model. A 784-128-10 MLP on MNIST and a two-layer Llama (grouped-query attention, RoPE, SwiGLU, RMSNorm) are built from
those pieces and trained in Bend, and both agree with PyTorch twins that start from the same weights and see the same
batches. Every gradient has the type of the value it is the gradient of, so a transposed gradient or two layers that do
not fit together do not compile, and the formulas of the backward pass are laws checked by Bend's kernel. It builds on
[bend-ml](https://github.com/nuxyel/bend-ml)'s matrices and on what [llama.bend](https://github.com/nuxyel/llama.bend)
learned about the Llama layers.

BendTorch is training: layers, losses, optimizers, the training step and checkpoints. Models and inference are
llama.bend's. The Llama models here (the two-layer Llama, SmolLM2's fine-tuning) are test twins, written to check
training against PyTorch, not llama.bend's model; the plan after v1 is for llama.bend's Llama to be built from these
layers, so that one implementation serves both.

Early research code: the CPU and one CUDA GPU, F32 training, matrices only (higher ranks are folded into rows and
columns). Bend **2.0.39**, MIT. On BendHub as `bendtorch` and three companion packages ([05 · Use it](#05--use-it)).

<p align="center"><img src="docs/media/learning.webp" alt="The two-layer Llama in Bend 2: what it writes after 0 to 2,000 AdamW steps, and its training loss falling from 4.228 to 2.359 (a mean of 50 steps) next to PyTorch's twin" width="90%"></p>

## 01 · A model is a layer

```python
# Linear(784, 128) -> ReLU -> Linear(128, 10), on batches of 100 rows
def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, X, TA.Mat<100n, 128n>, T.Two<X, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~X, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~X, ~TA.Mat<100n, 128n>, ~T.Linear(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))

def Model() -> T.Layer<P, X, Y, S>:
  T.Seq(~T.Two<T.Lin<784n, 128n>, Unit>, ~T.Lin<128n, 10n>, ~X, ~TA.Mat<100n, 128n>, ~Y, ~T.Two<X, TA.Mat<100n, 128n>>, ~TA.Mat<100n, 128n>, ~Hidden(), ~T.Linear(100n, 128n, 10n, 0n))

# one training step, the same for every model and loss
T.Train.step(~P, ~X, ~Y, ~S, ~List<&2, Nat>, ~Model(), ~ce, cfg, state, x, labels)
```

A layer is a record of four functions over its parameters `P`, its input `X`, its output `Y` and what its forward pass
saves, `S`. Its backward pass turns a gradient typed `Y` into one typed `X` and adds the parameters' gradient into a value
typed `P`. Layers are passed as templates (`~`), which Bend copies at compile time, so `Seq`, `Fork`, `Both` and
`Residual` compose any layers whose types meet. The Llama in [`demo/tiny_llama.bend`](demo/tiny_llama.bend) is written
the same way: `Residual(Seq(RMSNorm, attention))`, where attention is `Fork(q, Fork(k, v))`, then RoPE on q and k, the
attention core and the output projection.

A layer whose shapes do not meet is rejected before anything runs ([`tests/`](tests/)):

```
$ bend tests/bad_seq.bend      # Linear(784, 128) followed by Linear(64, 10)
- expected : T.Layer<T.Lin<64n, 10n>, TA.Mat<100n, 128n>, TA.Mat<100n, 10n>, TA.Mat<100n, 64n>>
- observed : T.Layer<T.Lin<64n, 10n>, TA.Mat<100n, 64n>, TA.Mat<100n, 10n>, TA.Mat<100n, 64n>>
```

## 02 · Laws

| Law | In plain language |
|---|---|
| `adjoint` | a linear layer's backward pass `dx = Wᵀ·dy` is the adjoint of its forward pass `y = W·x`: `dy·(W·x) = (Wᵀ·dy)·x` for every `W`, `dy`, `x`, which is what makes `dx` the gradient |
| `weight_adjoint` | the weight gradient `dW = dy ⊗ x` is the adjoint in `W`: `dy·(W·x) = <dW, W>` |
| `adjoint_seq` | `Seq`'s backward pass on two linear layers, `W1ᵀ·(W2ᵀ·dy)`, is the adjoint of `W2·(W1·x)`: the chain rule |
| `causal` | attention gives probability exactly 0 to every later position: the softmax of row i runs over exactly i + 1 positions, counted by its own loop |
| `causal_keys` | attention reads only keys 0..i for the query at position i, one score each |
| `causal_grad` | the backward pass sends no gradient to a key or value after the query's position: row i of the scores' gradient is computed over exactly i + 1 positions |
| `kv_of_group`, `gqa_head`, `kv_cols` | with `g` query heads per key/value head, query head `j·g + i` reads key/value head `j`; `kv_cols` says it of the columns both passes of attention read, which come from one function |
| `attn_par_fwd`, `attn_par_bwd` | attention's tree of parallel sequences gives exactly what running them one after another gives, at every depth and however the sequences are split: the parallel passes are the sequential ones |
| `seq_chain`, `fork_sum` | the `Seq` and `Fork` combinators' own code, run on linear layers, gives a backward pass that is the adjoint of the forward pass: the chain rule through `Seq`, the sum of the input gradients through `Fork` |
| `bc_sym` | broadcasting does not depend on the order of the operands: `x + y` and `y + x` accept the same shapes |
| `sum_numel`, `cat_numel`, `mat_numel` | summing over an axis, joining two tensors and viewing a matrix as a tensor keep the count of numbers right, for every shape |
| `narrow_rest`, `narrow_axis`, `permute_rank` | narrowing an axis changes that axis only, to exactly the length asked for; a permutation of the axes keeps the rank |
| `block_sum` | summing a row in blocks of 2^p cells, each block as a balanced tree of pairs and then the block sums as `tree_sum` adds them, makes the same additions in the same order as `tree_sum` over the whole row, whenever 2^p divides the row: the GPU's block sums give the CPU's bits |
| `unpair_pair` | packing 16-bit values two to a word and reading them back gives the same list, for any length |

They are stated in [`LAWS.bend`](LAWS.bend) and proved in [`PROOF.bend`](PROOF.bend); `bend PROOF.bend --verdict` checks
them again with Bend's kernel proved in Lean, with no `@unsafe`, no `?TODO` and no axioms. [`torch/README.md`](torch/README.md#the-laws-in-plain-language)
explains each proof. The linear laws are about an exact model where numbers are `Nat`; the F32 code computes the same
formulas and is tested, not proved, to do so.

98 deliberate faults planted in the code and the specification (`scripts/mutate.py`) each behave as expected: 96 are
caught by a law or a test, and two survive by design, because they still compute the right result. One is right by
accident (a one-cell array wraps every index to 0), and a bounds check catches the same fault; the other only changes
a GPU block size, and any block size that divides a row gives the same sum, bit for bit: the law `block_sum` proves it.

<p align="center"><img src="docs/media/mutants.webp" alt="98 planted bugs: 96 caught, 2 survive by design" width="80%"></p>

## 03 · It trains, and it computes what PyTorch computes

Against `torch.autograd` and `torch.optim` (PyTorch 2.14.1, CPU), with one tolerance for every number,
`|bend - torch| <= 1e-6 + 1e-5·|torch|` with NaN matching NaN, and bit-for-bit equality for the results that are a copy or
one or two F32 operations in PyTorch's order (relu, an embedding lookup, the MSE and NLL gradients).

| | numbers compared | bit-identical | worst error, as a share of the tolerance |
|---|---|---|---|
| every layer, forward and backward, 10 seeds ([`test_ops.py`](reference/test_ops.py)) | 18,140 | 13,435 | 0.43 |
| a 7-6-3 MLP: gradients, accumulation, 3 steps of SGD, momentum, Adam, Adam + L2, AdamW, 12 seeds ([`test_mlp.py`](reference/test_mlp.py)) | 6,588 | 4,219 | 0.06 |
| the two-layer Llama: the loss and all 21 parameter gradients ([`test_tiny.py`](reference/test_tiny.py)) | 22,753 | 3,916 | 0.02 |
| the same Llama after 20 AdamW steps: every loss and every final weight | 22,772 | 4,826 | 0.86 (0.79 against float64, the test's rule) |

- **MNIST** (784-128-10, SGD at lr 0.1, one epoch): loss 0.5148658 against PyTorch's 0.5148658, 9,108 test hits in both.
  50 batches under each optimizer give the same loss within the tolerance and the same hits.
- **The Llama** (65 characters, width 32, 4 query and 2 key/value heads, AdamW): the loss falls from 4.23 to 2.63 in
  500 steps on Tiny Shakespeare, as PyTorch's. The two runs agree to 1e-6 for 60 steps and then drift apart: Adam
  divides by `sqrt(v)`, which magnifies the rounding of gradients near zero ([`NOTES.md`](NOTES.md) traces where).
- **Generation**: PyTorch, given the weights BendTorch trained, continues a prompt with the same characters as BendTorch
  (300 out of 300 after 2,000 steps). The model is tiny and greedy decoding loops ("e than thar thar thar ..."): this
  shows that the two compute the same function, not that the model writes well.
- **Checkpoints**: 30 steps, a checkpoint, then 20 steps in a new process give the same bits as 50 steps without a stop.
- **SmolLM2-135M generation with a key/value cache** (`genkv` in [`demo/smol.bend`](demo/smol.bend)): every logit has the
  same bits as running the whole window again (49,152 a token, 285 tokens on five prompts), at about 80 ms a token on 16
  threads against 783 ms without the cache.
- The activations also match PyTorch on NaN, infinities, signed zeros, subnormals and overflowing inputs; dropout's mask
  is reproduced bit for bit by the twin; tied embeddings match a PyTorch table used twice.

## 04 · Speed

On an Intel Core Ultra 7 155H and an RTX 4050 Laptop GPU, idle machine, medians of 5 alternated rounds (SmolLM2: 3),
2026-10-24, Bend 2.0.39. One thread is pinned to a performance core:

| | BendTorch, 1 thread | PyTorch, 1 thread | BendTorch, 16 threads | PyTorch, 16 threads | BendTorch, GPU |
|---|---|---|---|---|---|
| MNIST, one epoch of SGD | 1.7 s | 0.33 s | 1.3 s | 0.39-1.12 s | 8.6 s |
| two-layer Llama, one AdamW step | ≤ 0.8 ms | 1.9-2.7 ms | ≤ 1.0 ms | 3.1-7.6 ms | ≤ 84 ms |
| one decoder block at SmolLM2-135M's sizes (512 rows), one AdamW step | 659 ms | 108 ms | 140 ms | 50 ms | 205 ms |
| SmolLM2-135M, one AdamW step on 2 x 128 tokens | 14.2 s | 2.48 s | 4.3 s | 0.98 s | - |

BendTorch is 5x to 6x slower than PyTorch on MNIST and on one thread of the SmolLM2-sized block and of SmolLM2-135M
itself (on 16 threads 2.8x on the block and 4.4x on SmolLM2), where the products dominate: bend-ml's products read cells of 8 numbers per array
access, but each sum is a chain of scalar F32 additions in a fixed order, far from a BLAS kernel. It is 2.4x to 3.5x
faster on the small Llama on one thread, where PyTorch's per-operation overhead dominates a model 32 wide (its
BendTorch cells are the whole program's wall time, start-up included, so they are upper bounds; PyTorch's range is
two harnesses, steps 2-200 on its clock and 200 steps minus start-up). Every loss in these runs is checked against
PyTorch's ([`BENCHMARK.md`](BENCHMARK.md) says how), and BendTorch gives the same bits on 1 and 16 threads.

On 2026-10-21 I found why a SmolLM2-135M step stopped scaling at about 6 threads: attention ran its 2 sequences one
after the other. With them in parallel (the same bits, by the laws already proved), idle machine, 3 alternated rounds,
a step takes 4.0 s on 16 threads against PyTorch's 0.93 s (4.3x) and 13.8 s on one thread against 2.43 s (5.7x). On
the GPU, with the embedding, the cross-entropy and a faster attention now on the device, it takes 7.0 s.

SmolLM2-1.7B trains here too, from its real weights, with AdamW's moments packed in BF16 so that the step fits in 32 GB:
on 2026-10-24, idle machine, 3 alternated rounds, a step takes 42.0 s on 16 threads against 13.3 s for PyTorch with the
same BF16 moments (3.2x), at a peak of 24.1 GB (PyTorch 23.6 GB), and both losses are within the tolerance of
PyTorch's ([`BENCHMARK.md`](BENCHMARK.md), W6).

The demos' linear layers, their optimizer step and the element-wise layers are the fast ones of
[`torch/fast.bend`](torch/fast.bend), whose parallel blocks share their arrays instead of copying them (outside the
proofs; tested bit for bit against the proved layers). The row-wise layers (attention, the norms, softmax, RoPE, the
embedding and the losses) run over arrays a row at a time, with the bits of the list versions they replaced. 16
threads do not help a step as small as the tiny Llama's; on the block they give 4.7x over one thread. On the GPU every
size here is slower than 16 CPU threads: most of a GPU step is Bend's runtime, not the arithmetic
([`docs/limitations.md`](docs/limitations.md#speed)). PyTorch's 16-thread times swing from round to round on this
machine, so those cells give the range of the rounds.

[`BENCHMARK.md`](BENCHMARK.md) has the same four workloads in eleven other systems (PyTorch, torch.compile, JAX,
tinygrad, TensorFlow, candle, burn, dfdx, Flux.jl, Lux.jl, llm.c), on the CPU and the GPU, each with its losses checked
against PyTorch's.

## 05 · Use it

Four packages on BendHub, all at version 1.1.0.0. Import the files you need in your own Bend program:

```python
import bendtorch@1.1.0.0/main.bend as T          # layers, combinators, losses, optimizers, Train
import bendtorch@1.1.0.0/tensor.bend as TT       # general tensors: any rank, broadcasting, their gradients
import bendtorch-fast@1.1.0.0/main.bend as F     # the same layers and bits, in parallel on the CPU
import bendtorch-gpu@1.1.0.0/main.bend as G      # the layers on one CUDA GPU (all but GELUErf)
import bendtorch-load@1.1.0.0/main.bend as L     # weights from flat binary and safetensors files, checkpoints
```

`bendtorch` holds the laws and their proofs. The other three are outside the proofs on purpose: `-fast` and `-gpu` share
arrays through Base's `@unsafe` `Array.fork`, and `-load` reads files with a temporary foreign reader until Bend reads
files natively. Each is tested against `bendtorch`'s bits; on the GPU they match except where CUDA's float `exp` rounds
some inputs differently ([`docs/limitations.md`](docs/limitations.md)). [`torch/README.md`](torch/README.md) has the whole API.

## 06 · Reproduce

Linux x86_64, clang, Python 3.14, `curl`. The Python venv with PyTorch takes about 1.2 GB.

```bash
make setup   # Bend 2.0.39 (SHA256-checked), Lean 4.34.0, the Python venv of reference/; no sudo
make data    # MNIST and Tiny Shakespeare into data/, the twins' initial weights and batches (seed 0)
make check   # the laws (--verdict), the rejected programs, every layer, the MLP, MNIST, the Llama and a SmolLM2-sized block against PyTorch (~2 min)
make mutate  # the 98 deliberate faults
make bench   # BendTorch against PyTorch on MNIST, the tiny Llama and the block (~6 min)
make check-js  # the same tests on the JavaScript build, run with Bun; the outputs must be the native build's bits
```

To train the tiny Llama on a text of your own (any language: the model reads bytes):

```bash
make own TEXT=lyrics.txt                  # 2,000 AdamW steps in a few seconds, then 400 characters in its style
make own TEXT=lyrics.txt STEPS=5000 PROMPT="the first line of a song"
```

It keeps the text's 65 most frequent bytes, trains from the twin's initial weights, writes every step's loss to
`.scratch/own/<file>/loss.tsv` and continues the prompt by drawing each character at temperature 0.8. A model of 22,752
parameters learns spelling and short words, not meaning; on Tiny Shakespeare, after 1,000 steps: "ROMEO: What light
through yonder, me by why to ham som o then co penen my I thow thout whe the so chot tror hos hom abl Lo por ...".

`main` will hold released versions; work happens on `devel`. Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 07 · Limits

The full list, each with its measurement or reason, is in [`docs/limitations.md`](docs/limitations.md). In short:

- **The laws cover structure, not numbers**: shapes, composition, the linear layers' backward formulas on an exact
  model, the causal mask, head grouping. Every F32 number and the non-linear gradients are tested against PyTorch.
- **Outside the proofs on purpose**: the parallel and GPU layers (`fast.bend`, `gpu.bend`) share arrays through Base's
  `@unsafe` `Array.fork`, and loading uses a temporary foreign reader; the tests check they give `main.bend`'s bits.
- **Slower than PyTorch** wherever products dominate, and on the GPU slower than 16 CPU threads at these sizes: most of a
  GPU step is the runtime (filling new arrays on one lane, a fixed cost between kernels), not the work.
- **Matrices only**, F32 training (BF16 storage, int8 inference), one CUDA GPU; SmolLM2-1.7B trains in 32 GB only with AdamW's moments packed in BF16, at 24.1 GB of peak memory.

## Where things are

| Path | What |
|---|---|
| [`torch/`](torch/) | the package: layers, combinators, losses, optimizers, training (`main.bend`); the exact model the laws use (`spec.bend`) |
| [`LAWS.bend`](LAWS.bend), [`PROOF.bend`](PROOF.bend) | the laws and their proofs |
| [`docs/media/`](docs/media/) | the banner and the README's images (`banner.mjs`) |
| [`demo/`](demo/) | MNIST, the two-layer Llama (training, generation, checkpoints), and the programs the tests run |
| [`tests/`](tests/) | programs that must not compile |
| [`reference/`](reference/) | the PyTorch twins and the tests |
| [`scripts/`](scripts/) | setup, data, the mutation run, the benchmark |
| [`docs/design.md`](docs/design.md), [`docs/plan.md`](docs/plan.md) | how it is built and why; the milestones |
| [`NOTES.md`](NOTES.md) | every finding, measurement and decision, by date |

<p align="center"><sub>MIT · Renan Vinícius (@nuxyel)</sub></p>
