# torch: BendTorch's package

Layers with typed forward and backward passes, composed generically and trained in Bend 2. A layer is a record of
functions over its parameters `P`, its input `X`, its output `Y` and what its forward pass saves, `S`:

```python
type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> OptState<P> -> Step<P>,
        zeros: Unit -> P, dump: P -> PL<P>, load: P -> Array<F32> -> U32 -> PA<P>, store: P -> Array<F32> -> U32 -> PA<P>,
        count: Unit -> U32}
```

`zeros` makes a gradient or a moment buffer. `Train.new(..., cfg, parameters)` allocates no moment buffers for SGD,
one for momentum and two for Adam/AdamW (`OptState<P>`: `NoMom`, `OneMom`, `TwoMom`). Keep the optimizer family fixed
for a training state; use a fresh state when switching families. The optimizer arithmetic is unchanged. Call
`T.OptCfg.check(cfg)` before training, as the demos do: it stops with a message when an Adam configuration's `c1`, `c2`
disagree with its `b1`, `b2`.

`dump` returns a flat list in a fixed order; `load` reads from one source array at an **element** offset and returns
`PA{parameters, source_array, next_offset}`. `count` gives the number of serialized F32 values, including non-learned
state such as Dropout's counter; it is a closure, like `zeros`, so building a composite layer does not build its
children's records just to add their counts (an eager field did, at every use of the template; NOTES, M9). Raw calls
to `Layer.load` require a source range large enough for that count; use the checked IO boundary below for files.
`store` is `load`'s inverse: it copies the parameters into an array from an element offset, in the same order.

The backward pass takes the gradient of the output, typed `Y`, and returns the gradient of the input, typed `X`, while
it adds the gradient of the parameters into a gradient of type `P`. So **a gradient always has the type of the value it
is the gradient of**: a backward pass that returns a transposed weight gradient, or composes two layers whose shapes do
not meet, does not compile ([`tests/`](../tests/)). Layers are passed as templates (`~l`), which Bend copies at compile
time, so `Seq`, `Fork`, `Both` and `Residual` compose any layers whose types meet, with no closure called twice and no
cost at run time.

- Bend: **2.0.39** · License: MIT · Depends on `bend-ml-tensor-array@0.1.14.0` (the matrices and their products) and
  `bend-ml-nat-lemmas@0.1.2.0`.
- `bend PROOF.bend --verdict` (from the repository's root) → `ALL PROOFS CHECK`, with no `@unsafe`, no `?TODO` and no
  axioms.
- On BendHub as `bendtorch@1.1.1.0`: `import bendtorch@1.1.1.0/main.bend as T`, and
  `import bendtorch@1.1.1.0/tensor.bend as TT` for the general tensors (below); `fast.bend`, `gpu.bend` and `load.bend`
  are the packages `bendtorch-fast`, `bendtorch-gpu` and `bendtorch-load`. The programs in `demo/` import the local copy
  (`import ../torch/main.bend as T`).

## What it has

| | |
|---|---|
| Layers with parameters | `Linear` (with bias, PyTorch's `out x in` layout), `LinearFirst` (the same, for a model's first layer: it skips the input gradient nobody reads), `LinearNB` (no bias), `Embedding`, `RMSNorm`, `LayerNorm` |
| Layers without | `ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU` (tanh form), `GELUErf` (the exact form, PyTorch's default `approximate="none"`), `Softmax`, `LogSoftmax`, `RoPE` (rotate-half), `Attention` (causal, grouped-query heads; the sequences of a batch in parallel), `Dropout` (a reproducible counter-based mask), `Add2`, `Mul2`, `Id` |
| Combinators | `Seq` (one after the other), `Fork` (one input to two layers, the input gradients summed), `Both` (a pair of inputs to two layers), `Residual` (`x + f(x)`), `Tie` (first, middle, last, where the first and the last share one parameter: tied embeddings) |
| Losses | `CrossEntropy` (mean, from logits; each row's softmax sum is a balanced tree, for vocabularies of 49,152), `NLL` (mean, from log-probabilities), `MSE` (mean) |
| Optimizers | `SGD`, SGD with momentum, `Adam` (with or without weight decay), `AdamW`; one pass per parameter matrix, in PyTorch's order of operations, the gradient zeroed on the way. `CAdam{lr, b1, b2, eps, wd, c1, c2}` and `CAdamW{...}` take `c1 = 1 - b1` and `c2 = 1 - b2` as their own constants (`0.1, 0.001` for the usual betas), as PyTorch computes them in float64, and the training state carries `1 - b^t`. `OptCfg.ok(cfg)` checks each pair: `0 < c <= 1` and `c` within 2^-23 of `1 - b` taken in F32 (always true for a `c` written as `1 - b`, false for a typo); `OptCfg.check(cfg)` stops an IO program with a message when it fails, and `Train.new` starts a failing configuration's state at NaN, so every loss after the first is NaN. `CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}` is AdamW with its first moment stored as packed BF16 (two values a `U32`, rounded to nearest even after each update, read back rounded by the weight update) and, when `v16`, its second moment too; the weights, the gradient and the arithmetic stay F32. Its state is `Packed{tree}`: the buffers of each matrix at a leaf, allocated on the matrix's first step, half the words of F32 moments; checkpoints write the tree with its sizes. A BF16 `v` loses most small updates of long runs (`b2 = 0.999` moves it by 0.1%, under half a BF16 step), so it drifts above the exact average: over 2,000 steps of the tiny Llama, 90.6% of v's updates were lost to rounding and the stored v sat 1.54x above the exact one at the median, while the losses stayed within M11's bound (0.0452 with m packed, 0.0234 with m and v, against 0.05). So it is an opt-in, not a default: m alone is safe, and m with v is for when memory decides, as for one step or a short fine-tune of SmolLM2-1.7B (AdamW's state 26.0 GiB in F32, 19.5 GiB with both packed) |
| Training | `Train.step`: forward, loss, backward and one optimizer step for any layer under any loss; `Train.save` and `load.bend`'s `train`: checkpoints of the parameters and the optimizer state (a resumed run gives the same bits) |
| General tensors (`tensor.bend`) | `Tensor<s>`, the shape `s` a list of sizes in the type, any rank (tested 0 to 4); `add`, `sub`, `mul`, `div` with PyTorch's broadcasting and their gradients (`grad2`, `sum_to`); `sum_axis`, `sum_axis_keep`, `mean_axis`, `sum_all`, `mean_all`; `reshape` (with a proof that the count does not change), `flatten`; `permute` (with `transpose_p`, the permutation that swaps two axes); `narrow` (with a proof that the window fits); `cat`; `bmm` (`[b, n, k] · [b, k, m]`), each with its backward function. `of_mat` and `to_mat` turn a `Mat<r, c>` into a `Tensor<[r, c]>` and back without a copy |

`erf(x)` is written in the package with F32 additions, multiplications and comparisons only (Base has no `erf`), so it
gives the same bits in C and JavaScript: a polynomial in x² below 0.84375, a correction polynomial around one of six
centres up to 3.92, and 1 from there on. Over all 2^32 inputs it is at most 0.93 ulp from float64's erf (the correctly
rounded F32 on 97.1% of them), NaN for NaN, `erf(-0) = -0`. `GELUErf` computes `(x * 0.5) * (1 + erf(x * 0.70710677))`
and the gradient `0.5 (1 + erf(x / √2)) + x · 0.3989423 · exp(-x²/2)`, both in PyTorch's order; the exponential is
`F32.exp`, as in SiLU. Not in `gpu.bend` yet.

```python
T.GELUErf(n, c)                         # Layer<Unit, Mat<n, c>, Mat<n, c>, Mat<n, c>>, as T.GELU(n, c)
T.erf(x)                                # F32 -> F32
```

A 784-128-10 MLP on MNIST and a two-layer Llama (grouped-query attention, RoPE, SwiGLU, RMSNorm) are built from these
pieces in [`demo/`](../demo/) and trained against PyTorch twins; see the repository's README.

## Fast layers (`fast.bend`)

[`fast.bend`](fast.bend) has `LinearF`, `LinearFirstF` and `LinearNBF`: `Linear`, `LinearFirst` and `LinearNB` with
their three products taken from `bend-ml-tensor-fork` instead of `bend-ml-tensor-array`, and their optimizer step in
parallel blocks; and `SiLU`, `ActIn`, `Mul2`, `Add2`, `Residual`, `Mat.dup` and `Mat.sum2` with their element-wise loops
in parallel blocks (tensor-fork's `par1`..`par4`). The types and the bits are `main.bend`'s: every number is computed by
`main.bend`'s own loop or tensor-array's own cells, only split into blocks that share their arrays instead of copying
them. Small matrices (below 131072 numbers, MNIST's and the tiny Llama's) run in one block. The demos use them.

```python
import ../torch/main.bend as T
import ../torch/fast.bend as F
F.LinearFirstF(100n, 784n, 128n, 4n)   # instead of T.LinearFirst(100n, 784n, 128n, 4n)
```

They are outside the proofs, and say so. bend-ml-tensor-fork shares arrays through Base's `Array.fork`, which Base marks
`@unsafe` (two handles to one array break the single owner the proofs rely on), so `bend fast.bend` names the defs
that reach it. No file with laws imports `fast.bend`, and `make laws` checks both that and the list of defs. What
connects them to the proved layers is a test, not a proof: [`demo/fast_test.bend`](../demo/fast_test.bend) runs each
fast piece and its proved twin on the same inputs (with -0, infinities, NaN and a subnormal mixed in), at MNIST's, the
tiny Llama's and larger sizes: the linear layers forward and backward, the optimizer step under SGD, momentum and AdamW,
SiLU, Mul2 and a Residual. Every output and gradient has the same bits, on 1 and 16 threads and in JavaScript. Seven
mutants of the fast code are caught by it (`make mutate`).

## GPU layers (`gpu.bend`)

[`gpu.bend`](gpu.bend) runs a whole training step as bangs (`!` calls), which run on the GPU in a binary built with CUDA
(`CUDA_HOME=... bend x.bend -o x`) and on the CPU threads otherwise: `LinearF`, `LinearFirstF` and `LinearNBF` with their
three products from bend-ml-tensor-gpu, and their optimizer step; SiLU, Mul2, Add2, Residual, `Mat.dup`, `Mat.sum2`,
RMSNorm, RoPE, Attention, Embedding and CrossEntropy. The names are `fast.bend`'s (which has `main.bend`'s RMSNorm,
RoPE, Attention, Embedding and CrossEntropy under the same names), so a program switches by its import line:

```python
import ../torch/main.bend as T
import ../torch/gpu.bend as F          # instead of ../torch/fast.bend
```

The layers other than the products are tensor-gpu's `run1`..`run4` (bangs over any leaf code) with leaves written here
in plain Bend, which compute every number with `main.bend`'s own expressions and row code. So the types and bits are
`main.bend`'s, except where CUDA's float `exp` is taken on the device: SiLU differs by up to 2 ulps in y and 4 in dX,
the cross-entropy by up to 4, attention by up to 176,128 on cells near zero (about 1e-7 of each output's largest
value), accepted because the cause is that `exp`: `make gpu` also builds every case with `exp_cr` (the host's
exponential in plain Bend, test-only) in place of `F32.exp`, and then each gives `main.bend`'s bits on the device. [`demo/gpu_test.bend`](../demo/gpu_test.bend) checks the linear layers and
[`demo/gpu_layers_test.bend`](../demo/gpu_layers_test.bend) the others, on the GPU and on 1 and 16 CPU threads (with
`+RTS --gpu off -RTS`, all of them exact), and `make gpu` trains MNIST, the tiny Llama and the SmolLM2-sized block on the GPU
against PyTorch and checks that MNIST's and the tiny Llama's outputs are `bench/bits.txt`'s. A mutant of the GPU layers
is caught by `gpu_test` (`make mutate`); tensor-gpu's own tests are bend-ml's. It is outside the proofs for the same
reason as `fast.bend` (`Array.fork`, inside tensor-gpu: this file writes no `@unsafe`), and `make laws` pins its defs.

The demos stay on `fast.bend`, because at these sizes the GPU is slower (RTX 4050 Laptop against 16 CPU threads, first
measurement, 2026-10-18): an MNIST epoch 7.8-8.4 s against 1.7-2.1 s, a tiny-Llama step 63-66 ms against 1.0-1.8 ms
(a SmolLM2-sized block's products never reached the GPU: its attention forks on the host first, and a bang after the
host's forks runs on the CPU, NOTES). Each product pays a bang's fixed cost
(about 1 ms) and the arrays move between host and device around it, since every other layer runs on the CPU; the
products alone are faster on the GPU only from SmolLM2's sizes up (bend-ml-tensor-gpu's README).


Measured (NOTES, 2026-10-17): MNIST's epoch 3.6 → 2.9 s on one thread with the fast products; a training step of one
decoder block at SmolLM2-135M's sizes 247 → 155 ms on 16 threads with everything above, one thread unchanged.

## Dtypes (`dtype.bend`)

BF16, F16 and int8 storage, and BF16 storage around F32 arithmetic for any model (plan M11). Everything is computed in
F32; the formats are bits in `U32` words, in plain Bend (no `@unsafe`, no foreign code).

- `BF16.of_f32`, `BF16.to_f32`, `F16.of_f32`, `F16.to_f32`: conversions with PyTorch's rounding (to nearest, ties to
  even), subnormals, infinities and signed zeros. NaN follows PyTorch 2.14.1's CPU kernels on this machine: every NaN
  becomes `0xFFFF` in BF16, and keeps its sign and the top of its payload, made quiet, in F16. `BF16.round(x)` is
  `x.to(torch.bfloat16).to(torch.float32)`.
- `HMat<r, c>`: a matrix packed two values a word, element `2k` in the low half of word `k` (the bytes of a torch
  tensor); `BF16.pack`/`unpack`, `F16.pack`/`unpack`.
- `Q8<r, c>`: int8 by rows for inference, `s_i = max_j |W[i, j]| / 127`, `q = round(W / s_i)` in `-127..127`, four a
  word as two's complement bytes; `Q8.quantize`, `Q8.dequantize` and `Q8.matmul_nt` (`y = s_i · Σ_j x[n, j] · q[i, j]`,
  reading the bytes from the words).
- `DW`: a two-F32 accumulator (`DW.add`, `DW.fma`; `Sum.dw`, `Dot.dw`), TwoSum and Dekker's product, for long sums.
- `CastOut(l)` rounds a layer's output to BF16 and the gradient arriving at it too; `CastP(l)` lets a one-matrix layer
  see its parameter in BF16 and rounds the gradient it accumulates, while the parameter itself stays F32 (the master
  weights the optimizer updates). Both keep the layer's types, so a model gets BF16 storage by wrapping its layers
  ([`demo/tiny_mixed.bend`](../demo/tiny_mixed.bend)). The values are BF16 values in F32 arrays: BF16's numbers, not
  yet its memory.

Tested by [`demo/dtype_test.bend`](../demo/dtype_test.bend) and [`reference/test_dtype.py`](../reference/test_dtype.py):
the conversions bit for bit against `tensor.to(torch.bfloat16)` and `torch.float16` over 1,347,584 F32 words (every
high half with the low halves that decide BF16's rounding, every F16 exponent with the bits that decide F16's, ties in
the subnormals, random words and named edge cases) and all 65,536 16-bit patterns both ways; packing, int8's words,
scales and dequantized weights bit for bit; int8's product within the shared tolerance.

## Run-time sizes and binary loading

`Seq.at` consumes two owned layer records and joins their corresponding closures once; `Train.step_at` consumes one
owned model for its forward, backward and optimizer step. Both accept dimensions captured at run time, with the same
shape checks as the template forms. The template forms and their existing laws remain in place; the new owned path
is tested. MNIST accepts an optional batch size (default 100), including a final partial batch:

```sh
.scratch/mnist 1 50 adam 0.001 37
```

Its reported training loss is the mean of the batch losses, including the final partial batch as one batch. Evaluation
covers all 10,000 examples. Batch sizes must be 1..60,000; a requested batch count cannot exceed the available batches.
`Pair3(A, B, C)` names `Two<Two<A, B>, C>`; `ResidualType(A)` names the paired type wrapped around a residual's
parameters or saved values, as used in the tiny Llama and the SmolLM2-sized block.

```python
import ../torch/load.bend as L
# P, X, Y, S and Model are the model's existing types and constructor.
# p : P <- L.weights(~P, ~X, ~Y, ~S, ~Model(), "weights.bin", zero_parameters)
# tr : T.Train<P> <- L.train(~P, ~X, ~Y, ~S, ~Model(), "checkpoint.bin", fresh_training_state)
# p : P <- L.safetensors(~P, ~X, ~Y, ~S, ~Model(), "model.safetensors", layout, zero_parameters)
#   layout : List<&2, L.Entry> = [L.Entry{"model.embed_tokens.weight", [49152, 576]}, ...]  (Layer.dump order)
```

`load.bend` reads an exact-size little-endian F32 file with `bend-ml-binary-read@0.2.0.0` (BendHub), checks the byte
count and fills the model through its array cursor. MNIST and the tiny Llama each load `data/<demo>/init.bin`, written
in `dump` order by their preparation scripts. The foreign C/JavaScript reader is a temporary loading-only exception: no
file with laws imports it, and `make laws` pins its definitions outside the proofs. It is replaced when Bend gains
native dense reads. Offsets and sizes are 64-bit (`L.W64`, two U32s), so a file may hold up to 2^32 - 1 numbers.

`L.safetensors` loads a Hugging Face `.safetensors` file: `layout` lists the model's tensors in `dump` order, each with
its name in the file and its shape. The header is parsed in plain Bend; each tensor must be in the file with that shape,
as F32 or BF16 (BF16 is widened exactly, as `bf16.to(torch.float32)`), and the shapes must add up to the model's
`Layer.count`; otherwise the program stops with a message naming the tensor. Other tensors in the file are ignored, so a
tied output head (SmolLM2's, which the file does not hold) is simply left out of the layout. `L.tensors(path, n,
layout)` gives the same numbers as one array.

`L.safetensors` reads every tensor into one array before `Layer.load` fills the model, and that array is as large as the
model: for SmolLM2-1.7B, a 2^31-cell block (8 GiB) whose memory the training step never reuses. The same can be done
piece by piece. `L.header(path)` parses the header once. `L.piece(m, path, h, layout, p)` fills one layer's parameters
from its own tensors, and the array they were read into is gone once it answers. `L.two(p, k1, k2)` fills a pair of
parameters one after the other. `demo/smol.bend`'s `load_st` uses them to fill the embedding, each block (`LStack`,
which `scripts/smol.sh` writes with the stack) and the final norm. `smol loss <model.safetensors>` gives the same loss
bits as `init.bin`, and `smol train <steps> <model.safetensors>` the same steps (`make smol`). On SmolLM2-1.7B the loss
from the real weights peaks at 10.1 GB instead of 18.1.

A checkpoint has no header; the caller supplies the same model and optimizer configuration on restore. Its layout is:

| Optimizer | F32 values in order |
|---|---|
| SGD | parameters only |
| Momentum | parameters, momentum buffer |
| Adam / AdamW | parameters, first moments, second moments, `1 - b1^t`, `1 - b2^t` |

`Train.save` writes this layout from one array filled by `store`, a chunk of 65,536 numbers per byte list (a list of
every byte would take several GB at SmolLM2's sizes); `Params.save(~model, path, p)` writes the parameters alone, in
dump order (`demo/smol.bend grads` writes gradients this way). Both use Base's `File.write_bytes`, not the foreign
reader. A file must stay under the reader's U32 limits to be read back: a 360M AdamW checkpoint is not. Adam/AdamW
checkpoints written before 2026-10-18 end in `b1^t`, `b2^t` and must be regenerated, as must old SGD/momentum
checkpoints containing unused buffers. Native loading preserves NaN
payloads in the copy test; JavaScript canonicalizes those payloads. Signed zeros, infinities and subnormals retain their
bits on both lanes; the existing numerical policy continues to match NaN with NaN.

## How to write a model

A model is a layer. Its types are written once, in small defs, and the layers are composed with templates:

```python
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# Linear(784, 128) -> ReLU on batches of 100 rows
def Hidden() -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, TA.Mat<100n, 784n>, TA.Mat<100n, 128n>, T.Two<TA.Mat<100n, 784n>, TA.Mat<100n, 128n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~Unit, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~T.Linear(100n, 784n, 128n, 3n), ~T.ReLU(100n, 128n))
```

`Seq` takes its seven types and its two layers. The types are explicit because Bend does almost no inference; they are
also the documentation of the model: `T.Two<T.Lin<784n, 128n>, Unit>` is the parameters of a linear layer next to a
ReLU's (none). Training is `T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, cfg, state, x, targets)`.

## The laws, in plain language

The laws are in [`LAWS.bend`](../LAWS.bend) at the repository's root, the proofs in [`PROOF.bend`](../PROOF.bend). The
five about linear maps (`seq_chain` and `fork_sum` through the package's own combinators) speak about
[`spec.bend`](spec.bend), a model of vectors and matrices as lists of `Nat`, where arithmetic is exact; `block_sum` speaks
about another model there, a sum as the tree of additions it makes. They say that the formulas of the backward pass are the right ones; whether the F32 code computes
them to F32's precision is tested against PyTorch, not proved.

| Law | What it says |
|---|---|
| `adjoint` | A linear layer computes `y = W·x` forward and `dx = Wᵀ·dy` backward. For every matrix `W` and vectors `dy` and `x`, `dy·(W·x) = (Wᵀ·dy)·x`: the backward pass is the *adjoint* of the forward pass. That equation is what makes `dx` the gradient of the loss with respect to `x` (if the loss changes by `dy·δy` when `y` moves by `δy`, it changes by `dx·δx` when `x` moves by `δx`). The proof goes row by row: each row `r` of `W` with its weight `dy[i]` gives `dy[i]·(r·x)` on one side and `(dy[i]·r)·x` on the other, and the sum of the weighted rows splits into the sum of their products with `x`. |
| `weight_adjoint` | The same for the weights: the weight gradient `dW = dy ⊗ x` (row `i` is `dy[i]·x`) satisfies `dy·(W·x) = <dW, W>`, the sum of the products of matching entries. So `dW` is the gradient with respect to `W`. |
| `adjoint_seq` | `Seq` runs the second layer's backward pass, then the first's. For two linear layers that is `W1ᵀ·(W2ᵀ·dy)`, and the law says it is the adjoint of the composed forward pass `W2·(W1·x)`: the chain rule, for linear maps. It follows from `adjoint` used twice. |
| `causal` | In `Attention`, row `i` of a head's probabilities is a softmax over exactly `i + 1` positions. `prow`, the function the attention runs for each row, writes the scores of query `i` from the start of its row and runs the softmax over as many positions as it wrote; the softmax's division loop counts the positions it divides, and the law says that count is `i + 1`. The probabilities start as zeros and nothing else writes row `i`, so its other `t - (i + 1)` positions stay 0: no probability lands on a later position. That last step is read from the code, not proved (the checker cannot follow writes into an array by index). The proof counts: each score written adds one, `qrow` gives `i + 1` scores, and each step of the division loop adds one (`cnt_div`); the loops before it (the maximum and the exponentials) are matched, not computed, so the count does not depend on any number. A softmax that ran over one position too few is now caught by this law (`make mutate`). |
| `causal_grad` | The same for the backward pass. Row `i` of a head's `G` (the gradient of its scores) is computed over exactly `i + 1` positions: `grow` reads the row of `dP` against values `0 .. i` only, writes it from the row's start and runs the softmax's backward pass over as many positions as it wrote. `G` starts as zeros, so its positions after `i` stay 0, and no gradient reaches a key or a value after the query's position. Before this law the backward pass computed the whole row and relied on the probabilities being 0 there; now it computes half the row and the restriction is a law (the same bits, and the tiny Llama about 2% faster). |
| `causal_keys` | `Attention` reads keys in one place, `qrow`, and the scores it builds for the query at position `i` come from keys `0 .. i`, one each: exactly `i + 1` of them. So no key after position `i` is even read. The proof follows the loop that reads them: each step reads one key and adds one score, so a loop started at `i` gives `i + 1` scores (an induction, with `1 + (a + b) = a + (1 + b)`). A version that read one key too many is caught by this law, by `causal` (the extra score would reach the softmax) and by the numbers (`make mutate`). |
| `kv_of_group` | `kv_of(hi, g)` is the one function both passes of `Attention` call to pick the key/value head of query head `hi`. For every group `j`, the heads `j·g + 0 .. j·g + g - 1` all get `j`. It is `gqa_head` (below) said about the package's own function, so a change to how the code picks the head breaks a law, not only a test. |
| `kv_cols` | The columns each query head reads come from one function, `head_cols`, in both passes: the head loops give the head's work only the columns, never the head's index, so it cannot read other columns. With `g = 1 + i + k` query heads per key/value head, query head `j·g + i` reads the keys and values of head `j`, from column `j·hd`. The proof is `kv_of_group` carried through the column arithmetic. A change that gives a head another key/value head's columns is caught by this law. |
| `attn_par_fwd` | `Attention` runs its sequences in a tree of parallel lets: `fpar` cuts the sequences in halves, each half again, `d` levels deep, and appends the leaves' outputs. Whatever the depth and however the sequences are split, the tree gives exactly what `fleaf` gives running them one after another, in order. So the parallel forward pass is the sequential one, and its output cannot depend on `d` or on the number of threads. The proof is an induction on the depth: a pair of leaves is the loop twice; the loop is "apply the head work to each sequence"; the map of two lists joined is the two maps joined; and `cut`'s halves joined again are the list it cut. The lists hold arrays, which Bend lets a proof use only once, so each fact hands its list back to the next one. |
| `attn_par_bwd` | The same for the backward pass: `bpar`'s tree gives each sequence's `dQ`, `dK` and `dV` exactly as `bleaf`, one sequence after another. |
| `seq_chain` | `Seq`'s own code, run on two linear layers of the exact model (`spec.bend`'s `LinL`, a `Layer` like any other), gives a backward pass that is the adjoint of its forward pass: `dy·y = dx·x`, for the `y` its forward pass returns and the `dx` its backward pass returns from the state the forward pass saved. `adjoint_seq` says this of the formulas; this law says it of the combinator: `Seq` hands the second layer's input gradient to the first, the chain rule. Every step of the combinator reduces, so the proof is `adjoint_seq`. |
| `fork_sum` | `Fork`'s own code, giving one input to two linear layers, returns as input gradient the sum of the two layers' input gradients, and that sum is the adjoint of the pair of outputs: `dy1·y1 + dy2·y2 = dx·x`. A residual block and attention's q, k, v projections are `Fork`s. The proof: the dot of a sum is the sum of the dots, and each is its layer's `adjoint`. A `Fork` that dropped one gradient is caught by it. |
| `bc_sym` | Broadcasting does not depend on the order of the operands: shapes `a` and `b` broadcast to `o` exactly when `b` and `a` do, so `x + y` and `y + x` accept the same shapes. The rule on one axis ("the sizes are equal, or one of them is 1") is written as "one way round, or the other", so swapping the operands only swaps the two halves of an "or"; the proof goes axis by axis. |
| `sum_numel` | Summing over axis `k` partitions the numbers: a tensor of shape `s` holds exactly `size(k)` times as many numbers as its sum over `k`, for every shape and every axis. The proof goes axis by axis and moves `size(k)` to the front with the rules `a·(b·c) = (a·b)·c` and `a·b = b·a`. |
| `narrow_rest` | Narrowing axis `d` (or giving it any size) changes only that axis: with axis `d` taken out, the shape is what it was. |
| `narrow_axis` | Narrowing axis `d` of a tensor that has it gives that axis exactly the length asked for. |
| `cat_numel` | Joining two tensors along an axis they have adds their counts: an axis of `x + y` holds as many numbers as one of `x` and one of `y` together. The proof is `(x + y)·n = x·n + y·n`, carried past the axes before it. |
| `mat_numel` | A `Tensor<[r, c]>` holds the `r·c` numbers of a `Mat<r, c>`, so `of_mat` and `to_mat` can reuse the array as it is. |
| `permute_rank` | Permuting the axes gives one axis per entry of the permutation, so a permutation keeps the rank. |
| `gqa_head` | With `g` query heads per key/value head, query head `j·g + i` (`i < g`) reads key/value head `(j·g + i) / g = j`. The forward pass picks key/value head `hi / g` for query head `hi`; the backward pass adds the gradients of heads `j·g + 0 .. j·g + g - 1` into key/value head `j`. The law says these are the same groups. The proof is llama.bend's: `j·g` is `j` whole blocks of `g`, and `i < g` more numbers do not complete another one. |
| `block_sum` | `tree_sum` adds a row in passes: neighbours in pairs, an odd last cell carried up as it is, until one number is left. The GPU's loss kernels split the row into blocks of `2^p` cells instead: one leaf sums each block as a balanced tree of pairs, then the host runs `tree_sum`'s remaining passes over the block sums. When `2^p` divides the row, both make exactly the same additions: the same pairs, in the same order. `spec.bend` writes a sum as the tree of its additions over the cells' positions (`(0+1)+(2+3)` rather than a number), so the law holds for F32 as for any arithmetic, with nothing assumed about rounding. The proof: one pass over `2k` blocks of `2^p` cells pairs block `2j` with block `2j+1`, which is a block of `2^(p+1)` cells as the balanced tree builds it (an induction on `k`); so `p` passes over a row of `2^p·k` cells leave its `k` blocks (an induction on `p`, with `2^(p+1)·k = 2^p·2k`), and the passes after that are the same on both sides. It is why the mutant that shrinks the cross-entropy's GPU blocks to one cell survives: one cell divides every row. That the GPU code builds these trees is tested (the GPU layers give `main.bend`'s bits), not proved. |
| `unpair_pair` (in [`dtype.bend`](dtype.bend), checked by `bend torch/dtype.bend` and `--verdict`) | Packing 16-bit values two to a word and reading back as many values as there were gives the same list, in the same order, for any length: an odd last value is paired with a 0 that the read leaves out. The proof takes two values at a time: the pair `HW{a, b}` unpacks to `a` then `b`, and the rest of the list is the induction hypothesis. That the bit operations which put a pair into a word and take it out are exact (`half_join`, `half_lo`, `half_hi`) is tested over every 16-bit value, not proved. |

`add4`, `mul_dist_l`, `dot_vadd`, `dot_scale`, `dot_comm`, `len_rcat`, `len_drow`, `sub0`, `cnt_div`, `cnt_psm3`,
`cnt_psm2`, `cnt_pw`, `cnt_drow`, `cnt_gw`, `cnt_gdrow`, `div_block`, `div_blocks`, `div_rest`, `or_comm`, `and_comm`,
`and_cong`, `or_cong`, `bc_dim_sym`, `bc_eq_sym`, `bc_rank_sym` and `mul_swap` are the lemmas the proofs use; `l1`,
`map_app`, `go_app`, `cg`, `tree` and their `b_` twins are the trees' (each returns the list it was given, with the
fact).

The tensor operations' shape conditions are `Bool` functions of the shapes, and each operation asks for the proof that
its condition is `True`: `bc_ok` for broadcasting, `narrow_ok`, `cat_ok`, `is_perm`, and for `reshape` the equality of
the two counts. For literal shapes the checker computes the condition, so a wrong broadcast, window or join does not
compile ([`tests/bad_broadcast.bend`](../tests/bad_broadcast.bend), [`bad_narrow`](../tests/bad_narrow.bend),
[`bad_cat`](../tests/bad_cat.bend); [`bad_bmm`](../tests/bad_bmm.bend) for the product, whose sizes are in its type).
For shapes read at run time, a program checks the condition once where they enter (`Bool.check`, `Nat.check_eq`), as
[`demo/tensor_test.bend`](../demo/tensor_test.bend) does.

Shape facts that a layer needs are arguments of its constructor, checked when the program is: `Attention` asks for proofs
that the rows hold `kv·g·head` numbers and that the rows are `b·t` positions, `RoPE` that a row is `h` heads of
`2·half`. For literal sizes the proof is `{==}`; a wrong size fails it ([`tests/bad_attention.bend`](../tests/bad_attention.bend)).

What is **not** proved:

- **The numbers.** `F32` is not a real number. Every layer is tested against `torch.autograd` and every optimizer against
  `torch.optim`, by the scripts in [`reference/`](../reference/).
- **That the F32 code implements the Nat model.** `adjoint` is about `spec.bend`'s `mv` and `mtv`; the package's `Linear`
  uses bend-ml's array products, which compute the same sums in another order. The connection is tested, not proved.
- **The gradients of the non-linear layers** (softmax, the norms, the activations, attention): their formulas are
  checked by the tests, not by a law.
- **The parallel trees that share arrays**: bend-ml-tensor-fork's products and blocks and `fast.bend`'s layers. In
  Base's model `Array.fork` returns two copies and `Array.join` keeps the first, so a law about the shared tree cannot be
  stated; they are tested bit for bit against the proved code. The same holds for tensor-array's parallel product: each
  block reads its own packed copy of its rows, and a law would need the contents of arrays written by index, which the
  checker cannot follow. Attention's trees hold no shared arrays, and those are laws (`attn_par_fwd`, `attn_par_bwd`).
- **The tensor operations' numbers and index maps**: every operation and its gradient is tested against
  `torch.autograd` on random shapes (`reference/test_tensor.py`), not proved; that a general permutation keeps the
  count of numbers is tested too (the law covers the rank).
- **The compiler and the runtime.** The kernel checks the translated Bend terms; it does not check Bend's translation to
  C, clang or the hardware.
