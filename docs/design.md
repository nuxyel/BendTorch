# Design

How BendTorch is built and why: each decision, the alternatives I weighed and the evidence behind the choice. The
measurements are in [`NOTES.md`](../NOTES.md); the milestones in [`plan.md`](plan.md).

## The goal

Train real networks in Bend 2 with general pieces (layers, losses, optimizers, a training step that does not know the
model), keep the guarantees Bend can give (shapes in the types, laws checked by the kernel), and check every number
against PyTorch. Speed matters, but is measured, never assumed.

## Tensors

**A tensor is `bend-ml-tensor-array`'s `Mat<r, c>`**: `r·c` F32 numbers in one flat `Array<F32>`, row by row, with `r`
and `c` erased type parameters. In the compiled program an `Array` is one block of memory and a read is an index mask
and a load (bend-ml NOTES, 2026-10-13), so a `Mat` is a contiguous buffer with its shape in the type.

- **Higher ranks are folded into rows and columns.** A batch of `b` sequences of `t` positions is `b·t` rows; `h` heads
  of `d` numbers are `h·d` columns. The layers that need the structure (RoPE, attention) take the sizes as arguments,
  with proofs that the folded sizes match (`n == b·t`, `w == kv·g·d`). This covers an MLP and a Llama; it does not give a
  general rank-N API.
- **Why not a dynamic tensor** (shape as a run-time value, checked at each operation)? It is the usual design and works in
  Bend, but it moves every shape error from the checker to run time, which gives up what Bend adds.
- **General tensors: `Tensor<s>`** (`torch/tensor.bend`, plan M7, 2026-10-16). The shape is a list of sizes in the type,
  any rank. A prototype measured the costs that were open: the checker takes 0.15 s for the whole module and for a
  program with literal rank-3 shapes, whose broadcast proof it computes; a type such as `Tensor<remove(s, k)>` reduces
  to a literal shape on its own; the laws about shapes are short inductions over the list (`--verdict` 0.5 s for every
  proof of the repository). **Decision: `Tensor<s>` is the layer of general operations, next to `Mat`, not a
  replacement.** `Tensor<[r, c]>` and `Mat<r, c>` are the same array under two types (`of_mat`/`to_mat`, no copy; law
  `mat_numel`), the layers stay on `Mat` and bend-ml's fast products, and M6 keeps working on `Mat`. Porting the layers
  would change nothing they compute and lose nothing to keep.
- **One kernel for many operations.** Every tensor operation is a gather (`out[j] = f(x[m(j)])`) or a scatter
  (`g[m(j)] += x[j]`) over an index map: for each position, the position to read or write, from per-axis sizes and
  strides (stride 0 on a broadcast axis), or "the position with one axis taken out" for reductions. The gradient of a
  broadcast is a scatter over the same map as its forward gather; the gradient of a reduction is a gather.
- **Shape conditions as Bool functions.** Broadcasting (`bc_ok`), a window (`narrow_ok`), a join (`cat_ok`), a
  permutation (`is_perm`) are functions of the shapes; an operation asks for the proof that its condition is `True`,
  which the checker computes for literal shapes and a program checks once (`Bool.check`) where shapes read at run time
  enter. `reshape` asks for the equality of the two counts.
- **Storage.** One owner per array: every operation hands back what it reads (`MMul{a, b, c}` returns both operands with
  the product). No views or strides: a reshape that keeps the row-major order is free in principle (the same array under
  another type, with a `product` proof, as bend-ml's `Mat.reshape`), a transpose is a copy. `Array.clone` is the only
  copy, and the code calls it only where a value is needed twice (the output an activation saves, the input `Fork` gives
  to two layers).
- **Rows over arrays.** The row-wise layers (attention, RMSNorm, LayerNorm, softmax, log-softmax, RoPE, the embedding,
  cross-entropy, MSE and NLL) work on the matrix's own array a row at a time, by index, in loops that keep the order of operations of
  the list versions they replaced (M6), so the bits did not move. A backward pass keeps the input matrix (the norms) or a
  copy of the output (softmax, log-softmax). Lists remain for ids, labels and the scores of one query (`qrow`), the
  things the laws count. The first version turned each matrix into a list of rows; taking those shared lists apart was
  two thirds of a tiny-Llama step.
- **Dtypes.** F32 arithmetic. Bend has `U32` and `F32` words and no F64, so `torch/dtype.bend` (M11, 2026-10-18) packs
  BF16 and F16 two to a `U32` with PyTorch's rounding (bit for bit against `tensor.to(torch.bfloat16)`), stores int8 by
  rows with a scale for inference, offers BF16 storage with F32 compute and master weights (mixed precision), and a
  two-F32 accumulator for long sums (NOTES, M11).

## Autograd: typed forward/backward pairs

```python
type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> OptState<P> -> Step<P>,
        zeros: Unit -> P, dump: P -> PL<P>, load: P -> Array<F32> -> U32 -> PA<P>, store: P -> Array<F32> -> U32 -> PA<P>,
        count: Unit -> U32}
```

(As in `torch/main.bend` since M9: an optimizer state per family, `OptState<P>`; weights loaded from and stored into one
flat `Array<F32>` at an offset; `count` a closure, so a composite does not build its children's records to count them.)

The alternatives:

| | What it is | In Bend 2 |
|---|---|---|
| **Typed pairs** (chosen) | each layer has a forward that returns its saved values `S` and a backward from `dY` to `dX` and `dP`; composition is code | The types of `dX` and `dP` are those of `X` and `P`, checked. What is saved is a visible record. No run-time graph. Needs the composition combinators. |
| Dynamic tape | the forward pass records operations on a list of nodes, the backward pass walks it | Nodes of different shapes need a dynamic tensor type, so shapes are checked at run time. A tape is first-order data, which makes laws about the sweep easy to state. |
| Pullback closures | each operation returns its result and a closure for its backward pass | Closures are affine: a value used twice needs an explicit copy, and a law cannot look inside a closure. |
| Forward mode (dual numbers) | each value carries its derivative along one direction | One forward pass per parameter: unusable for training. Good as an independent check. |

Typed pairs were the plan since 2026-10-05, and a first prototype of a linear layer showed a forward and backward pass
with no clone of the weights, the input or the gradient, matching PyTorch. What made them
general here is Bend's **templates**: a `Layer` record passed as `~l` is substituted at compile time, so
`Seq(~la, ~lb)` can call `la`'s and `lb`'s functions in both passes although each closure may run only once.

- **Composition.** `Seq` (pairs the parameters, saved values and optimizer state of its two layers; its backward runs
  the second layer's first), `Fork` (copies the input to two layers and sums their input gradients), `Both` (two layers
  on a pair), `Residual` (`Seq(Fork(Id, f), Add2)`). A Llama block is written with these, nothing model-specific.
- **Accumulation.** A backward pass adds into the gradient it receives (`P` typed), so repeated uses and two backward
  passes add up, as PyTorch's `.grad`; the optimizer step zeroes it.
- **What it does not do yet.** No graph built at run time, so control flow that depends on values is written in Bend
  around the layers, not recorded; no higher-order derivatives; no gradient checkpointing (a layer that recomputes its
  forward in its backward is possible, attention does it for its probabilities).

## Neural network API and training

- Layers: `Linear`, `LinearNB`, `Embedding`, `RMSNorm`, `LayerNorm`, the activations, `Softmax`, `LogSoftmax`, `RoPE`,
  `Attention`; losses `CrossEntropy`, `NLL`, `MSE`; optimizers SGD, momentum, Adam, AdamW, in PyTorch's order of
  operations.
- `Train.step(~model, ~loss, cfg, state, x, targets)` is the whole training step for any layer and any loss. The state
  holds the parameters, their gradient and `OptState<P>`: no buffers for SGD, one for momentum, two for Adam/AdamW;
  `1 - b1^t`, `1 - b2^t` (carried as `q + c (1 - q)`, never as a power) retain Adam's bias corrections. `Train.new` receives the configuration to allocate that state.
- **A generic walk over the parameters.** `dump` gives every number of a layer's parameters in a fixed order and `load`
  overwrites them from one `Array<F32>` at an element offset; the combinators thread the array and next offset.
  `Layer.count` gives the serialized size, so `load.bend` checks the file length before loading. `Train.save` writes
  only the optimizer buffers in use (plus Adam's 1 - b1^t and 1 - b2^t); `load.bend`'s `train` restores them, with identical resumed bits.
- **Run-time dimensions.** `Seq.at` composes owned records by distributing each child closure to its matching composite
  closure; `Train.step_at` consumes a model once across the three passes. MNIST uses both for its command-line batch
  size and its last partial batch. The shape checks still apply, including a rejected run-time-size composition.
  The existing laws refer to the template combinators; the owned path is numerically tested.
- **A loading-only module.** `torch/load.bend` imports the temporary foreign binary reader; the main package and every
  law stay independent of it. Files are raw little-endian F32 in dump order. SGD checkpoints contain parameters only,
  momentum adds one buffer, Adam/AdamW add two and their 1 - b^t. Array writing remains an M10 task.
- **State that is not learned** lives in `P` too when it must survive a step: `Dropout`'s step counter is its parameter,
  incremented by the optimizer step and kept by checkpoints, so the mask is reproducible.
- **Shared parameters.** `Tie(first, mid, last)` runs `first` and `last` with one parameter and adds both gradients into
  one accumulator (tied embeddings).
- **Work nobody reads is skipped**: `LinearFirst` does not compute the gradient of a model's input.

## Verification

### What is proved, tested and trusted

| Claim | Kind | Where |
|---|---|---|
| Every operation gets operands of the right shapes; every gradient has the shape of its value | **types**, checked | every program; rejected examples in `tests/` |
| Attention's and RoPE's folded sizes are consistent (`n == b·t`, `w == kv·g·d`, `w == h·2·half`) | **proof at construction**, erased | `Attention`, `RoPE` |
| A linear layer's backward pass is the adjoint of its forward, for the input and the weights; two linear layers compose by the chain rule | **law** over the exact `Nat` model | `LAWS.bend`: `adjoint`, `weight_adjoint`, `adjoint_seq` |
| Attention gives probability 0 to later positions: the softmax of row i runs over exactly i + 1 positions, the count its own division loop reports (that the rest of the row keeps its zeros is read from the code, not proved) | **law** about the package's own `prow` and `psmc` | `causal` |
| Attention reads only keys 0..i for the query at position i (the loop that reads them builds i + 1 scores) | **law** about the package's own `qrow` | `causal_keys` |
| The backward pass sends no gradient to a key or value after the query's position: row i of G is computed over i + 1 positions | **law** about the package's own `grow` | `causal_grad` |
| Query head `j·g + i` belongs to key/value head `j`, through the function both passes of attention call and the columns both read | **law** about the package's `kv_of`, `head_cols` (and `Nat.div`) | `kv_of_group`, `gqa_head`, `kv_cols` |
| Attention's tree of sequences gives the sequential result at every depth and split | **law** about the package's `fpar`/`bpar` against `fleaf`/`bleaf` | `attn_par_fwd`, `attn_par_bwd` |
| `Seq` and `Fork` compose backward passes correctly (chain rule, sum of input gradients) | **law** about the combinators' own code, on the `Nat` model's linear layers | `seq_chain`, `fork_sum` |
| Each layer's output and gradients equal PyTorch's within the tolerance | **tested**, 10 seeds, with IEEE edge cases for the activations | `reference/test_ops.py` |
| The optimizers equal `torch.optim` step by step | **tested** | `reference/test_mlp.py` |
| An Adam configuration's `c1`, `c2` agree with its `b1`, `b2` (`OptCfg.ok`); a program stops on one that does not (`OptCfg.check`), and without the check its state starts at NaN | **tested** at run time (F32 is opaque to the checker, so it cannot be a law): 5,486 configurations, 5 mutants | `reference/test_infra.py` |
| MNIST and a two-layer Llama train like their PyTorch twins | **tested** | `test_mnist.py`, `test_tiny.py` |
| The F32 code computes the formulas of the `Nat` model | **tested only** | the same tests |
| `torch/fast.bend`'s layers (over bend-ml-tensor-fork, `@unsafe` `Array.fork`) give `main.bend`'s bits: the linear layers forward and backward, the optimizer step (SGD, momentum, AdamW) and the element-wise layers (SiLU, Mul2, Residual's copies and sums) | **tested**, bit for bit, at MNIST's, the tiny Llama's and splitting sizes, 1 and 16 threads and JavaScript; 7 mutants | `demo/fast_test.bend` (12 cases); `make laws` pins its defs outside the proofs |
| `torch/gpu.bend`'s linear layers (over bend-ml-tensor-gpu, one `!` call a product, `@unsafe` `Array.fork`) give `main.bend`'s bits | **tested**, bit for bit, forward and backward at MNIST's, the tiny Llama's and SmolLM2-135M's sizes, on the GPU and 1 and 16 CPU threads; MNIST and the tiny Llama trained on the GPU give `bench/bits.txt`'s outputs; 1 mutant (tensor-gpu's code is tested in bend-ml) | `demo/gpu_test.bend` (8 cases), `make gpu`; `make laws` pins its defs outside the proofs |
| `torch/gpu.bend`'s other layers and optimizer step (leaves in plain Bend over tensor-gpu's `run1`..`run4`) give `main.bend`'s bits | **tested** at SmolLM2-135M's sizes: exact with `+RTS --gpu off -RTS` on 1 and 16 threads; on the GPU exact except SiLU (up to 4 ulps, CUDA's float `exp`; accepted, debt in NOTES); the block trained on the GPU against PyTorch; 5 mutants | `demo/gpu_layers_test.bend` (9 cases), `make gpu` |
| The non-linear layers' gradient formulas | **tested only** | `test_ops.py` |
| Bend's checker and kernel, the translation to C, clang, the CPU | **trusted** | |

### Laws

The laws are in `LAWS.bend` and their proofs in `PROOF.bend`, at the root, the convention of the Bend guide: the law is
the specification a reader checks, the proof is the code that discharges it. `make laws` runs the checker, then
`--verdict` (Bend's kernel proved in Lean), and requires every program in `tests/` to be rejected with its expected error.

The linear laws are about `torch/spec.bend`, where vectors are lists of `Nat`: exact arithmetic, so the statement is the
algebra of the formulas (`dy·(W·x) == (Wᵀ·dy)·x`), with no rounding to reason about. F32 is opaque to the checker
(`F32.mul(3.0, 3.0)` does not reduce), so a law about F32 values can only treat the operations as uninterpreted; no
IEEE 754 property is claimed anywhere.

`scripts/mutate.py` changes the code and the specification on purpose (57 mutants) and checks that a gate fails. A
mutant that survives a gate is listed with the reason (today one: a fault the numbers miss by accident and a bounds
check catches; since M8 the laws catch the two that used to slip past them, `kv_of` bypassed and a softmax over one
position too few).

## Numerical validation

- **Independent oracles.** The Python side (`reference/`) builds every model with PyTorch's own functions
  (`F.layer_norm`, `F.gelu(approximate="tanh")`, `torch.optim.AdamW`, ...); it never calls BendTorch, and the Bend
  programs never see a PyTorch result. Inputs and initial weights are written to files that both read.
- **Exact transport.** The Bend programs print each F32 as the decimal of its 32 bits, so nothing is lost between the
  two sides; the tests also count how many numbers are bit-identical.
- **One comparison policy for every test** (`reference/oracle.py`): `|bend - torch| <= 1e-6 + 1e-5·|torch|` element by
  element in float64; NaN matches NaN and an infinity matches the same infinity. On top of it, an **exact tier**: results
  that are a copy, a selection or one or two F32 operations in PyTorch's order (relu forward and backward, an embedding
  lookup, the gradients of MSE and NLL) must be the same bits, NaN included. Results whose order of additions or whose
  `exp`/`tanh` implementation differs from PyTorch's stay in the tolerance even when they happen to match today. The tolerance is never widened to make a
  test pass: when a comparison failed, the cause was found (relu's gradient at NaN) and the code fixed.
- **Trajectories.** Training is compared step by step for as long as the two runs are expected to stay together (50
  batches of MNIST, 20 steps of the Llama). Longer runs drift for a traced reason (NOTES.md, milestone 3); they are
  compared by statistics, and the drift is reported, not hidden.

## Execution and backends

- **Deterministic parallelism (Renan's call, 2026-10-16; not permanent).** Splitting work across threads never changes a
  bit of a result: blocks split rows, columns or heads, never the order of a sum, and the shape of every fork tree comes
  from the code and the sizes, not from the thread count. `make bits` (1 against 16 threads) and bend-ml's
  `tests/parseq.bend` (every split depth against depth 0) check it. A kernel that would need another order of sums to
  scale (split-k by thread count, reductions shaped by the core count, atomic adds on the GPU) is a decision for Renan,
  kernel by kernel, with its check moved from bits to the tolerance; a fixed split, the same at any thread count, keeps
  the rule.

- **CPU first.** Native code from Bend 2.0.39, clang. Products run in `2^par` parallel blocks (`par` is an argument of
  each linear layer; 8, 256 blocks, at SmolLM2's sizes, where 16 left the E-cores' blocks behind). In `torch/fast.bend`
  the optimizer step and the element-wise layers run in blocks of 16384 numbers from 131072 up, and the products pack
  their operands in blocks from 16384 cells up (bend-ml-tensor-fork); smaller work runs in one block, which is what the
  tiny Llama and MNIST get. Attention runs its sequences in parallel from 8 sequences. The norms, RoPE, softmax and
  the losses are sequential: RMSNorm and RoPE in blocks of rows gave the same bits and no measurable gain on a
  SmolLM2-sized block (NOTES, 2026-10-17).
- **GPU.** `torch/gpu.bend` (M12, 2026-10-18) runs every layer of a training step on CUDA through bend-ml-tensor-gpu's
  bangs, with plain-Bend leaves and no `@unsafe` of its own: the same bits as `main.bend` except CUDA's float `exp` in
  SiLU and softmax. The kernels sit behind the layer functions, so the GPU changed the kernels, not the protocol. It is
  slower than 16 CPU threads at the sizes trained here (`docs/limitations.md`), so the demos stay on `fast.bend`.
- **JavaScript.** `bend -o x.js` builds a program that runs sequentially under Bun (its file reads use `bun:ffi`, so not
  node). `make check-js` runs the test programs that way against the same oracles and compares the outputs of the
  training and generation programs with the native build's: the same bits (NOTES, 2026-10-16).

## Extensibility

A new layer is four functions and a constructor that builds the `Layer` record; a new optimizer is a constructor of
`Opt` and its per-element loop; a new loss is a function `Y -> T -> LossOut<Y>`. The tests follow one pattern: a section
in a Bend program that prints the results, and the same computation in PyTorch.
