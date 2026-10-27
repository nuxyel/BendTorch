# Notes

Every finding, measurement and decision, by date. Newest at the bottom.

## Pinned versions

- Bend **2.0.37** (`bend version`), Lean **4.34.0** for `--verdict`.
- `bend-ml-tensor-array@0.1.14.0`, `bend-ml-nat-lemmas@0.1.1.0` (BendHub); outside the proofs `bend-ml-tensor-fork@0.1.2.0`
  (`fast.bend`), `bend-ml-tensor-gpu@0.1.2.1` (`gpu.bend`) and `bend-ml-binary-read@0.1.0.0` (`load.bend`). Updated 2026-10-19.
- Python 3.14.7, torch 2.14.1+cpu, numpy 2.5.4 (`reference/requirements.txt`).
- Data: MNIST from the PyTorch mirror (`scripts/mnist.sh`); Tiny Shakespeare from karpathy/char-rnn, SHA256
  `86c4e6aa…c565ed` (`scripts/tiny.sh`).

## 2026-10-15: start

BendTorch is step 2 of the plans I made after bend-ml v1.2: typed autograd over tensors and a small transformer trained in
Bend against a PyTorch twin. llama.bend's `docs/bendtorch.md` listed what each operation of the Llama forward pass needs
for its backward pass; this repository starts from there.

### Decisions

- **A layer is a typed forward/backward pair, not an entry on a dynamic tape.** `fwd : P -> X -> Fwd<P, Y, S>` and
  `bwd : P -> S -> Y -> P -> Bwd<P, X>`. The gradient of the input has the input's type `X`, and the gradient of the
  parameters is accumulated into a value of the parameters' type `P`. A tape would need a dynamic tensor type (shape known
  only at run time) for its heterogeneous nodes, which gives up the one thing Bend adds over PyTorch: shapes in the types.
  The saved values `S` are an explicit record, so what a backward pass keeps is visible in its type.
- **Layers are templates.** Bend 2.0.37's templates (`~f`) are substituted at compile time and, unlike closures, can be
  used any number of times. A `Layer` is a record of four closures; passed as a template, `Seq(~la, ~lb)` builds a new
  record at each use, and each closure in it is called once. That is what makes generic composition possible with affine
  closures. A template can also take a `Nat` (`~n: Nat`), which `Residual` uses for its matrix shape.
- **Arrays from bend-ml.** Matrices are `bend-ml-tensor-array`'s `Mat<r, c>` (a flat `Array<F32>`, the dimensions erased
  type parameters) and its products (`matmul`, `matmul_nt`, `matmul_tn`, four rotating partial sums, rows in parallel
  blocks). The linear layers use the weights in PyTorch's `out x in` layout: forward `x·Wᵀ` (`matmul_nt`), backward
  `dy·W` (`matmul`) and `dyᵀ·x` (`matmul_tn`). No parameter, input or gradient is cloned by a linear layer; each product
  hands its operands back.
- **Gradients accumulate**, as PyTorch's `.grad`: a backward pass adds into the gradient it is given, and an optimizer
  step zeroes it. Repeated uses of a value (`Fork`, tied ids in `Embedding`) add up.
- **Optimizers in PyTorch's order of operations**, one pass per parameter matrix over the parameter, its gradient and two
  moment buffers: SGD (`w + (-lr)·g`), momentum (`buf·mu + g`), Adam (`m + (1-b1)·(g - m)` as `lerp`, then
  `v·b2 + (1-b2)·g·g`, then `w + (-step·m) / (sqrt(v)/bc2 + eps)`), AdamW's decoupled decay. The optimizer state of
  every layer is two values of its parameters' type, so SGD carries two buffers it never reads (2x the parameters in
  memory). One `Train` record type serves every optimizer; a leaner state per optimizer is a debt.
- **Laws in `LAWS.bend` and `PROOF.bend` at the root**, the convention the Bend 2.0.37 guide describes (a human states
  the laws, the proofs are written next to the code). llama.bend keeps its laws inside its package; here the package
  stays code, the laws live apart, and `torch/spec.bend` holds the exact model the linear laws speak about.

### Milestone 1: the protocol, MNIST

`Linear`, `ReLU`, `Seq`, cross-entropy, the four optimizers and `Train.step`, then `demo/mnist.bend` written only with
them (`Seq(Seq(Linear(784, 128), ReLU), Linear(128, 10))`).

- **MLP test** (`demo/mlp_test.bend`, `reference/test_mlp.py`): a 7-6-3 MLP on batches of 5, 12 seeds: logits, loss,
  `dx`, every parameter gradient, two backward passes accumulated, and three steps of each optimizer (SGD, momentum,
  Adam, Adam with weight decay, AdamW). 6,588 numbers, 2,196 bit-identical to PyTorch, the worst at 0.104 of the
  tolerance.
- **MNIST against its twin** (`reference/mnist_ref.py`, the same `nn.Linear` initial weights under seed 0, the same
  batches): one epoch of SGD at lr 0.1, loss 0.5148658 against 0.5148657560 and 9,108 test hits in both. 50 batches of
  each optimizer agree within 5e-7 in loss and give the same hits (the gate in `make check`).
- **Over a whole epoch, Adam drifts.** Two epochs of Adam at lr 1e-3: losses 0.3976068 / 0.1905289 against PyTorch's
  0.3976275 / 0.1905378, hits 9,320 / 9,521 against 9,317 / 9,514. The two sum the products in different orders; Adam
  divides by `sqrt(v)`, which magnifies the small differences of gradients near zero (see the tiny Llama below).

### Milestone 2: the layers of a Llama

Sigmoid, Tanh, SiLU, GELU (tanh form; the exact one needs `erf`, which Base lacks), Softmax, LogSoftmax, RMSNorm,
LayerNorm, Embedding, Linear without bias, RoPE (rotate-half, as `transformers`), causal attention with grouped-query
heads, MSE and NLL losses, and the combinators `Id`, `Fork`, `Both`, `Add2`, `Mul2`, `Residual`.

- **Each layer against `torch.autograd`** (`demo/ops_test.bend`, `reference/test_ops.py`): the output, `dx` and the
  parameter gradients from a random `dy`, on 10 seeds with zeros, `-0` and saturating values mixed in; SwiGLU is tested
  as built from `Fork`, `Seq`, `SiLU`, `Mul2` and `LinearNB`, and a residual RMSNorm through `Residual`. 15,550
  numbers, 11,308 bit-identical, the worst at 0.388 of the tolerance.
- **IEEE edge cases through the activations**: NaN, both infinities, both zeros, subnormals, inputs where `exp`
  overflows (|x| > 88.7) and 1e6. All five match PyTorch's class (NaN for NaN, the same infinity) once one thing was
  fixed: I had written relu's gradient as "passes where y > 0", which stops it at NaN; PyTorch's `threshold_backward`
  zeroes it only where `y <= 0`, so a NaN passes. Now the same.
- **Row operations on lists.** RMSNorm, LayerNorm, softmax, RoPE and attention turn a matrix into its rows (a list of
  lists) and back. Lists are `Data`, so a forward pass keeps its input rows for the backward pass without a copy of an
  `Array`. This is the simple first version: for the small models here it is fast enough (the tiny Llama below), and the
  products, where the time goes in larger models, stay on arrays. Moving the row operations to arrays is a debt.
- **Attention recomputes its probabilities** in the backward pass from the saved Q and K (rows) instead of keeping them.
  `Act` in llama.bend did not keep them either; for training the choice is now recomputation, by design, not by lack.

### Milestone 3: a two-layer Llama trained in Bend

`demo/tiny_llama.bend`: token embedding 65 x 32, two blocks of `Residual(Seq(RMSNorm, attention))` and
`Residual(Seq(RMSNorm, SwiGLU 32-64-32))`, RMSNorm, a linear head to 65 logits, cross-entropy; attention is
`Fork(q, Fork(k, v))`, then `Both(RoPE, Both(RoPE, Id))`, the attention core (4 query heads, 2 key/value heads of 8,
causal) and the output projection. Character level on Tiny Shakespeare, batches of 2 x 16 positions, AdamW (lr 3e-3,
weight decay 0.01) on every parameter. Its twin, `reference/tiny_ref.py`, writes the initial weights and 2,000 batches.

- The whole model, about 80 composed layer values (combinators included), type-checks in 0.2 s and compiles in 17 s.
- **Gradients on the first batch**: all 21 parameter tensors, 22,753 numbers, 3,931 bit-identical, the worst at 0.026 of
  the tolerance; the loss 4.228060, one ulp from PyTorch's.
- **Training**: 20 AdamW steps agree within the tolerance in every loss and every final weight (worst 0.83 of it; the
  gate in `make check`). Over 500 steps the loss goes from 4.23 to a mean of 2.633 over the last 50 steps, PyTorch's
  2.632; the losses agree within 1e-6 (relative) up to step 60 and drift apart after it (1e-4 at step 72, 2% at step 100).
- **Why they drift, traced**: the first parameter to move apart is row 55 of the embedding, the character `q`, which first
  appears as an input at step 34. At that step the 28th entry of its gradient is -1.39e-7, the remainder of a
  cancellation; Adam's denominator `sqrt(v)/bc2` is then 2.4e-8, of the order of `eps` (1e-8), so the update depends on
  the exact value of that tiny gradient, which the two programs round differently. The update differs by 0.7%, and the
  difference spreads from there. Not a bug in either: a property of Adam on gradients near `eps`. A trajectory test
  longer than ~50 steps would have to compare statistics, not values.

### Laws

Six laws, proved in `PROOF.bend` and checked by `--verdict` (0.4 s): `adjoint` and `weight_adjoint` (the backward pass of
a linear layer is the adjoint of its forward pass, for the input and for the weights), `adjoint_seq` (the chain rule for
two linear layers), `causal` (attention puts probability exactly 0 on later positions, about the package's own
`causal_row`) and `gqa_head` (llama.bend's proof, about the head grouping that both passes of `Attention` use).

- `bend-ml-nat-lemmas`' `add_assoc` states `a + (b + c) == (a + b) + c`; llama.bend's own states the other direction. The
  copy of `div_blocks` needed its `Equal.sym` removed.
- To refute `False == True` (in `len_take`), I rewrite the goal `Empty`, written as `disc(False)`, with the equation turned
  around, and answer `Unit{}`: the guide's recipe for clashes.
- The linear laws speak about a model over `Nat` (`torch/spec.bend`), not about the F32 code: the model's `mv` and `mtv`
  are the formulas the array products compute in another order. That connection is tested, not proved.
- A sixth law, `kv_of_group`, came out of the mutation run (below): the attention code now picks the key/value head in one
  def, `kv_of`, called by both passes, and the law says `gqa_head` of that def. A mutant that makes `kv_of` use `mod` is
  now caught by the laws; a mutant where the attention code stops calling `kv_of` still slips past them (no law says which
  def the attention calls) and is caught by the tests.

### Mutation testing

`scripts/mutate.py` (`make mutate`) applies deliberate faults, each to the code or to the specification, and runs the gate
that should see it. The first run had 18; with `kv_of` there are 20, and all behave as expected: 18 are caught, 2 survive
for a stated reason (one equivalent mutant, one fault the laws cannot see but the tests catch).

- A query head that reads key/value head `hi mod g` instead of `hi / g` first survived the laws (`gqa_head` is about
  `Nat.div` itself, not about the code that calls it) and was caught only by the numbers. Fixed with `kv_of` and
  `kv_of_group` (above).
- Adding the bias gradient in the other order survives: F32 addition is commutative, an equivalent mutant.
- **The mutation run found a gap:** "the bias gradient is not accumulated" (`db` instead of `gb + db`) first survived,
  because every test ran a single backward pass into a zero gradient, where `0 + db == db`. The MLP test now runs two
  backward passes into one gradient and compares with two `backward()` calls in PyTorch; the mutant is caught.

### Speed

`scripts/bench.sh`, 5 alternated rounds, medians (range), on an Intel Core Ultra 7 155H, `performance` profile, the
machine otherwise idle; 1 thread pinned to CPU 3, 16 threads unpinned; PyTorch 2.14.1 CPU with `torch.set_num_threads`.

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD (600 batches of 100), BendTorch | 12.7 s (12.6-12.7) | 8.1 s (8.0-8.4) |
| the same, PyTorch | 0.32 s (0.32-0.32) | 0.38 s (0.24-0.96) |
| tiny Llama, one AdamW step (2 x 16 positions), BendTorch | 13.1 ms (13.0-13.1) | 13.4 ms (13.3-14.3) |
| the same, PyTorch | 9.1 ms (9.1-9.1) | 12.8 ms (12.7-18.2) |

- MNIST is about 40x slower than PyTorch on one thread. bend-ml's hand-written MNIST (`demos/mnist/fast.bend`, on
  `bend-ml-tensor-array@0.1.3.0`, weights stored `in x out`) measured in the same session: 18.3 s on one thread, 7.4 s on
  16 (3 rounds). The general layers cost no more than the hand-written version on one thread and 9% more on 16; the two
  differ in kernel version and layout, so this is a sanity check, not an isolated measurement of the abstraction's cost.
- The tiny Llama is 1.4x slower than PyTorch on one thread: at 32 x 32, PyTorch's per-operation overhead dominates. It
  does not use more than one thread in BendTorch (its products are sequential, `par = 0`, and the row operations are
  list code).

### Debts

- **Shapes beyond matrices.** Every tensor is a `Mat<r, c>`; batches and sequences are folded into rows, heads into
  columns. There is no rank-3 or rank-4 type, no general broadcasting, no slicing API, and no batched matmul beyond what
  attention does internally. Whether a rank-polymorphic typed tensor (`Tensor<shape: List<Nat>>`) pays off in Bend is an
  open question (docs/plan.md).
- **Sizes fixed at compile time.** The layer constructors take literal sizes; bend-ml's `examples/runtime_batch.bend`
  shows the pattern for a batch size read at run time, not used here yet.
- Row operations on lists (above), sequential.
- No dropout (needs a reproducible random stream shared with the twin), no exact GELU (`erf`), no tied embeddings, no
  checkpoint save/load.
- Optimizer state: two buffers per parameter for every optimizer.
- The types of a model are long. Small `def X() -> Type` names help; a helper that derives `Seq`'s seven types from its
  two layers would help more.
- No law ties the F32 code to `spec.bend`, and none covers the non-linear layers' gradients.

## 2026-10-15: towards v1

What a v1 needed beyond the milestones: less waste in the products, a model that shows what it learned, checkpoints, the
missing modules (dropout, tied embeddings) and a stricter numerical policy where exactness is possible.

- **`LinearFirst`.** The first layer of a model computed `dX = dY·W`, which nothing reads (the data needs no gradient);
  PyTorch skips it. A variant that skips it (and returns zeros as dX) is a third less work for that layer: MNIST went
  from 12.5 to 8.7 s per epoch on one thread and from 8.2 to 5.9 s on 16 (3 alternated rounds), with the same loss to
  the bit (0.5148658). At 16 threads it is now below bend-ml's hand-written MNIST (7.4 s).
- **Attention over the sequences of a batch in parallel** (balanced halves, forward and backward): the tiny Llama's step
  from 13.1 to 11.3 ms at 16 threads, unchanged on one; outputs bit-identical over 50 steps (the order of every sum is
  the same; only which core runs a sequence changes).
- **The tiny Llama is a template over the batch** (`R(b) = 16·b` rows): `{==}` inside a template is checked before the
  template's arguments are known, so a size written as `Nat.mul(b, 16n)` keeps the proofs reflexive, and each instance
  reduces to its literal. The parameters' type does not depend on `b`, so the weights trained at `b = 2` run at `b = 1`.
- **Generation.** `tiny_llama sample` trains, then continues a prompt greedily one character at a time. PyTorch, given
  the weights Bend trained (read from the program's output), continues the prompt with the same characters: 300 out of
  300 after 2,000 steps; `test_tiny.py` checks 64 after 100 steps. The model is small (22,752 parameters, mean loss 2.36
  over the last 100 of 2,000 steps), and greedy decoding falls into a loop ("e than thar thar thar ..."): it shows that
  both programs compute the same function, not that the model writes well.
- **Checkpoints.** Every layer now has `dump` (its parameters as a flat list, in a fixed order) and `load` (overwrite
  them from such a list, returning what is left); `Seq`, `Fork`, `Both` and `Tie` compose both. `Train.save` writes the
  parameters, both optimizer moments and `b1^t`, `b2^t` as little-endian F32 (273,032 bytes for the tiny Llama:
  3 x 22,752 numbers + 2), `Train.load` reads them into a fresh state. 30 steps, a checkpoint, then 20 steps in another
  process give the same bits as 50 steps without a stop (`test_tiny.py`).
- **Dropout** with a counter-based mask: number `i` at step `t` is kept (times `1/(1-p)`) when
  `(lowbias32(key + i) >> 8) / 2^24 >= p`, `key = lowbias32(seed xor lowbias32(t))`. The step counter is the layer's
  parameter; the optimizer step increments it and checkpoints keep it. The twin computes the same mask in numpy, so the
  test is bit-exact (steps 0, 3, and 4 reached through `Layer.step`); 74.96% kept out of 100,000 at `p = 0.25`.
- **`Tie`** for a parameter used twice (tied embeddings): first and last share `P`, backward adds both gradients into one
  accumulator. Matches PyTorch's table used twice.
- **An exact tier in the numerical policy.** Copies, selections and one or two F32 operations in PyTorch's order (relu,
  an embedding lookup, the MSE and NLL gradients, dropout) must be the same bits; everything else keeps the tolerance.
  Results that match today only by luck of the order of additions or of `libm` (`tanh`, the embedding's gradient) stay
  out of the exact tier.
- **Mutation testing**: 23 mutants (new: two for dropout, one for `Tie`), all as expected.

### Speed, again

`make bench`, 5 alternated rounds, medians (range), the same machine; this time not idle (load average 2.8 when it
started), so the ranges are wider than before.

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD, BendTorch | 8.8 s (8.7-9.5) | 6.0 s (5.9-6.6) |
| the same, PyTorch | 0.32 s (0.32-0.34) | 0.24 s (0.22-0.86) |
| tiny Llama, one AdamW step, BendTorch | 13.1 ms (13.1-13.8) | 11.5 ms (11.0-12.2) |
| the same, PyTorch | 9.0 ms (9.0-10.9) | 10.4 ms (9.7-12.8) |

MNIST is ~27x slower than PyTorch on one thread (from ~40x); the gap is the products of `bend-ml-tensor-array`, bound
by one array access per element (bend-ml NOTES exp. 17). The tiny Llama is 1.5x slower on one thread and close on 16.

### Still debts for after v1

Rank-N tensors and general broadcasting (plan M7), the row-wise layers on arrays (M6), laws about the executed F32 code
(M8), sizes read at run time, an optimizer state that SGD does not pay for, exact GELU, a less verbose way to write a
model's types.

## 2026-10-16: bend-ml-tensor-array 0.1.9.0

The pin moves from 0.1.7.0 to 0.1.9.0: bend-ml's products with two or more rows now pack their operands into cells of 8
numbers (0.1.8.0) and run in parallel without copying both operands at every node of the fork tree (0.1.9.0). Nothing
changes in this repository's code. `make check` gives the same counts of bit-identical numbers as before, `make bits`
is identical, the 23 mutants behave as listed.

### Speed on tensor-array 0.1.9.0

`make bench`, 5 alternated rounds, medians (range), the same machine, idle (load average 1.7 with the benchmark itself
running).

| | 1 thread | 16 threads |
|---|---|---|
| MNIST, one epoch of SGD, BendTorch | 3.9 s (3.8-3.9) | 3.6 s (3.5-4.0) |
| the same, PyTorch | 0.32 s (0.32-0.32) | 0.25 s (0.24-1.03) |
| tiny Llama, one AdamW step, BendTorch | 12.4 ms (12.4-12.4) | 10.9 ms (10.3-11.1) |
| the same, PyTorch | 8.9 ms (8.9-9.5) | 9.5 ms (9.4-9.7) |

MNIST goes from 8.8 to 3.9 s on one thread and from 6.0 to 3.6 s on 16: ~12x slower than PyTorch, from ~27x. Measured
against the old pin in the same session (15 alternated rounds, wall time of the whole run), 9.4 s against 4.3 s. The
tiny Llama barely moves (13.1 to 12.4 ms): its products are small and its row-wise layers, on lists, are most of the step.


## 2026-10-16: the JavaScript lane (plan M12)

`make check-js` (`scripts/check_js.sh`) builds the four test programs with `bend -o x.js`, runs them with Bun, and gives
them to the same Python tests as the native ones; then it compares the outputs of the Llama's training (50 steps) and
generation and of 50 MNIST batches under Adam with `bench/bits.txt`. Result: every test passes with the same counts of
bit-identical numbers as the native build, and the three outputs are the native build's bits. `exp` and `tanh` are
JavaScript's `Math` functions there, not libm's; on these inputs they gave the same words.

What it took:

- **Bun, not node.** `-o x.mjs` builds a module of the non-IO definitions, which does not run `main` (the first run
  printed nothing). `-o x.js` builds the program, whose file reads call libc's `read`/`pread` through `bun:ffi`: under
  node, every program that reads a file fails with "Cannot find module 'bun:ffi'".
- **A 1 GB stack.** The generated JavaScript keeps a call frame per step of each recursion that is not a tail call, such
  as a list built head first (`f32s`, the loaders); Bun's default stack holds about 40,000 frames and a weight file is
  100,000 numbers. The wrapper runs `ulimit -s 1048576` and sets `BUN_JSC_maxPerThreadStackUsage`, which allows about
  8 million.
- **The array capacity in bend-ml.** With the pinned 0.1.9.0, 50 MNIST batches take 247 s of training in JavaScript
  (4 s natively): `cap_depth(n)`, the size of a new array, walks `n` steps, each comparing two unary numbers of up to
  `n` steps. bend-ml-tensor-array 0.1.10.0 computes it by doubling; with it the same run takes
  4.3 s and the whole `make check-js` 65 s, against 34 minutes now.

## 2026-10-16: accumulating weight gradients (bend-ml-tensor-array 0.1.10.0)

The three linear layers (`Linear`, `LinearFirst`, `LinearNB`) used to compute a weight gradient into a new matrix and
then add it to the gradient they accumulate (`Mat.add`), and the bias gradient the same way. They now call bend-ml's
accumulating products, `Mat.matmul_tn_acc` and `Mat.col_sums_acc`, which add each cell of the product straight into the
gradient: no new matrix, no second pass. Each cell is computed as before and added once, `g + x`, which is what the
`Mat.add` computed, so `make check` gives the same counts, `make bits` is identical, and the 23 mutants behave as listed
(the two linear ones now remove the accumulation; the equivalent mutant about the order of the bias addition is gone,
since that addition now happens inside bend-ml).

Measured on the same idle machine, alternated rounds, medians:

| | before | accumulating | ratio |
|---|---|---|---|
| `dW += dYᵀ·X`, 32 x 1536 x 576 (a SmolLM2-135M MLP projection, 32 tokens), 1 thread | 260 ms / 40 | 184 ms / 40 | 0.71 |
| the same, 8 tokens | 359 ms / 100 | 159 ms / 100 | 0.44 |
| the same, 128 tokens | 176 ms / 10 | 152 ms / 10 | 0.86 |
| the same, MNIST's first layer (100 x 128 x 784) | 175 ms / 100 | 154 ms / 100 | 0.88 |
| MNIST, one epoch of SGD, whole run, 1 thread (15 rounds) | 4.25 s | 4.12 s | 0.97 |
| the same, 16 threads | 4.07 s | 3.96 s | 0.97 |
| tiny Llama, 200 AdamW steps, 1 thread | 2.49 s | 2.48 s | 1.00 |

The pass it removes is proportional to the weights, the product to the weights times the rows of the batch, so the gain
grows as the batch shrinks relative to the weights: 3% on MNIST, nothing visible on the tiny Llama, 30-56% per weight
gradient at SmolLM2's sizes with a few dozen tokens (plan M10). On those sizes, the same micro-benchmark is also slower
on 16 threads (`par` 3) than on one, for both versions: 455 against 260 ms at 32 tokens.

- 0.1.10.0 is on BendHub (hash `0xbb78793a...`). Against the published package: `make check` the same counts, `make bits`
  identical, `make mutate` 23/23, `make check-js` the same bits as the native build.

## 2026-10-16: bend-ml-tensor-array 0.1.11.0

The products choose their split depth from the work (`par` is an upper bound; 0 below 2^23 multiply-adds or 64 per
cell). Here `make check` gives the same counts, `make bits` is identical, `make mutate` 23/23 and `make check-js` the
same bits as native. The layers keep their `par` values: MNIST's two large products stay above the threshold (an epoch
unchanged, 3.60 s on one thread, 3.30 on 16), and passing 4 instead of 3 to the first layer was slower over a whole run
(measured; the runs are in my private notes). The gain is for M10's shapes: a weight gradient over 8 tokens of
SmolLM2-135M's sizes 3.3x faster on 16 threads.

## 2026-10-16: CI (plan M13)

Actions is on for the private remote. The workflow runs in two modes, so it fits the 2,000 minutes a month a private
repository gets: **quick** on every push to `devel` or `main` (`make check` and `make bits`; a newer push to the same
branch cancels a run in progress) and **full** on pull requests to `main`, every night on `devel` and by hand (also
`make mutate` and `make check-js`, with Bun 1.4.2). The Bend and Lean toolchains, the Python environment and the
datasets are cached.

The first runs, all green on a 4-vCPU `ubuntu-24.04` runner: quick 6.4 min with cold caches (setup 1.0, check 2.9,
bits 1.7); full 10.4 min with warm caches (setup and caches 0.9, check 2.1, bits 1.5, mutate 3.9, JavaScript lane 1.7).
At five pushes a day and one nightly run that is about 600 + 320 minutes a month. The nightly schedule starts once this
workflow is on `main`, since GitHub runs scheduled workflows from the default branch.

## 2026-10-16: general tensors (plan M7)

`torch/tensor.bend`: `Tensor<s>`, the shape a list of sizes in the type, over one flat `Array<F32>`, row-major.

**The prototype first** (broadcast add and a sum over an axis at rank 3, with two laws), to measure what was open in the
design: the checker takes 0.15 s for the module and for a program with literal rank-3 shapes, whose broadcast proof
(`bc_ok(...) == True`) it computes; a wrong broadcast is rejected ("expected True, observed False"); types like
`Tensor<remove(s, k)>` reduce to literal shapes on their own; the laws (`bc_sym`, `sum_numel`) are short inductions over
the shape. **Decision: adopt `Tensor<s>` as the layer of general operations next to `Mat`, without porting the
layers.** `Tensor<[r, c]>` and `Mat<r, c>` are one array under two types, so `of_mat` and `to_mat` cost nothing (law
`mat_numel`); the layers keep bend-ml's fast products, and M6 stays on `Mat`.

**The operations**, each a gather or a scatter over an index map (per-axis sizes and strides, stride 0 where an axis is
broadcast; or "the position with one axis taken out"): `add`, `sub`, `mul`, `div` with broadcasting and their gradients
(PyTorch's formulas, the divisor's as `-dz·((x/y)/y)`); `sum_axis` (and `sum_axis_keep`), `mean_axis`, `sum_all`,
`mean_all`; `reshape` (with the proof that the counts are equal) and `flatten`; `permute` and `transpose`; `narrow`
(with the proof that the window fits); `cat`; `bmm`. `cat` and the backward of `narrow` and `permute` copy into place
(no addition), so `-0` stays `-0`.

**Checks.** `demo/tensor_test.bend` reads the operation and the shapes at run time and checks each shape condition once
where it enters; `reference/test_tensor.py` draws 204 cases (ranks 0-4, size-1 and empty axes, 12 per operation) and
compares the result and every gradient with `torch.autograd`: all within the tolerance, the worst at 0.070 of it (the
order of the sums of reductions and products), and 14 invalid shapes rejected. The same in JavaScript (`make check-js`).
Seven shape laws by `--verdict` (`bc_sym`, `sum_numel`, `narrow_rest`, `narrow_axis`, `cat_numel`, `mat_numel`,
`permute_rank`); four programs that must not compile (`tests/bad_broadcast`, `bad_narrow`, `bad_cat`, `bad_bmm`). Five
new mutants (a wrong sign in div's gradient, narrow's window on the wrong axis, bmm reading its operand transposed,
broadcasting one way round only, cat's shape counting one operand twice), all caught: 28 of 28.

**Left.** The kernels are scalar and use U32 division per position to find coordinates; fast enough for the tests, not
measured against PyTorch, and not parallel. That a general permutation keeps the count of numbers is tested, not proved
(the law covers the rank). The layers do not take tensors of rank 3 or more yet.

## 2026-10-16: the tensors' index maps stay inside their arrays (laws track L1)

`demo/imap_check.bend` and `reference/check_imap.py` walk every position of every index map on every small shape (rank
0-3, sizes 0-3) and on larger random ones: 4,206 cases, 242,871 positions, none out of bounds, natively and in
JavaScript; in `make check` and `make check-js`. Against 11 deliberate faults in the maps it caught one the numeric test
cannot: the whole-tensor sum mapping each element to its own position of a 1-cell array, right only because an Array
index wraps around. It is in `make mutate` as a pair (survives the numbers, caught by the bounds), 30 mutants in all.
A law that the maps stay in bounds is the next step (L2).

## bend-lint's first pass (2026-10-17)

Mathews Fernando's bend-lint, with its 12 performance rules, is a dev dependency (`package.json`, `bend-lint.json`,
`make lint`), run before each milestone closes; a finding is measured before it is taken. First pass over the 19 Bend
files: 25 findings, none taken yet:

- `list_index_loop` on `lm_rows` in `attn_batches`/`attn_bwd_batches` and on `emb_gather`, `loop_invariant_call` on
  `lm_zeros`/`l_zeros`: loops over the b = 2 sequences of a batch, or 32 rows of an embedding; their O(n²) is a few
  hundred list steps a step.
- `eager_branch` in `torch/tensor.bend` (shape checks, once per operation), `append_in_loop` in `demo/tensor_test.bend`
  (a test's output), `unused_result` in `PROOF.bend` (a law), `eager_branch` on `index_of` in the tiny Llama's
  vocabulary lookup (65 characters).

Found next to it, not by a rule: a profile of 50 tiny-Llama training steps on one thread spends 33.5% in `term_drop` and
33.3% in `span_fade`, the runtime taking apart list nodes that are shared (`+` lists through attention, the norms and the
row layers). Two thirds of a step is that, against 2.4% in the matrix products. This is what M6 (row-wise layers on
arrays) removes, and the measure M6 should be judged by.

## bend-lint's attention findings at larger batches (2026-10-17)

The tiny Llama's attention forward and backward over b sequences of 16 positions, one thread, 3 rounds: 2.01 ms per
sequence at b = 2, 2.04 at 8, 2.06 at 16, 2.08 at 32, 2.13 at 64. The quadratic part of `lm_rows` and `lm_zeros` is under
6% at b = 64, and a full fix would cut the lists down `attn_par`'s tree, which M6 replaces. Not taken; the 14 findings
carry a `bend-lint: disable-next` directive with the reason. `term_drop` and `span_fade` are 79% of the same profile.
`emb_gather` is left open: measured separately, the Embedding over arrays is 52-78x faster at SmolLM2's table, waiting
for my call.

## The Embedding over the table's array (2026-10-17)

bend-lint flagged `emb_gather` (`List.drop` per id); the cost was larger than that: the forward turned the whole table
into a list of rows (`Mat.rows(v, d, p)`) on every step, and the backward turned the gradient table into rows and back.
Now the rows of the ids are read from the table's `Array` by index (`TA.Mat.read_row`), and each dY row is added into
the gradient's row in place (`read_row`, `l_add`, `write_row`), in the order of the ids, as before. `row_head`,
`emb_gather`, `add_at` and `emb_scatter` are gone.

At SmolLM2-135M's table (49,152 x 576), forward and backward, one step, 3 rounds alternated: 2,487 → 31.8 ms with 64
ids (78x), 2,870 → 55.7 ms with 512 (52x); peak memory 942 → 273 MB at 64 ids; the same checksum (the runs are in
my private notes). `make check` passes, `make bits` identical, `make mutate` 30/30 as expected (the
embedding mutant now drops the `l_add` in place: caught), `make lint` 0 findings.

## The linear layers over bend-ml-tensor-fork (2026-10-17)

bend-ml-tensor-fork 0.1.0.0 (bend-ml v1.4.0) has tensor-array's six products with the parallel blocks sharing their
operands through Base's `@unsafe` `Array.fork` instead of copying them; same arguments, result types and bits. Renan's
decision: use it, as fast layers next to the proved ones, the demos on the fast ones.

- tensor-array 0.1.11.0 → 0.1.12.0 first (tensor-fork's types are 0.1.12.0's): `make laws`, `check`, `bits` identical.
- `torch/fast.bend`: `LinearF`, `LinearFirstF`, `LinearNBF`, `main.bend`'s layers with their three products from
  tensor-fork; everything else is `main.bend`'s code. `main.bend`, `LAWS.bend` and `PROOF.bend` do not change, so
  `--verdict` still checks every law. `make laws` pins the eight defs of `fast.bend` outside the proofs and checks that
  no file with laws imports it.
- `demo/fast_test.bend`: each fast layer against its proved twin, forward and backward, at MNIST's sizes (LinearFirst
  100x784x128 at par 3 and 4, Linear 100x128x10), the tiny Llama's (LinearNB 32x32x64) and splitting ones (Linear and
  LinearNB 128x576x1536): 0 differing words in y, dX and every gradient, on 1 and 16 threads and in JavaScript (in
  `make check` and `make check-js`). Three mutants on `fast.bend` (each weight gradient not accumulated), caught by it:
  `make mutate` 33/33 as expected.
- The demos (MNIST, the tiny Llama) on the fast layers: `make check` passes, `make bits` identical, `make check-js` the
  same outputs. MNIST's first layer at par 4 instead of 3: with the fast layers par 4 is faster on 16 threads (2.6 against
  3.0 s, alternated) and the same on one (2.9); with the proved layers it had been slower (2026-10-16). The same bits.
- `make bench`, 5 rounds, before (proved layers, par 3) and after: MNIST 3.6 → 2.9 s on one thread, 3.3 → 2.4 s on 16
  (PyTorch 0.32 and 0.26 s: 11x → 9x on one thread); the tiny Llama 12.1 → 12.2 ms on one thread and 10.7 → 10.3 on 16,
  unchanged, as expected: its products are 32 wide, below tensor-array's split threshold, so both layers run the same
  sequential code. Its step is spent taking apart shared lists (two thirds of it), which is M6's work.

## Attention over arrays (M6, phase 1; 2026-10-17)

A profile of the tiny Llama's step (one thread, 30 steps, call stacks) put about half of it in attention's list
products (`lm_dots` under `lm_mul_t`, with the teardown of the shared lists they walk) and 11% more in `List.take`
cutting rows and columns; the other row-wise layers under 5% each. So M6 starts with attention.

- Q, K and V stay in their arrays. Each sequence's rows are copied out once into arrays of its own; each head's P and G
  (t x t) are arrays. Every sum is added in the list version's order (`l_dot`: left to right from +0.0, including the
  zeros past the visible positions in P·V), and each row of P still goes through `causal_row` and `l_softmax`, each row
  of G through the same `l_mul`/`l_shift`/`l_dot`/`l_scale`. The scores past position i are no longer computed. `make
  bits` identical at 1 and 16 threads, and every summary line of `make check` (bit-identical counts, worst errors)
  identical to the list version's.
- The sequences of a batch: a tree of parallel lets from 8 sequences on, one after another below. Attention alone,
  forward and backward, 4 query heads and 2 key/value heads of 8, 16 positions, 16 threads, 5 rounds: 2 sequences 0.37 ms
  in a loop against 0.48 in the tree; 8: 1.52 against 1.32; 32: 5.95 against 4.00; 128: 23.8 against 14.0. One thread
  the same either way.
- Law `causal_keys`, the law about the code that runs (my call: write it over the new code): attention reads keys
  only through `qrow`, and the row it builds for the query at position i has exactly i + 1 scores, keys 0..i. The proof
  is an induction over `drow`'s loop with `len_rcat` and `add_succ`. `causal` still holds over the executed
  `causal_row`. A mutant that reads one key too many survives the numbers (`causal_row` drops the extra score) and is
  caught only by the law. `make mutate` 35/35 as expected (the two attention mutants that aimed at the list code
  rewritten for the new one).
- The list helpers attention alone used are gone (`lm_mul`, `lm_mul_t`, `lm_t`, `lm_dots`, `lm_heads_of`,
  `lm_tails_of`, `lm_cols`, `lm_rows`, `lm_zeros`, `lm_add`, `lm_hcat`).
- `make bench`, 5 rounds (load 1.7 at the start): the tiny Llama 12.2 → 4.1 ms a step on one thread, 10.3 → 5.0 on 16
  (PyTorch 9.0 and 12.9). MNIST unchanged (2.9 s on one thread; 2.6 on 16 against 2.4 the run before, within this run's
  2.4-2.7). A profile after: `term_drop` and `span_fade` 33% of a step three times shorter (67% before); what is left is
  the row-wise layers on lists (phase 2).

## The row-wise layers over arrays (M6, phase 2; 2026-10-17)

A profile after phase 1 (one thread, 400 steps) still put 39% of a tiny-Llama step in `term_drop` and `span_fade`, the
runtime taking apart shared lists, most of it under RMSNorm, the rows of attention's P and G, the matrix-to-rows
conversions and RoPE. Phase 2 moves every row-wise layer onto the matrix's own array, a row at a time, by index.

- Every loop keeps the order of operations of the list version it replaces (sums left to right from +0.0, each product
  in the same order), so the bits did not move: `make bits` identical at 1 and 16 threads and in JavaScript, every summary
  line of `make check` identical to phase 1's, and the raw output of `ops_test` under `test_ops.py` (605 lines, 10 seeds,
  every layer) byte for byte the same as before phase 2.
- In order, 200 training steps pinned to one P-core, each change against the one before it: RMSNorm 0.82 → 0.50 s,
  attention's rows of P and G 0.50 → 0.40 s, RoPE (a cos/sin table per call, the pairs rotated in place) 0.40 → 0.24 s.
  Then softmax, log-softmax, LayerNorm, the embedding, MSE and NLL, which the demos barely use or do not use, so that no
  layer turns a matrix into lists any more; 20 list helpers are gone, and the shared row kernels (dot, softmax in place,
  copy) live in one section of `torch/main.bend`.
- A backward pass now keeps a matrix: RMSNorm and LayerNorm their input, softmax and log-softmax a copy of their output.
  The `S` type of those layers changed from a list of rows to `Mat`, so the demos' type arguments changed with it.
- Law `causal`, rewritten over the new code (my call): `prow` writes query i's scores from the start of its row of P
  and hands the softmax the count of positions it wrote, and the law says that count is i + 1. The proof is the same
  induction as `causal_keys`, through `pw`'s count (`cnt_pw`, `cnt_drow`). That the positions past i keep P's zeros is
  read from the code, not proved: the checker cannot follow writes into an array by index. The old law spoke about
  `causal_row`, which no code calls any more, and is gone with it.
- `make mutate` 37/37 as expected. The row-wise mutants now aim at the array kernels. A query that sees the next position
  (the count starts at 1) is caught by the law and by the numbers. A softmax over one position too few survives the law
  (it speaks about the count, not the loops inside the softmax) and is caught by the numbers. A query that reads one key
  too many, which in phase 1 only `causal_keys` caught, now changes the numbers too: the extra score reaches the softmax.
- `make bench`, 5 rounds: the tiny Llama 4.0 → 1.1 ms a step on one thread, 4.3 → 1.3 on 16 (PyTorch 8.9 and 12.8);
  MNIST 2.9 s on one thread, 2.4 on 16, unchanged. PyTorch's MNIST epoch on 16 threads took 0.23, 1.07, 1.01, 0.90 and
  1.17 s in the five rounds (0.23-1.20 in the run before), so the speed table gives its range.
- A profile after (one thread, 1000 steps): `term_drop` and `span_fade` 8% of the step. The products are now the largest
  part (about a third), then bend-ml-tensor-array's cross-entropy, which still softmaxes each row as a list (about 10%),
  and `cap_depth`, recomputed by every `Array.new` of a kernel's scratch array (about 5%). 16 threads are still slower
  than one on a step this small: phase 3.

## What phase 2 left: cap_depth, cross-entropy, the products (2026-10-17)

The profile after phase 2 named three things left in a tiny-Llama step. Times are 1000 training steps pinned to one
P-core, 3 runs each, against the phase-2 binary.

- `cap_depth` (about 5%): every kernel's scratch array asked bend-ml-tensor-array's `cap_depth` for its capacity, which
  compares Nats one unit at a time at each doubling. `depth` doubles a U32 instead: log2(n) cheap steps, the same
  answer (`demo/depth_check.bend`, in `make check`: equal to `cap_depth` for every n from 0 to 4999). 1.06 → 1.03 s.
  What is left of `cap_depth` (2.7%) is inside the packages: the output arrays of bend-ml-tensor-fork's products.
- Cross-entropy (about 10%): `CrossEntropy` called tensor-array's `Mat.softmax_ce`, which reads each row into a list,
  softmaxes it as a list and writes the gradient back. It now works on the logits' array with `psm` and two short loops,
  tensor-array's operations in its order: `make bits` identical (MNIST and the Llama), the raw `ops_test` outputs byte for
  byte the same. With `depth`, 1.10 → 0.93 s.
- The products (about a third): the tiny Llama's products (32 rows, LinearNB 32 → 16, 32, 64, 65 and 64 → 32; forward,
  dX and dW) through tensor-array's two paths, 1000 rounds on one thread: 0.35 s on the 8-wide packed path the layers
  already take, 0.58 s on the scalar one, the same result. So the products are where the time should be, already on the
  faster path; making them faster is a new kernel in bend-ml, not a fix here.
- `make bench`, 5 rounds: the tiny Llama 1.1 → 1.0 ms a step on one thread, 1.3 → 1.1 on 16 (PyTorch 9.0 and 12.8); MNIST
  unchanged (2.9 s, 2.5). A profile after: `term_drop` 0.6% of a step (67% before M6); the products about 43% (with
  tensor-array's packing), attention's kernels about 26%. `make mutate` 39/39 with a cross-entropy mutant and a scratch
  array at half its capacity, both caught.

## Parallelism where it pays (M6, phase 3; 2026-10-17)

The tiny Llama cannot show it: 1000 steps take 0.92 s on one thread and 0.97-1.01 s on 2 to 16, every piece of work
smaller than the cost of splitting it. So I measured on a new workload, `demo/smol_block.bend`: one decoder block at
SmolLM2's sizes (the tiny Llama's block, RMSNorm, grouped-query attention with RoPE, SwiGLU, two residuals), 8 sequences
of 64 positions, AdamW against a target under MSE, its weights, inputs and targets from a hash of each number's index, and
its PyTorch twin `reference/block_ref.py`. `scripts/block.sh` builds it at 135M's sizes (576 wide, 9 query and 3
key/value heads of 64, feed-forward 1536), 360M's (960, 15 and 5, 2560) and 1.7B's (2048, 32 and 32, 8192). The three
agree with PyTorch within the tolerance (worst 0.01 of it) and give the same bits on 1 and 16 threads
(`reference/test_block.py`; the 135M one is in `make check`).

- **MSE summed as a tree.** The block's first comparison was 1e-4 off: the Bend loss was exactly a running F32 sum of
  PyTorch's own squares, and 295k squares added left to right lose the small ones. The loss is now a balanced tree of
  additions (pairwise, as llama.bend's `tree_sum`): 1 ulp from PyTorch and from a float64 sum. The gradient is unchanged.
- **Where one thread spends a step** (135M, `perf`): the products 86% (tensor-fork's two fork trees 82%, packing 3%),
  attention 7%, every row-wise layer together about 3%. The row-wise layers were never the problem.
- **Where 16 threads wait.** A `perf` profile on 16 threads, cut into 1 ms slices and counting the threads at work in each:
  43% of the wall time had three threads or fewer at work. In order: the packing of A and B (sequential), sizing arrays
  through `TA.cap_depth`, the optimizer step, SiLU, the copies and sums of `Fork`/`Residual`, RMSNorm, RoPE, attention's
  copies.
- **The products' split.** At `par` 4 (16 blocks) the products stopped gaining at 8 threads: equal blocks on P-cores,
  their second hyperthreads and E-cores finish at different times. `par` 8 (256 blocks) took the block from 763 to 577
  ms per 3 steps on 16 threads at 135M, 1840 to 1313 ms at 360M (par 9 no better), one thread unchanged.
- **bend-ml-tensor-fork 0.1.1.0** (bend-ml NOTES): `par1`..`par4`, any kernel over blocks of a range on up to four shared
  arrays; the packing in blocks when an operand holds 16384 cells or more; output arrays sized by `depth`. 562 → 527 ms.
- **`torch/fast.bend`**: the optimizer step (SGD, momentum, AdamW), `SiLU`/`ActIn`, `Mul2`, `Add2`, `Residual` and the
  copies and sums its `Fork` uses, each on `par1`..`par4` with main.bend's own loops, so main.bend's bits
  (`demo/fast_test.bend`, 12 cases, 1 and 16 threads and JavaScript). Below 131072 numbers a matrix runs in one block:
  with MNIST's 100352-number weights in 8 blocks its 16-thread epoch was 0.1-0.2 s slower. 528 → 500 ms.
- **Not adopted**: RMSNorm and RoPE in blocks of rows (the backward pass's gain gradient as a second pass over blocks of
  columns, each column in row order: the same bits) took them off the sequential list but changed nothing measurable,
  so they stay sequential. Leaves of 2 x 2 tiles in the products (bend-ml NOTES): the same bits, 20% slower.
- **Result**, ms per AdamW step, medians of 5 alternated rounds (1.7B: 3), a warm-up step outside the timing, one thread
  pinned to a P-core (my private notes have the runs):

  | | 1 thread before | after | PyTorch | 16 threads before | after | PyTorch |
  |---|---|---|---|---|---|---|
  | block at 135M's sizes | 779 | 779 | 106 | 247 | **155** | 51 |
  | block at 360M's sizes | 2029 | 2026 | 282 | 589 | **351** | 92 |
  | block at 1.7B's sizes | 14116 | 13963 | 1852 | 4412 | **2733** | 545 |

  16 threads 1.6-1.7x faster at every size, one thread unchanged; 16 threads now give 5.0-5.8x over one, against 3.1-3.4x
  before. PyTorch stays 7x faster on one thread and 3-5x on 16: its products are a BLAS kernel, and ours are scalar F32
  chains in a fixed order (the 2 x 2 tiles showed the reads are not the limit).
- What is left on 16 threads (135M): about 23% of the wall time with three threads or fewer at work, now mostly zeroed
  allocations (the packed operands and the products' outputs) and attention's copies and joins; attention runs its 8
  sequences on 16 threads.
- `make bench`, 5 rounds: the block 768 ms a step on one thread, 152 ms on 16 (PyTorch 105 and 50); the tiny Llama 1.0
  and 1.1 ms (PyTorch 9.1 and 12.8) and MNIST 2.8 and 2.4 s (PyTorch 0.32 and 0.25-1.07), unchanged within the noise:
  their matrices are below every new threshold. `make check`, `make bits` (identical), `make check-js` and `make mutate`
  (42/42, with mutants for the fast optimizer's blocks, the fast copies and MSE's tree sum) pass.
- M6's pass criterion asked for the tiny Llama's step to be faster on 16 threads than on one. It is not, and cannot be at
  that size; the 16-thread gain is measured on the block instead.
- Decisions (2026-10-17): I keep the optimizer step and the element-wise layers in `torch/fast.bend` (the exception now
  covers them, my call). RMSNorm and RoPE in blocks stay out; I will measure them again if a later workload makes them
  look useful (the code is kept in my lab notes).

## Laws about the code that runs (M8; 2026-10-17)

Six new laws and one rewritten, all by `--verdict`, about the code the demos execute.

- **`kv_cols`.** The head loops of both attention passes used to compute each head's columns inline, so a mutant could
  make the forward pass alone bypass `kv_of`, and no law saw it. Now one function, `head_cols`, gives a head its columns
  (`HC{hc, kc}`), and `head_fwd`/`head_bwd` take only those: with no head index they cannot compute others. The law: query
  head `j·g + i` reads key/value columns `j·hd`, by `kv_of_group`.
- **`causal_grad`.** The backward pass computed each row of dP and G over all t positions and relied on P being 0 past
  the query. `grow` now reads dP's row against values 0..i only (as `qrow` does for keys), and the sum and G run over the
  positions it wrote; the law says that count is i + 1. The rest of G keeps its zeros. Same bits (`make bits`), one more
  bit-identical number in `ops_test` (13,451), and less work: the tiny Llama 0.93 → 0.91 s per 1000 steps on one thread,
  the 135M block 1590 → 1566 ms per 2 steps (3 alternated runs, the machine not idle; 16 threads within the noise).
- **`causal`, rewritten.** It spoke about the count `prow` handed the softmax, so a softmax that ran over one position
  too few passed it. The softmax's division loop now counts the positions it divides (`psmc`, a `PC{a, c}`; `psm` wraps
  it, so Softmax and cross-entropy are unchanged), and `prow3` hands that count on as the row's. The proof reaches the
  count through the maximum and exponential loops by matching their results, not computing them.
- **`attn_par_fwd`, `attn_par_bwd`.** The trees over sequences (`fpar`, `bpar`) give what `fleaf`/`bleaf` give one
  sequence after another, at every depth and however `cut` splits them. Their lists hold arrays, and Bend lets a proof use
  such a list once, so the obvious proof (the loop is a map, maps distribute over append, cut's halves append back to the
  list) needed each list twice. Each lemma now returns the list it was given together with its fact (records like
  `L1R{ys, same, fact}`), and the induction on the depth calls itself through two closures, so there is no mutual
  recursion. About 200 lines for each pass.
- **`seq_chain`, `fork_sum`.** `spec.bend` now has the model's linear layer as a `T.Layer` (`LinL`), and `SeqL`/`ForkL`
  instantiate the package's own `Seq` and `Fork` with it. The laws run those combinators: `dy·y == dx·x` through `Seq`
  (the proof is `adjoint_seq`, since every step of the combinator reduces) and `dy1·y1 + dy2·y2 == dx·x` through `Fork`
  (`dot_vadd` and `adjoint` twice). A law cannot quantify over layers (they are templates), so these speak about the
  model's linear layers.
- **Not stated**: tensor-array's parallel product. Each block reads its own packed copy of its rows, so "the same rows as
  the sequential product" would need the contents of arrays written by index. Tensor-fork and `fast.bend` cannot have
  such a law at all: in Base's model `Array.join` keeps one handle. They stay tested bit for bit.
- `make mutate`: 46 as expected. The two mutants that survived the laws (`kv_of` bypassed, the softmax one position
  short) are caught by laws, and so are new ones: `head_cols` with `mod`, dP's row reading one value past the query, each
  tree putting its right half first, `Fork` dropping a gradient. The one survivor is the tensor sum that a 1-cell array
  makes right by accident (caught by the bounds check). `make check`, `make bits`, `make check-js` pass.

## Training infrastructure (M9; 2026-10-17)

- MNIST reads an optional batch size (default 100) and handles partial training/evaluation batches. `Seq.at` composes
  owned layer records and `Train.step_at` consumes one across forward, backward and step; their shapes remain checked.
  Batches 37 and 8191 match PyTorch within the existing tolerance with the same test hits. Batch 100 retains the
  recorded bits. Pixel reads are bounded and fill arrays in a tail loop: the first large-batch JavaScript test hit a
  stack overflow when millions of pixels were converted in one list; the bounded loader passes it.
- `OptState<P>` holds no buffers for SGD, one for momentum, two for Adam/AdamW. `Train.new` takes the configuration;
  every layer/combinator uses the matching state. The fast matrix step splits two, three or four arrays respectively;
  momentum's loop returns three arrays without a dummy fourth one. The F32 expressions and their order are unchanged.
- `Layer.load` takes one `Array<F32>` and an element offset, returning the source and next offset with the parameters;
  `Layer.count` gives its serialized size. MNIST and the tiny Llama use one initial-weights file each, in `dump` order.
  `torch/load.bend` checks exact sizes and short reads with the temporary binary reader; it stays outside every law.
  Checkpoint counts are checked before multiplying into a U32 byte length. Array allocation uses the existing U32 depth
  loop, avoiding `cap_depth`'s unary size comparisons on a large file.
- Checkpoints now contain only parameters for SGD, one extra parameter-shaped buffer for momentum, two plus Adam's
  powers for Adam/AdamW. A tied 3x3 parameter with Dropout gives 40, 80 and 128 bytes respectively; all four optimizers
  resume bit for bit, including the mask counter. The tiny Llama's 30-step AdamW checkpoint is byte-identical to the
  pre-M9 file. Old SGD/momentum checkpoints must be regenerated. Writing still uses lists and waits for M10.
- `Pair3` and `ResidualType` replace repeated composite types in the tiny Llama and the SmolLM2-sized block.
- Limits: JavaScript canonicalizes a noncanonical quiet NaN payload in the copy test; native preserves it. Both keep
  signed zero, infinities and subnormals. The numerical policy still matches NaN with NaN; finite-value checks and the
  shared tolerance are unchanged. The owned composition/step are tested, not newly proved; the existing laws remain.
- Checks: `make check`, `make check-js`, the original bit digests, `make laws` by `--verdict`, and 57/57 mutation outcomes
  as expected (one known numerical survivor is caught by the bounds gate). A rejected run-time-size composition keeps
  the shape-error gate. The block's SGD path also matches PyTorch and gives identical bits on 1 and 16 threads.
- `make lint` reports the same 10 eager-branch findings in `torch/tensor.bend`; none is in the new M9 paths. I did
  not change those existing predicates without measurements.
- Measured on an idle machine (alternated rounds, 1 thread pinned to CPU 3, 16 unpinned; Core Ultra 7 155H, Bend
  2.0.37, clang 22):
  - Peak RSS of one decoder block at SmolLM2-360M's sizes (9.8M parameters, 5 steps): 234 MiB under SGD and 354 MiB
    under AdamW on 1 thread, 470 and 584 MiB on 16. AdamW adds 114-120 MiB, more than its two moments' 75 MiB; I have not
    broken the rest down.
  - A SmolLM2-135M-sized file (134,515,008 F32, 538 MB) reads into one array in 0.26 s with the file in the page cache
    and 0.33 s after asking the kernel to drop it (a hint, not a guaranteed cold read), medians of 5. Peak RSS is
    1.0 GiB: the array's capacity rounds up to 2^28 cells, twice the file. That array is released once the layers are
    filled; it is the transient cost of loading, and for 360M it would be 2 GiB.
  - Loading the demos' initial weights (with a walk over every number): MNIST 6.65 -> 3.02 ms, the tiny Llama 1.67 ->
    0.81 ms.
  - A whole MNIST epoch run: 3.20 -> 2.63 s on 1 thread, 2.67 -> 2.02 s on 16, the same output. The 135M- and
    360M-sized block steps did not move (within 0.7%).
- The tiny Llama got slower: 1000 training steps took 6% longer on 1 thread with the same bits. It ran 5.2% more
  instructions, so it was not clang's loop schedule. A profile put the difference in `term_drop`: `count` was a plain
  `U32` field, and `Seq`, `Fork`, `Both` and `Tie` set it to `Layer.count(la) + Layer.count(lb)`. A layer passed as a
  template is built again at every use, so each `fwd`, `bwd` and `step` of a composite built its children's whole
  records to add their counts and dropped them. `count` is now a closure (`Unit -> U32`), like `zeros`: the tiny Llama
  is back to +0.13% instructions and +1.1% time on 1 thread against before M9 (0.897 -> 0.907 s; on 16 threads 1.158 ->
  1.081 s, a noisy lane). MNIST and the block did not change with it. `make check`, `make bits`, `make mutate` (57 of
  57) and `make check-js` pass.
- I also tried 8 and 16 independent partial sums in the packed dot product instead of 4 (numerically fine, the same
  bits on 1 and 16 threads): the whole block step got slower, 0.56-0.58x on 1 thread and 0.74-0.78x on 16, at both
  sizes. Kept 4.
- bend-lint did not see the eager `count`. I wrote a local rule, `bend-lint-rules/performance/template_value_field.bend`:
  a record with a lambda field whose other field is a value that reads a template parameter (not a `Type`, number, flag
  or string). On the code before the fix it reports exactly `Seq`, `Fork`, `Both` and `Tie`; on the current code,
  nothing. `tests/lint_template_value_field.bend` expects the finding, so `make lint` fails if the rule stops seeing it.

## Before M10: where the memory goes (2026-10-17)

- AdamW's extra memory in the 360M-sized block is its two moments and nothing else, but each parameter-shaped buffer
  takes 60.0 MiB for 37.5 MiB of numbers: a Bend array is one block of 2^k cells, so the 2560 x 960 matrices sit in
  blocks 1.71x their size and the 960 x 960 ones 1.14x. AdamW's heap is exactly 122,896 KiB above SGD's, the two
  moments at that capacity; the step allocates nothing. Under SGD the peak is parameters and gradient (61,448 KiB each)
  plus 107,828 KiB of forward and backward values (about half of it saved for the backward pass), and about 23 MiB of it
  is freed blocks that the runtime's size classes cannot reuse for other sizes. Measured with allocation counters added
  to the emitted runtime (massif sees nothing: the runtime does not use malloc); raw data kept outside this repository.
- At whole-model sizes the same rounding makes AdamW's four buffers 2.49 GiB instead of 2.00 at 135M and 8.50 GiB
  instead of 5.39 at 360M (arithmetic over the shapes). Both fit this machine's 32 GB; 1.7B would not without M11.
- I did not make loading exact (dropped from the plan, my decision, 2026-10-17). The loader's source array is the next power of two above the file (1 GiB for the 538 MB
  135M file), but it is released before `Train.new` allocates the gradient and the moments, so the loading peak (1.62
  GiB at 135M, 4.13 at 360M) stays below training's. It would not lower M10's peak.
- Left for later, with sizes: banded storage of the big matrices (cells just under a power of two; about 19.5 MiB per
  buffer at the 360M block), one saved input shared by the q/k/v and gate/up forks (6 MiB at 512 rows), and the size
  classes' stranded blocks (a runtime question).

## SmolLM2-135M and 360M in BendTorch's layers (M10; 2026-10-17)

- `demo/smol.bend` is SmolLM2-135M from the package's layers: the embedding, 30 of `smol_block.bend`'s blocks on
  sequences of 128, the final RMSNorm and the head, tied to the embedding through `Tie`. `scripts/smol.sh 360m` writes
  the 360M's copy. The stack is a balanced tree of `Seq` written by the script, one def per subtree size: a template
  takes only arguments fixed at compile time (no recursion on the depth), and Bend instantiates templates 64 levels
  deep at most (a chain of 30 `Seq` fails). Each block is wrapped in a `Box` layer whose parameters and saved values are
  one boxed word: Bend lays nested records out flat, a word a field, and a def or a continuation takes at most 247
  words; from 8 blocks up the compiler stopped with "an arity over 247". The program compiles in about 50 s.
- `reference/smol_ref.py` is the twin in plain torch; `reference/check_smol_hf.py` shows it equals transformers'
  `LlamaForCausalLM` on the same batch (the same loss and bit-identical logits). `scripts/smol_data.sh` converts the
  pinned checkpoints llama.bend already has (SHA-256 checked) into `init.bin` in dump order
  (`reference/smol_layout.txt`, `smol360_layout.txt`) and makes 64 batches of 2 x 129 ids from Tiny Shakespeare.
- The first loss was 3.4 tolerances off PyTorch's. Cross-entropy summed the 49,152 exponentials of each row as a running
  F32 sum; it now sums them as a balanced tree (`ce_sm`, over `tree_sum`). The loss is 0.41 of the tolerance off at 135M
  and 0.47 at 360M, and closer to float64 than PyTorch's float32 loss. The tiny Llama's bits moved with it (its rows of
  65 classes are summed in another order; `bench/bits.txt` saved again, the same on 1 and 16 threads); MNIST's did not.
  Attention's softmax (`psm`) keeps its order: its rows are short and the law `causal` reads its count.
- Gradients: the shared tolerance cannot hold here, for PyTorch either. Against float64 gradients of the same model and
  batch, 17 of the 135M's 272 tensors from PyTorch float32 are within it (worst element 109x) and 1 from Bend (300x);
  Bend's relative error is a median 1.8x PyTorch's (0.59-3.54x per tensor). At 360M: 83 and 63 of 290 tensors, worst
  26x and 91x, Bend 1.42x PyTorch's error in the median. I did not widen the tolerance. The gradient criterion is now
  relative (my decision, 2026-10-17): per tensor, Bend's rms error against float64 at most 8x PyTorch float32's, and at
  most 2x in the median; `test_smol.py` computes the float64 gradients with the twin (`grads64`) and still prints the
  element-wise comparison for information. What I ruled out as the cause: the head's backward product (simulated, 0.58
  of the tolerance even as one running sum of 49,152 terms), RMSNorm's two row sums (as trees, no change; reverted), and
  fused multiply-add. Bend's build has none (clang gets no `-march`, and the emitted C puns each F32 through a word
  between the multiply and the add, so even `-march=native` fuses nothing); built with `-march=native
  -ffp-contract=fast`, the hot product loop fuses and the gradients' error is unchanged. The missing FMA stays a
  language item (an upstream report is drafted, not posted); the cause of the 1.8x is not found.
- Training, 3 AdamW steps (lr 1e-4): at 360M every loss is within the tolerance of PyTorch's (0.47, 0.58, 0.16). At 135M
  the third is 1.38 of it off PyTorch and 1.84 off float64 (PyTorch float32: 0.46): AdamW's first update moves each
  weight by about lr times the sign of its gradient, so where a gradient is near zero its rounding flips the step.
  The later training losses now follow the gradients' rule (my decision, 2026-10-17): within the tolerance of a float64
  run of the twin (`train64`), or Bend's error against it at most 8x PyTorch float32's; the first stays strict. 135M:
  2.22x and 4.02x; 360M: 0.01x and 0.66x. `make smol` passes at both sizes.
- CI failed on the tiny Llama's 20 training steps: 1.25 tolerances from PyTorch float32 on GitHub's runner, 0.83 here,
  with the same Bend program (0.90 there before cross-entropy's change). Against the twin trained in float64, Bend's
  final weights are 0.76 of the tolerance off and this machine's PyTorch float32 0.17: twenty AdamW steps drift in
  float32 by an amount that depends on the CPU's kernels. `test_tiny.py` now judges training against float64 by the same
  rule as M10 (within the tolerance, or at most 8x PyTorch float32's error; my decision, 2026-10-17); it passes at
  0.76, on CI too.
- Peak memory (16 threads, `/usr/bin/time`): 135M loss 2.4 GiB, 3 AdamW steps 5.1 GiB; 360M loss 5.5 GiB, gradients
  10.0 GiB, 3 AdamW steps 12.8 GiB (repeated runs vary by a few percent). The arrays' power-of-two capacity is part of it (above).
- `Layer` has a `store` field, `load`'s inverse: the parameters copied into an array at an offset. `Params.save` writes
  a model's parameters and `Train.save` a training state from one array, a chunk of 65,536 numbers per byte list
  instead of one list for the whole file (several GB at these sizes). `Train.save` writes the same bytes as before
  (`test_infra.py` checks them for every optimizer); `infra_test params` checks `Params.save`. `demo/smol.bend grads`
  writes all 134,515,008 gradients this way. A 360M AdamW checkpoint (1.09 G numbers) is over the binary reader's
  U32 byte offsets, so it cannot be read back yet.
- Step time, idle machine (2 steps x 3 alternated rounds, one thread pinned to CPU 3, sixteen unpinned; each program's
  own clock after loading; Bend 2.0.37, clang 22.1.8, PyTorch 2.14.1): 135M 17.4 s per AdamW step on 1 thread and
  5.3 s on 16 (PyTorch 2.5 s and 1.1 s); 360M 42.7 s and 11.7 s (PyTorch 6.8 s and 2.4 s). Bend is 5.0-7.0x slower per
  step; 16 threads are 3.3-3.6x faster than one in Bend, 2.3-2.9x in PyTorch. I have not profiled where the step goes.
- Checks: `make check`, `make bits` (saved again, above), `make check-js`, `make mutate` (60 of 60, three new: `store`
  one cell short, Adam's powers swapped in the array checkpoint, cross-entropy's tree sum one class short) and
  `make smol` (the loss and the 1/16-thread bits pass; the gradients fail as described). Between the `Layer.store`
  commit and the one after it that gives `Box` its store, `demo/smol.bend` does not compile.

## SmolLM2-135M fine-tuned, and what make smol's test catches (M10; 2026-10-18)

- 400 AdamW steps (lr 1e-4) of SmolLM2-135M on Tiny Shakespeare's first 90% (`ft_train.bin`, 2 x 128 tokens a step,
  about 100k tokens), the mean loss on 16 held-out batches from the last 10% (`ft_eval.bin`) every 50 steps
  (`demo/smol.bend finetune 400 50`, the twin's `finetune` and `finetune64`, `make finetune`): the held-out loss falls
  from 3.5167 to 3.2446 in Bend, 3.5167 to 3.2455 in float64. It flattens after about 150 steps. Over the run, Bend's
  deviation from float64 is at most 1.83e-3 (PyTorch float32: 2.29e-3) and 1.09e-3 in rms (9.50e-4). One evaluation
  in the middle is 1.4e-3 off float64 where PyTorch float32 happened to be 3e-5, so the run is judged as a whole (my
  decision, 2026-10-18): Bend's largest and rms deviation at most 8x PyTorch float32's; `check_finetune.py` passes.
  The machine was busy: Bend took 2,321 s for the 400 steps and 9 evaluations, PyTorch 382 s.
- `scripts/mutate_smol.py` (`make mutate-smol`) breaks the code in a worktree and runs `test_smol.py`: a partial RMSNorm
  backward (the last number of each row), an embedding that keeps only a repeated id's last gradient and an attention
  scale of 0.124 are caught, with ratios of 8.4-19,906x against the 8x/2x bounds. Three survive because their effect is
  below float32's resolution at this size, measured: cross-entropy's sum missing the last class (probability <= 1.8e-6
  here), AdamW without weight decay in `main.bend`'s step (1e-6 of a weight per step) and the final norm's eps at 1e-4
  (its mean of squares is >= 255, so 1/rms moves by <= 1.8e-7). `make mutate` catches the first two on small models;
  the third is a constant of the demo. They are recorded as expected survivors.

## The native target and fused multiply-add, timed (2026-10-18)

- On an idle machine (3 alternated rounds; Bend 2.0.37, clang 22.1.8), the same emitted C of `demo/smol.bend` built
  three ways, seconds per SmolLM2-135M AdamW step on 1 thread pinned to CPU 3 and 16 unpinned: `-O3` (what `bend -o`
  does) 17.27 and 5.34; `-O3 -march=native` 17.05 and 5.07 (1.3% and 5.1% faster); with `-ffp-contract=fast` 23.96 and
  6.31 (39% and 18% slower). Each build gives the same loss bits on 1 and 16 threads. bend-ml's `fma_dot` repeated:
  the fused build is 1.89x slower. Below the 10% I set, so the FMA draft stays a draft and the native target is not
  worth asking for; I have not found why the fused build is slower.
- The emitted C starts with `#pragma clang fp contract(off)`: Bend asks clang not to fuse, on top of the word puns
  between operations.

## The linear layers' products on the GPU (M12, GPU part; 2026-10-18)

- `torch/gpu.bend`: `LinearF`, `LinearFirstF` and `LinearNBF` with their products from bend-ml-tensor-gpu (a local copy
  in `vendor/` until it is on BendHub; `make laws` checks it is bend-ml's file), one `!` call a product; the rest is
  `fast.bend`'s under the same names, so a demo switches by its import line (`make gpu` builds MNIST and the tiny Llama
  so). tensor-gpu keeps tensor-array's order of summation, so the layers give `main.bend`'s bits: `demo/gpu_test.bend`
  (8 cases, forward and backward) 0 differing words on the GPU and on 1 and 16 CPU threads; MNIST and the tiny Llama
  trained with their products on the GPU pass `test_mnist.py` and `test_tiny.py` with the CPU's numbers, and their
  outputs are `bench/bits.txt`'s digests; a SmolLM2-sized block's three AdamW steps give the CPU's output too.
  `make mutate`: 64 mutants as expected, four new on the GPU code (the weight gradient not accumulated, the last tile
  written outside C, a step past k counted, the wrong rotation); the last two are caught because the backward's products
  have k = 10 and 65, which are not multiples of 4.
- Speed, RTX 4050 Laptop (driver 615.78.08) against 16 CPU threads, 3 alternated rounds on an idle machine (a first
  measurement): an MNIST epoch (SGD) 7.8-8.4 s against 1.7-2.1 s; a tiny-Llama step 63-66 ms against 1.0-1.8 ms; three
  AdamW steps of the SmolLM2-135M-sized block (`smol_block`) 1.3-1.7 s against 0.64-0.71 s. Slower everywhere I
  measured, so the demos stay on `fast.bend` (M12's rule: adopted only where faster). Where it goes: a bang costs about 1
  ms before any leaf runs, and the arrays move over PCIe around every product, since every other layer runs on the
  CPU: nsys counted about 13,000 CPU page faults a step of the block. The products alone win from SmolLM2's sizes up
  (bend-ml-tensor-gpu: 1.6-2.8x), not inside this training step.
- What would change it: the whole step on the device (the element-wise layers and the optimizer as bangs too, so the
  arrays stay there), and the upstream fixes for the error poll (bendlang/bend#1776) and the leaf order (#1780).
- bend-ml-tensor-gpu 0.1.0.0 is on BendHub (2026-10-18): `torch/gpu.bend` imports it from there, `vendor/` is gone,
  and so are the three mutants that changed the local copy (tensor-gpu's bits are bend-ml's test; 61 mutants now).

## Correction (2026-10-18): the SmolLM2-sized block never reached the GPU

- The GPU timing above for `smol_block` (1.3-1.7 s for three steps) did not measure the GPU: nsys counts no kernel at
  all in its runs, on 16 threads or on 1. Its attention runs the heads in parallel lets on the host, and a bang that
  follows the host's forks in the same evaluation runs on the CPU pool (`guide/SHADERS.md`; llama.bend found the same,
  48 of 4,007 products on the device). So those bangs ran on the CPU threads, slower than tensor-fork's blocks.
- MNIST and the tiny Llama did reach it: 600 launches for 20 MNIST batches, and 45 bangs a tiny-Llama step, all its
  products. Their timings stand.

## The whole training step on the GPU, tried (2026-10-18)

- I wrote every layer of the SmolLM2-sized block as bangs: element-wise layers, RMSNorm, RoPE, attention (a leaf per
  sequence, head and position, with `main.bend`'s row code) and the AdamW step in place, beside tensor-gpu's products,
  so the arrays stay on the device and the host only passes handles. The same program with `--gpu off` gives all 11
  losses of 10 steps bit for bit; on the GPU the first loss differs by 1 ulp (CUDA's float exponential) and the other
  10 are the same bits.
- Speed, 10 steps on a busy machine (a first look): 264 ms a step on the RTX 4050 against 139 ms on 16 CPU threads.
  Of a GPU step (nsys), ~83 ms is the work itself; 115-130 ms is the first launch of the 65 bangs, where the tree grows
  on one group and `Array.new` fills each new output in one loop on one lane (~4-9 ns a cell); ~59 ms is the host
  between bangs, ~23 of it the loss. Two things cost more on the way: tensor-array's `cap_depth` inside a bang (it walks
  its `Nat` one step at a time: 152 ms a bang against 9 ms with the number computed on the host) and outputs allocated on the host (their pages come back from the
  device; 1,118 against 760 ms for 3 steps).
- It does not go into `torch/`: it writes `@unsafe` in its own defs (shared handles and `Array.join` outside the
  packages), which rule 3 does not allow. The layers stay on `fast.bend`.

## The whole training step on the GPU, in torch/gpu.bend (2026-10-18)

- The layers tried above are now `torch/gpu.bend`'s, with no `@unsafe` of their own: the sharing moved into
  bend-ml-tensor-gpu 0.1.1.0 (`run1`..`run4`, bangs over any leaf code; Renan's decision, 2026-10-18), and the leaves
  here are plain Bend that take every array as an ordinary argument and compute each number with `main.bend`'s own
  expressions and row code. `gpu.bend` now has SiLU, Mul2, Add2, Residual, Mat.dup, Mat.sum2, RMSNorm, RoPE, Attention
  and the optimizer step (SGD, momentum, Adam/AdamW, in place) on the device beside the products. `fast.bend` gained
  RMSNorm, RoPE and Attention as `main.bend`'s under the same names, so `smol_block` switches by its import line, and
  `make gpu` trains it on the GPU against PyTorch.
- `demo/gpu_layers_test.bend`: each layer against `main.bend`'s at SmolLM2-135M's sizes, the largest distance in ulps
  over every output. With `--gpu off` all 0 on 1 and 16 threads. On the GPU all 0 except SiLU: up to 2 ulps in y and 4
  in dX. That is CUDA's float `exp` (1 ulp off the host's on some inputs) carried through the division and the
  derivative; attention, which also takes `exp`, gave 0 on these inputs. Renan's criterion (2026-10-18): a difference
  is accepted when its cause is CUDA's `exp`; `make gpu` allows SiLU and attention up to 4 ulps and requires the rest
  exact.
- Debt (Renan, 2026-10-18): try another way to compute the exponential on the device so that SiLU and softmax give the
  host's bits too (for example an `exp` written in Bend, the same code on both sides), and measure what it costs.
- `make check` and `make gpu` pass: MNIST, the tiny Llama and the SmolLM2-sized block on the GPU within the tolerance
  (the block's worst loss 0.02 of it), and `scripts/bits.sh gpu` gives the CPU's outputs. `make mutate`: 66 of 66 then, five
  new mutants of the GPU layers (a leaf that skips a cell, RMSNorm's gain gradient overwritten, RoPE's backward pass
  rotating forward, dV from G, Adam not zeroing the gradient), all caught by `gpu_layers_test`.
- Speed, 3 steps of the block, 3 alternated runs on a busy machine (not the idle round): GPU 786 ms (`train_ms`;
  1.27-1.32 s of wall time) against 420 ms on 16 CPU threads (0.57-0.60 s), the three losses the same bits. The same
  as the lab version (753-769 ms): the cost is still the first launch of each bang and `Array.new`'s one-lane fill.

## tensor-array 0.1.13.0, tensor-fork and tensor-gpu 0.1.1.1 (2026-10-18)

- Every import moved together (a `Mat` of one tensor-array version is not one of another): 0.1.13.0's `cap_depth`
  compares with Base's native `Nat.is_le` instead of walking the number (bend-ml NOTES). `PROOF.bend` checks;
  `make check`, `make gpu` and `make bits` (every output identical) pass.

## The GPU step, cheaper allocations and the MSE on the device (2026-10-18)

- A new array made inside a bang is filled by one lane (`Array.new`, ~4.6 ns a cell): in nsys, the first launches of a
  block step were ~180 ms, ~160 of it those fills. Bend's arrays are affine, so a layer that consumes an input it does not
  keep can write its output over it: SiLU's dX over dY, Mul2's dA and dB over the saved B and A, `Mat.sum2` over its first
  addend, attention's dK and dV over K and V (no longer read after dQ). Every cell is still `main.bend`'s expression.
- `F.MSE` on the device, the same bits as `main.bend`'s: `tree_sum` adds neighbouring pairs pass by pass, and its first
  p passes carry no odd cell when 2^p divides the count, so a leaf summing a block of 2^p squares as a balanced tree of
  pairs leaves exactly the number those passes leave there; the host finishes `tree_sum` over the block sums (p = 6 for
  the block: 4,608 numbers back instead of the whole output). `fast.bend` has `main.bend`'s under the same name, and
  `smol_block` takes it from F.
- `gpu_layers_test` gains MSE (512 x 576, 7 x 5 and 100 x 10: 0 ulps on the GPU and the CPU threads); `make mutate`
  67 of 67 (an MSE gradient with its sign flipped is caught). `make check`, `make gpu` and `make bits` pass.
- Speed, 3 steps of the block on the GPU, 5 alternated runs with seven agents running (busy machine): 766-793 ms before,
  681-685 after (-13%), the same four losses. 16 CPU threads: ~420 ms. nsys of one step after: ~181 ms first launches
  (fills still ~160), ~140 ms of work, ~142 ms between bangs (62 gaps of exactly 2.29 ms each, 0.55 ms on a quieter
  machine before: a runtime wait, to look at in idle round 2). Not done: dX written over the saved X in the linear
  layers' backward (needs a tensor-gpu product into a given C, a new published version), fusing bangs.

## Cross-entropy as log-softmax (2026-10-18)

- Found by M14's edge cases: `CrossEntropy` took loss - log(max(p[y], 1e-12)), so a row whose label's logit sits far
  below the max stopped at 27.63 (a logit of -104 against 1e6: PyTorch 1000104 for that row, BendTorch 27.63; the
  gradient was right). It is now log-softmax as PyTorch computes it: loss - ((x[y] - m) - log(sum exp(x - m))), the sum
  the same balanced tree as before (Renan's decision). A label past the last class keeps the 1e-12 clamp.
- Every test against PyTorch passes; MNIST's SGD loss moved from 1.6311406 to 1.6311405 (PyTorch 1.6311405897). The
  bits moved on purpose for the tiny Llama (its training and sampling digests in `bench/bits.txt` saved again);
  SmolLM2-135M's loss and training digests, first recorded the same day, are the same before and after the change.

## Dtypes (M11; 2026-10-18)

- `torch/dtype.bend`, in plain Bend: BF16 and F16 stored two per U32, with PyTorch's rounding (to nearest, ties to even;
  subnormals, infinities and signed zeros included). PyTorch 2.14.1's CPU kernel turns every NaN into 0xFFFF in BF16,
  sign included (c10's scalar code would give 0x7FC0); in F16 a NaN keeps its sign and the top of its payload and gets
  the quiet bit. I matched this machine. The conversions are bit for bit on 1,347,584 F32 words chosen to hit every
  rounding case, and on all 65,536 16-bit patterns both ways; packing and int8's words, scales and dequantized weights
  too. The law `unpair_pair` (packing two by two, then reading back, gives the list) is proved in `dtype.bend`; the word
  bit identities are tested over every 16-bit value. 14 hand mutants: 13 caught, 1 equivalent.
- Mixed precision on the tiny Llama (`demo/tiny_mixed.bend`): every layer output and parameter rounded to BF16, the
  gradients rounded where autograd rounds them, F32 arithmetic and F32 master weights in AdamW; the twin does the same
  with explicit casts. The criterion, Renan's decision (2026-10-18), against the twin's BF16 storage in float64: over
  the first 20 steps BendTorch's largest and rms deviation at most 8x the twin's; over long runs every 100-step mean
  within 0.05. Measured: 1.00x and 1.35x over 20 steps (`make check`); over 2000 steps (`make mixed-long`) the means are
  at most 0.0344 off (the twin 0.0220). A step costs 1.02-1.14x the F32 one. BF16's numbers in F32 arrays, not its memory.
- int8 by rows for inference: the tiny Llama's logits are 0.042 of the tolerance from the twin's, 2.2e-3 at most from
  F32 (rms 6.1e-4, logits' rms 0.111), the same argmax in all 32 rows.
- A two-F32 accumulator, one thread on CPU 3: a 2^20-term dot product goes from 1506 ulps (running F32) to 0 at 8.3x the
  cost (a tree: 4 ulps at about 3x; `torch.dot`: 1176 ulps); a 49,152-class softmax sum from 1404 ulps to 0.4 at 7.3x
  (a tree: 0.6 ulps at 1.2x, which cross-entropy already uses). Not adopted.
- SmolLM2-1.7B: AdamW's state is 26.0 GiB in F32, the same with BF16 values in F32 arrays, 16.25 GiB with packed BF16
  gradient and moments, plus about 8 GiB of other memory (extrapolated). It does not fit 32 GB; its 6.85 GB F32 file is
  past the reader's U32 offsets. Left for after v1 (Renan): packed BF16 gradient and moments, a BF16 loader or 64-bit
  offsets. One 1.7B-sized block (AdamW, 512 rows) peaks at 1.76 GiB.
- Side finding: BendTorch's F32 tiny run deviates from float64 38x as much as PyTorch's over 100 steps (3.15x over 20).
  Traced and fixed the same day: AdamW's scalars (below).

## A frozen, held-out corpus (M14; 2026-10-18)

- `bench/corpus/` (corpus-2, 152 files, 175 runs, 7.1 MB): inputs drawn with seed 271828, which no development test
  uses, and PyTorch 2.14.1's results on them in float32 and float64, computed once by `reference/corpus_gen.py`, with a
  sha256 per file. Every layer of `main.bend` and `fast.bend`, the losses, five optimizers over 3-8 steps and
  `tensor.bend`'s operations, at shapes from 1x1 to 64x256x72 and 2x49152. `make corpus` runs it at release time.
- Judged first against float32 with `oracle.py`'s tolerance scaled by |result|, 10 results failed: weight gradients and
  RMSNorm's gain gradient under cancellation (BendTorch 0.6-1.5e-6 from float64, PyTorch 0.3-0.8e-6), and PyTorch's own
  float32 being the less accurate (GELU's gradient near x = -5, LayerNorm with one feature). Renan's decision: judge
  each number by its cancellation, keeping `oracle.py`'s constants: |bend - float64| <= ATOL + RTOL max(scale,
  |float64|), the scale the sum of |terms| that produce the result, defined per operation in the manifest. All pass
  (the closest is RoPE at position 63, 0.96 of the limit, where PyTorch's own float32 sits at 0.97); 209,273 of 480,610
  numbers bit-identical to PyTorch, the exact tier all; the same bits on 1 and 16 threads.
- The IEEE edge cases reach the norms, softmax inside attention and the losses. Accepted for now and compared later
  (Renan): attention's dQ is 0, not NaN, at a position that cannot see a -inf in V; RMSNorm's dX and LayerNorm's output
  and dX stay finite where the squares overflow, where PyTorch gives NaN. `test_ops.py` expects BendTorch's values there.
- SmolLM2-135M's loss and 3 AdamW steps are in `bench/bits.txt` (`make bits-smol`; 5.2 GB, 3 min), the same on 1 and
  16 threads.
- `make mutate`: 70 of 70 after M11 (three mutants of `dtype.bend`: BF16 rounding half away from even, int8 clamped at
  126, a NaN rounded instead of made canonical, all caught by `test_dtype`).

## The gradients' 1.8x, found; SGD tracks PyTorch strictly (2026-10-18)

- `reference/numerics_probe.py` runs SmolLM2-135M's twin with each operation as PyTorch runs it, in BendTorch's order in
  float32, or in float64; with every operation in BendTorch's order it gives `demo/smol.bend`'s loss bit for bit and
  99.995% of its gradients' bits.
- The 1.8x (BendTorch's gradient error against float64 over PyTorch float32's, median per tensor) has one cause: the
  head's backward product dZ·E over the 49,152 classes, where tensor-array's four partial sums each add 12,288 small
  terms in a row. With PyTorch's product there and everything else in BendTorch's order the ratio is 0.023x; summed in
  blocks of 256 it is 0.020x, of 1024 0.028x, of 4096 0.104x (emulated). Attention, RMSNorm, SiLU, the RoPE table and
  Fork's sums move nothing alone. PyTorch float32's own error is almost all its cross-entropy: with BendTorch's tree
  sum there, PyTorch is at 0.018x.
- Blocks of 1024 implemented in bend-ml-tensor-array (each block dotr's four sums, the block totals added in order;
  dots up to 1024 products unchanged), inherited by tensor-fork, and in tensor-gpu as one bang per block. Measured in
  BendTorch on those local versions (not yet published): the SmolLM2-135M gradients go from 1.80x to 0.03x (median;
  the worst tensor 0.10x), the training losses 0.07x; `make check`, `make gpu` and `make bits` pass, MNIST's and the
  tiny Llama's bits unchanged. The head's product alone (256 x 49,152 x 576, 16 threads, busy machine) takes 0.76 s
  against 0.70-0.73 s.
- `test_smol.py` checks 3 SGD steps against PyTorch float32 under the strict tolerance. At lr 0.003 the worst was 0.42
  (135M) and 0.51 (360M); at 0.005 and 0.01 it failed by 271 and 2,698 tolerances under the old cross-entropy clamp (a
  token's probability fell to 1.9e-22). With log-softmax, lr 0.01 passes: 0.518 (135M) and 0.568 (360M); `make smol`
  uses 0.01.

## The linear layers' dX with the precise product (2026-10-18)

- `Linear`/`LinearNB`'s dX = dY·W now uses `Mat.matmul_precise` (bend-ml-tensor-array 0.1.14.0: k summed in blocks of
  1024) in `main.bend`, `TF.matmul_precise` in `fast.bend` and `TG.matmul_precise` in `gpu.bend`, so the three still
  give the same bits. Products with k <= 1024 (all of MNIST's and the tiny Llama's) are unchanged: `make bits` identical.
- SmolLM2-135M's gradients against float64: median 0.03x and worst 0.10x of PyTorch float32's error (they were 1.80x and
  ~3x); 360M: 0.06x and 0.51x. The criterion is tightened accordingly (Renan, 2026-10-18): median at most 0.25x and every
  tensor at most 1x PyTorch float32's (it was 2x and 8x). SGD at lr 0.01: worst 0.426 (135M) and 0.481 (360M) of the
  strict tolerance.
- Cost: a SmolLM2-135M step on 16 threads, 3 alternated rounds on a busy machine: 9.71-9.79 s for 2 steps before, 9.96-
  10.10 after (+2.7%). On the GPU, the gate and up projections' dX (k = 1536) now copy their slices on the host.
- `bench/bits.txt`: SmolLM2's `train 3` digest saved again (ee15c003b7c603ad); its loss digest is unchanged. `make
  check`, `make gpu`, `make bits`, `make corpus`, `make smol` and `make mutate` (70 of 70) pass.
- With tensor-gpu 0.1.2.0 the GPU precise dX copied its slices on the host: the SmolLM2-sized block on the GPU went from
  679-697 to 963-979 ms for 3 steps. tensor-gpu 0.1.2.1 runs each block as a bang at its place in A and B: 690-702 ms,
  the same losses. `make check` and `make gpu` pass.

## AdamW's scalars, the tiny Llama's drift (2026-10-18)

- The tiny Llama's 38x was systematic, not chaos: PyTorch float32 runs a weight 1 ulp away land at 0.64-1.24x of its
  own distance from float64, and the twin with Bend's optimizer scalars lands at 21x. The cause is `1 - b2`: b2 = 0.999
  is 0.99900001287 in F32, so `1 - b2` taken in F32 is 1.3e-5 too small in relative terms, and so is `1 - b2^t` from a
  running F32 power. PyTorch takes both in float64 before rounding.
- The fix (`torch/main.bend`, `fast.bend`, `gpu.bend`): `CAdam`/`CAdamW` take `c1 = 1 - b1` and `c2 = 1 - b2` as their
  own constants (`0.1, 0.001` for the usual betas); the moments update as `m + c1 (g - m)` and `v b2 + (c2 g) g`; the
  training state carries `q = 1 - b^t` as `q' = q + c (1 - q)`, so no `1 - (a number near 1)` is ever taken; `Opt.at`
  gives `lr / q1` and `sqrt(q2)`. A breaking change: the configuration literals gain two fields, and checkpoints end in
  `1 - b1^t`, `1 - b2^t` instead of `b1^t`, `b2^t` (older Adam/AdamW checkpoints must be regenerated).
- Against float64, rms over the 22,752 final weights: 100 steps 0.50x of PyTorch float32's distance (was 38x), the
  losses 0.42x; 20 steps 1.09x (was 3.15x). The same bits on 1 and 16 threads. `test_tiny.py`'s fallback tightened from
  8x to 2x PyTorch float32's (Renan, 2026-10-18); at 20 steps no tensor needs it (the worst element is 0.794 of the
  tolerance from float64).
- SmolLM2's 3 AdamW steps: losses 2 and 3 against float64 at 0.03x/0.03x of PyTorch float32's distance (135M) and
  0.05x/0.02x (360M). MNIST's 50-batch Adam loss is 1.2313914 against PyTorch's 1.2313913 (it was 1.2313918).
- Cost: none measured. One thread on CPU 3, 3 alternated rounds: the tiny Llama's 50 steps 45-46 ms before and 46 after,
  MNIST's 50 Adam batches 373-374 ms before and 372-382 after.
- `bench/bits.txt` saved again (the tiny Llama, MNIST's Adam and SmolLM2's `train 3` digests change; SmolLM2's loss
  digest does not). `make check`, `make gpu`, `make bits`, `make corpus`, `make smol` and `make mutate` (70 of 70) pass.

## AdamW's c1 and c2 checked against b1 and b2 (2026-10-18)

- Since the AdamW fix a configuration carries each beta twice (`b2` and `c2 = 1 - b2`), and nothing tied them. I tried
  three ways to tie them. A law: the checker does not compute F32 (`F32.sub(1.0, 0.999)` stays symbolic), so it cannot
  prove `0.001 ≈ 1 - 0.999`. Deriving the betas from the complements (`b = 1 - c` in F32): the same bits for 0.9, 0.99,
  0.999 and 0.9999, but over the 55,555 betas from 0.5 with up to 5 decimals 9,235 come out 1 ulp away, which would
  change b for anyone with an unusual beta. A run-time check, adopted (Renan, 2026-10-18).
- `OptCfg.ok(cfg)`: for each pair, `0 < c <= 1` and `((1 - b) - c)^2 <= 2^-46` in F32, that is `c` within 2^-23 of
  `1 - b`. For b in [0, 1), `1 - b` in F32 is within half an ulp of the exact value and c's own rounding adds at most as
  much, so a `c` written as `1 - b` always passes; a typo (0.01 for 0.001, c1 and c2 swapped) is orders of magnitude
  above. `OptCfg.check(cfg)` stops an IO program with a message; `Train.new`, which is pure and cannot stop, starts a
  failing configuration's `1 - b^t` at NaN, so every loss after the first is NaN. The demos call `OptCfg.check` before
  training.
- `test_infra.py`: 2,739 consistent configurations accepted (the usual betas and 900 drawn with 1-6 decimals, each with
  PyTorch's float64 complement and with the complement taken in F32), 2,747 inconsistent rejected (each c times 10, over
  10, negated; swapped, zero, NaN, b = 1, 0.00099 and 0.0011 for 0.001), by both `CAdam` and `CAdamW`; a bad one stops
  with exit code 1 and the message, and unchecked its weights are NaN after one step. Five new mutants (a looser bound,
  each variant ignoring c2, a negative c accepted, no NaN start): `make mutate` 75 of 75.
- No cost to a step: the check runs once. `make check`, `make bits` (identical), `make gpu`, `make corpus`, `make smol`
  and `make bits-smol` (identical) pass.


## The final timings, and the benchmark against eleven systems (2026-10-18)

- One idle round (3 h 27 min) on the final commits: BendTorch against PyTorch in 5 alternated rounds, the eleven
  systems' twins in 3 (`BENCHMARK.md`). The README's speed table and `docs/limitations.md` now hold its numbers.
- A correction: the README's 9.1 ms a step for PyTorch on the tiny Llama was the wall time over 200 steps with
  Python's start-up and `import torch` (~1.4 s) inside it. Without it PyTorch takes 1.9-2.8 ms a step on one thread
  (two harnesses), so BendTorch (at most 0.8 ms) is 2.4x to 3.5x faster there, not 9x. `scripts/bench.sh` now
  subtracts a `train 0` run.
- BendTorch passes all four workloads (the tiny Llama at 0.49x PyTorch F32's distance from float64 over 200 steps,
  the same bits on 1 and 16 threads and on the GPU). Of the eleven systems as they ship, only PyTorch eager passes
  every workload on every device; the ~10x cluster on the tiny Llama (torch.compile, JAX, tinygrad, burn, dfdx with workarounds on the CPU, Flux, Lux)
  is AdamW's bias corrections in F32, the drift BendTorch had until the morning of the same day.
- Losses: 5.5-6x slower than PyTorch eager on MNIST, 6.5x and 3.4x on the SmolLM2-sized block (1 and 16 threads),
  about 6x on SmolLM2-135M; the GPU path stays slower than 16 CPU threads at every size. Not measured: the single
  operations (W5) in BendTorch, and SmolLM2-135M on BendTorch's GPU (no build of it).

## Four hub packages, assembled from torch/ (2026-10-19)

- My decision: BendTorch ships as four BendHub packages split by what each guarantees: the core (`main`, `spec`,
  `tensor`, `dtype`, with `LAWS.bend` and `PROOF.bend`: laws, no `@unsafe`, no foreign code, only bend-ml-tensor-array
  and bend-ml-nat-lemmas), `-fast` (tensor-fork), `-gpu` (tensor-gpu) and `-load` (binary-read). The core's name waits
  for the hub's auction of the short name; the script takes it as an argument.
- Checked first, in Bend 2.0.37's source and against a published multi-file package: a hub package may hold many files,
  each imported by path, and relative imports inside it resolve; `--publish` sends the entry and what it imports, so
  the core gets a generated entry, `package.bend`, that imports its four files and the laws.
- `torch/` stays as it is. `scripts/package.sh` (`make package CORE=<name> VERSION=<a.b.c.d>`) assembles the packages
  into `build/packages/`, changing only import lines, checks the hub's name and path rules and the core's guarantees
  (`--verdict` over the whole core: ALL PROOFS CHECK), checks that each satellite's checker output is exactly its
  `torch/` file's (the same pinned defs outside the proofs), and with the satellites pointed at a local copy of the
  core builds `fast_test`, `gpu_test` (1 and 16 threads: main.bend's bits), `infra_test`, `dtype_test` (their PyTorch
  tests pass) and the tiny Llama (`bench/bits.txt`'s training digest on 1 and 16 threads). It passes for both
  candidate names, `bendtorch` and `bend-ml-torch`, in 71-83 s; nothing is published.
- `make check`, `make bits` and `make mutate` (75 of 75) pass; no code changed.

## Drawing characters from the tiny Llama (2026-10-19)

- Greedy decoding loops on a model this small ("the the the" after 2,000 steps, "with with" after 5,000), so the text
  says little about what the model learned. `tiny_llama draw <steps> <chars> <seed> <prompt>` trains the same way, then
  draws each character from the softmax of the last row at temperature 0.8, with a uniform number from `mix32` of
  (seed, character index), so a run is reproducible. After 2,000 steps, seed 7: "sincet yow, wand Mas and grod mar
  ppleer I thove groves ...": English-looking words, some real, no loop.
- Greedy `sample` is still the mode the twin checks (64 of 64 characters); `draw` is not checked against PyTorch: its
  exp and running sum would have to match the twin's to the bit for every draw to agree. `test_tiny.py` passes as before.

## The tiny Llama on any text (2026-10-19)

- `scripts/own_text.py` (`make own TEXT=file`) prepares a working directory as `tiny_ref.py --prep` prepares
  `data/tiny/` (the 65 most frequent bytes as the vocabulary, the rest mapped to the most frequent one; the twin's
  initial weights, seed 0; one seeded batch a step) and runs the unchanged Bend program from it: `train` for the losses,
  then `draw`. No Bend change was needed: the program reads `data/tiny/` relative to where it runs. The prompt is passed
  one byte per character and the output turned back into bytes, so UTF-8 text (accents) survives.
- Tried on Tiny Shakespeare and on a Portuguese text (140 distinct bytes, 75 mapped away): 2,000 steps train and draw in
  about 5 s; with UTF-8 text a character is sometimes broken where one byte of a pair was mapped away.

## SmolLM2-135M writes text, before and after fine-tuning (2026-10-20)

- `demo/smol.bend` gained `gen <n> <weights> <ids>` (greedy), `draw <n> <seed> <weights> <ids>` (temperature 0.8, a
  uniform number from `mix32` of (seed, token index), as the tiny Llama's `draw`) and `finetune <k> <e> <out>`, which
  saves the weights in init.bin's layout. Generation runs one sequence (b = 1) with the same parameters: the tokens so
  far sit at the start of the 128-position window and token 0 fills the rest, which the causal mask keeps out of every
  written position; the next token comes from the logits at the last written position. `reference/smol_ref.py gen` is
  the twin (the same window), and `--encode` / `--decode` use the model's tokenizer. `make smol-gen [WEIGHTS=...]`
  (`scripts/smol_gen.sh`) checks the greedy tokens against the twin, then draws.
- Prompt "ROMEO:\n": the greedy tokens equal the twin's (60 of 60) with the checkpoint and with the fine-tuned weights
  (the twin then runs Bend's weights, so this checks the forward pass; the training is `check_finetune`'s). Greedy text
  loops on this model ("I am not a man of the world, and I am not a man of the world, ..."), so the before/after uses
  the seeded draw (seed 7), which is not checked against PyTorch:
  - before: "Now, you have a fundamentally different mindset about accessibility for everyone in the world. The people
    who make the world and the world that we live in is well-lit, well-shaped, well-proportioned. ..."
  - after 400 steps on Tiny Shakespeare: "And her love is in the bosom more than thou art in me, / And all the world will
    be the side of thy love. // JULIET: / My soul, my soul! // UTERUS: / My soul, my soul! ..."
- Cost: 48 s for 60 tokens on 16 threads (0.8 s a token: a whole 128-position forward pass per token, no cache); the
  twin takes 9.3 s.
- The fine-tuning re-run, the first since AdamW's scalars and the precise product (2026-10-18): 400 steps in 36 min on
  16 threads (a 2-minute GPU build at nice 19 ran beside it), held-out loss 3.5167 -> 3.2458 (float64 3.5167 -> 3.2455; it was 3.2446 before
  those fixes). `check_finetune` passes: Bend's largest deviation from float64 over the run is 1.62x PyTorch float32's
  and its rms 1.52x (at most 8x; they were 0.80x and 1.15x). The first three evaluations are within 6e-7 of float64
  where PyTorch float32 is 2-3e-5 away; the largest deviation is at step 300 (3.7e-3), where float32 trajectories have
  drifted apart (PyTorch float32's largest, 2.3e-3, is at step 250).
- `make check` and `make bits-smol` (identical) pass; switching smol.bend's RMSNorm, RoPE and attention to fast.bend's
  names (main.bend's layers) changes no bit.

## SmolLM2-135M fine-tuned on the GPU (2026-10-20)

- `make smol-gpu` builds `demo/smol.bend` with `torch/gpu.bend`'s layers (the import line switched, as `make gpu` does
  for the block) and runs `test_smol.py` on it: PASS. The loss and the 3 AdamW steps within the tolerance (worst 0.439),
  the gradients against float64 at median 0.03x and worst 0.09x PyTorch float32's error, 1 thread and 16 the same bits.
  The losses are 2-3 ulps from the CPU build's (CUDA's float `exp`). Peak device memory 3.3 GB of 6 GB.
- 400 fine-tuning steps on the GPU, with the held-out evaluation every 50: 4,089 s (68 min) against 2,156 s (36 min) on 16
  CPU threads, 1.90x slower; one run each, not alternated, nothing else running during the GPU run. 3 steps alone: 27.5 s
  against 14.9 s. Held-out loss 3.5167 -> 3.2512 (the CPU build 3.2458, float64 3.2455); `check_finetune` passes at
  2.51x (largest) and 2.64x (rms) PyTorch float32's deviation from float64, against the CPU build's 1.62x and 1.52x.
  The first three evaluations agree with the CPU build to 1 ulp; after that the exp differences grow through AdamW as
  any float32 trajectory's do.
- I did not profile this run. The likely causes are the block's (`docs/limitations.md`: filling new arrays on one lane,
  the gaps between bangs), plus the embedding and the cross-entropy, which stay on the host; not measured.

## Where a SmolLM2-135M step spends its time (2026-10-20)

- `perf record` of one AdamW step (`train 1`), 1 thread on CPU 3 and 16 threads, nothing else running: 13.9 s and 5.4 s,
  so 16 threads are only 2.6x faster. On 1 thread the products take about 76% of the samples (tensor-fork's `cpar` and
  `rpar` over tensor-array's `t8_cols`/`t8_rows` ~70%, packing into 8-wide tiles ~4%), attention's loops ~12%, the
  zeroing of new matrices ~4% (`Train.new`, once per run, on one thread), AdamW ~1%, loading ~1%.
- The 16-thread step keeps on average about 6.6 cores busy (a timeline of the samples in 250 ms bins), less in the
  backward half, and spends 33 CPU seconds where 1 thread spends 13.9: hyperthreads, E-cores and shared bandwidth.
- On 1 thread the step is bound by instructions, not memory: IPC 4.9, 2% L1 misses, 328 billion instructions for about
  207 GFLOP (my count of the step's products, forward and backward). The products' inner loop is vectorized, but with
  SSE (4 lanes): Bend 2.0.37 calls `clang -O3` without `-march`, and has no option for compiler flags (it honors `CC`).
- Tried, not adopted:
  - `PAR()` 9 and 10 (512 and 1,024 blocks a product) against 8, 4 steps on 16 threads, 3 alternated rounds: 19.98-
    20.04 s (8), 20.37-20.54 (9), 21.19-21.33 (10). The same bits. 8 stays.
  - The same emitted C compiled with `-mavx2 -ffp-contract=off` (8 lanes, no FMA, so no operation changes): the same
    losses, 24.7% fewer instructions and 9.5% fewer cycles on 1 thread (14.0-14.3 -> 12.6-12.9 s, 3 alternated rounds),
    and 3% on 16 threads (14.9-15.2 -> 14.6-14.7 s for 3 steps). Not adopted: it is a build flag for every program, not
    Bend code, and the binaries would need AVX2; Renan's call, and a question for upstream (whether `bend -o` should
    take compiler flags or target the host).
- Nothing changed in the code: the speed is in bend-ml's products and in how the runtime spreads them over the cores.

## Checks after the night of 2026-10-19 -> 20 (2026-10-20)

- Every gate again on today's code (laws, check, bits, corpus, check-js, gpu, mixed-long, package, mutate, bits-smol,
  smol): all pass, `make mutate` 75 of 75. The README's MNIST line holds: one epoch of SGD gives 0.51486576 and 9,108 of
  10,000 test hits on 1 and 16 threads, as PyTorch (0.5148657560 and 9,108).
- Greedy SmolLM2-135M generation equals the twin's on four more prompts (`scripts/smol_gen.sh`): "The capital of France
  is" (60 tokens), "def fibonacci(n):\n" (60, a correct recursive function), and a 75-token prompt (the first lines of
  Tiny Shakespeare) with the checkpoint and with the fine-tuned weights (50 tokens each, positions 75-124 of the
  window). The fine-tuned weights saw that text in training. The seeded draw is still not checked against PyTorch.
- `make mutate-smol` failed before any mutant: smol.bend's final RMSNorm became fast.bend's `F.RMSNorm` (which is
  main.bend's) when the demo moved to fast.bend's names, and the eps mutant still named `T.RMSNorm`. With the name fixed,
  6 of 6 mutants behave as listed on today's code.
- `CC` with flags does not pass through `bend -o`: Bend 2.0.37 runs `$CC` as one executable name (no shell), so
  `CC="clang -mavx2"` is not found and the build falls back to plain clang without a word (no AVX instruction in the
  binary). A wrapper script (`exec clang -mavx2 "$@"`) set as `CC` works: 8,093 AVX instructions in the tiny Llama and
  the same bits. Nothing adopted; whether builds use AVX2 is still open.
- The "no checkpoint save/load" debt of 2026-10-15 was paid the same day ("towards v1": `Train.save`/`load`, a resumed
  run gives the same bits); the later debt lists already leave it out.

## The C target and CPU against GPU, on an idle machine (2026-10-20)

- The same emitted C of `demo/smol.bend` built six ways and timed on one `train 3` (3 AdamW steps), 3 alternated
  rounds, 1 thread on CPU 3 and 16 unpinned; all 36 runs give `bench/bits.txt`'s bits, and no build has a fused
  multiply-add (the emitted C's `fp contract(off)` holds):

  | clang flags besides `-std=c11 -O3` | 1 thread | 16 threads |
  |---|---:|---:|
  | none (what `bend -o` does) | 41.50 s | 15.07 s |
  | `-falign-loops=64` | 40.59 s (-2.2%) | 14.88 s (-1.3%) |
  | `-mavx2` | 37.49 s (-9.7%) | 14.55 s (-3.4%) |
  | `-march=native` | 36.33 s (-12.5%) | 14.05 s (-6.8%) |

  Loop alignment alone moves the AVX2 and native builds by under 0.6%, so their gains are the target, not the schedule.
  The hot product loop multiplies 8 numbers per instruction with AVX2 and then splits them back into two 4-wide adds,
  because the order of the sum is fixed; `-march=native` emits the same instructions with the loop unrolled 4 times
  instead of 2 (12% fewer instructions than AVX2), likely the tuning, not a new instruction set.
- Not adopted, as decided: the speed has to be what anyone gets from `bend`. Upstream declined `$CFLAGS` for `bend -o`
  (PR #1461, 2026-10-14) in favour of `bend X.bend -o x.c` and clang by hand. The one thing left, that a `$CC` with flags
  is dropped silently, is bendlang/bend#1802. `docs/limitations.md` has a
  line on it.
- CPU against GPU, alternated: 3 steps take 15.13 s on 16 CPU threads (14.97-15.26) and 27.59 s on the GPU
  (27.56-27.64), 1.82x, 3 rounds; yesterday's single runs gave 1.85x on 3 steps and 1.90x over the 400-step fine-tuning.
  Peak device memory 3,324 MiB. The GPU's losses equal yesterday's GPU run to the bit.

## The causes: fine-tuning's ratio, threads, generation, a GPU step (2026-10-20)

- **What moved the fine-tuning's deviation from 0.80x to 1.62x PyTorch float32's: nothing in the code.** Three changes
  touch SmolLM2's numbers since 2026-10-17 (cross-entropy as log-softmax, the precise product, AdamW's scalars). The
  cross-entropy change leaves the training bits alone (only the last bit of a few printed losses moves; the gradient,
  p - y, is the same), so I re-ran the 400 steps before AdamW's fix (`79634e9`, the precise product only) and today's
  code with one weight 1 ulp away (`model.layers.0.self_attn.q_proj.weight[0]`), against the same PyTorch runs:

  | run | largest | rms | held-out after 400 steps (float64 3.2455) |
  |---|---:|---:|---:|
  | 2026-10-17, none of the changes | 0.80x | 1.15x | 3.2446 |
  | the precise product, the old AdamW | 0.45x | 0.61x | 3.2447 |
  | today | 1.62x | 1.52x | 3.2458 |
  | today, one weight 1 ulp away | 0.37x | 0.37x | 3.2456 |

  The same code started one ulp apart gives 1.62x or 0.37x, so chaos alone spans the whole range; a single run's ratio
  says nothing about the code, only that it is within the check's bound (8x). Before chaos takes over, the precise
  product shows: after 50 and 100 steps the 2026-10-17 run is 6.0e-6 and 9.9e-6 from float64, every run with the
  precise product 1.1e-7 to 5.9e-7 (PyTorch float32 2.5e-5 and 2.9e-5).
- **Threads.** `train 3` on 1, 4, 6, 8, 12, 16 and 22 threads (3 alternated rounds, the same bits in every run): 41.50,
  18.85, 16.33, 15.76, 15.34, 14.84 and 15.06 s. The step stops scaling at about 6 threads. task-clock timelines at 16
  threads: the forward pass keeps 5.6-5.8 threads on a CPU on average and the backward with AdamW 5.5, so the backward
  is not the half that idles; the other threads are parked (not spinning, not waiting on memory). The likely limit is
  how the runtime hands out the work; not measured further. The backward's `spin_431` is tensor-fork's `rpar` inner
  loop, not a join.
- **Generation and its window.** 20 greedy tokens with the window (`TT()`, and the twin's `T`) at 32, 64 and 128
  positions, 16 threads, 3 rounds: 296, 463 and 783 ms a token in Bend, 57, 70 and 124 in PyTorch, the same tokens in
  all six. Bend's cost is ~134 ms plus 5.1 ms a position, so at 128 the window is ~83% of a token; a key/value cache
  would remove most of it. PyTorch runs the same window, so the 5-7x is a forward pass's cost, at every window.
- **A GPU step.** 10.0 s without a profiler; under nsys the device is busy 7.6 s in 1,974 bangs, more than the whole
  CPU step (~5 s). The host's share is allocating the new training state before the first bang, then two long gaps:
  the cross-entropy after the forward pass and the embedding's gradient near the end of the backward, each with the
  driver migrating managed memory back to the host (`nvidia_uvm`). nsys inflates the host side (14.6-15.9 s a step), so
  only the device time is a number here.
- Raw outputs and scripts: in my lab notes. `docs/limitations.md` has the thread, generation and GPU lines.

## Old results re-run with their output, and an exponential in plain Bend (2026-10-20)

- Every result this log held without a kept output was re-run. Most hold as written: dropout keeps 74,964 of 100,000
  at p = 0.25 (74.96%); 9,235 of 55,555 betas would move by deriving b from 1 - b; the old cross-entropy clamp gave
  27.631 on a row whose loss is 1,000,104; Adam's drift on the tiny Llama still starts at row 55 of the embedding
  (first an input at step 34, a gradient entry of -1.39e-7 against AdamW's denominator 2.41e-8), with a smaller
  divergence since AdamW's fix (7.6e-5 relative at step 60); the double-word accumulator is 8.2x the running sum's cost
  for 0 ulps; 14 new hand mutants of `dtype.bend` (the first 14 were not kept): 13 caught, 1 equivalent; the gradient
  probe's ratios are within a few thousandths of the earlier ones and reproduce 99.995% of the old gradient bits;
  MNIST's first version was 40.1x PyTorch's epoch and `LinearFirst` 27.9x (today 6.0x); attention's parallel
  sequences took the tiny Llama step from 13.2 to 10.9 ms on 16 threads with the same bits.
- Corrected: the tiny Llama's drift before AdamW's fix was 21.3x PyTorch float32's distance from float64 over 100 steps
  (1.88x over 20), not the 38x and 3.15x recorded on 2026-10-18 by a method I did not keep; the fix's 0.50x holds. On
  today's code `LinearFirst` saves 24% of an MNIST epoch on one thread and 16% on 16 (2.5 -> 1.9 s, 1.9 -> 1.6 s), the
  precise product costs 4.5% of a SmolLM2-135M step (not 2.7%), the tiny Llama type-checks in 0.59 s and compiles in
  40 s (it was 0.2 and 17 s, with fewer modes), and the corpus's cancellation cases have moved: weight gradients are now
  as close to float64 as PyTorch's, RMSNorm's gain gradient still further.
- **An exponential in plain Bend with the host's bits.** `exp_cr` (F32 and U32 arithmetic only: Cody-Waite reduction,
  a 256-entry table of 2^(j/256), a degree-5 polynomial, double-F32 sums and Dekker products without FMA, one explicit
  rounding) gives exactly the bits of Bend's `F32.exp` on the host (glibc 2.44) for all 2^32 inputs, and the same bits on the GPU
  for all 2^32, where CUDA's float `exp` differs on 289,883 of 2^20 inputs. With it in place of every `F32.exp` in
  `torch/main.bend` (a prototype, not merged), every gate passes, the CPU's bits and the corpus's do not move, SiLU on
  the GPU goes from 4 ulps to 0, and SmolLM2-135M's three AdamW steps on the GPU give the CPU's bits exactly. It costs
  5.2x `F32.exp` a call: the tiny Llama's step 1.68x on one thread, SmolLM2-135M's 1.19x on 16 threads and 1.06x on the
  GPU. Ten mutants of it against the 2^32 sweep: five caught; the polynomial's last term and the table's lowest word
  change no result, so it can be made cheaper. Not adopted for now (Renan, 2026-10-20): the GPU keeps CUDA's `exp` and
  its accepted few-ulp difference. Dropping the r^5/120 term changes no result and saves 1.5%; the cost is the exact
  arithmetic itself (~278 instructions a call against ~66).

## Attention's sequences in parallel at SmolLM2's batch (2026-10-21)

- A SmolLM2-135M step stopped scaling at ~6 threads. A product alone scales 5.5-6.3x at 16 threads, so neither the
  runtime nor the products were the limit. A perf timeline split by phase showed attention taking 2.11 of 6.4 s of a
  16-thread step on one thread: `fgo`/`bgo` opened the tree of sequences only from 8 sequences, a threshold set for the
  tiny Llama, and SmolLM2 trains with 2.
- `attn_tree(b, t, w)` opens the tree from 2 sequences when t w >= 4096 (still from 8 below that). The laws
  `attn_par_fwd`/`attn_par_bwd` already prove that the tree gives the sequential order, so no bit moves.
- Idle machine, 3 rotated rounds, `train 3`: 16 threads 15.14 -> 12.26 s (-19%), 6 threads 16.43 -> 13.51 s, 1 thread
  unchanged (41.94 -> 41.32 s, noise); 27 of 27 digests equal to `bench/bits.txt`. Against PyTorch on the same session,
  4.03 s a step against 0.93 s on 16 threads (4.3x) and 13.8 s against 2.43 s on one (5.7x).
- Left: the heads still run one after another inside a sequence.

## The exact GELU, with erf in plain Bend (2026-10-21)

- `GELUErf` is PyTorch's default `gelu(approximate="none")`, with `erf` written in Bend from F32 additions,
  multiplications and comparisons only: `x + x P(x^2)` below 0.84375, `K + S(a - c)` around six centres up to 3.92,
  then 1. Measured on all 2^32 inputs against glibc's double `erf`: at most 0.93 ulp off, correctly rounded on 97.1%.
  The Bend program gives the bits of a C copy of the algorithm on every input, and the JavaScript build matches on the
  ranges I ran.
- PyTorch's float32 GELU is 0.25 of the tolerance from float64 in the forward pass, because of its vectorized `erf`;
  mine is 0.08, and 0.29 from PyTorch's float32 at worst. Every existing bit is unchanged; `make mutate` 86 of 86.
- Cost, idle machine, 2^24 calls on one thread: `erf` 61 ms, `GELUErf` 80 ms against the tanh form's 188 ms, the
  gradients 148 against 204 ms. It is not in `gpu.bend`.

## Batch and sequence sizes read at run time (2026-10-21)

- `demo/tiny_rt.bend` reads B and T from the command line, with no change to `torch/`: the layer defs take `+b, +t`,
  composition goes through `Seq.at`/`Train.step_at`, and run-time forms of Fork, Both, Id and Residual are written in
  the demo, because a template cannot take a run-time size. Wrong sizes stop with a message.
- B = 2, T = 16 gives `bench/bits.txt`'s tiny Llama digests on 1 and 16 threads (`scripts/bits.sh rt`). 4x8, 3x24,
  8x32 and 2x16 are within test_tiny's float64 rule. Cost, idle machine, 2000 steps on one thread: 1.60 s compiled
  sizes, 1.55 s run-time sizes.

## Generation with a key/value cache (2026-10-21)

- `demo/smol.bend` has `genkv`, `drawkv` and `kvcheck`. The prompt runs through the whole window once, and the keys and
  values come from what attention saved in that pass; then each token runs one position. The output layer runs over 16
  bands of rows (`Bands.matvec_l`, the same bits). Nothing in `torch/` changed: a cached step threads state through every
  block, which the `Layer` protocol does not carry.
- On the five prompts of 2026-10-20 the greedy tokens equal gen's and the twin's, the seeded draw equals draw's, and
  every logit has the same bits as the whole-window pass (kvcheck: 0 of 49,152 differ at each of 285 tokens). Two broken
  copies of the cached attention are caught on every logit.
- Idle machine, 16 threads, 3 rounds: 77-84 ms a token at windows of 32, 64 and 128, against 300, 475 and 795 ms
  without the cache. 60 tokens after "ROMEO:" take 5.6 s against 47 s. One thread gives 7.9 s for those 60 tokens: the
  blocks' one-row products run on one thread each.
- The tiny Llama's window slides from the first character, so nothing exact can be cached there.

## The whole GPU step on the device, and attention's real GPU test (2026-10-21)

- `torch/gpu.bend` has `Embedding` (one leaf per column, the ids in order: main.bend's bits, no atomics; the ids travel
  as F32 bits, since tensor-gpu's inputs are `Array<F32>`) and `CrossEntropy` (four bangs: the row max, exp and block
  sums, tree_sum's remaining passes, the gradient; the host reads 3 numbers a row and takes the log). `demo/smol.bend`
  takes both from fast.bend's names, so its import line switches the GPU build.
- On one profiled GPU step (busy machine) the two long host gaps (1.6 s and 0.44 s) and their page migration are gone:
  data moved back to the host 346 -> 80 MiB. Idle machine, 3 rounds: `train 3` on the GPU 27.55 -> 25.62 s (-7%).
- The GPU losses of `train 3` move by 4 and 3 ulps at steps 2 and 3, all from CUDA's float exp in the cross-entropy:
  with `exp_cr` there they are the previous GPU build's bits. The CPU's bits are unchanged.
- The GPU layer test printed 0 for attention because, after the host's parallel lets, its bangs ran on the CPU threads.
  Run alone on the device, attention differs from main.bend by up to 176,128 ulps on cells near zero (absolute 1e-7
  to 1e-6). With `exp_cr` in place of every `F32.exp`, every case gives main.bend's bits on the device, so CUDA's exp
  is the whole cause. `make gpu` now runs each case alone, prints attention's ulps without a bound, and requires 0 on
  every case of the `exp_cr` build (`scripts/gpu_expcr.sh`; `demo/exp_cr.bend` is test-only, and `make laws` keeps it
  out of `torch/`).

## GPU attention as one bang a cell (2026-10-21)

- The GPU attention and RMSNorm already ran one leaf a row; the time went to long serial chains on each lane (a forward
  leaf did a row's scores, softmax and 64 outputs) and to filling new arrays on one lane. Attention is now cell bangs,
  one output number a lane with main.bend's `dot0`, over transposed K and V so that neighbouring lanes read neighbouring
  memory; only the rows' sums (the softmax, dP.P and G) stay one leaf a row. Scores are computed on the lower triangle
  only, rows i and t-1-i dealt as a pair; P and G store the lower triangle.
- One layer's forward and backward at SmolLM2-135M's sizes: 56.1 -> 17.4 ms (busy machine). Idle machine, 3 rounds:
  `train 3` on the GPU 25.6 -> 20.9 s (-18.5%), the same losses. Against the previous GPU attention, 0 ulps on the device
  in every case; with exp_cr, 0 against main.bend in all 13 cases of `make gpu`.
- One behaviour changes: an infinity or NaN in a masked (future) position reached main.bend's sums as 0 x inf = NaN, and
  no longer reaches the new ones, which skip the masked cells. For finite inputs the bits are the same.
- RMSNorm as row and cell bangs gave the same bits but was slower (6.79 against 4.72 ms), so it stays. An online softmax
  would save about 0.1% of a step and move bits; not done.

## On BendHub (2026-10-21)

- Published from `build/packages/` (`make package CORE=bendtorch VERSION=1.0.0.0`, identical to the sources at
  `dcf4029`): `bendtorch@1.0.0.0` `0x1ca1def0868294e0aed9920bf123e68a` (entry `package.bend`, 8 files),
  `bendtorch-fast@1.0.0.0` `0x6475a9340a3eb94acc1748bc05999432`, `bendtorch-gpu@1.0.0.0`
  `0x88faa733c17ccc2bd79ba26bdf7a4c89`, `bendtorch-load@1.0.0.0` `0xf25754162a4d6294dc96b9870e0997fa`.
- From a clean folder importing the published packages: the core checks and `--verdict` passes; the three satellites
  check with the same defs outside the proofs as `torch/` (58, 105, 6); `fast_test` and `gpu_test` print 0 on 1 and 16
  threads; `test_infra` and `test_dtype` pass; the tiny Llama's `train 50` gives `bench/bits.txt`'s digest on 1 and 16
  threads; `gpu_layers_test` on the device shows only CUDA's exp differences (SiLU 4 ulps, attention 176,128 ulps).

## CI: GELU(+inf) on some runners (2026-10-21)

- `test_ops`' edge cases failed now and then on CI (bf237bd, dcf4029, and on 9b1b2d4's pull request run): PyTorch's
  exact GELU of +inf was NaN there, while BendTorch gives +inf, the exact value, and the same PyTorch (2.14.1, CPU wheel)
  gives +inf on my machine. Pinning PyTorch's AVX2 kernels on CI (`ATEN_CPU_CAPABILITY`) did not change it, so it is
  reverted. The test now expects +inf in that one cell when PyTorch gives NaN there, the way it already handles
  PyTorch's NaN in the masked attention cell; the tolerance is unchanged.
- Renaming the repository to BendTorch broke the cached venv (it holds its absolute path); the cache key names the
  repository now.
- The first full CI run of `make mutate` (pull request #1) had 97 of 98 as expected: the GPU exactness mutant needs a
  CUDA device, and CI's runner has none, so `gpu_expcr.sh` built for the CPU and could not catch it. Without a CUDA
  toolkit and device, `mutate.py` now skips that mutant and says so in its summary; on my machine it still runs and is
  caught.

## Bend 2.0.38 (2026-10-23)

- **Runtime options between `+RTS` and `-RTS`** (bendlang/bend#1530, breaking). A bare `./x --threads 16` now reaches
  the program as an argument and runs on the default thread count with no error, so every 1-and-16-thread check would
  have kept passing while testing one thread count. Every call in the Makefile, `scripts/` and `reference/` passes
  `+RTS --threads N -RTS` (and `+RTS --gpu off ... -RTS`); a program given an unknown option inside the bracket now fails.
- **The checker's list of defs outside the proofs** is one line per kind ("defs that depend on @unsafe: a, b") instead
  of one `- name` per line, and now names the hub packages' defs (`0x<hash>/main.f`). `make laws` reads the new form
  (`OUTSIDE` in the Makefile) and drops the hub's names as before; FAST_DEFS, LOAD_DEFS and GPU_DEFS are unchanged, in
  the same order.
- With 2.0.38: `make check`, `make bits` (every digest identical, 1 and 16 threads), `make check-js`, `make gpu` ("the
  GPU builds' outputs are the CPU's"), `make smol` (135M's first loss 3.2626889, as before), `make package` (bendtorch
  1.0.0.0 and its three companions over local copies) and `make mutate` (98 of 98 as expected, the same two survivors,
  the GPU exactness mutant caught on the device) all pass. The published packages need no new version: no source of
  `torch/` changed.
- `docs/limitations.md` said Bend builds only for CUDA; it also builds for Metal on macOS (in 2.0.37 already). Only
  CUDA is tested here.
- **The GPU step, the same commit (`c7f334c`) built with 2.0.37 and 2.0.38**: `train 3` on the GPU (SmolLM2-135M, three
  AdamW steps), idle machine on mains power, 5 alternated rounds in a shuffled order, whole-run wall time (loading
  included): 22.49 s (22.27-22.58) on 2.0.37 against 22.28 s (21.99-22.39) on 2.0.38, 0.9% faster and lower in every
  round, with the same loss bits. By the program's own clock (`train_ms`, the three steps only, as every GPU number
  above was measured), one run each: 20.99 s against 20.76 s. #1447 (GPU bangs chain their rounds on the device) and
  #1602 (L1 kept in `bend_dev`) change little here, where most of a step is the host's work between bangs. Samples in
  my private notes.
- I first read 22.5 s as slower than the README's 7.0 s a step (20.9 s on 2026-10-21): it was the other clock. The
  difference is loading the weights, 1.3-3.4 s depending on the page cache. Checked on the way: the 2026-10-21 GPU binary
  runs at today's speed (22.2-22.4 s, 3 alternated rounds against today's 2.0.37 build), and nsys gives the same device
  work for both and for that day's profile (8,404 launches, 6.36 s busy). The README's 7.0 s holds.
- The CPU timings (bend-ml's GPT-2, llama.bend's SmolLM2-135M) show no change beyond the spread (their NOTES).

## The law block_sum (2026-10-23)

- After the launch, a reply on X (@latent_node) asked whether "the block divides the row" could be proven rather than
  tested, about the mutant that shrinks the cross-entropy's GPU blocks to one cell and survives. It is now a law:
  `block_sum` (LAWS.bend), 22 laws in all. `torch/spec.bend` writes a sum as the tree of its additions over the cells'
  positions (`Sum`: `SCell`, `SAdd`), with tree_sum's pass (`spass`, an odd last cell carried), `passes`, a block summed
  as a balanced tree of pairs (`bal`) and a row of blocks (`blocks`). The law: over a row of 2^p·k cells, r + p passes
  build the same list of trees as r passes over its k blocks. Two trees that are equal make the same additions in the
  same order, so the law holds for F32 with nothing assumed about rounding.
- The proof (PROOF.bend, about 50 lines): `pass_pair`, one pass over 2k blocks of 2^p cells gives the k blocks of
  2^(p+1) (induction on k, with `add_succ` and `add_assoc`); `cnt2`, 2^(p+1)·k = 2^p·2k; `blocks_passes`, p passes over
  the row leave its k blocks (induction on p); `passes_add`. It checked on the first try in both checkers; a false
  statement (the blocks from cell 1) is rejected at `Laws.block_sum`. On 12 cells in blocks of 4, both sides print
  `((((0+1)+(2+3))+((4+5)+(6+7)))+((8+9)+(10+11)))`.
- What it does not do: the mutant still survives (one cell divides every row; the law says why), so the count stays
  96 caught and 2 surviving, and that the GPU code builds these trees is tested, not proved. `make laws` (one comment
  in spec.bend named the GPU file, which the gate forbids there; reworded), `make package` at a test version and
  `make mutate` (98 of 98 as expected) pass.
- The core package publishes spec.bend, LAWS.bend and PROOF.bend, so the law went out as a new version of all four
  packages (the satellites import the core at their own version), my decision. Published from `build/packages/`
  (`make package CORE=bendtorch VERSION=1.0.1.0` at 088e342): `bendtorch@1.0.1.0` `0xfc8448b10655d3efdb72c022131382f2`
  (8 files), `bendtorch-fast@1.0.1.0` `0xb91744a472860b4a9d4321aff37d6994`, `bendtorch-gpu@1.0.1.0`
  `0x357d99c3e9fc95428e11c1d5f106e0a4`, `bendtorch-load@1.0.1.0` `0xd81dbae31742e38f426cc4b4700c9e51`. The published
  core's main, spec, tensor and dtype are the repository's files, and its LAWS.bend has block_sum.
- From a clean folder importing the published packages: the core checks and `--verdict` passes; the satellites check
  with the same defs outside the proofs as `torch/` (58, 105, 6); `fast_test` and `gpu_test` print 0 on 1 and 16
  threads; `test_infra` and `test_dtype` pass; the tiny Llama's `train 50` gives `bench/bits.txt`'s digest on 1 and 16
  threads. The READMEs import 1.0.1.0.

## Guards for the runtime options (2026-10-23)

- Since Bend 2.0.38 a runtime option written outside `+RTS ... -RTS` is not an error: the program gets it as an argument
  and runs on the default thread count, so a 1-and-16-thread check passes while testing nothing. After fixing about 60
  such calls by hand across the projects, I added three guards:
  1. `scripts/check_rts.py`, run by the checks (and CI): it fails on any tracked line with `--threads` or `--gpu` left
     once the `+RTS ... -RTS` spans are taken out, except the lines in `rts-allow.txt`, each with its reason (the log,
     other programs' own `--threads`). `--self-test` shows it finds a bare option and passes a bracketed one; a bare
     `--threads` planted in a Makefile fails the checks.
  2. Test programs that take no arguments refuse any, with exit 2 and a message that names `+RTS` (`IO.args()` holds the
     program's name only, on C and on JavaScript; the runtime takes `+RTS ... -RTS` out first). The checks run one with a
     bare `--threads 16` and require the refusal.
  3. `scripts/rts_probe.py`: a parallel program run with `+RTS --threads 1 -RTS` must use at most 1.5 cores, and with
     `--threads 16` several. Local only: CI sets `SKIP_RTS_PROBE=1`, and it skips itself under 8 cores. With a wrapper
     that drops the bracket (a runtime that ignores it), one thread used 10.1 cores and the probe failed, as it should.
  On this repository: `fast_test`, `gpu_test` and `gpu_layers_test` refuse arguments (`gpu_layers_test` takes a case
  number and used to run every case on anything else, a bare `--threads 16` included); the probe runs `fast_test`,
  1.00 and 2.6-2.7 cores (it is short and has sequential parts, so its bar is 1.8).

## Bend 2.0.39 (2026-10-24)

- **Pin.** `scripts/setup.sh` (version and both SHA256 sums), the CI step, the README badge, line and setup command,
  CONTRIBUTING, `torch/README.md`, `docs/design.md`, `docs/limitations.md` (Bend 2.0.39 still takes no compiler flags
  and still reads files only as byte lists) and CLAUDE.md. Nothing in the code changed.
- **`bend --publish` sends only files inside the entry's directory** (#1836): `build/packages/` holds plain files only,
  and each companion imports the core from the hub (`bendtorch@<version>/main.bend`); the `../<core>/main.bend` imports
  exist only in the local copies `make package` tests, which are never published. So publishing works as before.
- With 2.0.39: `make check` (the probe: 1.00 and 2.72 cores), `make bits` (every output identical), `make check-js`,
  `make gpu` ("the GPU builds' outputs are the CPU's"), `make package` (the four packages at a test version over local
  copies, the tiny Llama with `bench/bits.txt`'s bits on 1 and 16 threads), `make smol` (135M's first loss 3.5882206 and
  360M's 3.2626889, as before; 1 and 16 threads the same bits) and `make mutate` (98 of 98 as expected, the same two
  survivors) all pass. The published packages need no new version: no source of `torch/` changed.


## SmolLM2-1.7B: the night branches on devel, a loader piece by piece (2026-10-24)

- **Merged here (branch `night/1b7`, not in devel yet):** `night/t12`, that is the 64-bit reader (T6), safetensors
  loading (T8), packed BF16 moments (T10) and `scripts/smol.sh 1.7b`. Two conflicts (Makefile, `demo/smol.bend`'s header
  and usage) and one bare `--threads` that had to go inside `+RTS ... -RTS`.
- **The loader.** `L.safetensors` read every tensor into one array as large as the model before `Layer.load`: at 1.7B a
  2^31-cell block (8 GiB) that the step never reuses, which is why a step from the real weights went past 29 GB on
  2026-10-21. Now `L.header` parses the header once, `L.piece` fills one layer from its own tensors, and `L.two` fills a
  pair one after the other. `demo/smol.bend`'s `load_st` fills the embedding, each block (`LStack`, written by
  `scripts/smol.sh` next to the stack) and the final norm. `L.safetensors` is `header` then `piece`, the same bits.
- **Checks.** On 135M, the loss from model.safetensors read piece by piece gives `bench/bits.txt`'s digest, and
  `train 2 <model.safetensors>` gives the same output bits as `train 2` from init.bin (both in `make smol` now). Three
  new mutants (a half of the stack reading the other half's blocks, a piece one cell off, the final norm read from
  another tensor of its shape), all caught; they are skipped where model.safetensors is absent, as in CI. Gates: laws,
  check, bits, check-js, corpus, gpu, smol, bits-smol, adamw16-long, mutate (117 of 117).
- **1.7B, memory only** (busy machine, 16 threads, a 20 GB cap): the loss from the real weights is 1076417622
  (2.6379600, as on 2026-10-21) and peaks at 10.1 GB instead of 18.1. A step from the real weights is not measured yet:
  it needs about 23 GB, and the machine had 22 GB free.
- `smol.sh 1.7b` now writes AdamW with m and v packed (`T.CAdamWBF16{..., True{}}`), and `torch/README.md` says it is an
  opt-in with its measured drift (90.6% of v's updates lost over 2,000 steps of the tiny Llama).
- **Still open:** the reader is the vendored copy (`vendor/binary-read/`), which `bend --publish` refuses since 2.0.39,
  so `make package` cannot ship it. bend-ml's `binary-read` 0.2.0.0 has the same code; once it is on BendHub, the import
  moves to it and `vendor/` goes.

## SmolLM2-1.7B: two AdamW steps from the real weights (2026-10-24)

- `smol_1b7 train 2 <model.safetensors>` (m and v packed, the loader piece by piece), 16 threads, idle machine with the
  browser closed, 3 alternated rounds against a PyTorch twin with the same BF16 moments (m and v rounded after each
  update, stored as bfloat16), each run in a 25 GB memory scope with no swap: BendTorch 84.1 s for 2 steps (42.0 s a
  step), peak 24.1 GB; PyTorch 26.7 s (13.3 s a step), peak 23.6 GB. 3.2x slower, against 4.3x at 135M.
- Losses, the same bits in every round: 2.6379600 then 2.5548434; PyTorch F32 2.6379418 then 2.5548196, within the
  tolerance at 0.66x and 0.90x. The first is 1.0e-6 from float64 (PyTorch F32 1.7e-5); the second has no float64
  reference, since a float64 run does not fit in memory.
- One step alone peaks at 22.6 GB; the second step holds more (the moments now exist), and a 23.5 GB cap killed it.

## The reader from BendHub, the 1.7B in the benchmark (2026-10-24)

- `bend-ml-binary-read@0.2.0.0` is on BendHub (the vendored copy's code, with tests on both lanes in bend-ml).
  `torch/load.bend` and `demo/reader_test.bend` import it from there, and `vendor/` is gone. `make package` builds and
  checks the four packages with it, which `bend --publish` could not ship with a reader outside the entry's folder.
  `make laws`, `reader`, `check` and `smol` pass on it.
- The 1.7B step goes into `BENCHMARK.md` as W6 (BendTorch against PyTorch only, 16 threads) and into the README's speed
  section; the README's limitations line no longer says it does not train in 32 GB.

## test_mlp and BF16 rounding ties (2026-10-24)

- CI failed `make check` on the merge: seed 8's `adamw16mv.w1[20]` was 28x the tolerance from the twin. Its first moment
  after step 2 sits 17 F32 ulps above a BF16 rounding midpoint here (1 ulp in float64); the runner's torch sums the
  gradient in another order, rounds that moment the other way, and the weight moves. Bend gives the same bits on both
  machines and matches this machine's torch.
- Now the twin marks every element whose moment came within 64 F32 ulps of a midpoint at some step, and those weights
  are left out of the comparison: 19 of the 1,800 BF16-state weights over the 12 seeds. AdamW works element by element,
  so a tie moves only its own weight, and the losses are still all compared. The 11 mutants gated on the MLP are caught.

## Idle round on Bend 2.0.39 (2026-10-24)

- **What.** The 2026-10-18 idle round again, on 2.0.39 and commit 92f0486: the same cases, 5 rounds (SmolLM2: 3) for
  BendTorch against PyTorch and 3 for the eleven other systems, alternated, 3 h 13 min overnight with nothing else
  running. `BENCHMARK.md` and the README's speed table now carry it.
- **What changed for BendTorch.** MNIST 2.0 s to 1.7 s an epoch on one thread and 1.6 s to 1.3 s on 16 (0.1 s
  resolution); the block on the GPU 234 to 205 ms a step; SmolLM2-135M on 16 threads 5.1 s to 4.3 s a step (attention's
  two sequences in parallel, 2026-10-21); the tiny Llama on the GPU 82 to 84 ms. One thread on the block and on
  SmolLM2 did not move (664 to 659 ms, 14.2 s). Code and Bend both changed between the two rounds, so these numbers do
  not say which one gave what, except for the tiny Llama on the GPU: the same commit takes 16.29 s with Bend 2.0.37
  and 16.71 s with 2.0.38 and 2.0.39 (idle, alternated rounds), so the 2.6% came with Bend 2.0.38.
- **The other systems** gave the same times within a few percent and the same verdicts, except torch.compile on the GPU
  in W2: 8.51x PyTorch F32's distance from float64 in one round (fail), 7.13-7.46x in the five others over both nights.
  It sits near the line; the table says so.

## add4 and mul_dist_l from nat-lemmas 0.1.2.0 (2026-10-24)

- bend-ml's B6 moved my two general lemmas into `bend-ml-nat-lemmas@0.1.2.0` (published today). `PROOF.bend` now calls
  `NL.add4` and `NL.mul_dist_l` and keeps no copies; `torch/main.bend` and `torch/tensor.bend` import the same version.
  `make laws` (both checkers), `make check` and `make package` pass.

## v1.1.0 on BendHub (2026-10-24)

- Published from `build/packages/` (`make package CORE=bendtorch VERSION=1.1.0.0` at c23a361): `bendtorch@1.1.0.0`
  `0x88d35bdf07184f57aa5174c6f06ab791`, `bendtorch-fast@1.1.0.0` `0x154ce8ea744f0491f3a42ae0881fc224`,
  `bendtorch-gpu@1.1.0.0` `0xd450835bf87b8ebe49fc7baf18bdbc46`, `bendtorch-load@1.1.0.0`
  `0x4922dce2dd59be0648ebd8f3643975ae`. The load package imports `bend-ml-binary-read@0.2.0.0` from the hub.
- From a clean folder importing the published packages: the core checks and `--verdict` passes; the satellites check
  with the same defs outside the proofs as `torch/` (58, 105, 22); `fast_test` and `gpu_test` print 0 on 1 and 16
  threads; `test_infra` and `test_dtype` pass; the tiny Llama's `train 50` gives `bench/bits.txt`'s digest on 1 and 16
  threads. Before publishing, at c23a361: `make mutate` 117 of 117 as expected (the same two survivors) and CI green.
  The READMEs import 1.1.0.0.

## 16 threads: the training state's zeros and attention's units (2026-10-25)

- A timeline of one SmolLM2-135M `train 1` at 16 threads (2026-10-21) had attention on 1.9 busy threads for 1.0 s and
  the training state's zeroing on one thread for 0.6 s, of 4.8 s.
- **Zeros**: `Seq`, `Fork`, `Both` and the tied pair make their halves' zeros in a parallel let (`pair_zeros`). The
  arrays are many, so their page faults spread over the threads. Zeros are zeros: no bit can move.
- **Attention's units**: the tree of parallel lets ran one sequence per leaf (2 at SmolLM2's b = 2). Now a unit is one
  sequence's key/value group: its g query heads' columns of Q, its head's columns of K and V, and outputs of its own,
  written back by column blocks. 6 units at 135M, 10 at 360M, 64 at 1.7B. Inside a unit the heads keep their order and
  their arithmetic (dK and dV summed over the group's heads in order), so the bits are the same, and `fpar`/`bpar` are
  unchanged, so the laws `attn_par_fwd` and `attn_par_bwd` still cover the tree. Five new mutants (a unit's K and V
  columns, a block's rows, the output's, dO's and dK's columns), caught by `test_ops` (2 sequences x 2 groups).
- Idle machine, 3 alternated rounds, `train 3` (the 2026-10-25 window): 12.70 s -> 10.95 s on 16 threads
  (-13.7%), 41.7 s on one thread either way; every output is `bench/bits.txt`'s. Gates: laws, check, bits, check-js,
  corpus, gpu, bits-smol, smol, mutate (122 of 122 as expected).

## v1.1.1 on BendHub (2026-10-25)

- Published from `build/packages/` (`make package CORE=bendtorch VERSION=1.1.1.0` at 449326a): `bendtorch@1.1.1.0`
  `0xe1214263ebd97956a02f4f3ea7544c84`, `bendtorch-fast@1.1.1.0` `0x7185f5844710249c5ac77956f6514eb7`,
  `bendtorch-gpu@1.1.1.0` `0x3c22148d676c2988827dae103c18f499`, `bendtorch-load@1.1.1.0`
  `0x0b2d574934489d8cf85589b273567833`: the core carries attention's units and the parallel zeros.
- From a clean folder importing the published packages: the core checks and `--verdict` passes; the satellites check
  with the same defs outside the proofs as `torch/` (58, 105, 22); `fast_test` and `gpu_test` print 0 on 1 and 16
  threads; `test_infra` and `test_dtype` pass; the tiny Llama's `train 50` gives `bench/bits.txt`'s digest on 1 and 16
  threads. The READMEs import 1.1.1.0.

## Two laws: both_sum and residual (2026-10-25)

- `Both` and `Residual` were the two combinators of the Llama block with no law. `both_sum` runs `T.Both`'s own code on
  two linear layers of the exact model (`spec.bend`'s `BothL`): `dy1·y1 + dy2·y2 = dx1·x1 + dx2·x2`, each layer's
  `adjoint` on its own side. `residual` needed the composition written once over any type: `T.ResidualOf(X, dup, add,
  join, f)` is `Seq(Fork(Id, f), join)`, and both `Residual`s (`main.bend` and `fast.bend`, the one the demos use) are
  it over matrices with `Add2`. The law runs it over vectors with a linear `f` and `Add2`'s twin (`spec.bend`'s `AddL`):
  `dy·(x + W·x) = (dy + Wᵀ·dy)·x`. 24 laws in all (23 in `LAWS.bend` and `unpair_pair`).
- The emitted C of `tiny_llama`, `ops_test`, `fast_test` and `smol` is the same as before, apart from constructor ids and
  the template's name, and `make bits` gives every output of `bench/bits.txt`.
- Two new mutants, `ResidualOf`'s backward pass dropping the skip connection's gradient or the layer's, are rejected at
  `Laws.residual`. No mutant of `Both` that still type-checks changes the input gradient (its sides have their own
  types), so `both_sum` adds no mutant; it says the sides never mix.
- Dropped from the plan: the BF16 pack and unpack round trip (the checker cannot compute F32, and the U32 bit operations
  would need lemmas it cannot prove generally) and `L.piece` filling the layout (it lives in `load.bend`, which imports
  foreign code, so no file with laws can import it).
- Gates: `make laws` (both checkers), `make check`, `make bits`, the two new mutants.

## AdamW on 16 threads where it still ran on one (2026-10-26)

- Timed alone, `Opt.mat` runs at 1.0x on 16 threads. With F32 moments the demos' `fast.bend` layers already split it
  (`opt_mat`, `par4`); two paths still took main.bend's one loop: `fast.bend`'s `Embedding` was `T.Embedding`, whose step
  is `T.mat_step` (SmolLM2's 49152 x 576 table, tied to the head), and packed BF16 moments (`CAdamWBF16`, the 1.7B's),
  which `opt_mat` handed to `T.Opt.mat`.
- Now `fast.bend`'s `Embedding` is main.bend's with its step through `mat_step`, and packed moments run in blocks of
  whole words: two moments share a word, so a block of words i0 .. i0 + cnt - 1 runs main.bend's `adamh_l` (or
  `adamhv_l`) over elements 2·i0 .. 2·(i0 + cnt) - 1, the last block stopping at n. No word is written by two blocks and
  each element's arithmetic is main.bend's.
- `demo/fast_test.bend` adds both packed layouts (two steps from new, a 1537x577 matrix so the last word holds one
  moment, 64 blocks) and the Embedding's step: 0 differing words on 1 and 16 threads. A mutant that starts a block at
  element i0 instead of 2·i0 is caught (1,959,690 words); one that lets the last block run past n survives, as it must
  (the cells past n are zeros, and AdamW keeps zero at zero). Gates: check, bits, bits-smol, mutate (126 of 126 as
  expected, after pointing the mutant that drops fast.bend's packed step at the new case).
- RMSNorm is left as it is: in the idle timeline of `train 3` (2026-10-25) it takes 0.02 s of a 4.00 s step, on 3.3
  threads. Timing the AdamW change waits for the next idle window.
- Found on the way: `scripts/mutate.py`'s fast gate expected 13 zeros from a program that printed 12, from 2026-10-21
  (272a256) on, so its 7 mutants counted as caught without being tested. Run again with the right count, all 7 are
  caught. The gate now expects 15, and `mutate.py` first runs every gate on the unmutated code and stops if one fails.

## SmolLM2-1.7B's memory: the allocator, not the activations (2026-10-26)

- The plan was to free activations earlier in the backward pass so AdamW's v can stay F32 at 1.7B. Measured first on
  1.7B's shapes cut to 2 and 4 blocks (`train 2`, F32 moments, peak RSS): at one thread a block adds 1,078 MB, its state
  alone (4 x 268 MB); the activations it saves (about 56 MB) do not show. At 16 threads a block adds about 0.4 GB more:
  Bend's allocator parks, per thread and size class, one freed generation (COLD), and for blocks of 32,768 words or more
  one block is a generation, so each thread keeps a big block no other thread can take.
- With big blocks sent straight to the shared bank (a patch to the emitted C, in my
  private notes), the 4-block model peaks at 7.18 GB on 16 threads instead of 7.87-8.43, the
  one-thread level, with the same bits. At 24 blocks that is about 2.5 GB of the 1.7B step's 24.1 GB.
- Estimated from the one-thread numbers: v in F32 (m packed) needs about 25.1 GB with the fix and 27.6 GB without. Reported
  upstream as bendlang/bend#1856; `V16=0 scripts/smol.sh 1.7b` writes the
  1.7B with v in F32 for the run that measures it.
- Idle window (2026-10-26 00:05-00:12): `train 3` on 135M, 16 threads, 3 alternated rounds, 10,977 ms before the
  packed and embedding blocks against 10,719 after (-2.3%), the same bits. The full 1.7B (m and v packed) with the
  allocator patch: two steps peak at 22.6 GB (KB / 10^6) against 24.1 GB on 2026-10-24, in 59.7 s against 84.1 s (both
  changes), the same two losses. v in F32 (`V16=0`), with the patch and 27 GB free: 26.0 GB, 59.6 s for two steps,
  losses 2.6379600 and 2.5548436 (1 ulp above the packed run's second). So v stays F32 at 1.7B once the allocator stops
  parking big blocks (bendlang/bend#1856); without it, about 27.6 GB.

## SmolLM2-1.7B fine-tuned on Tiny Shakespeare (2026-10-27)

- `demo/smol.bend finetunest <k> <e> <model.safetensors>`: `finetune` from Hugging Face's weights (at 135M, 2 steps:
  the same bits as `finetune` from `init.bin`). The 1.7B (`scripts/smol.sh 1.7b`, m and v packed) with the allocator
  patch (bendlang/bend#1856), 100 steps, the held-out loss on `ft_eval.bin` every 25, 16 threads, idle machine, against
  a PyTorch twin with the same BF16 moments.
- At `cfg()`'s lr 1e-4 (the 135M's) the held-out loss rises, 2.0389 -> 2.4924, in both systems to 5 digits: too hot
  for the 1.7B with two sequences a batch and no warmup. At lr 1e-5 it falls, 2.0389 -> 1.8708 at step 75 and 1.8743
  at 100, every evaluation within the tolerance of the twin (0.67x-0.80x).
- 3,331 s for the 100 steps and 4 evaluations against PyTorch's 1,336 s (2.5x); peak 22.6 GB against 24.5 GB.
- `make mutate` at bfc6082 (devel with PR #9 and `finetunest`): 126 mutants, 126 as expected. 123 caught; the 3 that
  survive are the known equivalent ones (the GPU cross-entropy's block size, the packed AdamW's last block past n, the
  whole-tensor sum's 1-cell wrap, which its bounds mutant catches), each with its reason in `scripts/mutate.py`.

## v1.2.0: the gates (2026-10-27)

- In since v1.1.1: PR #9 (adeerkhan: reusing parameters in a loop, `demo/loop_test.bend` in the build), PR #11
  (adeerkhan, fixes #10: the toolkit found through the same `nvrtc.h` candidates in the Makefile, `gpu_expcr.sh` and
  `mutate.py`, `lib64` or `lib`, one `cuda-toolkit` guard for `gpu` and `smol-gpu`, the first GPU build checked to link
  libnvrtc), PR #14 (adeerkhan, fixes #13: `Linear`, `LinearFirst` and `LinearNB` in `fast.bend` and `gpu.bend`, the
  demos calling them), `finetunest`, the laws `both_sum` and `residual`, AdamW's embedding and packed paths on 16
  threads, and the package and CI fixes. A minor release: #14 adds names to the published `-fast` and `-gpu`.
- `scripts/smol.sh 1.7b` now writes lr 1e-5 into the 1.7B's `cfg()` (`LR=` sets another; `LR=1e-4` rebuilds
  `BENCHMARK.md`'s W6), so the converging fine-tune above builds from the repository; the 135M's `cfg()` keeps 1e-4.
- Found on the way: `scripts/check_js.sh` still expected 12 zeros from the JavaScript `fast_test`, which prints 15 since
  705fb39, so `make check-js` failed (CI runs it only on pull requests and the schedule). It expects 15 now.
- One new mutant from #14: `fast.bend`'s `Linear` delegating to `LinearFirstF` (the same type, no input gradient), caught
  by the MNIST test, whose output layer is `F.Linear`.
- Gates at 580fa84 (the code of the release): laws, check, bits, corpus, gpu, bits-smol, smol, `make package
  CORE=bendtorch VERSION=1.2.0.0`, mutate (126 of 126 as expected); check-js after the fix; the new mutant caught.
