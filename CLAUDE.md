# BendTorch

Neural networks trained in Bend 2 with general pieces: layers with typed forward and backward passes (`Layer<P, X, Y, S>`),
combinators (`Seq`, `Fork`, `Both`, `Residual`), losses, SGD/momentum/Adam/AdamW and one training step for any model.
An MLP on MNIST and a two-layer Llama train against PyTorch twins; the backward formulas of the linear layers, the
causal mask and the head grouping are laws checked by Bend's kernel. Step 2 of the plans after bend-ml v1.2 (the
"BendTorch" of those plans). **Private until Renan is satisfied with a v1; then it goes public**, so everything in
its files and its history must be fit to publish. Nothing is pushed without him. Sensitive material (analysis and
method beyond a one-line result, anything about other people's projects) never goes into this repository: it lives in
Renan's private repositories that never go public.

## Where things are

- `NOTES.md`: every finding, measurement and decision, by date. Read it before changing anything.
- `docs/design.md`: the architecture and the reasons; the table of what is proved, tested and trusted.
- `docs/plan.md`: the milestones, each with its pass criterion.
- `torch/main.bend`: the package (layers, combinators, losses, optimizers, `Train`); `torch/spec.bend`: the exact `Nat`
  model the linear laws speak about; `torch/README.md`: the API and the laws in plain language.
- `LAWS.bend` / `PROOF.bend` at the root: the laws (claims a human reads) and their proofs, the convention of the Bend
  guide. `make laws` runs both checkers and the rejected programs in `tests/`.
- `demo/`: MNIST, the tiny Llama and the programs the tests run. `reference/`: the PyTorch twins and tests;
  `oracle.py` holds the single comparison policy. `scripts/`: setup, data, `mutate.py`, `bench.sh`.
- bend-ml (`~/Repos/Bend2/bend2-ml`) and llama.bend (`~/Repos/Bend2/llama.bend`) are the parent projects; their
  `NOTES.md` hold the performance experiments this builds on. Renan's planning notes are private and live outside this
  repository; never link them from it.

## Branches and versions

Work on `devel`; `main` only receives releases through a pull request `devel` -> `main`. Version numbers follow
bend-ml's `docs/VERSIONING.md`.

## About Renan

Software developer at Sidia (Samsung R&D), where he trains NLP, ML and CV models, puts them into production and builds
agentic systems over LLMs; writes Python and TypeScript; studies ML formally. No background in type theory: explain each
proof in plain language, in a comment and in chat. Machine: Core Ultra 7 155H (22 threads; P-cores are CPUs 0-11, pin
one-thread timings with `taskset -c 3`), 32 GB RAM, RTX 4050 6 GB.

## Rules

1. Bend is pinned to **2.0.39** (and Lean 4.34.0 for `--verdict`). Before writing code, read `bend guide`,
   `bend base <name>` and the Base source (`~/.bend/bend2/base.bend`). A version bump goes in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second. `make check` before a commit that changes code.
3. Every proof is real: no `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet is recorded in `NOTES.md`
   as debt and left out of `LAWS.bend`. One exception, Renan's decision (2026-10-17): `torch/fast.bend`, layers over
   bend-ml-tensor-fork (Base's `@unsafe` `Array.fork`, all Bend), which the demos use: the linear layers' products and,
   since M6 phase 3 (confirmed by Renan, 2026-10-17), their optimizer step and the element-wise layers over its
   `par1`..`par4` blocks. No file with laws imports it, `make laws` pins its defs outside the proofs, and
   `demo/fast_test.bend` checks it gives `main.bend`'s bits. A second exception, temporary (Renan, 2026-10-17):
   `bend-ml-binary-read` (foreign C and JavaScript) to load weights, only in loading code and never in a file with laws,
   because Bend 2.0.39 reads files only as byte lists; Taelin said native fast reads will come, and then it is replaced
   by 100% native Bend. A third exception (Renan, 2026-10-18, launch plan scope): `torch/gpu.bend`, the layers over
   bend-ml-tensor-gpu (its products and its `run1`..`run4` bangs, Base's `@unsafe` `Array.fork` inside the package, all
   Bend), which `make gpu` and the GPU demos use. Like `fast.bend`: no file with laws imports it, `make laws` pins its
   defs (`GPU_DEFS`) and it writes no `@unsafe` of its own (its leaves are plain Bend); `demo/gpu_test.bend` and
   `demo/gpu_layers_test.bend` check it gives `main.bend`'s bits, except CUDA's float `exp` on the device (SiLU,
   softmax), accepted by its cause (Renan, 2026-10-18). A new exception is Renan's call.
4. Prove structure (shapes, composition); test numbers against PyTorch with the one tolerance in `reference/oracle.py`.
   Never widen it to make a test pass: find the cause, fix it or record it.
5. Every proved law appears in `torch/README.md` in plain language. A new layer gets a section in `demo/ops_test.bend`
   and `reference/test_ops.py`; a change to `torch/` or the laws gets a `make mutate` run.
6. Benchmarks are honest: hardware and versions written down, alternated runs, one thread pinned to a P-core, losses
   shown.
7. Everything in the repository is in English. Commits are small, with clear messages and no attribution lines (no
   `Co-Authored-By`, no "Generated with").
8. Public text (READMEs, `NOTES.md`, `docs/`) is written in Renan's first person singular: "I measured", never "we".
   This file stays in the third person.

## Bend 2.0.39 pitfalls (paid for in bend-ml and llama.bend)

- A compiled program's runtime options go between `+RTS` and `-RTS` (since 2.0.38): `./x +RTS --threads 16 -RTS`, or
  `BENDRTS="--threads 1"` in the environment. A bare `./x --threads 16` passes `--threads 16` to the program and runs on
  the default thread count with no error, so a thread-count test silently tests nothing. `--gpu` goes in the same bracket.
  Three guards (2026-10-23): `scripts/check_rts.py` (in the checks, with `rts-allow.txt`) fails on one written outside
  the bracket; test programs that take no arguments refuse any; `scripts/rts_probe.py` checks, locally, that 1 and 16
  threads use 1 and several cores.
- `match` only inspects a parameter or a pattern variable; a computed value (including a destructuring
  `let` of a call) needs its own def or a parameter computed by the caller. Matching several parameters at
  once (`match b small:`) is the way to branch on a computed `Bool` and a value together.
- No mutual recursion. Termination reads arguments left to right: each must be passed unchanged until one
  shrinks. Erased (`-x`) parameters are skipped, so put a changing size after the shrinking argument or
  make it erased.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type counts as a use.
- `%e : P` rewrites: `_` marks where the right-hand side of `e` sits in the current goal, and the new goal
  has the left-hand side there; use `Equal.sym` to go the other way. `Nat.sub(1n+p, 1n)` does not reduce to
  `p` in the checker: pass `p` itself.
- Parallel lets (`a b = f(..) g(..)`) suffer from non-tail recursive helpers in the fork tree (use tail
  loops with an accumulator). Check scaling with `+RTS --threads 1` against `+RTS --threads 16`, and read the emitted
  C (`bend X.bend -o x.c`): the joins of each parallel let are there. (Up to 2.0.36 they silently became
  sequential when the def's last parameter was erased; fixed in 2.0.37 by bendlang/bend#1377.)
- On the GPU a lane has a 2,048-word stack: a non-tail recursion inside a `!` faults past it ("memory fault (machine
  stack overflow?)", e.g. `List.append` over 1,023 elements) while `+RTS --gpu off` runs; use tail loops in bangs (bendlang/bend
  #1393). Every device loop also polls for errors, which costs 4-20x on loops with little work a step (a draft issue
  in bend-ml's `docs/upstream/gpu-error-poll.md`); loops that read memory pay 1.2-1.9x.
- On CUDA a bang deals a balanced tree's leaves to threads in a run-time order: each block of 128 threads gets one
  aligned range of 128 leaves, but inside it the order changes from run to run, so "leaf k on lane k" layouts read
  uncoalesced (16 sectors a warp instead of 4). Check layouts with `ncu` (sectors per request); a renumbering of
  the low 7 bits helps a 2^14 tree (an issue draft in bend-ml's `docs/upstream/gpu-leaf-order.md`).
- `IO.fork` gives concurrency, not parallelism (by design, bendlang/bend#1415). Use parallel lets in pure
  code for several cores.
- Proofs follow the affine rules too: a lemma about values that hold an `Array` cannot mention them twice at
  run-time positions; pass the expression in an erased argument (`-s`) and the arrays once. A lemma
  quantified over a single-constructor value (`g: G`) can match it, which avoids computing the call that
  produced it.
- `Array` is a binary tree in Base (`ALeaf`/`ANode`) but one flat block in the compiled program: `get`/`set` are
  an index mask and a load or store (no descent), while matching an array copies each half into its own block
  (BendRT, "Arrays"), so never walk an array by `match`; `Array.clone` is O(n); `Array.fork`/`Array.join` are
  `@unsafe`. Indexes wrap around, so never write past a capacity. Corrected 2026-10-13 (bend-ml NOTES).
- `File.read_at` returns a `List<U32>` with one cell per byte; producing and walking it is the cost of
  loading (bend-ml exp. 13), and erasing a large unread list costs more than walking it.
- An `import ... as M` does not re-export. A BendHub package can hold several files, each imported by path
  (`pkg@ver/dir/file.bend`), and relative imports inside it resolve; `--publish` sends the entry file and what it
  imports, so every file must be reachable from the entry (checked 2026-10-19; bend-ml's CLAUDE.md has the rules).
  Relative imports (`import ./x.bend as X`) work for local copies.
- `Nat` is unary in the checker: `{==}` on a count of millions overflows its stack. State large counts in
  `U32` (the checker computes them at once; mind the 2^32 wrap) and keep `Nat` for sizes in types.
- The pair `A & B` has kind `Type`; a value that must be `Data` (reused, like a cache read once per head) needs its
  own datatype (`Snoc<A>`). A type can be computed by recursion (`def Cache(t: Nat) -> Data`).
- Scrutinees of a multi-`match` follow parameter order: `match fuel queue left`, never `match fuel left queue`
  when `queue` is declared first.
- A loop that must match a computed result (a forward pass at each position) recurses inside a closure: pass the
  result to a `then`-style def that matches it and calls a lambda (`Out.then(o, lg => ... => loop(p, ...))`). The
  termination checker accepts the recursive call inside the lambda, and there is no mutual recursion.
- A running F32 sum loses the small terms next to a large one (Llama's massive activations): sum long reductions
  as a balanced tree (`tree_sum` in `llama/main.bend`), and compare against a float64 run before blaming PyTorch.
- A template (`~f`) is substituted at compile time and may be called any number of times; a closure only once. Pass a
  record of closures (a `Layer`) as a template to use it in both passes. A template can take a `Nat` (`~n: Nat`); its
  function-typed parameters must spell reusable arguments (`~f: @+a: F32 -> F32`) when the def it receives has `+`.
- A template record is built again at every use, and a plain-value field of it is computed then. A composite's value
  field that reads its children (`count: Layer.count(la) + Layer.count(lb)`) builds the children's whole records on
  every `fwd`/`bwd`/`step` and drops them: +5% instructions in the tiny Llama (M9). Make such fields closures
  (`count: Unit -> U32`), like `zeros`; `make lint` reports them (local rule `performance/template_value_field`).
- A def must be declared above its callers (no forward references), and two defs may not call each other: a loop that
  needs a helper which loops back has to be one def (`match k tr:`, both cases inline).
- `bend-ml-nat-lemmas`' `add_assoc` states `a + (b + c) == (a + b) + c`, the opposite of llama.bend's own; check the
  direction before copying a rewrite.
