# Changelog

BendTorch's releases, newest first. Version numbers follow bend-ml's `docs/VERSIONING.md`: MAJOR for a change in what
the packages are or in how they are used, MINOR for a new capability, PATCH for fixes, speed-ups and version bumps. The
four BendHub packages (`bendtorch`, `bendtorch-fast`, `bendtorch-gpu`, `bendtorch-load`) carry the release's number
with a fourth digit (`2.0.0.0`). The engineering log behind each entry, measurement by measurement, is `NOTES.md`.

## 2.0.0 (unreleased)

A major release: the optimizer configuration has one owner, the training state. Until 1.3.0 a `Train` was made with
one configuration and every step was given another, and the step's won; now `Train.new` takes the configuration once,
every step runs under it, and a step whose configuration does not fit the state's buffers is refused. It breaks how the
package is used (`Train.step`'s arguments and the record's fields), hence the major. Reported by @adeerkhan in #24;
the change is #27.

### What was wrong in 1.3.0 (#24)

- **The learning rate given to `Train.new` was never read.** `Train.new(cfg)` used its configuration only to choose
  the buffers (and for `OptCfg.ok`), and `Train.step(cfg)` ran under its own argument: three `Train`s made at three
  learning rates and stepped at one gave one curve, digit for digit. A learning-rate sweep that set the rate in
  `Train.new` measured nothing.
- **A configuration of another family than the state's buffers did nothing, with no error.** It fell to `Opt.mat`'s
  last case, which hands the weights back unchanged: a `Train` made for SGD or momentum and stepped with AdamW kept its
  loss for every step (0.272812 for 20 steps in the test below), no error and no NaN.
- **Or it mixed two optimizers.** A `Train` made for AdamW and stepped with SGD ran SGD, with the moments frozen.
- **A configuration that `OptCfg.ok` rejects ran unchecked at the step**, since the check ran only in `Train.new`.

### Breaking changes

- `type Train<P>` has a seventh field, the configuration, after `st`:
  `Train{p: P, g: P, st: OptState<P>, cfg: OptCfg, p1: F32, p2: F32, loss: F32}`. Every pattern on the record and every
  construction of it gains one field.
- `Train.step(~P, ~X, ~Y, ~S, ~T, ~m, ~loss, tr, x, t)` and `Train.step_at(P, X, Y, S, Target, m, loss, tr, x, t)` no
  longer take a configuration: they run under the record's.
- `q_start` is removed. `Train.new` starts Adam's 1 - b1^t and 1 - b2^t at 0 for every configuration; a configuration
  that `OptCfg.ok` rejects is refused at every step instead (below), so every loss after the first is still NaN.

### New

- `Train.with_cfg(P, cfg, state)`: the same training state under another configuration from the next step on, for a
  schedule (warmup, decay) or a deliberate change mid-run. It keeps the parameters, the gradient, the moments and Adam's
  1 - b1^t and 1 - b2^t, so a change of learning rate continues the run. It must be of the state's family and pass
  `OptCfg.ok`; otherwise the next step is refused, since the moments of another optimizer cannot be made up.
- `Opt.guard(P, cfg, q1, q2, st) -> Guarded<P>`: the check every step makes. The step runs under its configuration's
  optimizer (`Opt.at`) only when the configuration is of the state's family (SGD on `NoMom`, momentum on `OneMom`,
  Adam or AdamW on `TwoMom`, AdamW with BF16 moments on `Packed`) and passes `OptCfg.ok`. Any other step runs with the
  optimizer of the state's own family with every number NaN (`SGD{nan()}`, `nan_mom()`, `nan_adam()`), so the
  parameters and every loss after it are NaN: a dead run, never a wrong one that looks right.
- `OptCfg.kind(cfg) -> U32` (0 SGD, 1 momentum, 2 Adam or AdamW, 3 AdamW with BF16 moments), `Opt.when(ok, cfg, q1,
  q2, refused) -> Opt`, `type Guarded<P>` with `Guarded.opt`, and `nan()`, `nan_mom()`, `nan_adam()`: the pieces of the
  guard, public so that the laws can speak about them.

### Laws

Four new laws in `LAWS.bend`, proved in `PROOF.bend` and checked again by `bend PROOF.bend --verdict`, 33 laws in all
(with `dtype.bend`'s `unpair_pair`):

- `guard_nomom`, `guard_onemom`, `guard_twomom`, `guard_packed`: for every configuration and each kind of state (no
  buffers, one, two, two packed in BF16), the optimizer `Opt.guard` hands the step is
  `Opt.when(OptCfg.kind(cfg) == <the state's kind> && OptCfg.ok(cfg), cfg, q1, q2, <the state's refused optimizer>)`:
  the configuration's own exactly when it is of the state's family and passes `OptCfg.ok`, the NaN one otherwise. Each
  proof takes the five configurations one by one, and in each both sides compute to the same optimizer. The guard is
  written as that `Opt.when` so that they do. What is read from the code rather than proved: that `Train.step` and
  `Train.step_at` hand the guard's optimizer to `Layer.step` (`ts4g`, `train_at_g`).

### Tests

- `demo/cfg_test.bend` with `reference/test_cfg.py`, in `make check` and on the JavaScript lane (`make check-js`): a
  Linear(4 -> 2) under MSE against `torch.optim.AdamW` on the same weights, input and target.
  - A learning-rate sweep with the rate given once to `Train.new` (0.1, 0.01, 0.001) gives three runs, each equal to
    PyTorch's.
  - A warmup (0.002·(k + 1) for five steps, then 0.01) and a change from 0.01 to 0.001 after ten steps, through
    `Train.with_cfg`, match PyTorch with `param_groups[0]["lr"]` set before each step: the worst number is 0.027 of the
    shared tolerance (`reference/oracle.py`).
  - A checkpoint written after five steps and read back resumes bit for bit: into a `Train` at the same learning rate
    it gives the uninterrupted run's bits, into one at another rate it gives the bits of a `Train.with_cfg` change at
    that step (the configuration is the `Train`'s, not the file's).
  - Five refused steps, SGD -> AdamW, AdamW -> SGD, momentum -> AdamW, AdamW -> AdamW with BF16 moments and a `c1`
    that disagrees with `b1`: the first loss is finite, every later loss and every parameter NaN. A learning-rate change
    within AdamW is not refused.
  - The program takes no arguments and refuses any (a runtime option outside `+RTS ... -RTS` would be ignored).
- On the JavaScript lane a NaN made by arithmetic has its sign bit set (`0xffc00000` against the native `0x7fc00000`);
  the test reads a NaN as a NaN.
- `reference/test_infra.py` passes unchanged: a configuration `OptCfg.ok` rejects still stops a program that checks it
  (`OptCfg.check`), and without the check its weights are NaN after a step; checkpoint sizes are unchanged.

### Mutation testing

`make mutate`: 159 mutants, 159 as expected (149 at 1.3.0); 156 caught, and the same three surviving by design.

- Seven new against `cfg_test`: the guard letting another family run on a `NoMom` state (1.3.0's silent no-op) and on
  a `TwoMom` state (SGD over AdamW's moments), the guard skipping `OptCfg.ok`, `Train.with_cfg` keeping the old
  configuration or restarting 1 - b^t, `Train.new` keeping a fixed configuration instead of its argument, and a restored
  Adam state taking a fixed configuration instead of the `Train`'s.
- Three new against the laws: a `NoMom` state taking a configuration of another family, a packed state skipping
  `OptCfg.ok`, and AdamW's kind set to the packed state's.
- Moved: the mutant that started a rejected configuration's 1 - b^t at 0 instead of NaN (`q_start`) survived once the
  guard refused that configuration at every step; an equivalent mutant is dead code, so `q_start` is gone and the
  mutant now lets the guard run a rejected configuration, against `test_infra`.
- Updated: three checkpoint mutants spell the record's new field.

### Compatibility

- **Bits**: `bench/bits.txt` does not move: the tiny Llama's training and sampling, MNIST and SmolLM2-135M's loss and
  steps give 1.3.0's bits on 1 and 16 threads, and the GPU builds give the CPU's. A program that passed the same
  configuration to `Train.new` and to every step gets the same bits after moving; one that passed each step its own
  (a schedule) gets the same bits through `Train.with_cfg`.
- **Checkpoints**: the file layout is unchanged (the parameters, the moments in use, then Adam's 1 - b1^t and
  1 - b2^t); a 1.x checkpoint reads into 2.0.0. The configuration is not in the file: a checkpoint resumes under the
  configuration of the `Train` it is read into. One edge: a `Train` made with a configuration `OptCfg.ok` rejects and
  saved before its first step holds 0 for 1 - b^t where 1.3.0 held NaN; any step on it is still refused.
- **Lanes**: `bendtorch-fast` and `bendtorch-gpu` are unchanged; their layers' optimizer steps receive the guard's
  optimizer from the core's `Train.step`. `bendtorch-load`'s `train` carries the record's new field.
- **Toolchain**: Bend 2.0.39, Lean 4.34.0, bend-ml-tensor-array 0.1.18.0, bend-ml-tensor-fork and bend-ml-tensor-gpu
  0.1.5.0, bend-ml-binary-read 0.2.0.0, all as in 1.3.0.

### Performance

One thread pinned to a performance core, the versions alternated in each round, the same bits in every run:

| | 1.3.0 | 2.0.0 |
|---|---:|---:|
| tiny Llama, one AdamW step (median of 5, range) | 0.821 ms (0.815-0.834) | 0.817 ms (0.803-0.853) |
| MNIST, one SGD epoch | 1.7 s | 1.7 s |
| Linear(4 -> 2), instructions a step (100,000 AdamW steps, 10-30 runs) | 14,060 | 14,246 |

On a real model the difference is inside the runs' spread. On the smallest step there is, carrying the configuration in
the record costs about 180 instructions a step (1.3%); the guard itself costs nothing measurable. For comparison, option
(2) of #24, a step that compares its configuration with the record's field by field, measured 690 a step (4.9%) there.

### Why this design

The four candidates were built on branches and run through every gate: (1) the record carries the configuration and
the steps lose their argument; (2) the record carries it and a step given another one is refused; (3) (2) plus a step
without the argument, the old one to go later; (4) no API change, the guard alone. Adding a field to the record breaks
every pattern on it, not only the calls to the step (42 in the demos, 8 in `load.bend`, about 30 in `main.bend`), so
(2) and (3) break callers as much as (1) and keep two sources of one value; (4) breaks nothing but leaves the learning
rate given to `Train.new` unread. 2.0.0 is (1) with (4)'s guard, which is also what makes `Train.with_cfg` safe. Getters
in place of patterns would not help in Bend 2 (a `match` only inspects parameters, so every caller would need a helper
def for the getter's result), and an `OptState` indexed by the optimizer's kind would change `Layer`'s `step` type and
with it every layer. Every option passed every gate except one mutant, the same on all four: `q_start`'s, above.

### Moving from 1.x

A training loop moves in about five lines. The configuration goes to `Train.new` once:

```python
# 1.x
T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, cfg, state, x, targets)
# 2.0.0
T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, state, x, targets)
```

A pattern on the record gains `cfg` after `st` (any name; it is passed back as it came):

```python
# 1.x
case T.Train{p, g, st, p1, p2, +l}: ... T.Train{p, g, st, p1, p2, l}
# 2.0.0
case T.Train{p, g, st, c, p1, p2, +l}: ... T.Train{p, g, st, c, p1, p2, l}
```

A helper that receives the record's fields one by one and builds it again takes the configuration too (the repository's
`demo/smol.bend` had one, `ev_print`).

A schedule, which gave each step its own configuration, sets it with `Train.with_cfg` before the step, with the same
numbers, and gets the same bits:

```python
# 1.x
T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, adamw(lr(k)), state, x, targets)
# 2.0.0
T.Train.step(~P, ~X, ~Y, ~S, ~T, ~Model(), ~loss, T.Train.with_cfg(P, adamw(lr(k)), state), x, targets)
```

Switching optimizer family still needs a fresh state (`Train.new`): `Train.with_cfg` to another family is refused at the
next step. Checkpoints need no change.

### Also in this release

- `docs/design.md`, `docs/limitations.md`, `torch/README.md` (the API, the rule, the move from 1.x, the laws in plain
  language) and the README (the example, 33 laws, 159 planted faults, the mutants figure rendered again from
  `scripts/mutate.py`) describe the new design.
- `docs/media/mutants.mjs`: the figure's tally keeps its spaces under librsvg (`xml:space="preserve"`), and the header
  says how to render it without a browser.
- CI: the job's limit is 90 minutes (the full run, with the mutants and the JavaScript lane, took 57 to 60 minutes at
  1.3.0, and the 2026-11-02 nightly passed every step and was cut at 60).

Thanks to @adeerkhan for #24: the report, the repro and its measurements.

## Earlier releases

Their notes, in full, are on GitHub: https://github.com/nuxyel/BendTorch/releases

- **1.3.0** (2026-11-01): `GroupNorm` and `FullAttention` on the CPU and GPU lanes, the tensor stack at
  tensor-array 0.1.18.0, 29 laws, 149 mutants.
- **1.2.0** (2026-10-27): the fast and GPU lanes take `main.bend`'s names, so a program switches lanes by its import
  line; SmolLM2-1.7B fine-tunes from the repository at a rate that converges.
- **1.1.1** (2026-10-25): two parts of a training step that ran on one or two threads run in parallel.
- **1.1.0** (2026-10-25): weights from safetensors files, AdamW with BF16 moments, SmolLM2-1.7B training on a 32 GB
  machine.
- **1.0.2** (2026-10-23): the law `block_sum`, the packages at 1.0.1.0, guards for the runtime options.
- **1.0.1** (2026-10-23): Bend 2.0.38.
- **1.0.0** (2026-10-22): the first public release: typed layers, combinators, losses, four optimizers and one training
  step for any model, MNIST and a two-layer Llama trained against PyTorch twins, the backward formulas as laws.
