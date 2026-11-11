# Changelog

BendTorch's releases, newest first. Version numbers follow bend-ml's `docs/VERSIONING.md`: MAJOR for a change in what
the packages are or in how they are used, MINOR for a new capability, PATCH for fixes, speed-ups and version bumps. The
four BendHub packages (`bendtorch`, `bendtorch-fast`, `bendtorch-gpu`, `bendtorch-load`) carry the release's number
with a fourth digit (`2.0.0.0`). The engineering log behind each entry, measurement by measurement, is `NOTES.md`.

## Unreleased

### Gradient accumulation (#36)

- **`Train.accumulate` and `Train.flush`**: one optimizer step over k micro-batches, for an effective batch larger than
  the batch a program's types allow (a 64x64 image example cannot type more than two images, bendlang/bend#1071).
  `Train.accumulate` runs the forward pass, the loss and the backward pass into the state's gradient and stops there;
  `Train.flush` runs one optimizer step on the summed gradient, under the state's configuration, and zeroes it. Adam's t
  advances once a flush, as in PyTorch. `LossOut.div(n, c, k, r)` divides a micro-batch's loss and its gradient by k,
  PyTorch's `loss / k`, so the k gradients add up to the gradient of the mean over all k·B rows. `Train.accumulate_at`
  and `Train.flush_at` are the owned forms, as `Train.step_at`. Asked for by @adeerkhan.
- **Tests**: `demo/accum_test.bend` with `reference/test_accum.py` (`make check-accum`). With k = 1 the two calls give
  `Train.step`'s bits, in both forms. Four micro-batches of 5 rows a step against PyTorch's accumulated step, 3 steps of
  SGD, momentum, Adam, AdamW and AdamW with BF16 moments, 8 seeds: 6,578 numbers, 3,494 of them bit-identical, the worst
  0.04 of the tolerance. Against one step on the 20 rows at once, the same up to the order of the sums: the worst 0.02 of
  the tolerance, 1.2e-7 at most on a parameter. The owned forms and `fast.bend`'s linear layers give the same bits.
- **Over the parallel and device lanes**: `demo/accum_lanes.bend` with `reference/accum_lanes.py` (`make
  check-accum-lanes`, and in `make gpu`). Five small models (LinearFirst, RMSNorm and LinearNB; a SwiGLU block; an
  attention block with RoPE; GroupNorm, SiLU and Linear; Embedding and LinearNB under cross-entropy), each trained two
  AdamW steps four ways from one source whose layers come from `main.bend`, `fast.bend` or `gpu.bend`: four
  micro-batches a step, one step on their rows at once, k = 1, and `Train.step`. On the CPU threads both lanes give
  `main.bend`'s bits on 1 and 16 threads; k = 4 is the one big step within 0.03 of the tolerance. On the RTX 4050 the
  linear stack is exact, k = 1 is `Train.step` to the bit, and the other four differ by 80 to 468 ulps after two steps,
  all of it CUDA's float `exp` (with `exp_cr` every number is `main.bend`'s).
- **Mutants**: eight more (the 1/k left out of the gradient or the loss, a micro-batch starting from a zero gradient in
  either form, a step or an advance of t at every micro-batch, a flush that leaves t where it was, in either form), and
  four over the lanes (a `Fork` whose second layer starts its backward pass from a zero gradient, which only an
  accumulated step can see; the GroupNorm step trading the gain's and the bias's gradients in `gpu.bend` and in
  `fast.bend`; `gpu.bend`'s Residual dropping its skip connection).
- **Moving from 2.0.2**: nothing changes for a program that does not call the new functions.

### Attention (#28)

- **Attention sums only what a position can see** (#28). Its forward pass's `O = P·V` and its backward pass's
  `dQ = G·K`, `dK += Gᵀ·Q` and `dV += Pᵀ·dO` used to sum every key and every row, the masked half included, where `P`
  and `G` are 0; now row `i` sums keys `0 .. i` and key `j`'s row sums rows `j .. t-1`. On finite numbers the bits are
  the same (`bench/bits.txt` unchanged, and `make check-triangle` compares the two forms on every cell); counted with
  Cachegrind on one thread, SmolLM2-135M's attention runs 30% fewer instructions forward and 34% forward and backward,
  the ViT block's one-head attention (t = 16) 24% and 26%. `FullAttention` is unchanged. One difference: an infinity
  or a NaN in V or K at a position a row cannot see no longer makes that row's output or dQ NaN, as PyTorch's does
  (`docs/limitations.md`).
- **`fast.bend`'s `Attention` runs its forward pass in parallel blocks over shared arrays** (#28): `main.bend`'s units
  (one sequence's key/value group each) read Q, K and V and write the output where they are, instead of being copied
  out and back in on one thread. The same bits (`demo/fast_test.bend` compares y, dQ, dK and dV with `main.bend`'s).
  The core alone at the ViT block's shape (64 sequences of 16, one head of 64), on an idle machine: 0.27 ms a pass on
  16 threads against 0.73 with `main.bend`'s forward (0.77 before the triangle sums), and 0.54 against 0.89 on one;
  at SmolLM2-135M's, 3.4 against 4.1 ms on 16 threads and about the same on one. The backward pass is still
  `main.bend`'s.

### Laws

Four new laws in `LAWS.bend`, proved in `PROOF.bend` and checked again by `bend PROOF.bend --verdict`, 39 laws in all
(with `dtype.bend`'s `unpair_pair`). Two by @adeerkhan (#32), and two for attention's sums:

- `causal_pv`: every number of row `i` of attention's `O = P·V` (and `dQ = G·K`) is a sum of exactly `i + 1` products,
  the count `trow0` gives each of the row's dots.
- `causal_kv`: every number of key `j`'s row of `dK` and `dV` is a sum of exactly `1 + r` products when `r` keys come
  after `j`, so `t - j`, the count `tcol0` gives each dot.
- `conv_roundtrip_window`: the round trip at **any** stride and padding, per cell. Slot `(ch, ky, kx)` of output
  position `(oy, ox)` is image cell `(oy·s + ky - p, ox·s + kx - p)`, channel `ch`, so the gather and the scatter name
  the same cell for every kernel, stride and padding — the UNet's `k = 3, s = 2, p = 1` among them. The decode does
  not care about `s` or `p`, so this is one subtraction lemma (`(p + t) - p = t`) away from `conv_roundtrip_ks`, which
  is now proved as its instance (`s = k`, `p = 0`) rather than by a second proof. The bounds are four hypotheses:
  `k = 1 + ky + a`, `k = 1 + kx + b`, and the window being inside the image, `oy·s + ky = p + ty`, `ox·s + kx = p + tx`.
- `conv_roundtrip_ks`: the im2col/col2im round trip at a stride equal to the kernel size. With `s = k` and no
  padding, slot `(ch, ky, kx)` of output position `(oy, ox)` is image cell `(oy·k + ky, ox·k + kx)` of channel `ch`,
  so the gather and the scatter name the same cell — `p_src` both times. That needs the mixed-radix decode
  `i = (p_ch·k + p_ky)·k + p_kx`, which is Bend's unary division twice: the proof of `Nat.divmod.go`'s invariant
  (`divmod_go_spec`, a `def` beside `divmod_one`) then one block count plus a rest per level. `ky < k` and
  `kx < k` are `k = 1 + ky + a` and `k = 1 + kx + b`, as `gqa_head` writes one bound; the bound that `k·k` leaves
  for the inner cell is `radix_bound`. #29's numbers at `k = s = 2, 3, 4` were already exact; the law is what makes
  them so for every `k`.

Two more in the same file, proved the same way, **41 laws in all** (with `dtype.bend`'s `unpair_pair`), both by
@adeerkhan (#34):

- `tile_row`: the `t` output rows of image row `i` are all image row `i`, in order — `i·t + j` divided by `t` is `i`
  for `j ≤ s`, the division fact under the broadcast and under the backward's sum (`div_blocks` then `div_rest`).
- `tile_sum`: every one of the `t` rows of image `i` reads (forward) and adds into (backward) the same cell of image
  row `i`, column `k`, which is what makes the backward a **sum** of the `t` rows rather than the last of them.

Three more in the same file, proved the same way, **44 laws in all** (with `dtype.bend`'s `unpair_pair`), all by
@adeerkhan (#33):

- `rope2_partition`: a head's `2·half_y + 2·half_x` numbers are tiled by `RoPE2d`'s two runs with no gap and no
  overlap, each pair once — the y run starts at `0` and the x run one `half_y` past it, at `j = half_y`.
- `rope2_is_rope`: at `tw = 1` and `half_x = 0` the y run **is** `RoPE`: the same pair `(c, c + half)` of head `head`
  in row `r`, at the same address `r·w + head·(half+half) + c`.
- `rope2_is_rope_tab`: the same for the table entry: at `tw = 1`, `(r mod (th·1))/1 · half + j` is `RoPE`'s
  `(r mod th)·half + j`.

### New

- **`Conv2dNB`**: `Conv2d` without the bias, for the image reference's no-bias convolutions (BendTorch#39). The same
  `im2col` and `col2im` around `LinearNB`'s forward, backward and optimizer step; `P` is the `co x c*k*k` weight
  alone, so the parameter count drops by `co`. Tested against `F.conv2d(..., bias=None)` and its autograd on the
  three shapes `reference/conv2d_check.py` uses: y, dX and dW are **0.000e+00** on all three
  (`reference/conv2dnb_check.py`). By @adeerkhan (#40).

- **`Tile`**: a per-image row broadcast, for adaLN-conditioned blocks (BendTorch#19). Row `i` of a `b x c` input
  becomes rows `i·t .. i·t+t-1` of a `(b·t) x c` output, and the backward sums each image's `t` rows back into its
  row, so `x * (1 + scale)` with a per-image scale is `Tile` then `Mul2` with the tiling on the device. The addressing
  is two `Nat` defs the loop calls through `U32.from_nat` (no division in the hot path: the walk is images → rows →
  row → cells). Two laws: `tile_row` (the `t` rows of image `i` are image row `i` — a division fact, `div_blocks`
  then `div_rest`) and `tile_sum` (all `t` rows read and add into the same cell, so the backward is a sum). Tested
  against PyTorch `repeat_interleave` and its autograd: `0.000e+00` on both, with a last-row backward off by
  `2.000e+00`. Not in this change: the device lane (no CUDA toolkit here).

- **`RoPE2d`**: 2D rotary positions, for an image transformer (BendTorch#18). A head's `2*half_y` numbers rotate
  by the row's `y` and its `2*half_x` by its `x`, so rows `(0,1)` and `(1,0)` of a 2x2 grid rotate by different
  angles at the same distance — the case one 1D rotation cannot express. `RoPE` itself is untouched. The pairs are
  walked in two runs over disjoint dimensions rather than one run with a per-pair choice, which also removes a
  `F32.pow` per pair (`Bool.pick` evaluates both branches). Tested against a hand-built 2D rotation
  (`reference/rope2d_check.py`): `1.192e-07` forward and `1.192e-07` backward, tolerance `1e-6`. Three laws:
  `rope2_partition` (the two runs tile the head: no gap, no overlap, each pair once) and `rope2_is_rope` /
  `rope2_is_rope_tab` (at `tw = 1`, `half_x = 0` the y run is `RoPE`, address and entry). Stated over the addressing
  defs in `main.bend` (`rope2_ht`, `rope2_yy`, `rope2_xx`, `rope2_yaddr`, `rope2_xaddr`, `rope2_yentry`,
  `rope2_xentry`), which the layer's own `U32` code calls through `U32.from_nat`: no comparison, no pick, no `U32`
  arithmetic left in the addressing, so the laws are about the code and not a copy of it. That move is free:
  `1024x64`, 20 passes, 0.02 s before and 0.02 s after (median of three). Still not in this change: the device twin (no CUDA
  toolkit here).

### GPU lane

- `Tile` and `RoPE2d` in `torch/gpu.bend`: both passes give `main.bend`'s bits on the RTX 4050 and on 1 and 16 CPU
  threads (`demo/gpu_layers_test.bend`, cases 19 to 22, so `make gpu` runs 23 cases). RoPE2d's angle tables are built
  on the host, as `RoPE`'s are: CUDA's float `cos` and `sin` round 1 ulp away from the host's on 6 to 20% of a table's
  entries.
- `Conv2dNB` in `torch/gpu.bend`: both passes give `main.bend`'s bits on the RTX 4050 and on 1 and 16 CPU threads, at
  `Conv2d`'s four shapes (`demo/gpu_layers_test.bend`, case 23, so `make gpu` runs 24 cases).

### Tests

- Twelve more mutants: three device mutants each of `Tile` and `RoPE2d`, and six on the CPU (`Tile`'s sum, its source
  row and its destination row; `RoPE2d`'s x position, the x run's offset and the y run's table entry).
- Six mutants of `Conv2dNB`, three on each lane: the forward pass with a zero weight, the input gradient through a
  zero weight, and the weight gradient from zero patches (CPU) or from a zero starting gradient (device).
- 204 mutants in all, with gradient accumulation's twelve: 201 caught, 3 surviving by design.
- `reference/rope2d_check.py` takes the probe from the command line, as `tile_check.py` does. It always read
  `.scratch/rope2d_probe` before, so `make mutate` tested a mutant of `RoPE2d`'s CPU code against the unmutated binary.

## 2.0.2 (2026-11-07)

A patch release: the tensor packages bend-ml published on 2026-11-06, and the speed numbers on Bend 2.0.40. No layer,
law or API changes, and every output is `bench/bits.txt`'s.

- **bend-ml-tensor-array 0.1.20.0, -tensor-fork 0.1.7.1 and -tensor-gpu 0.1.5.2** in every import (from 0.1.18.0,
  0.1.5.0 and 0.1.5.0), which take the same `Mat` again. What changed under BendTorch: tensor-array's dot product
  unrolled by four and `Mat.to_list` without a reverse (0.1.19.0), tensor-fork's `softmax_ce` (0.1.6.0) and its
  products' cells read with `Array.get_unchecked` (0.1.7.x, which needs Bend 2.0.40); tensor-gpu's code is 0.1.5.0's.
  The products keep their bits.
- **The speed table and `BENCHMARK.md` on Bend 2.0.40**, from a round on an idle machine on 2026-11-07: SmolLM2-135M's
  step on 16 threads 4.3 s to 3.3 s, the decoder block 12% faster on 16 threads and 8% on the GPU; one thread did not
  move. Bend and the code both changed since the last round (2026-10-24), so the numbers do not say which gave what.
  The block's first loss on the GPU is now the CPU's bits; it was 1 ulp off before.
- **Checks on Bend 2.0.40**: laws, check, bits, corpus, gpu, smol, smol-gpu, bits-smol, reader, check-js and
  mutate-smol, and every mutant in CI.
- **Moving from 2.0.1**: the import lines only, `2.0.1.0` to `2.0.2.0` in all four packages; checkpoints read as before.

## 2.0.1 (2026-11-06)

A patch release: Bend 2.0.40, and the weights reader on bend-ml-binary-read 0.2.0.1. No layer, law or API changes,
and every output is `bench/bits.txt`'s.

- **Bend 2.0.40.** The pin moves from 2.0.39; `make setup` installs it, checked by SHA256.
- **The reader on bend-ml-binary-read 0.2.0.1.** Bend 2.0.40 removed a runtime binding that binary-read 0.2.0.0's
  JavaScript lane called (bendlang/bend#2063), so on 2.0.40 its reads fail there with `sys.pread is not a function`,
  and `bendtorch-load@2.0.0.0` imports 0.2.0.0. 0.2.0.1 reads through `fs`, as Base's own reads do; the C lane is the
  same code. `torch/load.bend` and the reader test import it.
- **Checks on Bend 2.0.40**: laws, check, bits, corpus, gpu, smol, smol-gpu, bits-smol, reader and check-js, and every
  mutant in CI.
- **CI** runs what a change reads on pushes and pull requests to `devel` (`make affected`, `scripts/affected.py`);
  releases, Sundays and manual runs still run everything.
- **Moving from 2.0.0**: the import lines only, `2.0.0.0` to `2.0.1.0` in all four packages; checkpoints read as before.

## 2.0.0 (2026-11-03)

A major release, with two changes. The optimizer configuration has one owner, the training state: until 1.3.0 a
`Train` was made with one configuration and every step was given another, and the step's won; now `Train.new` takes the
configuration once, every step runs under it, and a step whose configuration does not fit the state's buffers is
refused (#24, #27). And convolution: `Conv2d` on the CPU and GPU lanes, with `im2col` and `col2im` (#22), which with
`GroupNorm` and `FullAttention` completes the layers an unconditional image model (a UNet) needs (#15). The first breaks
how the package is used (`Train.step`'s arguments and the record's fields), hence the major; the second is new names
only. Both by or with @adeerkhan.

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

### Conv2d (#22, for #15)

- `Conv2d(n, c, co, k, hs, ws, p, s1, par)`: a 2-D convolution of `n` images, `c` channels in and `co` out, a `k x k`
  kernel, padding `p` and stride `1 + s1` (so it is never 0), images channels last. The image's size is written as the
  kernel plus a surplus, `h + 2p = k + hs` (and `ws` for the width), so a window bigger than its image cannot be
  written and no call site needs a lemma; a UNet's `k = 3`, `s = 2`, `p = 1` at 64 x 64 is `hs = ws = 63`.
- It is `im2col` then `Linear`'s three products, so its parameters are `Lin<c·k·k, co>` and the optimizer, the zero
  parameters and the checkpoints serve it unchanged; the patch matrix is the forward's state, not rebuilt in the
  backward.
- `im2col` and `col2im` in `bendtorch`, the gather and its scatter, channel-major inside a patch as PyTorch's `unfold`;
  a padded cell reads 0 and no out-of-bounds cell is touched. In `bendtorch-gpu` both are cell bangs; a device
  `col2im` leaf sums the patches that read its cell in the CPU lane's order, so the device gives the CPU's bits.
- Laws `im2col_rows` (the gathered list holds `n·oh·ow·c·k·k` numbers) and `conv_roundtrip` (with a 1 x 1 kernel, no
  padding and stride 1, `col2im(im2col(x)) = x` cell for cell). The general `k = s` round trip and Conv2d's adjoint law
  are tested, not proved: #29.
- Tests: `demo/conv2d_probe.bend` with `reference/conv2d_check.py` (y, dX, dW and db against `F.conv2d` and its
  autograd at three shapes: a padded border, stride 2 with an even and an odd surplus, where the last window floors;
  every difference 0), `demo/im2col_probe.bend` with `reference/im2col_check.py` (against `F.unfold`, and the round
  trips at `k = 1` and `k = s = 2`), and case 16 of `demo/gpu_layers_test.bend` (2 images of 5 x 3 and of CIFAR's
  32 x 32, 3 x 3, padding 1): the device lane gives the CPU lane's bits, 0 ulps, and 0 with `exp_cr` too.
- The two probes are built with the demos and passed to their checks, so `make check` tests the current code (before,
  a probe that already existed was not rebuilt).

### Laws

Six new laws in `LAWS.bend`, proved in `PROOF.bend` and checked again by `bend PROOF.bend --verdict`, 35 laws in all
(with `dtype.bend`'s `unpair_pair`): Conv2d's two (above) and the guard's four:

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

`make mutate`: 167 mutants, 167 as expected (149 at 1.3.0); 164 caught, and the same three surviving by design.

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
- Eight new for Conv2d: on `main.bend`, the gather reading the padding cells, a patch's row taking the kernel's column
  offset, the channel inside a patch read from the kernel's row, `col2im` adding padding cells into the image (against
  `conv2d_check`), and `col2im`'s scatter starting one gathered cell late (against `im2col_check`: every case of
  `conv2d_check` is padded, so its first gathered cell is padding and the shift there changes nothing); on the device
  lane, the gather reading the padding, a cell's sum skipping its last candidate along x, and the candidates along y
  starting one row late (against the GPU layers gate).

### Compatibility

- **Bits**: `bench/bits.txt` does not move: the tiny Llama's training and sampling, MNIST and SmolLM2-135M's loss and
  steps give 1.3.0's bits on 1 and 16 threads, and the GPU builds give the CPU's. A program that passed the same
  configuration to `Train.new` and to every step gets the same bits after moving; one that passed each step its own
  (a schedule) gets the same bits through `Train.with_cfg`.
- **Checkpoints**: the file layout is unchanged (the parameters, the moments in use, then Adam's 1 - b1^t and
  1 - b2^t); a 1.x checkpoint reads into 2.0.0. The configuration is not in the file: a checkpoint resumes under the
  configuration of the `Train` it is read into. One edge: a `Train` made with a configuration `OptCfg.ok` rejects and
  saved before its first step holds 0 for 1 - b^t where 1.3.0 held NaN; any step on it is still refused.
- **Lanes**: `bendtorch-fast` is unchanged; `bendtorch-gpu` adds `im2col`, `col2im` and `Conv2d` and is otherwise
  unchanged. Both lanes' optimizer steps receive the guard's optimizer from the core's `Train.step`. `bendtorch-load`'s
  `train` carries the record's new field. `Conv2d` is new names only: nothing that worked before changes.
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

- `docs/design.md`, `docs/limitations.md`, `torch/README.md` (the API, the rule, the move from 1.x, Conv2d, the laws in
  plain language) and the README (the example, Conv2d in place of "no convolution yet", 35 laws, 167 planted faults, the
  mutants figure rendered again from `scripts/mutate.py`) describe the new design.
- `docs/media/mutants.mjs`: the figure's tally keeps its spaces under librsvg (`xml:space="preserve"`), and the header
  says how to render it without a browser.
- CI: the job's limit is 90 minutes (the full run, with the mutants and the JavaScript lane, took 57 to 60 minutes at
  1.3.0, and the 2026-11-02 nightly passed every step and was cut at 60).

Thanks to @adeerkhan for #22, Conv2d and its laws on both lanes, and for #24, the report, the repro and its
measurements.

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
