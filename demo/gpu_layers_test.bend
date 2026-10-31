# torch/gpu.bend's element-wise layers, RMSNorm, RoPE, attention, optimizer step, losses and embedding against
# torch/main.bend's, on the same inputs at SmolLM2-135M's sizes (8 sequences of 64 positions: 512 rows; width 576, 9
# query heads and 3 key/value heads of 64; MLP width 1536). Prints, for each case, the largest distance in ulps between the two results over every
# output (forward y, backward dX and the gradient; for the optimizer, the weights, the zeroed gradient and the moments):
#   SiLU, Mul2, Mat.sum2 and Mat.dup, RMSNorm, RoPE, Attention, SGD, momentum, AdamW, MSE (loss and gradient),
#   Embedding (rows and gradient), CrossEntropy (loss and gradient), attention at small and odd sizes, GroupNorm
# With `+RTS --gpu off -RTS` or without CUDA every case is 0 (the same bits). On the GPU, SiLU, attention and cross-entropy take
# CUDA's float exp, whose last bit can differ from the host's: 1 ulp is accepted for an exponential (my decision,
# 2026-10-18), and the differences that follow from it downstream are what this test shows; every other case is 0.
#   gpu_layers_test        every case, one line          gpu_layers_test <k>   case k alone (0..13, in the order above)
import Base
import ../torch/main.bend as T
import ../torch/gpu.bend as G
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# deterministic values in [-1, 1), no specials (softmax and the norms would carry a NaN through a whole row)
def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  ((U32.to_f32(h) / 500.0 : F32) - 1.0 : F32)

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def arr(+cnt: Nat, +seed: U32) -> Array<F32>:
  fill(cnt, 0, seed, Array.new(F32, TA.cap_depth(cnt), 0.0))

def mat(+r: Nat, +c: Nat, +seed: U32) -> TA.Mat<r, c>:
  TA.Mat{arr(Nat.mul(r, c), seed)}

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def u32_max(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_lt(a, b), b, a)

# the distance in ulps between two floats: 0 for equal bits or two NaNs, across zero when the signs differ
def ulps(+x: F32, +y: F32) -> U32:
  +a = bits(x)
  +b = bits(y)
  +ma = U32.and(a, 2147483647)
  +mb = U32.and(b, 2147483647)
  Bool.pick(U32, Bool.or(U32.is_eq(a, b), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, Bool.pick(U32, U32.is_eq(U32.shrn(a, 31n), U32.shrn(b, 31n)), Bool.pick(U32, U32.is_lt(ma, mb), (mb - ma : U32), (ma - mb : U32)), (ma + mb : U32)))

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), u32_max(acc, ulps(x, y)), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def du(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

def dm(+r: Nat, +c: Nat, x: TA.Mat<r, c>, y: TA.Mat<r, c>) -> U32:
  match x y:
    case TA.Mat{a} TA.Mat{b}: du(Nat.mul(r, c), a, b)

def max3(+a: U32, +b: U32, +c: U32) -> U32:
  u32_max(u32_max(a, b), c)

def max4(+a: U32, +b: U32, +c: U32, +d: U32) -> U32:
  u32_max(u32_max(a, b), u32_max(c, d))

# one forward and one backward of a layer: its output, its dX and its gradient
type RR<-P: Type, -X: Type, -Y: Type> is Type:
  RR{y: Y, dx: X, g: P}

def run_b(-P: Type, -X: Type, -Y: Type, y: Y, r: T.Bwd<P, X>) -> RR<P, X, Y>:
  match r:
    case T.Bwd{p, dx, g}: RR{y, dx, g}

def run_f(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>, dy: Y, g: P, r: T.Fwd<P, Y, S>) -> RR<P, X, Y>:
  match r:
    case T.Fwd{p, y, s}: run_b(P, X, Y, y, T.Layer.bwd(P, X, Y, S, l, p, s, dy, g))

def run(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>, p: P, x: X, dy: Y, g: P) -> RR<P, X, Y>:
  run_f(~P, ~X, ~Y, ~S, ~l, dy, g, T.Layer.fwd(P, X, Y, S, l, p, x))

# ---- the element-wise layers ----

def c_m(+n: Nat, +c: Nat, a: RR<Unit, TA.Mat<n, c>, TA.Mat<n, c>>, b: RR<Unit, TA.Mat<n, c>, TA.Mat<n, c>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: u32_max(dm(n, c, y1, y2), dm(n, c, d1, d2))

def silu() -> U32:
  c_m(512n, 1536n, run(~Unit, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~T.SiLU(512n, 1536n), Unit{}, mat(512n, 1536n, 1), mat(512n, 1536n, 2), Unit{}), run(~Unit, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~TA.Mat<512n, 1536n>, ~G.SiLU(512n, 1536n), Unit{}, mat(512n, 1536n, 1), mat(512n, 1536n, 2), Unit{}))

def c_mul(+n: Nat, +c: Nat, a: RR<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>>, b: RR<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>>) -> U32:
  match a b:
    case RR{y1, T.Two{a1, b1}, g1} RR{y2, T.Two{a2, b2}, g2}: max3(dm(n, c, y1, y2), dm(n, c, a1, a2), dm(n, c, b1, b2))

def two(+r: Nat, +c: Nat, +s: U32) -> T.Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  T.Two{mat(r, c, s), mat(r, c, (s + 1 : U32))}

def mul2() -> U32:
  c_mul(512n, 1536n, run(~Unit, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~TA.Mat<512n, 1536n>, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~T.Mul2(512n, 1536n), Unit{}, two(512n, 1536n, 11), mat(512n, 1536n, 13), Unit{}), run(~Unit, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~TA.Mat<512n, 1536n>, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~G.Mul2(512n, 1536n), Unit{}, two(512n, 1536n, 11), mat(512n, 1536n, 13), Unit{}))

def c_two(+r: Nat, +c: Nat, a: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>, b: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>) -> U32:
  match a b:
    case T.Two{a1, b1} T.Two{a2, b2}: u32_max(dm(r, c, a1, a2), dm(r, c, b1, b2))

def sumdup() -> U32:
  u32_max(dm(512n, 576n, T.Mat.sum2(512n, 576n, mat(512n, 576n, 21), mat(512n, 576n, 22)), G.Mat.sum2(512n, 576n, mat(512n, 576n, 21), mat(512n, 576n, 22))), c_two(512n, 576n, T.Mat.dup(512n, 576n, mat(512n, 576n, 23)), G.Mat.dup(512n, 576n, mat(512n, 576n, 23))))

# ---- RMSNorm: y, dx and the gain's gradient (added into a gradient that is not zero); and 7 rows of 5 ----

def c_rms(+n: Nat, +d: Nat, a: RR<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>, b: RR<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: max3(dm(n, d, y1, y2), dm(n, d, d1, d2), dm(1n, d, g1, g2))

def rms1(+s: U32) -> U32:
  c_rms(7n, 5n, run(~TA.Mat<1n, 5n>, ~TA.Mat<7n, 5n>, ~TA.Mat<7n, 5n>, ~TA.Mat<7n, 5n>, ~T.RMSNorm(7n, 5n, 0.00001), mat(1n, 5n, s), mat(7n, 5n, (s + 1 : U32)), mat(7n, 5n, (s + 2 : U32)), mat(1n, 5n, (s + 3 : U32))), run(~TA.Mat<1n, 5n>, ~TA.Mat<7n, 5n>, ~TA.Mat<7n, 5n>, ~TA.Mat<7n, 5n>, ~G.RMSNorm(7n, 5n, 0.00001), mat(1n, 5n, s), mat(7n, 5n, (s + 1 : U32)), mat(7n, 5n, (s + 2 : U32)), mat(1n, 5n, (s + 3 : U32))))

def rms() -> U32:
  u32_max(c_rms(512n, 576n, run(~TA.Mat<1n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~T.RMSNorm(512n, 576n, 0.00001), mat(1n, 576n, 31), mat(512n, 576n, 32), mat(512n, 576n, 33), mat(1n, 576n, 34)), run(~TA.Mat<1n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~G.RMSNorm(512n, 576n, 0.00001), mat(1n, 576n, 31), mat(512n, 576n, 32), mat(512n, 576n, 33), mat(1n, 576n, 34))), rms1(35))

# ---- RoPE on the 9 query heads ----

def c_rope(+n: Nat, +w: Nat, a: RR<Unit, TA.Mat<n, w>, TA.Mat<n, w>>, b: RR<Unit, TA.Mat<n, w>, TA.Mat<n, w>>) -> U32:
  match a b:
    case RR{y1, d1, g1} RR{y2, d2, g2}: u32_max(dm(n, w, y1, y2), dm(n, w, d1, d2))

def rope() -> U32:
  c_rope(512n, 576n, run(~Unit, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~Unit, ~T.RoPE(512n, 576n, 64n, 9n, 32n, 100000.0, {==}), Unit{}, mat(512n, 576n, 41), mat(512n, 576n, 42), Unit{}), run(~Unit, ~TA.Mat<512n, 576n>, ~TA.Mat<512n, 576n>, ~Unit, ~G.RoPE(512n, 576n, 64n, 9n, 32n, 100000.0, {==}), Unit{}, mat(512n, 576n, 41), mat(512n, 576n, 42), Unit{}))

# ---- attention: y, dQ, dK and dV; and, apart, smaller sizes: odd t (P's lower triangle is dealt in pairs of rows, the
# middle row of an odd t alone), t = 1 and 2, one group, one head a group ----

def qkv(+n: Nat, +w: Nat, +kvw: Nat, +s: U32) -> T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>:
  T.Two{mat(n, w, s), T.Two{mat(n, kvw, (s + 1 : U32)), mat(n, kvw, (s + 2 : U32))}}

def c_att(+n: Nat, +w: Nat, +kvw: Nat, a: RR<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>>, b: RR<Unit, T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>>) -> U32:
  match a b:
    case RR{y1, T.Two{q1, T.Two{k1, v1}}, g1} RR{y2, T.Two{q2, T.Two{k2, v2}}, g2}: max4(dm(n, w, y1, y2), dm(n, w, q1, q2), dm(n, kvw, k1, k2), dm(n, kvw, v1, v2))

def attn_o0() -> U32:
  c_att(14n, 16n, 8n, run(~Unit, ~T.Two<TA.Mat<14n, 16n>, T.Two<TA.Mat<14n, 8n>, TA.Mat<14n, 8n>>>, ~TA.Mat<14n, 16n>, ~T.AttnS, ~G.Attention(14n, 16n, 8n, 2n, 7n, 2n, 2n, 4n, 0.5, {==}, {==}, {==}), Unit{}, qkv(14n, 16n, 8n, 101), mat(14n, 16n, 104), Unit{}), run(~Unit, ~T.Two<TA.Mat<14n, 16n>, T.Two<TA.Mat<14n, 8n>, TA.Mat<14n, 8n>>>, ~TA.Mat<14n, 16n>, ~T.AttnS, ~T.Attention(14n, 16n, 8n, 2n, 7n, 2n, 2n, 4n, 0.5, {==}, {==}, {==}), Unit{}, qkv(14n, 16n, 8n, 101), mat(14n, 16n, 104), Unit{}))

def attn_o1() -> U32:
  c_att(3n, 6n, 2n, run(~Unit, ~T.Two<TA.Mat<3n, 6n>, T.Two<TA.Mat<3n, 2n>, TA.Mat<3n, 2n>>>, ~TA.Mat<3n, 6n>, ~T.AttnS, ~G.Attention(3n, 6n, 2n, 3n, 1n, 1n, 3n, 2n, 0.5, {==}, {==}, {==}), Unit{}, qkv(3n, 6n, 2n, 105), mat(3n, 6n, 108), Unit{}), run(~Unit, ~T.Two<TA.Mat<3n, 6n>, T.Two<TA.Mat<3n, 2n>, TA.Mat<3n, 2n>>>, ~TA.Mat<3n, 6n>, ~T.AttnS, ~T.Attention(3n, 6n, 2n, 3n, 1n, 1n, 3n, 2n, 0.5, {==}, {==}, {==}), Unit{}, qkv(3n, 6n, 2n, 105), mat(3n, 6n, 108), Unit{}))

def attn_o2() -> U32:
  c_att(2n, 15n, 15n, run(~Unit, ~T.Two<TA.Mat<2n, 15n>, T.Two<TA.Mat<2n, 15n>, TA.Mat<2n, 15n>>>, ~TA.Mat<2n, 15n>, ~T.AttnS, ~G.Attention(2n, 15n, 15n, 1n, 2n, 3n, 1n, 5n, 0.5, {==}, {==}, {==}), Unit{}, qkv(2n, 15n, 15n, 109), mat(2n, 15n, 112), Unit{}), run(~Unit, ~T.Two<TA.Mat<2n, 15n>, T.Two<TA.Mat<2n, 15n>, TA.Mat<2n, 15n>>>, ~TA.Mat<2n, 15n>, ~T.AttnS, ~T.Attention(2n, 15n, 15n, 1n, 2n, 3n, 1n, 5n, 0.5, {==}, {==}, {==}), Unit{}, qkv(2n, 15n, 15n, 109), mat(2n, 15n, 112), Unit{}))

def attn_o3() -> U32:
  c_att(66n, 6n, 3n, run(~Unit, ~T.Two<TA.Mat<66n, 6n>, T.Two<TA.Mat<66n, 3n>, TA.Mat<66n, 3n>>>, ~TA.Mat<66n, 6n>, ~T.AttnS, ~G.Attention(66n, 6n, 3n, 2n, 33n, 1n, 2n, 3n, 0.5, {==}, {==}, {==}), Unit{}, qkv(66n, 6n, 3n, 113), mat(66n, 6n, 116), Unit{}), run(~Unit, ~T.Two<TA.Mat<66n, 6n>, T.Two<TA.Mat<66n, 3n>, TA.Mat<66n, 3n>>>, ~TA.Mat<66n, 6n>, ~T.AttnS, ~T.Attention(66n, 6n, 3n, 2n, 33n, 1n, 2n, 3n, 0.5, {==}, {==}, {==}), Unit{}, qkv(66n, 6n, 3n, 113), mat(66n, 6n, 116), Unit{}))

def attn() -> U32:
  # gpu.bend's run first: main.bend's attention runs its sequences in parallel lets on the host, and a bang that follows
  # them in one evaluation runs on the CPU threads, so with main.bend's first the device was never used here
  c_att(512n, 576n, 192n, run(~Unit, ~T.Two<TA.Mat<512n, 576n>, T.Two<TA.Mat<512n, 192n>, TA.Mat<512n, 192n>>>, ~TA.Mat<512n, 576n>, ~T.AttnS, ~G.Attention(512n, 576n, 192n, 8n, 64n, 3n, 3n, 64n, 0.125, {==}, {==}, {==}), Unit{}, qkv(512n, 576n, 192n, 51), mat(512n, 576n, 54), Unit{}), run(~Unit, ~T.Two<TA.Mat<512n, 576n>, T.Two<TA.Mat<512n, 192n>, TA.Mat<512n, 192n>>>, ~TA.Mat<512n, 576n>, ~T.AttnS, ~T.Attention(512n, 576n, 192n, 8n, 64n, 3n, 3n, 64n, 0.125, {==}, {==}, {==}), Unit{}, qkv(512n, 576n, 192n, 51), mat(512n, 576n, 54), Unit{}))

# the smaller sizes, a 13th number (gpu.bend's run first, as above): 0 on the CPU threads; on the GPU CUDA's exp moves their small numbers (dK, dV)
# by more ulps than at the sizes above, as in the old attention (night/gpu-rows' notes), so make gpu prints it
def attn_small() -> U32:
  max4(attn_o0(), attn_o1(), attn_o2(), attn_o3())

# ---- the optimizer step on a 1536 x 576 matrix: weights, gradient (zeroed) and moments ----

def c_st(+r: Nat, +c: Nat, a: T.OptState<TA.Mat<r, c>>, b: T.OptState<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.OneMom{m1} T.OneMom{m2}: dm(r, c, m1, m2)
    case T.TwoMom{m1, v1} T.TwoMom{m2, v2}: u32_max(dm(r, c, m1, m2), dm(r, c, v1, v2))
    case _ _: 0

def c_step(+r: Nat, +c: Nat, a: T.Step<TA.Mat<r, c>>, b: T.Step<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.Step{p1, g1, s1} T.Step{p2, g2, s2}: max3(dm(r, c, p1, p2), dm(r, c, g1, g2), c_st(r, c, s1, s2))

# second moments are not negative
def sq(k: Nat, +i: U32, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{a, x}: a
    case 1n+p Tuple{a, +x}: sq(p, (i + 1 : U32), Array.get(F32, Array.set(F32, a, i, (x * x : F32)), (i + 1 : U32)))

def arr2(+cnt: Nat, +seed: U32) -> Array<F32>:
  sq(cnt, 0, Array.get(F32, arr(cnt, seed), 0))

def st_of(+r: Nat, +c: Nat, +k: Nat, +s: U32) -> T.OptState<TA.Mat<r, c>>:
  match k:
    case 0n: T.NoMom{}
    case 1n: T.OneMom{mat(r, c, s)}
    case _: T.TwoMom{mat(r, c, s), TA.Mat{arr2(Nat.mul(r, c), (s + 1 : U32))}}

def step(+op: T.Opt, +k: Nat) -> U32:
  c_step(1536n, 576n, T.mat_step(1536n, 576n, op, mat(1536n, 576n, 61), mat(1536n, 576n, 62), st_of(1536n, 576n, k, 63)), G.mat_step(1536n, 576n, op, mat(1536n, 576n, 61), mat(1536n, 576n, 62), st_of(1536n, 576n, k, 63)))

# ---- MSE: the loss and the gradient, at the block's size and at counts with fewer powers of two ----

def c_loss(+n: Nat, +c: Nat, a: T.LossOut<TA.Mat<n, c>>, b: T.LossOut<TA.Mat<n, c>>) -> U32:
  match a b:
    case T.LossOut{+l1, d1} T.LossOut{+l2, d2}: u32_max(ulps(l1, l2), dm(n, c, d1, d2))

def mse1(+n: Nat, +c: Nat, +s: U32) -> U32:
  c_loss(n, c, T.MSE(n, c, mat(n, c, s), mat(n, c, (s + 1 : U32))), G.MSE(n, c, mat(n, c, s), mat(n, c, (s + 1 : U32))))

def mse() -> U32:
  max3(mse1(512n, 576n, 71), mse1(7n, 5n, 73), mse1(100n, 10n, 75))

# ---- Embedding: the rows looked up and the gradient (added into one that is not zero), ids repeated so that rows are
# added into the same gradient row in order; and fewer ids than rows ----

# k ids below m, some repeated
def idl(k: Nat, +i: U32, +m: U32, +seed: U32) -> List<&2, Nat>:
  match k:
    case 0n: Nil{}
    case 1n+p: U32.to_nat(U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), m)) <> idl(p, (i + 1 : U32), m, seed)

def et_b(-n: Nat, -v: Nat, -d: Nat, y: TA.Mat<n, d>, r: T.Bwd<TA.Mat<v, d>, List<&2, Nat>>) -> T.Two<TA.Mat<n, d>, TA.Mat<v, d>>:
  match r:
    case T.Bwd{p, dx, g}: T.Two{y, g}

def et_f(+n: Nat, +v: Nat, +d: Nat, lb: T.Layer<TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>>, dy: TA.Mat<n, d>, g: TA.Mat<v, d>, r: T.Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>) -> T.Two<TA.Mat<n, d>, TA.Mat<v, d>>:
  match r:
    case T.Fwd{p, y, s}: et_b(n, v, d, y, T.Layer.bwd(TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>, lb, p, s, dy, g))

# the layer's rows and its gradient (la runs the forward pass, lb the backward: two copies of one layer)
def et(+n: Nat, +v: Nat, +d: Nat, la: T.Layer<TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>>, lb: T.Layer<TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>>, +k: Nat, +m: U32, +s: U32) -> T.Two<TA.Mat<n, d>, TA.Mat<v, d>>:
  et_f(n, v, d, lb, mat(n, d, (s + 1 : U32)), mat(v, d, (s + 2 : U32)), T.Layer.fwd(TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>, la, mat(v, d, s), idl(k, 0, m, s)))

def c_emb(+n: Nat, +v: Nat, +d: Nat, a: T.Two<TA.Mat<n, d>, TA.Mat<v, d>>, b: T.Two<TA.Mat<n, d>, TA.Mat<v, d>>) -> U32:
  match a b:
    case T.Two{y1, g1} T.Two{y2, g2}: u32_max(dm(n, d, y1, y2), dm(v, d, g1, g2))

def emb1(+n: Nat, +v: Nat, +d: Nat, +k: Nat, +m: U32, +s: U32) -> U32:
  c_emb(n, v, d, et(n, v, d, T.Embedding(n, v, d), T.Embedding(n, v, d), k, m, s), et(n, v, d, G.Embedding(n, v, d), G.Embedding(n, v, d), k, m, s))

def emb() -> U32:
  max3(emb1(256n, 1000n, 576n, 256n, 97, 81), emb1(7n, 9n, 5n, 7n, 9, 83), emb1(7n, 9n, 5n, 4n, 3, 85))

# ---- CrossEntropy: the loss and the gradient over SmolLM2's 49,152 classes (3 * 2^14, so tree_sum carries a block),
# odd sizes, a label past the last class, fewer labels than rows, one class ----

def ce1(+n: Nat, +c: Nat, +ls: List<&2, Nat>, +s: U32) -> U32:
  c_loss(n, c, T.CrossEntropy(n, c, mat(n, c, s), ls), G.CrossEntropy(n, c, mat(n, c, s), ls))

def ce() -> U32:
  max4(u32_max(ce1(64n, 49152n, idl(64n, 0, 49152, 91), 91), ce1(9n, 1000n, idl(9n, 0, 1000, 92), 92)), u32_max(ce1(7n, 5n, [3n, 9n, 0n, 4n, 1n, 2n, 2n], 93), ce1(5n, 6n, [5n, 0n, 3n], 94)), ce1(3n, 1n, [0n, 0n, 0n], 95), ce1(6n, 13n, idl(6n, 0, 13, 96), 96))

# ---- GroupNorm: y, dx and the gain's and bias's gradients (added into gradients that are not zero); 8 images of 64
# pixels in 8 groups of 16 channels, and 3 images of 5 pixels in 3 groups of 2 ----

def c_gn(+m: Nat, +d: Nat, a: RR<T.LN<d>, TA.Mat<m, d>, TA.Mat<m, d>>, b: RR<T.LN<d>, TA.Mat<m, d>, TA.Mat<m, d>>) -> U32:
  match a b:
    case RR{y1, d1, T.LN{g1, b1}} RR{y2, d2, T.LN{g2, b2}}: max4(dm(m, d, y1, y2), dm(m, d, d1, d2), dm(1n, d, g1, g2), dm(1n, d, b1, b2))

def lnp(+d: Nat, +s: U32) -> T.LN<d>:
  T.LN{mat(1n, d, s), mat(1n, d, (s + 1 : U32))}

def gn1(+s: U32) -> U32:
  c_gn(15n, 6n, run(~T.LN<6n>, ~TA.Mat<15n, 6n>, ~TA.Mat<15n, 6n>, ~TA.Mat<15n, 6n>, ~T.GroupNorm(3n, 5n, 3n, 2n, 0.00001), lnp(6n, s), mat(15n, 6n, (s + 2 : U32)), mat(15n, 6n, (s + 3 : U32)), lnp(6n, (s + 4 : U32))), run(~T.LN<6n>, ~TA.Mat<15n, 6n>, ~TA.Mat<15n, 6n>, ~TA.Mat<15n, 6n>, ~G.GroupNorm(3n, 5n, 3n, 2n, 0.00001), lnp(6n, s), mat(15n, 6n, (s + 2 : U32)), mat(15n, 6n, (s + 3 : U32)), lnp(6n, (s + 4 : U32))))

def gn() -> U32:
  u32_max(c_gn(512n, 128n, run(~T.LN<128n>, ~TA.Mat<512n, 128n>, ~TA.Mat<512n, 128n>, ~TA.Mat<512n, 128n>, ~T.GroupNorm(8n, 64n, 8n, 16n, 0.00001), lnp(128n, 101), mat(512n, 128n, 103), mat(512n, 128n, 104), lnp(128n, 105)), run(~T.LN<128n>, ~TA.Mat<512n, 128n>, ~TA.Mat<512n, 128n>, ~TA.Mat<512n, 128n>, ~G.GroupNorm(8n, 64n, 8n, 16n, 0.00001), lnp(128n, 101), mat(512n, 128n, 103), mat(512n, 128n, 104), lnp(128n, 105))), gn1(111))

# every case, on one line
def all_cases() -> IO(Unit):
  IO.print(U32.show(silu()) ++ " " ++ U32.show(mul2()) ++ " " ++ U32.show(sumdup()) ++ " " ++ U32.show(rms()) ++ " " ++ U32.show(rope()) ++ " " ++ U32.show(attn()) ++ " " ++ U32.show(step(T.SGD{0.01}, 0n)) ++ " " ++ U32.show(step(T.Momentum{0.01, 0.9}, 1n)) ++ " " ++ U32.show(step(T.Opt.at(T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}, 0.1, 0.001), 2n)) ++ " " ++ U32.show(mse()) ++ " " ++ U32.show(emb()) ++ " " ++ U32.show(ce()) ++ " " ++ U32.show(attn_small()) ++ " " ++ U32.show(gn()))

# case k alone (0 = SiLU .. 11 = CrossEntropy, 12 = attention at small sizes, 13 = GroupNorm). In the line above, a case's bangs can follow host work in the same
# evaluation and then run on the CPU threads: under nsys, the whole line made 15 bangs on the device, and cross-entropy
# read 0 ulps there against 4 alone. So make gpu runs each case as its own program (nsys: every case's bangs on the
# device, 2026-10-21).
def one_13(k: Nat) -> U32:
  match k:
    case 0n: gn()
    case 1n+q: 4294967295

def one_12(k: Nat) -> U32:
  match k:
    case 0n: attn_small()
    case 1n+q: one_13(q)

def one_11(k: Nat) -> U32:
  match k:
    case 0n: ce()
    case 1n+q: one_12(q)

def one_10(k: Nat) -> U32:
  match k:
    case 0n: emb()
    case 1n+q: one_11(q)

def one_9(k: Nat) -> U32:
  match k:
    case 0n: mse()
    case 1n+q: one_10(q)

def one_8(k: Nat) -> U32:
  match k:
    case 0n: step(T.Opt.at(T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}, 0.1, 0.001), 2n)
    case 1n+q: one_9(q)

def one_7(k: Nat) -> U32:
  match k:
    case 0n: step(T.Momentum{0.01, 0.9}, 1n)
    case 1n+q: one_8(q)

def one_6(k: Nat) -> U32:
  match k:
    case 0n: step(T.SGD{0.01}, 0n)
    case 1n+q: one_7(q)

def one_5(k: Nat) -> U32:
  match k:
    case 0n: attn()
    case 1n+q: one_6(q)

def one_4(k: Nat) -> U32:
  match k:
    case 0n: rope()
    case 1n+q: one_5(q)

def one_3(k: Nat) -> U32:
  match k:
    case 0n: rms()
    case 1n+q: one_4(q)

def one_2(k: Nat) -> U32:
  match k:
    case 0n: sumdup()
    case 1n+q: one_3(q)

def one_1(k: Nat) -> U32:
  match k:
    case 0n: mul2()
    case 1n+q: one_2(q)

def one(k: Nat) -> U32:
  match k:
    case 0n: silu()
    case 1n+q: one_1(q)

# an argument other than a case number is an error: since Bend 2.0.38 a runtime option goes between +RTS and -RTS, and
# one written without them arrives here while the program runs with the default options
def bad_arg(a: String) -> IO(Unit):
  do IO<Unit>:
    IO.print_err("unexpected argument " ++ a ++ ": a case number or none (runtime options go between +RTS and -RTS)")
    IO.exit(2)

def one_m(m: Maybe<&2, Nat>, a: String) -> IO(Unit):
  match m:
    case Some{k}: IO.print(U32.show(one(k)))
    case None{}: bad_arg(a)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Nil{}: all_cases()
    case Con{prog, Nil{}}: all_cases()
    case Con{prog, Con{+a, Nil{}}}: one_m(Nat.read(a), a)
    case Con{prog, Con{a, Con{b, t}}}: bad_arg(a)

# no argument: every case on one line; `gpu_layers_test <k>`: case k alone; anything else is an error
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
