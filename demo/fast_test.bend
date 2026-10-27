# torch/fast.bend's layers against torch/main.bend's, bit for bit: forward (y) and backward (dX and every gradient)
# of each pair on the same inputs, at MNIST's and the tiny Llama's sizes and at sizes large enough to split into
# parallel blocks. Prints the number of differing words of each case; every one must be 0:
#   LinearFirst 100x784x128 par 3 and par 4 (MNIST's first layer), Linear 100x128x10 par 0 (its second),
#   Linear 128x576x1536 par 4, LinearNB 32x32x64 par 0 (the tiny Llama), LinearNB 128x576x1536 par 3;
#   then the optimizer step (SGD, momentum, AdamW) on a 1536x576 matrix and AdamW on a 576 -> 1536 Linear (weights and
#   bias), the parameter, gradient and both moments after it; SiLU and Mul2 on 512x1536 and SiLU on 32x64, forward and
#   backward; Residual(SiLU) on 512x576, forward and backward (its copies and sums); AdamW with packed BF16 moments
#   (m and v packed, then m packed and v in F32) on a 1537x577 matrix, two steps from new, odd so the last word holds one
#   moment, in 64 blocks; the Embedding's AdamW step on a 1537x577 table.
import Base
import ../torch/main.bend as T
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# deterministic values, with -0, infinity, NaN and a subnormal mixed in by index (as bend-ml's tests/split.bend)
def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  Bool.pick(F32, U32.is_eq(h, 7), (0.0 - 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 13), (1.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 17), (0.0 / 0.0 : F32), Bool.pick(F32, U32.is_eq(h, 23), 0.00000000000000000000000000000000000001, (((U32.to_f32(h) / 250.0 : F32) - 2.0 : F32))))))

def fill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+p: fill(p, (i + 1 : U32), seed, Array.set(F32, a, i, val(i, seed)))

def mat(+r: Nat, +c: Nat, +seed: U32) -> TA.Mat<r, c>:
  TA.Mat{fill(Nat.mul(r, c), 0, seed, Array.new(F32, TA.cap_depth(Nat.mul(r, c)), 0.0))}

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def cmp(k: Nat, +i: U32, acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmp(p, (i + 1 : U32), (acc + Bool.pick(U32, Bool.or(U32.is_eq(bits(x), bits(y)), Bool.and(F32.is_ne(x, x), F32.is_ne(y, y))), 0, 1) : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def diff(+cnt: Nat, x: Array<F32>, y: Array<F32>) -> U32:
  cmp(cnt, 0, 0, Array.get(F32, x, 0), Array.get(F32, y, 0))

def cmpu(k: Nat, +i: U32, acc: U32, ra: Array<U32> & U32, rb: Array<U32> & U32) -> U32:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}: acc
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      cmpu(p, (i + 1 : U32), (acc + Bool.pick(U32, U32.is_eq(x, y), 0, 1) : U32), Array.get(U32, a, (i + 1 : U32)), Array.get(U32, b, (i + 1 : U32)))

def diffu(+cnt: Nat, x: Array<U32>, y: Array<U32>) -> U32:
  cmpu(cnt, 0, 0, Array.get(U32, x, 0), Array.get(U32, y, 0))

# one forward and one backward of a layer: its output, its dX and its gradient
type R<-P: Type, -n: Nat, -i: Nat, -o: Nat> is Type:
  R{y: TA.Mat<n, o>, dx: TA.Mat<n, i>, g: P}

def run_b(-P: Type, -n: Nat, -i: Nat, -o: Nat, y: TA.Mat<n, o>, r: T.Bwd<P, TA.Mat<n, i>>) -> R<P, n, i, o>:
  match r:
    case T.Bwd{p, dx, g}: R{y, dx, g}

def run_f(~n: Nat, ~i: Nat, ~o: Nat, ~P: Type, ~l: T.Layer<P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>, dy: TA.Mat<n, o>, g: P, r: T.Fwd<P, TA.Mat<n, o>, TA.Mat<n, i>>) -> R<P, n, i, o>:
  match r:
    case T.Fwd{p, y, s}: run_b(P, n, i, o, y, T.Layer.bwd(P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>, l, p, s, dy, g))

def run(~n: Nat, ~i: Nat, ~o: Nat, ~P: Type, ~l: T.Layer<P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>, p: P, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: P) -> R<P, n, i, o>:
  run_f(~n, ~i, ~o, ~P, ~l, dy, g, T.Layer.fwd(P, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>, l, p, x))

def d_lin(+n: Nat, +i: Nat, +o: Nat, a: R<T.Lin<i, o>, n, i, o>, b: R<T.Lin<i, o>, n, i, o>) -> U32:
  match a b:
    case R{TA.Mat{y1}, TA.Mat{d1}, T.Lin{TA.Mat{w1}, TA.Mat{b1}}} R{TA.Mat{y2}, TA.Mat{d2}, T.Lin{TA.Mat{w2}, TA.Mat{b2}}}:
      (((diff(Nat.mul(n, o), y1, y2) + diff(Nat.mul(n, i), d1, d2) : U32) + diff(Nat.mul(o, i), w1, w2) : U32) + diff(o, b1, b2) : U32)

def d_nb(+n: Nat, +i: Nat, +o: Nat, a: R<TA.Mat<o, i>, n, i, o>, b: R<TA.Mat<o, i>, n, i, o>) -> U32:
  match a b:
    case R{TA.Mat{y1}, TA.Mat{d1}, TA.Mat{w1}} R{TA.Mat{y2}, TA.Mat{d2}, TA.Mat{w2}}:
      ((diff(Nat.mul(n, o), y1, y2) + diff(Nat.mul(n, i), d1, d2) : U32) + diff(Nat.mul(o, i), w1, w2) : U32)

def lp(+i: Nat, +o: Nat, +s: U32) -> T.Lin<i, o>:
  T.Lin{mat(o, i, s), mat(1n, o, (s + 1 : U32))}

def first3() -> U32:
  d_lin(100n, 784n, 128n, run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~T.LinearFirst(100n, 784n, 128n, 3n), lp(784n, 128n, 1), mat(100n, 784n, 3), mat(100n, 128n, 4), lp(784n, 128n, 5)), run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~F.LinearFirstF(100n, 784n, 128n, 3n), lp(784n, 128n, 1), mat(100n, 784n, 3), mat(100n, 128n, 4), lp(784n, 128n, 5)))

def first4() -> U32:
  d_lin(100n, 784n, 128n, run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~T.LinearFirst(100n, 784n, 128n, 4n), lp(784n, 128n, 11), mat(100n, 784n, 13), mat(100n, 128n, 14), lp(784n, 128n, 15)), run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~F.LinearFirstF(100n, 784n, 128n, 4n), lp(784n, 128n, 11), mat(100n, 784n, 13), mat(100n, 128n, 14), lp(784n, 128n, 15)))

def second() -> U32:
  d_lin(100n, 128n, 10n, run(~100n, ~128n, ~10n, ~T.Lin<128n, 10n>, ~T.Linear(100n, 128n, 10n, 0n), lp(128n, 10n, 21), mat(100n, 128n, 23), mat(100n, 10n, 24), lp(128n, 10n, 25)), run(~100n, ~128n, ~10n, ~T.Lin<128n, 10n>, ~F.LinearF(100n, 128n, 10n, 0n), lp(128n, 10n, 21), mat(100n, 128n, 23), mat(100n, 10n, 24), lp(128n, 10n, 25)))

def big_lin() -> U32:
  d_lin(128n, 576n, 1536n, run(~128n, ~576n, ~1536n, ~T.Lin<576n, 1536n>, ~T.Linear(128n, 576n, 1536n, 4n), lp(576n, 1536n, 31), mat(128n, 576n, 33), mat(128n, 1536n, 34), lp(576n, 1536n, 35)), run(~128n, ~576n, ~1536n, ~T.Lin<576n, 1536n>, ~F.LinearF(128n, 576n, 1536n, 4n), lp(576n, 1536n, 31), mat(128n, 576n, 33), mat(128n, 1536n, 34), lp(576n, 1536n, 35)))

def tiny_nb() -> U32:
  d_nb(32n, 32n, 64n, run(~32n, ~32n, ~64n, ~TA.Mat<64n, 32n>, ~T.LinearNB(32n, 32n, 64n, 0n), mat(64n, 32n, 41), mat(32n, 32n, 43), mat(32n, 64n, 44), mat(64n, 32n, 45)), run(~32n, ~32n, ~64n, ~TA.Mat<64n, 32n>, ~F.LinearNBF(32n, 32n, 64n, 0n), mat(64n, 32n, 41), mat(32n, 32n, 43), mat(32n, 64n, 44), mat(64n, 32n, 45)))

def big_nb() -> U32:
  d_nb(128n, 576n, 1536n, run(~128n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~T.LinearNB(128n, 576n, 1536n, 3n), mat(1536n, 576n, 51), mat(128n, 576n, 53), mat(128n, 1536n, 54), mat(1536n, 576n, 55)), run(~128n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~F.LinearNBF(128n, 576n, 1536n, 3n), mat(1536n, 576n, 51), mat(128n, 576n, 53), mat(128n, 1536n, 54), mat(1536n, 576n, 55)))

# ---- the optimizer step ----

def d_mstate(+r: Nat, +c: Nat, a: T.OptState<TA.Mat<r, c>>, b: T.OptState<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.NoMom{} T.NoMom{}: 0
    case T.OneMom{TA.Mat{x}} T.OneMom{TA.Mat{y}}: diff(Nat.mul(r, c), x, y)
    case T.TwoMom{TA.Mat{x}, TA.Mat{u}} T.TwoMom{TA.Mat{y}, TA.Mat{v}}:
      (diff(Nat.mul(r, c), x, y) + diff(Nat.mul(r, c), u, v) : U32)
    case T.Packed{T.PLeaf{+n1, x, u}} T.Packed{T.PLeaf{+n2, y, v}}:
      ((Bool.pick(U32, U32.is_eq(n1, n2), 0, 1) + diffu(Nat.div(Nat.add(Nat.mul(r, c), 1n), 2n), x, y) : U32) + diffu(Nat.div(Nat.add(Nat.mul(r, c), 1n), 2n), u, v) : U32)
    case T.Packed{T.PLeafV{+n1, x, u}} T.Packed{T.PLeafV{+n2, y, v}}:
      ((Bool.pick(U32, U32.is_eq(n1, n2), 0, 1) + diffu(Nat.div(Nat.add(Nat.mul(r, c), 1n), 2n), x, y) : U32) + diff(Nat.mul(r, c), u, v) : U32)
    case _ _: 1

def d_step(+r: Nat, +c: Nat, a: T.Step<TA.Mat<r, c>>, b: T.Step<TA.Mat<r, c>>) -> U32:
  match a b:
    case T.Step{TA.Mat{w1}, TA.Mat{g1}, s1} T.Step{TA.Mat{w2}, TA.Mat{g2}, s2}:
      ((diff(Nat.mul(r, c), w1, w2) + diff(Nat.mul(r, c), g1, g2) : U32) + d_mstate(r, c, s1, s2) : U32)

def mstate(+r: Nat, +c: Nat, op: T.Opt, +s: U32) -> T.OptState<TA.Mat<r, c>>:
  match op:
    case T.SGD{lr}: T.NoMom{}
    case T.Momentum{lr, mu}: T.OneMom{mat(r, c, (s + 2 : U32))}
    case _: T.TwoMom{mat(r, c, (s + 2 : U32)), mat(r, c, (s + 3 : U32))}

def st_nb(+op: T.Opt, +s: U32) -> U32:
  d_step(1536n, 576n, T.mat_step(1536n, 576n, op, mat(1536n, 576n, s), mat(1536n, 576n, (s + 1 : U32)), mstate(1536n, 576n, op, s)), F.mat_step(1536n, 576n, op, mat(1536n, 576n, s), mat(1536n, 576n, (s + 1 : U32)), mstate(1536n, 576n, op, s)))

def d_lstate2(+i: Nat, +o: Nat, a: T.Two<T.OptState<TA.Mat<o, i>>, T.OptState<TA.Mat<1n, o>>>, b: T.Two<T.OptState<TA.Mat<o, i>>, T.OptState<TA.Mat<1n, o>>>) -> U32:
  match a b:
    case T.Two{w1, b1} T.Two{w2, b2}: (d_mstate(o, i, w1, w2) + d_mstate(1n, o, b1, b2) : U32)

def d_lstate(+i: Nat, +o: Nat, a: T.OptState<T.Lin<i, o>>, b: T.OptState<T.Lin<i, o>>) -> U32:
  d_lstate2(i, o, T.OptState.split(TA.Mat<o, i>, TA.Mat<1n, o>, T.OptState.map(T.Lin<i, o>, T.Two<TA.Mat<o, i>, TA.Mat<1n, o>>, a, x => T.lin_pair(i, o, x), x => T.lin_pair(i, o, x))), T.OptState.split(TA.Mat<o, i>, TA.Mat<1n, o>, T.OptState.map(T.Lin<i, o>, T.Two<TA.Mat<o, i>, TA.Mat<1n, o>>, b, x => T.lin_pair(i, o, x), x => T.lin_pair(i, o, x))))

def d_lstep(+i: Nat, +o: Nat, a: T.Step<T.Lin<i, o>>, b: T.Step<T.Lin<i, o>>) -> U32:
  match a b:
    case T.Step{T.Lin{TA.Mat{w1}, TA.Mat{b1}}, T.Lin{TA.Mat{gw1}, TA.Mat{gb1}}, s1} T.Step{T.Lin{TA.Mat{w2}, TA.Mat{b2}}, T.Lin{TA.Mat{gw2}, TA.Mat{gb2}}, s2}:
      ((((diff(Nat.mul(o, i), w1, w2) + diff(o, b1, b2) : U32) + diff(Nat.mul(o, i), gw1, gw2) : U32) + diff(o, gb1, gb2) : U32) + d_lstate(i, o, s1, s2) : U32)

def adamw() -> T.Opt:
  T.Adam{0.001, 0.9, 0.999, 0.00000001, 0.01, True{}, 0.0100000005, 0.044710837, 0.1, 0.001}

def opt_steps() -> U32:
  ((st_nb(T.SGD{0.1}, 61) + st_nb(T.Momentum{0.1, 0.9}, 65) : U32) + st_nb(adamw(), 69) : U32)

def lin_adamw() -> U32:
  d_lstep(576n, 1536n, T.lin_step(576n, 1536n, adamw(), lp(576n, 1536n, 71), lp(576n, 1536n, 73), T.TwoMom{lp(576n, 1536n, 75), lp(576n, 1536n, 77)}), F.lin_step(576n, 1536n, adamw(), lp(576n, 1536n, 71), lp(576n, 1536n, 73), T.TwoMom{lp(576n, 1536n, 75), lp(576n, 1536n, 77)}))

# ---- AdamW over packed BF16 moments, two steps from new, and the Embedding's step ----

def pk_t2(+op: T.Opt, r: T.Step<TA.Mat<1537n, 577n>>) -> T.Step<TA.Mat<1537n, 577n>>:
  match r:
    case T.Step{w, g, st}: T.mat_step(1537n, 577n, op, w, mat(1537n, 577n, 103), st)

def pk_f2(+op: T.Opt, r: T.Step<TA.Mat<1537n, 577n>>) -> T.Step<TA.Mat<1537n, 577n>>:
  match r:
    case T.Step{w, g, st}: F.mat_step(1537n, 577n, op, w, mat(1537n, 577n, 103), st)

def packed(+v16: Bool) -> U32:
  d_step(1537n, 577n, pk_t2(adamw(), T.mat_step(1537n, 577n, adamw(), mat(1537n, 577n, 101), mat(1537n, 577n, 102), T.Packed{T.PNew{v16}})), pk_f2(adamw(), F.mat_step(1537n, 577n, adamw(), mat(1537n, 577n, 101), mat(1537n, 577n, 102), T.Packed{T.PNew{v16}})))

def emb_step() -> U32:
  d_step(1537n, 577n, T.Layer.step(TA.Mat<1537n, 577n>, List<&2, Nat>, TA.Mat<4n, 577n>, List<&2, Nat>, T.Embedding(4n, 1537n, 577n), adamw(), mat(1537n, 577n, 105), mat(1537n, 577n, 106), T.TwoMom{mat(1537n, 577n, 107), mat(1537n, 577n, 108)}), T.Layer.step(TA.Mat<1537n, 577n>, List<&2, Nat>, TA.Mat<4n, 577n>, List<&2, Nat>, F.Embedding(4n, 1537n, 577n), adamw(), mat(1537n, 577n, 105), mat(1537n, 577n, 106), T.TwoMom{mat(1537n, 577n, 107), mat(1537n, 577n, 108)}))

# ---- layers without parameters: output, then dX ----

type RU<-X: Type, -n: Nat, -c: Nat> is Type:
  RU{y: TA.Mat<n, c>, dx: X}

def ru_b(-X: Type, -n: Nat, -c: Nat, y: TA.Mat<n, c>, r: T.Bwd<Unit, X>) -> RU<X, n, c>:
  match r:
    case T.Bwd{p, dx, g}: RU{y, dx}

def ru_f(~X: Type, ~n: Nat, ~c: Nat, ~S: Type, ~l: T.Layer<Unit, X, TA.Mat<n, c>, S>, dy: TA.Mat<n, c>, r: T.Fwd<Unit, TA.Mat<n, c>, S>) -> RU<X, n, c>:
  match r:
    case T.Fwd{p, y, s}: ru_b(X, n, c, y, T.Layer.bwd(Unit, X, TA.Mat<n, c>, S, l, p, s, dy, Unit{}))

def run_u(~X: Type, ~n: Nat, ~c: Nat, ~S: Type, ~l: T.Layer<Unit, X, TA.Mat<n, c>, S>, x: X, dy: TA.Mat<n, c>) -> RU<X, n, c>:
  ru_f(~X, ~n, ~c, ~S, ~l, dy, T.Layer.fwd(Unit, X, TA.Mat<n, c>, S, l, Unit{}, x))

def d_u1(+n: Nat, +c: Nat, a: RU<TA.Mat<n, c>, n, c>, b: RU<TA.Mat<n, c>, n, c>) -> U32:
  match a b:
    case RU{TA.Mat{y1}, TA.Mat{d1}} RU{TA.Mat{y2}, TA.Mat{d2}}:
      (diff(Nat.mul(n, c), y1, y2) + diff(Nat.mul(n, c), d1, d2) : U32)

def d_u2(+n: Nat, +c: Nat, a: RU<T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, n, c>, b: RU<T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, n, c>) -> U32:
  match a b:
    case RU{TA.Mat{y1}, T.Two{TA.Mat{d1}, TA.Mat{e1}}} RU{TA.Mat{y2}, T.Two{TA.Mat{d2}, TA.Mat{e2}}}:
      ((diff(Nat.mul(n, c), y1, y2) + diff(Nat.mul(n, c), d1, d2) : U32) + diff(Nat.mul(n, c), e1, e2) : U32)

def silu(~n: Nat, ~c: Nat, +s: U32) -> U32:
  d_u1(n, c, run_u(~TA.Mat<n, c>, ~n, ~c, ~TA.Mat<n, c>, ~T.SiLU(n, c), mat(n, c, s), mat(n, c, (s + 1 : U32))), run_u(~TA.Mat<n, c>, ~n, ~c, ~TA.Mat<n, c>, ~F.SiLU(n, c), mat(n, c, s), mat(n, c, (s + 1 : U32))))

def mul2() -> U32:
  d_u2(512n, 1536n, run_u(~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~512n, ~1536n, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~T.Mul2(512n, 1536n), T.Two{mat(512n, 1536n, 81), mat(512n, 1536n, 82)}, mat(512n, 1536n, 83)), run_u(~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~512n, ~1536n, ~T.Two<TA.Mat<512n, 1536n>, TA.Mat<512n, 1536n>>, ~F.Mul2(512n, 1536n), T.Two{mat(512n, 1536n, 81), mat(512n, 1536n, 82)}, mat(512n, 1536n, 83)))

def RS() -> Type:
  T.Two<T.Two<Unit, TA.Mat<512n, 576n>>, Unit>

def RP() -> Type:
  T.Two<T.Two<Unit, Unit>, Unit>

def rr_b(y: TA.Mat<512n, 576n>, r: T.Bwd<RP, TA.Mat<512n, 576n>>) -> RU<TA.Mat<512n, 576n>, 512n, 576n>:
  match r:
    case T.Bwd{p, dx, g}: RU{y, dx}

def rr_f(~l: T.Layer<RP, TA.Mat<512n, 576n>, TA.Mat<512n, 576n>, RS>, dy: TA.Mat<512n, 576n>, r: T.Fwd<RP, TA.Mat<512n, 576n>, RS>) -> RU<TA.Mat<512n, 576n>, 512n, 576n>:
  match r:
    case T.Fwd{p, y, s}: rr_b(y, T.Layer.bwd(RP, TA.Mat<512n, 576n>, TA.Mat<512n, 576n>, RS, l, p, s, dy, T.Two{T.Two{Unit{}, Unit{}}, Unit{}}))

def run_r(~l: T.Layer<RP, TA.Mat<512n, 576n>, TA.Mat<512n, 576n>, RS>, x: TA.Mat<512n, 576n>, dy: TA.Mat<512n, 576n>) -> RU<TA.Mat<512n, 576n>, 512n, 576n>:
  rr_f(~l, dy, T.Layer.fwd(RP, TA.Mat<512n, 576n>, TA.Mat<512n, 576n>, RS, l, T.Two{T.Two{Unit{}, Unit{}}, Unit{}}, x))

def residual() -> U32:
  d_u1(512n, 576n, run_r(~T.Residual(~512n, ~576n, ~Unit, ~TA.Mat<512n, 576n>, ~T.SiLU(512n, 576n)), mat(512n, 576n, 91), mat(512n, 576n, 92)), run_r(~F.Residual(~512n, ~576n, ~Unit, ~TA.Mat<512n, 576n>, ~F.SiLU(512n, 576n)), mat(512n, 576n, 91), mat(512n, 576n, 92)))

def test_main() -> IO(Unit):
  do IO<Unit>:
    IO.print(U32.show(first3()) ++ " " ++ U32.show(first4()) ++ " " ++ U32.show(second()) ++ " " ++ U32.show(big_lin()) ++ " " ++ U32.show(tiny_nb()) ++ " " ++ U32.show(big_nb()))
    IO.print(U32.show(opt_steps()) ++ " " ++ U32.show(lin_adamw()) ++ " " ++ U32.show(silu(~512n, ~1536n, 85)) ++ " " ++ U32.show(mul2()) ++ " " ++ U32.show(silu(~32n, ~64n, 87)) ++ " " ++ U32.show(residual()))
    IO.print(U32.show(packed(True{})) ++ " " ++ U32.show(packed(False{})) ++ " " ++ U32.show(emb_step()))

# This program takes no arguments. Since Bend 2.0.38 a runtime option goes between +RTS and -RTS, and one written
# without them arrives here as an argument while the program runs on the default thread count, so a check of 1 and 16
# threads would pass without testing them: any argument is an error.
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, test_main())
