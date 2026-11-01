# torch/gpu.bend's linear layers against torch/main.bend's, bit for bit: forward (y) and backward (dX and every gradient)
# of each pair on the same inputs. Prints the number of differing words of each case; every one must be 0:
#   LinearFirst 100x784x128 (twice, two seeds; MNIST's first layer), Linear 100x128x10 (its second), Linear
#   128x576x1536, LinearNB 32x32x64 (the tiny Llama), LinearNB 128x576x1536, LinearNB 256x576x1536 (a SmolLM2-135M
#   projection) and LinearNB 32x32x65 (the tiny Llama's head). Built with CUDA, the products run on the GPU; with
#   `+RTS --gpu off -RTS` or without CUDA, on the CPU threads; the bits are the same.
import Base
import ../torch/main.bend as T
import ../torch/gpu.bend as G
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

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
  d_lin(100n, 784n, 128n, run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~T.LinearFirst(100n, 784n, 128n, 3n), lp(784n, 128n, 1), mat(100n, 784n, 3), mat(100n, 128n, 4), lp(784n, 128n, 5)), run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~G.LinearFirstF(100n, 784n, 128n, 3n), lp(784n, 128n, 1), mat(100n, 784n, 3), mat(100n, 128n, 4), lp(784n, 128n, 5)))

def first4() -> U32:
  d_lin(100n, 784n, 128n, run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~T.LinearFirst(100n, 784n, 128n, 4n), lp(784n, 128n, 11), mat(100n, 784n, 13), mat(100n, 128n, 14), lp(784n, 128n, 15)), run(~100n, ~784n, ~128n, ~T.Lin<784n, 128n>, ~G.LinearFirstF(100n, 784n, 128n, 4n), lp(784n, 128n, 11), mat(100n, 784n, 13), mat(100n, 128n, 14), lp(784n, 128n, 15)))

def second() -> U32:
  d_lin(100n, 128n, 10n, run(~100n, ~128n, ~10n, ~T.Lin<128n, 10n>, ~T.Linear(100n, 128n, 10n, 0n), lp(128n, 10n, 21), mat(100n, 128n, 23), mat(100n, 10n, 24), lp(128n, 10n, 25)), run(~100n, ~128n, ~10n, ~T.Lin<128n, 10n>, ~G.LinearF(100n, 128n, 10n, 0n), lp(128n, 10n, 21), mat(100n, 128n, 23), mat(100n, 10n, 24), lp(128n, 10n, 25)))

def big_lin() -> U32:
  d_lin(128n, 576n, 1536n, run(~128n, ~576n, ~1536n, ~T.Lin<576n, 1536n>, ~T.Linear(128n, 576n, 1536n, 4n), lp(576n, 1536n, 31), mat(128n, 576n, 33), mat(128n, 1536n, 34), lp(576n, 1536n, 35)), run(~128n, ~576n, ~1536n, ~T.Lin<576n, 1536n>, ~G.LinearF(128n, 576n, 1536n, 4n), lp(576n, 1536n, 31), mat(128n, 576n, 33), mat(128n, 1536n, 34), lp(576n, 1536n, 35)))

def tiny_nb() -> U32:
  d_nb(32n, 32n, 64n, run(~32n, ~32n, ~64n, ~TA.Mat<64n, 32n>, ~T.LinearNB(32n, 32n, 64n, 0n), mat(64n, 32n, 41), mat(32n, 32n, 43), mat(32n, 64n, 44), mat(64n, 32n, 45)), run(~32n, ~32n, ~64n, ~TA.Mat<64n, 32n>, ~G.LinearNBF(32n, 32n, 64n, 0n), mat(64n, 32n, 41), mat(32n, 32n, 43), mat(32n, 64n, 44), mat(64n, 32n, 45)))

def big_nb() -> U32:
  d_nb(128n, 576n, 1536n, run(~128n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~T.LinearNB(128n, 576n, 1536n, 3n), mat(1536n, 576n, 51), mat(128n, 576n, 53), mat(128n, 1536n, 54), mat(1536n, 576n, 55)), run(~128n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~G.LinearNBF(128n, 576n, 1536n, 3n), mat(1536n, 576n, 51), mat(128n, 576n, 53), mat(128n, 1536n, 54), mat(1536n, 576n, 55)))


def smol_nb() -> U32:
  d_nb(256n, 576n, 1536n, run(~256n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~T.LinearNB(256n, 576n, 1536n, 3n), mat(1536n, 576n, 61), mat(256n, 576n, 63), mat(256n, 1536n, 64), mat(1536n, 576n, 65)), run(~256n, ~576n, ~1536n, ~TA.Mat<1536n, 576n>, ~G.LinearNBF(256n, 576n, 1536n, 3n), mat(1536n, 576n, 61), mat(256n, 576n, 63), mat(256n, 1536n, 64), mat(1536n, 576n, 65)))

def tiny_head() -> U32:
  d_nb(32n, 32n, 65n, run(~32n, ~32n, ~65n, ~TA.Mat<65n, 32n>, ~T.LinearNB(32n, 32n, 65n, 0n), mat(65n, 32n, 71), mat(32n, 32n, 73), mat(32n, 65n, 74), mat(65n, 32n, 75)), run(~32n, ~32n, ~65n, ~TA.Mat<65n, 32n>, ~G.LinearNBF(32n, 32n, 65n, 0n), mat(65n, 32n, 71), mat(32n, 32n, 73), mat(32n, 65n, 74), mat(65n, 32n, 75)))

def test_main() -> IO(Unit):
  IO.print(U32.show(first3()) ++ " " ++ U32.show(first4()) ++ " " ++ U32.show(second()) ++ " " ++ U32.show(big_lin()) ++ " " ++ U32.show(tiny_nb()) ++ " " ++ U32.show(big_nb()) ++ " " ++ U32.show(smol_nb()) ++ " " ++ U32.show(tiny_head()))

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
