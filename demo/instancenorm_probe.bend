# InstanceNorm probe (BendTorch#43): nn.InstanceNorm1d is GroupNorm with one channel per group, so it needs no layer of
# its own. n = 2 sequences of 5 steps and 3 channels, channels last (a 10 x 3 matrix), GroupNorm(2n, 5n, 3n, 1n, eps).
#
#   off   gain 1 and bias 0, InstanceNorm1d's default (affine=False)
#   on    a gain and a bias per channel (affine=True)
#
# Each case prints y, dx and the gain's and bias's gradients as raw words; reference/instancenorm_check.py builds the
# same tensors and compares with F.instance_norm and its autograd.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

# a second pattern, for x: vals' rows are shifts of each other
def vals2(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.add(Nat.mul(Nat.mul(i, i), 5n), Nat.mul(i, 3n)), 17n)) - 8.0 : F32) / 4.0 : F32), vals2(rest, Nat.add(i, 1n))}

def fill(+left: Nat, +v: F32) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{v, fill(rest, v)}

def M() -> Type:
  TA.Mat<10n, 3n>

def L() -> T.Layer<T.LN<3n>, M, M, M>:
  T.GroupNorm(2n, 5n, 3n, 1n, 0.00001)

def x0() -> M:
  TA.Mat.from_list(10n, 3n, vals2(30n, 0n))

def dy0() -> M:
  TA.Mat.from_list(10n, 3n, vals(30n, 8n))

def row(+v: F32) -> TA.Mat<1n, 3n>:
  TA.Mat.from_list(1n, 3n, fill(3n, v))

# the parameters: gain 1 and bias 0 (off), or a gain and a bias per channel (on)
def params(+on: Bool) -> T.LN<3n>:
  match on:
    case True{}:
      T.LN{TA.Mat.from_list(1n, 3n, vals(3n, 4n)), TA.Mat.from_list(1n, 3n, vals(3n, 9n))}
    case False{}:
      T.LN{row(1.0), row(0.0)}

def show_g(+t: String, l: T.LN<3n>) -> IO(Unit):
  match l:
    case T.LN{g, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(1n, 3n, t ++ ".dg", g)
        T.Mat.print(1n, 3n, t ++ ".db", b)

def show_b(+t: String, r: T.Bwd<T.LN<3n>, M>) -> IO(Unit):
  match r:
    case T.Bwd{p, d, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(10n, 3n, t ++ ".dx", d)
        show_g(t, g)

def bwd_of(r: T.Fwd<T.LN<3n>, M, M>) -> T.Bwd<T.LN<3n>, M>:
  match r:
    case T.Fwd{p, y, s}:
      T.Layer.bwd(T.LN<3n>, M, M, M, L(), p, s, dy0(), T.LN{row(0.0), row(0.0)})

def show_y(+t: String, r: T.Fwd<T.LN<3n>, M, M>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      T.Mat.print(10n, 3n, t ++ ".y", y)

def fwd(+on: Bool) -> T.Fwd<T.LN<3n>, M, M>:
  T.Layer.fwd(T.LN<3n>, M, M, M, L(), params(on), x0())

def run(+t: String, +on: Bool) -> IO(Unit):
  do IO<Unit>:
    Unit <- show_y(t, fwd(on))
    show_b(t, bwd_of(fwd(on)))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- run("off", False{})
    run("on", True{})
