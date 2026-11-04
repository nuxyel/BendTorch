# RoPE2d probe: a 2x2 image (th = tw = 2) of 4 tokens, one head of 4 numbers split 2 for y and 2 for x, so the two
# axes rotate by different angles at the same distance — the case a single 1D rotation cannot pass. Prints the
# forward ("y") and, from a second forward of the same input, the backward of it ("back", which must give x again).
# reference/rope2d_check.py builds the hand rotation and compares both.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 5n), 11n)) - 5.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def x0() -> TA.Mat<4n, 4n>:
  TA.Mat.from_list(4n, 4n, vals(16n, 0n))

def L() -> T.Layer<Unit, TA.Mat<4n, 4n>, TA.Mat<4n, 4n>, Unit>:
  T.RoPE2d(4n, 4n, 2n, 2n, 1n, 1n, 1n, 10000.0, {==})

def fwd() -> T.Fwd<Unit, TA.Mat<4n, 4n>, Unit>:
  T.Layer.fwd(Unit, TA.Mat<4n, 4n>, TA.Mat<4n, 4n>, Unit, L(), Unit{}, x0())

def showBwd(b: T.Bwd<Unit, TA.Mat<4n, 4n>>) -> IO(Unit):
  match b:
    case T.Bwd{p2, d, g}:
      T.Mat.print(4n, 4n, "back", d)

def showY(r: T.Fwd<Unit, TA.Mat<4n, 4n>, Unit>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      T.Mat.print(4n, 4n, "y", y)

def showBack(r: T.Fwd<Unit, TA.Mat<4n, 4n>, Unit>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      showBwd(T.Layer.bwd(Unit, TA.Mat<4n, 4n>, TA.Mat<4n, 4n>, Unit, L(), p, s, y, Unit{}))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- showY(fwd())
    showBack(fwd())
