# Tile probe: 2 images of 4 numbers each, broadcast over t = 3 tokens, so the forward is a 6 x 4 matrix and the
# backward sums the three copies of each image back into its row. Prints both as raw words;
# reference/tile_check.py builds the same tensors and compares with repeat_interleave and its autograd.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def x0() -> TA.Mat<2n, 4n>:
  TA.Mat.from_list(2n, 4n, vals(8n, 0n))

def dy0() -> TA.Mat<6n, 4n>:
  TA.Mat.from_list(6n, 4n, vals(24n, 3n))

def L() -> T.Layer<Unit, TA.Mat<2n, 4n>, TA.Mat<6n, 4n>, Unit>:
  T.Tile(2n, 3n, 4n)

def fwd() -> T.Fwd<Unit, TA.Mat<6n, 4n>, Unit>:
  T.Layer.fwd(Unit, TA.Mat<2n, 4n>, TA.Mat<6n, 4n>, Unit, L(), Unit{}, x0())

def showY(r: T.Fwd<Unit, TA.Mat<6n, 4n>, Unit>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      T.Mat.print(6n, 4n, "y", y)

def showDx(b: T.Bwd<Unit, TA.Mat<2n, 4n>>) -> IO(Unit):
  match b:
    case T.Bwd{p, d, g}:
      T.Mat.print(2n, 4n, "dx", d)

def showB(r: T.Fwd<Unit, TA.Mat<6n, 4n>, Unit>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      showDx(T.Layer.bwd(Unit, TA.Mat<2n, 4n>, TA.Mat<6n, 4n>, Unit, L(), p, s, dy0(), Unit{}))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- showY(fwd())
    showB(fwd())
