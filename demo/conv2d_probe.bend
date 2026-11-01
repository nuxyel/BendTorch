# Conv2d probe: n=1, h=w=4, c=2, co=3, k=3, p=1, stride 1. Prints y, dX, dW, db as
# raw words; reference/conv2d_check.py compares all four against F.conv2d.
import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def mkx() -> TA.Mat<16n, 2n>:
  TA.Mat.from_list(16n, 2n, vals(32n, 0n))

def mkw() -> TA.Mat<3n, 18n>:
  TA.Mat.from_list(3n, 18n, vals(54n, 5n))

def mkb() -> TA.Mat<1n, 3n>:
  TA.Mat.from_list(1n, 3n, vals(3n, 11n))

def mkdy() -> TA.Mat<16n, 3n>:
  TA.Mat.from_list(16n, 3n, vals(48n, 17n))

def P() -> Type:
  T.Lin<18n, 3n>

def X() -> Type:
  TA.Mat<16n, 2n>

def Y() -> Type:
  TA.Mat<16n, 3n>

def S() -> Type:
  TA.Mat<16n, 18n>

def L() -> T.Layer<P(), X(), Y(), S()>:
  T.Conv2d(1n, 4n, 4n, 2n, 3n, 3n, 1n, 0n, 0n)

def bwd(r: T.Bwd<P(), X()>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      match g:
        case T.Lin{w, b}:
          do IO<Unit>:
            T.Mat.print(16n, 2n, "dx", dx)
            T.Mat.print(3n, 18n, "dw", w)
            T.Mat.print(1n, 3n, "db", b)
            return Unit{}

def fwd(r: T.Fwd<P(), Y(), S()>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(16n, 3n, "y", y)
        T.Mat.print(16n, 3n, "t", mkdy())
        bwd(T.Layer.bwd(P(), X(), Y(), S(), L(), p, s, mkdy(), T.Lin{TA.Mat.zeros(3n, 18n), TA.Mat.zeros(1n, 3n)}))

def main() -> IO(Unit):
  do IO<Unit>:
    fwd(T.Layer.fwd(P(), X(), Y(), S(), L(), T.Lin{mkw(), mkb()}, mkx()))
    return Unit{}
