# im2col probe: n=1, h=w=4, c=2, k=3, p=1, s=1 -> oh=ow=4, 16 rows of 18 numbers.
# reference/im2col_check.py builds the same x and compares with F.unfold.
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def main() -> IO(Unit):
  do IO<Unit>:
    T.Mat.print(16n, 18n, "col", T.im2col(1n, 4n, 4n, 2n, 3n, 1n, 1n, TA.Mat.from_list(16n, 2n, vals(32n, 0n))))
    return Unit{}
