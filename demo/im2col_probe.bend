# im2col probe: n=1, h=w=4, c=2, k=3, p=1, s=1 -> oh=ow=4, 16 rows of 18 numbers; then the round trip at k=1, at
# k=s=2 and at k=s=4 on the 4x4 image, and at k=s=3 on a 6x6 one (oh=ow=2, 4 patch rows of 18 numbers). The
# general `k = s` round trip of BendTorch#29 is stated for every kernel that tiles the image; these are the cases
# that tiling has, and the Python check compares every one against the input.
# reference/im2col_check.py builds the same x and compares with F.unfold.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def mkx() -> TA.Mat<16n, 2n>:
  TA.Mat.from_list(16n, 2n, vals(32n, 0n))

# a 6x6 image of 2 channels: 72 cells, so k = s = 3 tiles it (oh = ow = 2)
def mkx6() -> TA.Mat<36n, 2n>:
  TA.Mat.from_list(36n, 2n, vals(72n, 0n))

def main() -> IO(Unit):
  do IO<Unit>:
    T.Mat.print(16n, 18n, "col", T.im2col(1n, 4n, 4n, 2n, 3n, 1n, 1n, mkx()))
    T.Mat.print(16n, 2n, "id", T.col2im(1n, 4n, 4n, 2n, 1n, 0n, 1n, T.im2col(1n, 4n, 4n, 2n, 1n, 0n, 1n, mkx())))
    # the non-overlapping case with a real kernel: k = s = 2, p = 0 (oh = ow = 2, 4 patch rows of 8 numbers)
    T.Mat.print(16n, 2n, "id2", T.col2im(1n, 4n, 4n, 2n, 2n, 0n, 2n, T.im2col(1n, 4n, 4n, 2n, 2n, 0n, 2n, mkx())))
    # one window covers the image: k = s = 4 at 4x4 (oh = ow = 1, one row of 32 numbers)
    T.Mat.print(16n, 2n, "id4", T.col2im(1n, 4n, 4n, 2n, 4n, 0n, 4n, T.im2col(1n, 4n, 4n, 2n, 4n, 0n, 4n, mkx())))
    # k = s = 3 at 6x6: the mixed-radix case the law has to cover (oh = ow = 2)
    T.Mat.print(36n, 2n, "id3", T.col2im(1n, 6n, 6n, 2n, 3n, 0n, 3n, T.im2col(1n, 6n, 6n, 2n, 3n, 0n, 3n, mkx6())))
    return Unit{}
