# MaxPool2d probe: three shapes, each printing y and the backward's dX as raw words.
# reference/maxpool_check.py compares both against F.max_pool2d(..., return_indices=True)
# and its autograd, including a ties case and overlapping windows (s < k).
#
#   c0  1 x 2 x 4 x 4, k 2, s 2        (windows tile the image; ties in the value sequence)
#   c1  1 x 2 x 5 x 5, k 3, s 2        (windows overlap by one row and column)
#   c2  2 x 1 x 4 x 3, k 3, s 1        (every output overlaps its neighbours)
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

def tag(+t: String, +name: String) -> String:
  String.append(t, String.append(".", name))

def show(~rows: Nat, ~c: Nat, +t: String, r: T.Bwd<Unit, TA.Mat<rows, c>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      T.Mat.print(rows, c, tag(t, "dx"), dx)

def one(~rows: Nat, ~c: Nat, ~orows: Nat, ~k: Nat, ~s1: Nat,
  ~l: T.Layer<Unit, TA.Mat<rows, c>, TA.Mat<orows, c>, TA.Mat<rows, c>>,
  +t: String, dy: TA.Mat<orows, c>, r: T.Fwd<Unit, TA.Mat<orows, c>, TA.Mat<rows, c>>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(orows, c, tag(t, "y"), y)
        show(~rows, ~c, t, T.Layer.bwd(Unit, TA.Mat<rows, c>, TA.Mat<orows, c>, TA.Mat<rows, c>, l, Unit{}, s, dy, Unit{}))

# ---- c0: 1 x 2 x 4 x 4, k 2, s 2 (oh = 2, ow = 2) ----

def c0_L() -> T.Layer<Unit, TA.Mat<16n, 2n>, TA.Mat<4n, 2n>, TA.Mat<16n, 2n>>:
  T.MaxPool2d(1n, 2n, 4n, 4n, 2n, 2n)

def c0() -> IO(Unit):
  one(~16n, ~2n, ~4n, ~2n, ~2n, ~c0_L(), "c0",
    TA.Mat.from_list(4n, 2n, vals(8n, 11n)),
    T.Layer.fwd(Unit, TA.Mat<16n, 2n>, TA.Mat<4n, 2n>, TA.Mat<16n, 2n>, c0_L(), Unit{},
      TA.Mat.from_list(16n, 2n, vals(32n, 0n))))

# ---- c1: 1 x 2 x 5 x 5, k 3, s 2 (oh = 2, ow = 2) ----

def c1_L() -> T.Layer<Unit, TA.Mat<25n, 2n>, TA.Mat<4n, 2n>, TA.Mat<25n, 2n>>:
  T.MaxPool2d(1n, 2n, 5n, 5n, 3n, 2n)

def c1() -> IO(Unit):
  one(~25n, ~2n, ~4n, ~3n, ~2n, ~c1_L(), "c1",
    TA.Mat.from_list(4n, 2n, vals(8n, 17n)),
    T.Layer.fwd(Unit, TA.Mat<25n, 2n>, TA.Mat<4n, 2n>, TA.Mat<25n, 2n>, c1_L(), Unit{},
      TA.Mat.from_list(25n, 2n, vals(50n, 3n))))

# ---- c2: 2 x 1 x 4 x 3, k 3, s 1 (oh = 2, ow = 1; every output overlaps) ----

def c2_L() -> T.Layer<Unit, TA.Mat<24n, 1n>, TA.Mat<4n, 1n>, TA.Mat<24n, 1n>>:
  T.MaxPool2d(2n, 1n, 4n, 3n, 3n, 1n)

def c2() -> IO(Unit):
  one(~24n, ~1n, ~4n, ~3n, ~1n, ~c2_L(), "c2",
    TA.Mat.from_list(4n, 1n, vals(4n, 23n)),
    T.Layer.fwd(Unit, TA.Mat<24n, 1n>, TA.Mat<4n, 1n>, TA.Mat<24n, 1n>, c2_L(), Unit{},
      TA.Mat.from_list(24n, 1n, vals(24n, 5n))))

# ---- c3: the tie, 1 x 1 x 2 x 2, k 2, s 1, [1, 1, 0, 0]: the first max wins ----

def c3_L() -> T.Layer<Unit, TA.Mat<4n, 1n>, TA.Mat<1n, 1n>, TA.Mat<4n, 1n>>:
  T.MaxPool2d(1n, 1n, 2n, 2n, 2n, 1n)

def c3() -> IO(Unit):
  one(~4n, ~1n, ~1n, ~2n, ~1n, ~c3_L(), "c3",
    TA.Mat.from_list(1n, 1n, [1.0]),
    T.Layer.fwd(Unit, TA.Mat<4n, 1n>, TA.Mat<1n, 1n>, TA.Mat<4n, 1n>, c3_L(), Unit{},
      TA.Mat.from_list(4n, 1n, [1.0, 1.0, 0.0, 0.0])))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- c0()
    Unit <- c1()
    Unit <- c2()
    c3()
