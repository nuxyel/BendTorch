# Conv2d probe: three shapes, each printing y, the backward's dX, dW and db as raw words.
# reference/conv2d_check.py compares all four against F.conv2d and its autograd.
#
#   c0  1 image of 4 x 4, 2 channels, 3 out, 3 x 3, padding 1, stride 1   (a padded border all round)
#   c1  1 image of 7 x 5, 2 channels, 3 out, 3 x 3, padding 1, stride 2   (h + 2p - k = 6, even: the last
#       window lands on the stride, and col2im's guards skip nothing)
#   c2  1 image of 7 x 6, 2 channels, 3 out, 3 x 3, padding 1, stride 2   (h + 2p - k = 5, odd: the last window
#       floors and col2im's guards skip the candidate that runs off the end — the case the review asked for)
#
# The image is written as the kernel plus a surplus (main.bend's `cimg`): h + 2p = k + hs, so c0 is hs = ws = 3,
# c1 is hs = 6, ws = 4, and c2 is hs = 6, ws = 5.
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

# the backward's three results, printed under <t>.dx, <t>.dw and <t>.db
def show(~rows: Nat, ~c: Nat, ~co: Nat, ~ckk: Nat, +t: String, r: T.Bwd<T.Lin<ckk, co>, TA.Mat<rows, c>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      match g:
        case T.Lin{w, b}:
          do IO<Unit>:
            Unit <- T.Mat.print(rows, c, tag(t, "dx"), dx)
            Unit <- T.Mat.print(co, ckk, tag(t, "dw"), w)
            T.Mat.print(1n, co, tag(t, "db"), b)

# one case: forward under <t>.y, then the backward from the dY written beside it
def fwd_of(~rows: Nat, ~c: Nat, ~orows: Nat, ~co: Nat, ~ckk: Nat,
  ~S: Type, ~l: T.Layer<T.Lin<ckk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S>,
  p: T.Lin<ckk, co>, x: TA.Mat<rows, c>) -> T.Fwd<T.Lin<ckk, co>, TA.Mat<orows, co>, S>:
  T.Layer.fwd(T.Lin<ckk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S, l, p, x)

def one(~rows: Nat, ~c: Nat, ~orows: Nat, ~co: Nat, ~ckk: Nat,
  ~S: Type, ~l: T.Layer<T.Lin<ckk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S>,
  +t: String, dy: TA.Mat<orows, co>, g: T.Lin<ckk, co>,
  r: T.Fwd<T.Lin<ckk, co>, TA.Mat<orows, co>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p2, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(orows, co, tag(t, "y"), y)
        show(~rows, ~c, ~co, ~ckk, t, T.Layer.bwd(T.Lin<ckk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S, l, p2, s, dy, g))

# ---- c0: 1 image of 4 x 4, 2 channels, 3 out, 3 x 3, padding 1, stride 1 ----

def c0_P() -> Type:
  T.Lin<18n, 3n>

def c0_X() -> Type:
  TA.Mat<16n, 2n>

def c0_Y() -> Type:
  TA.Mat<16n, 3n>

def c0_S() -> Type:
  TA.Mat<16n, 18n>

def c0_L() -> T.Layer<c0_P(), c0_X(), c0_Y(), c0_S()>:
  T.Conv2d(1n, 2n, 3n, 3n, 3n, 3n, 1n, 0n, 0n)

def c0() -> IO(Unit):
  one(~16n, ~2n, ~16n, ~3n, ~18n, ~c0_S(), ~c0_L(), "c0",
    TA.Mat.from_list(16n, 3n, vals(48n, 17n)),
    T.Lin{TA.Mat.zeros(3n, 18n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~16n, ~2n, ~16n, ~3n, ~18n, ~c0_S(), ~c0_L(),
      T.Lin{TA.Mat.from_list(3n, 18n, vals(54n, 5n)), TA.Mat.from_list(1n, 3n, vals(3n, 11n))},
      TA.Mat.from_list(16n, 2n, vals(32n, 0n))))

# ---- c1: 1 image of 7 x 5, 2 channels, 3 out, 3 x 3, padding 1, stride 2 (hs = 6, ws = 4, oh = 4, ow = 3) ----

def c1_P() -> Type:
  T.Lin<18n, 3n>

def c1_X() -> Type:
  TA.Mat<35n, 2n>

def c1_Y() -> Type:
  TA.Mat<12n, 3n>

def c1_S() -> Type:
  TA.Mat<12n, 18n>

def c1_L() -> T.Layer<c1_P(), c1_X(), c1_Y(), c1_S()>:
  T.Conv2d(1n, 2n, 3n, 3n, 6n, 4n, 1n, 1n, 0n)

def c1() -> IO(Unit):
  one(~35n, ~2n, ~12n, ~3n, ~18n, ~c1_S(), ~c1_L(), "c1",
    TA.Mat.from_list(12n, 3n, vals(36n, 19n)),
    T.Lin{TA.Mat.zeros(3n, 18n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~35n, ~2n, ~12n, ~3n, ~18n, ~c1_S(), ~c1_L(),
      T.Lin{TA.Mat.from_list(3n, 18n, vals(54n, 7n)), TA.Mat.from_list(1n, 3n, vals(3n, 13n))},
      TA.Mat.from_list(35n, 2n, vals(70n, 3n))))

# ---- c2: 1 image of 7 x 6, 2 channels, 3 out, 3 x 3, padding 1, stride 2 (hs = 6, ws = 5, oh = 4, ow = 3) ----

def c2_P() -> Type:
  T.Lin<18n, 3n>

def c2_X() -> Type:
  TA.Mat<42n, 2n>

def c2_Y() -> Type:
  TA.Mat<12n, 3n>

def c2_S() -> Type:
  TA.Mat<12n, 18n>

def c2_L() -> T.Layer<c2_P(), c2_X(), c2_Y(), c2_S()>:
  T.Conv2d(1n, 2n, 3n, 3n, 6n, 5n, 1n, 1n, 0n)

def c2() -> IO(Unit):
  one(~42n, ~2n, ~12n, ~3n, ~18n, ~c2_S(), ~c2_L(), "c2",
    TA.Mat.from_list(12n, 3n, vals(36n, 29n)),
    T.Lin{TA.Mat.zeros(3n, 18n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~42n, ~2n, ~12n, ~3n, ~18n, ~c2_S(), ~c2_L(),
      T.Lin{TA.Mat.from_list(3n, 18n, vals(54n, 21n)), TA.Mat.from_list(1n, 3n, vals(3n, 25n))},
      TA.Mat.from_list(42n, 2n, vals(84n, 9n))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- c0()
    Unit <- c1()
    c2()
