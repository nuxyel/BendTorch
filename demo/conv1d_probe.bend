# Conv1d probe: five shapes covering standard, odd-surplus stride, dilation, grouped and depthwise convolutions.
# Prints y, dX, dW and db as raw F32 words; reference/conv1d_check.py compares all four against F.conv1d and autograd.
#
#   c0  n=2, t=5, g=1, cg=2, cog=3, k=3, p=1, s=1, d=1  (standard padded conv1d, batch 2)
#   c1  n=1, t=6, g=1, cg=2, cog=3, k=3, p=1, s=2, d=1  (odd surplus t+2p-eff_k = 5, last window floors)
#   c2  n=1, t=7, g=1, cg=2, cog=3, k=3, p=2, s=1, d=2  (dilated eff_k = 5, ot = 7)
#   c3  n=2, t=8, g=2, cg=2, cog=3, k=3, p=1, s=2, d=2  (grouped + stride + dilation + padding, eff_k = 5, ts = 5, ot = 3)
#   c4  n=1, t=9, g=4, cg=1, cog=1, k=7, p=3, s=1, d=1  (depthwise k=7, tiny-voca Block pattern)
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

def show(~rows: Nat, ~c: Nat, ~co: Nat, ~cgk: Nat, +t: String, r: T.Bwd<T.Lin<cgk, co>, TA.Mat<rows, c>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      match g:
        case T.Lin{w, b}:
          do IO<Unit>:
            Unit <- T.Mat.print(rows, c, tag(t, "dx"), dx)
            Unit <- T.Mat.print(co, cgk, tag(t, "dw"), w)
            T.Mat.print(1n, co, tag(t, "db"), b)

def fwd_of(~rows: Nat, ~c: Nat, ~orows: Nat, ~co: Nat, ~cgk: Nat,
  ~S: Type, ~l: T.Layer<T.Lin<cgk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S>,
  p: T.Lin<cgk, co>, x: TA.Mat<rows, c>) -> T.Fwd<T.Lin<cgk, co>, TA.Mat<orows, co>, S>:
  T.Layer.fwd(T.Lin<cgk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S, l, p, x)

def one(~rows: Nat, ~c: Nat, ~orows: Nat, ~co: Nat, ~cgk: Nat,
  ~S: Type, ~l: T.Layer<T.Lin<cgk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S>,
  +t: String, dy: TA.Mat<orows, co>, g: T.Lin<cgk, co>,
  r: T.Fwd<T.Lin<cgk, co>, TA.Mat<orows, co>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p2, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(orows, co, tag(t, "y"), y)
        show(~rows, ~c, ~co, ~cgk, t, T.Layer.bwd(T.Lin<cgk, co>, TA.Mat<rows, c>, TA.Mat<orows, co>, S, l, p2, s, dy, g))

# ---- c0: n=2, t=5, g=1 (g1=0), cg=2, cog=3, k=3 (k1=2), p=1, s=1 (s1=0), d=1 (d1=0), ts=4, ot=5 ----

def c0_L() -> T.Layer<T.Lin<6n, 3n>, TA.Mat<10n, 2n>, TA.Mat<10n, 3n>, TA.Mat<10n, 6n>>:
  T.Conv1d(2n, 0n, 2n, 3n, 2n, 4n, 1n, 0n, 0n, 0n, {==})

def c0() -> IO(Unit):
  one(~10n, ~2n, ~10n, ~3n, ~6n, ~TA.Mat<10n, 6n>, ~c0_L(), "c0",
    TA.Mat.from_list(10n, 3n, vals(30n, 17n)),
    T.Lin{TA.Mat.zeros(3n, 6n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~10n, ~2n, ~10n, ~3n, ~6n, ~TA.Mat<10n, 6n>, ~c0_L(),
      T.Lin{TA.Mat.from_list(3n, 6n, vals(18n, 5n)), TA.Mat.from_list(1n, 3n, vals(3n, 11n))},
      TA.Mat.from_list(10n, 2n, vals(20n, 0n))))

# ---- c1: n=1, t=6, g=1 (g1=0), cg=2, cog=3, k=3 (k1=2), p=1, s=2 (s1=1), d=1 (d1=0), ts=5, ot=3 ----

def c1_L() -> T.Layer<T.Lin<6n, 3n>, TA.Mat<6n, 2n>, TA.Mat<3n, 3n>, TA.Mat<3n, 6n>>:
  T.Conv1d(1n, 0n, 2n, 3n, 2n, 5n, 1n, 1n, 0n, 0n, {==})

def c1() -> IO(Unit):
  one(~6n, ~2n, ~3n, ~3n, ~6n, ~TA.Mat<3n, 6n>, ~c1_L(), "c1",
    TA.Mat.from_list(3n, 3n, vals(9n, 19n)),
    T.Lin{TA.Mat.zeros(3n, 6n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~6n, ~2n, ~3n, ~3n, ~6n, ~TA.Mat<3n, 6n>, ~c1_L(),
      T.Lin{TA.Mat.from_list(3n, 6n, vals(18n, 7n)), TA.Mat.from_list(1n, 3n, vals(3n, 13n))},
      TA.Mat.from_list(6n, 2n, vals(12n, 3n))))

# ---- c2: n=1, t=7, g=1 (g1=0), cg=2, cog=3, k=3 (k1=2), p=2, s=1 (s1=0), d=2 (d1=1), eff_k=5, ts=6, ot=7 ----

def c2_L() -> T.Layer<T.Lin<6n, 3n>, TA.Mat<7n, 2n>, TA.Mat<7n, 3n>, TA.Mat<7n, 6n>>:
  T.Conv1d(1n, 0n, 2n, 3n, 2n, 6n, 2n, 0n, 1n, 0n, {==})

def c2() -> IO(Unit):
  one(~7n, ~2n, ~7n, ~3n, ~6n, ~TA.Mat<7n, 6n>, ~c2_L(), "c2",
    TA.Mat.from_list(7n, 3n, vals(21n, 29n)),
    T.Lin{TA.Mat.zeros(3n, 6n), TA.Mat.zeros(1n, 3n)},
    fwd_of(~7n, ~2n, ~7n, ~3n, ~6n, ~TA.Mat<7n, 6n>, ~c2_L(),
      T.Lin{TA.Mat.from_list(3n, 6n, vals(18n, 21n)), TA.Mat.from_list(1n, 3n, vals(3n, 25n))},
      TA.Mat.from_list(7n, 2n, vals(14n, 9n))))

# ---- c3: n=2, t=8, g=2 (g1=1), cg=2, cog=3, k=3 (k1=2), p=1, s=2 (s1=1), d=2 (d1=1), eff_k=5, ts=5, ot=3 ----

def c3_L() -> T.Layer<T.Lin<6n, 6n>, TA.Mat<16n, 4n>, TA.Mat<6n, 6n>, TA.Mat<12n, 6n>>:
  T.Conv1d(2n, 1n, 2n, 3n, 2n, 5n, 1n, 1n, 1n, 0n, {==})

def c3() -> IO(Unit):
  one(~16n, ~4n, ~6n, ~6n, ~6n, ~TA.Mat<12n, 6n>, ~c3_L(), "c3",
    TA.Mat.from_list(6n, 6n, vals(36n, 31n)),
    T.Lin{TA.Mat.zeros(6n, 6n), TA.Mat.zeros(1n, 6n)},
    fwd_of(~16n, ~4n, ~6n, ~6n, ~6n, ~TA.Mat<12n, 6n>, ~c3_L(),
      T.Lin{TA.Mat.from_list(6n, 6n, vals(36n, 11n)), TA.Mat.from_list(1n, 6n, vals(6n, 23n))},
      TA.Mat.from_list(16n, 4n, vals(64n, 2n))))

# ---- c4: n=1, t=9, g=4 (g1=3), cg=1, cog=1, k=7 (k1=6), p=3, s=1 (s1=0), d=1 (d1=0), eff_k=7, ts=8, ot=9 ----

def c4_L() -> T.Layer<T.Lin<7n, 4n>, TA.Mat<9n, 4n>, TA.Mat<9n, 4n>, TA.Mat<36n, 7n>>:
  T.Conv1d(1n, 3n, 1n, 1n, 6n, 8n, 3n, 0n, 0n, 0n, {==})

def c4() -> IO(Unit):
  one(~9n, ~4n, ~9n, ~4n, ~7n, ~TA.Mat<36n, 7n>, ~c4_L(), "c4",
    TA.Mat.from_list(9n, 4n, vals(36n, 41n)),
    T.Lin{TA.Mat.zeros(4n, 7n), TA.Mat.zeros(1n, 4n)},
    fwd_of(~9n, ~4n, ~9n, ~4n, ~7n, ~TA.Mat<36n, 7n>, ~c4_L(),
      T.Lin{TA.Mat.from_list(4n, 7n, vals(28n, 15n)), TA.Mat.from_list(1n, 4n, vals(4n, 37n))},
      TA.Mat.from_list(9n, 4n, vals(36n, 6n))))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- c0()
    Unit <- c1()
    Unit <- c2()
    Unit <- c3()
    c4()
