# AdaIN probe (BendTorch#43): adaptive instance normalization built from layers BendTorch already has, no new layer in
# torch/. b = 2 sequences of t = 5 steps, c = 4 channels (n = 10 rows, channels last), a style of 3 numbers a sequence.
#
#   mod    y = Tile(sh) + Tile(m) * InstanceNorm(x)     InstanceNorm is GroupNorm(2n, 5n, 4n, 1n, eps) at gain 1, bias 0
#   id     mod with m = 1 and sh = 0, beside InstanceNorm alone (they must give the same numbers)
#   adain  mod with m and sh two Linear(3, 4) layers of the style, as StyleTTS2's AdaIN1d: (1 + gamma) * norm(x) + beta
#
# m is AdaIN1d's 1 + gamma: the 1 is folded into its Linear's bias, and its one Linear of width 2c is two of width c
# here (the same weights, split by rows). Each case prints y, every input gradient and every parameter gradient as raw
# words; reference/adain_check.py builds the same tensors in PyTorch and compares.
#
# `T.Rewire` and `T.two_swap` come from torch/; `Fma` ((a, (b, c)) to b + a * c) is the adaLN probe's helper.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

# a second pattern, for x and the style: vals' rows are shifts of each other
def vals2(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.add(Nat.mul(Nat.mul(i, i), 5n), Nat.mul(i, 3n)), 17n)) - 8.0 : F32) / 4.0 : F32), vals2(rest, Nat.add(i, 1n))}

def ones(+left: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{1.0, ones(rest)}

def mat(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals(Nat.mul(r, c), s))

def mat2(+r: Nat, +c: Nat, +s: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, vals2(Nat.mul(r, c), s))

def tag(+t: String, +name: String) -> String:
  String.append(t, String.append(".", name))

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# ---- Fma: (a, (b, c)) to b + a * c, over n x 4 ----

def X3(n: Nat) -> Type:
  T.Two<M(n, 4n), T.Two<M(n, 4n), M(n, 4n)>>

def MulS(n: Nat) -> Type:
  T.Two<M(n, 4n), M(n, 4n)>

def FmaP() -> Type:
  T.Two<Unit, T.Two<T.Two<Unit, Unit>, Unit>>

def FmaS(n: Nat) -> Type:
  T.Two<Unit, T.Two<T.Two<Unit, MulS(n)>, Unit>>

def MA(~n: Nat) -> T.Layer<T.Two<Unit, Unit>, X3(n), T.Two<M(n, 4n), M(n, 4n)>, T.Two<Unit, MulS(n)>>:
  T.Both(~Unit, ~Unit, ~M(n, 4n), ~T.Two<M(n, 4n), M(n, 4n)>, ~M(n, 4n), ~M(n, 4n), ~Unit, ~MulS(n), ~T.Id(~M(n, 4n)), ~T.Mul2(n, 4n))

def MAA(~n: Nat) -> T.Layer<T.Two<T.Two<Unit, Unit>, Unit>, X3(n), M(n, 4n), T.Two<T.Two<Unit, MulS(n)>, Unit>>:
  T.Seq(~T.Two<Unit, Unit>, ~Unit, ~X3(n), ~T.Two<M(n, 4n), M(n, 4n)>, ~M(n, 4n), ~T.Two<Unit, MulS(n)>, ~Unit, ~MA(~n), ~T.Add2(n, 4n))

def Fma(~n: Nat) -> T.Layer<FmaP, X3(n), M(n, 4n), FmaS(n)>:
  T.Seq(~Unit, ~T.Two<T.Two<Unit, Unit>, Unit>, ~X3(n), ~X3(n), ~M(n, 4n), ~Unit, ~T.Two<T.Two<Unit, MulS(n)>, Unit>,
    ~T.Rewire(~X3(n), ~X3(n), ~(x => T.two_swap(M(n, 4n), M(n, 4n), M(n, 4n), x)), ~(x => T.two_swap(M(n, 4n), M(n, 4n), M(n, 4n), x))),
    ~MAA(~n))

def unitsF() -> FmaP:
  T.Two{Unit{}, T.Two{T.Two{Unit{}, Unit{}}, Unit{}}}

def lin(+i: Nat, +o: Nat, +s: Nat) -> T.Lin<i, o>:
  T.Lin{mat(o, i, s), mat(1n, o, Nat.add(s, 40n))}

def show_lin(+i: Nat, +o: Nat, +t: String, l: T.Lin<i, o>) -> IO(Unit):
  match l:
    case T.Lin{w, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(o, i, tag(t, "dw"), w)
        T.Mat.print(1n, o, tag(t, "db"), b)

def show_ln(+d: Nat, +t: String, l: T.LN<d>) -> IO(Unit):
  match l:
    case T.LN{g, b}:
      do IO<Unit>:
        Unit <- T.Mat.print(1n, d, tag(t, "dg"), g)
        T.Mat.print(1n, d, tag(t, "db"), b)

# =====================================================================================================
# mod: (m, (sh, x)) to Tile(sh) + Tile(m) * InstanceNorm(x)
# =====================================================================================================

def IN() -> T.Layer<T.LN<4n>, M(10n, 4n), M(10n, 4n), M(10n, 4n)>:
  T.GroupNorm(2n, 5n, 4n, 1n, 0.00001)

# InstanceNorm1d's default, affine=False: gain 1 and bias 0
def in_p() -> T.LN<4n>:
  T.LN{TA.Mat.from_list(1n, 4n, ones(4n)), TA.Mat.zeros(1n, 4n)}

def MX() -> Type:
  T.Two<M(2n, 4n), T.Two<M(2n, 4n), M(10n, 4n)>>

def TNP() -> Type:
  T.Two<Unit, T.LN<4n>>

def TNS() -> Type:
  T.Two<Unit, M(10n, 4n)>

def TN() -> T.Layer<TNP, T.Two<M(2n, 4n), M(10n, 4n)>, T.Two<M(10n, 4n), M(10n, 4n)>, TNS>:
  T.Both(~Unit, ~T.LN<4n>, ~M(2n, 4n), ~M(10n, 4n), ~M(10n, 4n), ~M(10n, 4n), ~Unit, ~M(10n, 4n), ~T.Tile(2n, 5n, 4n), ~IN)

def M1P() -> Type:
  T.Two<Unit, TNP>

def M1S() -> Type:
  T.Two<Unit, TNS>

def M1() -> T.Layer<M1P, MX, X3(10n), M1S>:
  T.Both(~Unit, ~TNP, ~M(2n, 4n), ~T.Two<M(2n, 4n), M(10n, 4n)>, ~M(10n, 4n), ~T.Two<M(10n, 4n), M(10n, 4n)>, ~Unit, ~TNS, ~T.Tile(2n, 5n, 4n), ~TN)

def MP() -> Type:
  T.Two<M1P, FmaP>

def MS() -> Type:
  T.Two<M1S, FmaS(10n)>

def Mod() -> T.Layer<MP, MX, M(10n, 4n), MS>:
  T.Seq(~M1P, ~FmaP, ~MX, ~X3(10n), ~M(10n, 4n), ~M1S, ~FmaS(10n), ~M1, ~Fma(~10n))

def mod_p() -> MP:
  T.Two{T.Two{Unit{}, T.Two{Unit{}, in_p}}, unitsF}

def mod_dp(g: MP) -> T.LN<4n>:
  match g:
    case T.Two{a, u}:
      match a:
        case T.Two{u2, tn}:
          match tn:
            case T.Two{u3, l}:
              l

def mod_dx(+t: String, d: MX) -> IO(Unit):
  match d:
    case T.Two{dm, r}:
      match r:
        case T.Two{dsh, dx}:
          do IO<Unit>:
            Unit <- T.Mat.print(2n, 4n, tag(t, "dm"), dm)
            Unit <- T.Mat.print(2n, 4n, tag(t, "dsh"), dsh)
            T.Mat.print(10n, 4n, tag(t, "dx"), dx)

def mod_b(+t: String, r: T.Bwd<MP, MX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- mod_dx(t, dx)
        show_ln(4n, tag(t, "in"), mod_dp(g))

def mod_f(+t: String, r: T.Fwd<MP, M(10n, 4n), MS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(10n, 4n, tag(t, "y"), y)
        mod_b(t, T.Layer.bwd(MP, MX, M(10n, 4n), MS, Mod, p, s, mat(10n, 4n, 95n), T.Layer.zeros(MP, MX, M(10n, 4n), MS, Mod)))

def mod() -> IO(Unit):
  mod_f("mod", T.Layer.fwd(MP, MX, M(10n, 4n), MS, Mod, mod_p, T.Two{mat(2n, 4n, 21n), T.Two{mat(2n, 4n, 25n), mat2(10n, 4n, 31n)}}))

# ---- id: mod with m = 1 and sh = 0, then InstanceNorm alone on the same x and dy ----

def in_b(r: T.Bwd<T.LN<4n>, M(10n, 4n)>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(10n, 4n, "in.dx", dx)
        show_ln(4n, "in.in", g)

def in_f(r: T.Fwd<T.LN<4n>, M(10n, 4n), M(10n, 4n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(10n, 4n, "in.y", y)
        in_b(T.Layer.bwd(T.LN<4n>, M(10n, 4n), M(10n, 4n), M(10n, 4n), IN, p, s, mat(10n, 4n, 95n), T.Layer.zeros(T.LN<4n>, M(10n, 4n), M(10n, 4n), M(10n, 4n), IN)))

def id() -> IO(Unit):
  do IO<Unit>:
    Unit <- mod_f("id", T.Layer.fwd(MP, MX, M(10n, 4n), MS, Mod, mod_p, T.Two{TA.Mat.from_list(2n, 4n, ones(8n)), T.Two{TA.Mat.zeros(2n, 4n), mat2(10n, 4n, 31n)}}))
    in_f(T.Layer.fwd(T.LN<4n>, M(10n, 4n), M(10n, 4n), M(10n, 4n), IN, in_p, mat2(10n, 4n, 31n)))

# =====================================================================================================
# adain: (style, x) to mod(m, sh, x), with (m, sh) two Linear(3, 4) of the style
# =====================================================================================================

def L34() -> Type:
  T.Lin<3n, 4n>

def Lc() -> T.Layer<L34, M(2n, 3n), M(2n, 4n), M(2n, 3n)>:
  T.Linear(2n, 3n, 4n, 0n)

# style to (m, sh); the two input gradients are summed
def K2() -> T.Layer<T.Two<L34, L34>, M(2n, 3n), T.Two<M(2n, 4n), M(2n, 4n)>, T.Two<M(2n, 3n), M(2n, 3n)>>:
  T.Fork(~L34, ~L34, ~M(2n, 3n), ~M(2n, 4n), ~M(2n, 4n), ~M(2n, 3n), ~M(2n, 3n), ~(m => T.Mat.dup(2n, 3n, m)), ~(a => b => T.Mat.sum2(2n, 3n, a, b)), ~Lc, ~Lc)

def AX() -> Type:
  T.Two<M(2n, 3n), M(10n, 4n)>

def A1Y() -> Type:
  T.Two<T.Two<M(2n, 4n), M(2n, 4n)>, M(10n, 4n)>

def A1P() -> Type:
  T.Two<T.Two<L34, L34>, Unit>

def A1S() -> Type:
  T.Two<T.Two<M(2n, 3n), M(2n, 3n)>, Unit>

# (style, x) to ((m, sh), x)
def A1() -> T.Layer<A1P, AX, A1Y, A1S>:
  T.Both(~T.Two<L34, L34>, ~Unit, ~M(2n, 3n), ~M(10n, 4n), ~T.Two<M(2n, 4n), M(2n, 4n)>, ~M(10n, 4n), ~T.Two<M(2n, 3n), M(2n, 3n)>, ~Unit, ~K2, ~T.Id(~M(10n, 4n)))

def regroup2(p: T.Two<M(2n, 4n), M(2n, 4n)>, x: M(10n, 4n)) -> MX:
  match p:
    case T.Two{m, sh}:
      T.Two{m, T.Two{sh, x}}

# ((m, sh), x) to (m, (sh, x))
def regroup(v: A1Y) -> MX:
  match v:
    case T.Two{p, x}:
      regroup2(p, x)

def ungroup2(m: M(2n, 4n), r: T.Two<M(2n, 4n), M(10n, 4n)>) -> A1Y:
  match r:
    case T.Two{sh, x}:
      T.Two{T.Two{m, sh}, x}

# regroup's inverse, for the gradients
def ungroup(v: MX) -> A1Y:
  match v:
    case T.Two{m, r}:
      ungroup2(m, r)

def A2P() -> Type:
  T.Two<Unit, MP>

def A2S() -> Type:
  T.Two<Unit, MS>

def A2() -> T.Layer<A2P, A1Y, M(10n, 4n), A2S>:
  T.Seq(~Unit, ~MP, ~A1Y, ~MX, ~M(10n, 4n), ~Unit, ~MS, ~T.Rewire(~A1Y, ~MX, ~(v => regroup(v)), ~(v => ungroup(v))), ~Mod)

def AP() -> Type:
  T.Two<A1P, A2P>

def AS() -> Type:
  T.Two<A1S, A2S>

def AdaIN() -> T.Layer<AP, AX, M(10n, 4n), AS>:
  T.Seq(~A1P, ~A2P, ~AX, ~A1Y, ~M(10n, 4n), ~A1S, ~A2S, ~A1, ~A2)

def adain_p() -> AP:
  T.Two{T.Two{T.Two{lin(3n, 4n, 9n), lin(3n, 4n, 11n)}, Unit{}}, T.Two{Unit{}, mod_p}}

def show_k2(+t: String, k: T.Two<L34, L34>) -> IO(Unit):
  match k:
    case T.Two{lm, ls}:
      do IO<Unit>:
        Unit <- show_lin(3n, 4n, tag(t, "m"), lm)
        show_lin(3n, 4n, tag(t, "sh"), ls)

def show_a1(+t: String, a: A1P) -> IO(Unit):
  match a:
    case T.Two{k, u}:
      show_k2(t, k)

def show_a2(+t: String, a: A2P) -> IO(Unit):
  match a:
    case T.Two{u, mp}:
      show_ln(4n, tag(t, "in"), mod_dp(mp))

def show_ap(+t: String, g: AP) -> IO(Unit):
  match g:
    case T.Two{a1, a2}:
      do IO<Unit>:
        Unit <- show_a1(t, a1)
        show_a2(t, a2)

def adain_dx(d: AX) -> IO(Unit):
  match d:
    case T.Two{ds, dx}:
      do IO<Unit>:
        Unit <- T.Mat.print(2n, 3n, "adain.ds", ds)
        T.Mat.print(10n, 4n, "adain.dx", dx)

def adain_b(r: T.Bwd<AP, AX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- adain_dx(dx)
        show_ap("adain", g)

def adain_f(r: T.Fwd<AP, M(10n, 4n), AS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(10n, 4n, "adain.y", y)
        adain_b(T.Layer.bwd(AP, AX, M(10n, 4n), AS, AdaIN, p, s, mat(10n, 4n, 97n), T.Layer.zeros(AP, AX, M(10n, 4n), AS, AdaIN)))

def adain() -> IO(Unit):
  adain_f(T.Layer.fwd(AP, AX, M(10n, 4n), AS, AdaIN, adain_p, T.Two{mat2(2n, 3n, 23n), mat2(10n, 4n, 33n)}))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- mod()
    Unit <- id()
    adain()
