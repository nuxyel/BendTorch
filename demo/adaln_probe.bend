# adaLN probe (BendTorch#19): a conditioned block built from layers BendTorch already has, no new layer in torch/.
# b = 2 images of t = 3 tokens, width c = 4 (n = 6 rows), a condition of cd = 3 numbers per image.
#
#   gate   y = h + Tile(g) * f(h)                  the gated residual: Tile, Fork(Id, f), Mul2, Add2
#   mod    y = Tile(sh) + Tile(m) * LayerNorm(h)    LayerNormMod: Tile, LayerNorm, Mul2, Add2
#   id     mod with m = 1 and sh = 0, beside LayerNorm alone (they must give the same numbers)
#   block  y = h + Tile(g) * f(mod(m, sh, h)), with g, m and sh three Linear layers of silu(cond), as DiT's adaLN
#
# f is a Linear(4, 4). m is the multiplier, DiT's 1 + scale: the 1 is folded into its Linear's bias.
# Each case prints y, every input gradient and every parameter gradient as raw words; reference/adaln_check.py builds
# the same tensors in PyTorch and compares.
#
# The one piece not in torch/ is `Rewire`, a layer with no parameters that only moves the parts of a pair around (its
# backward pass moves the gradients back); a Fork and a Both cannot regroup a pair's halves.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

def vals(+left: Nat, +i: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.mul(i, 7n), 13n)) - 6.0 : F32) / 4.0 : F32), vals(rest, Nat.add(i, 1n))}

# a second pattern, for h and cond: vals' rows are shifts of each other, which LayerNorm maps to one row
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

# ---- Rewire: a layer that moves the parts of its input, f forward and b (its inverse) on the gradient ----

def Rewire(~X: Type, ~Y: Type, ~f: X -> Y, ~b: Y -> X) -> T.Layer<Unit, X, Y, Unit>:
  T.Layer{p => x => T.Fwd{p, f(x), Unit{}}, p => s => dy => g => T.Bwd{p, b(dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

def sw2(-A: Type, -B: Type, -C: Type, a: A, r: T.Two<B, C>) -> T.Two<B, T.Two<A, C>>:
  match r:
    case T.Two{b, c}:
      T.Two{b, T.Two{a, c}}

# (a, (b, c)) to (b, (a, c)); its own inverse
def sw(-A: Type, -B: Type, -C: Type, x: T.Two<A, T.Two<B, C>>) -> T.Two<B, T.Two<A, C>>:
  match x:
    case T.Two{a, r}:
      sw2(A, B, C, a, r)

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
    ~Rewire(~X3(n), ~X3(n), ~(x => sw(M(n, 4n), M(n, 4n), M(n, 4n), x)), ~(x => sw(M(n, 4n), M(n, 4n), M(n, 4n), x))),
    ~MAA(~n))

def unitsF() -> FmaP:
  T.Two{Unit{}, T.Two{T.Two{Unit{}, Unit{}}, Unit{}}}

# ---- f: Linear(4, 4) over the 6 rows ----

def L44() -> Type:
  T.Lin<4n, 4n>

def Fl() -> T.Layer<L44, M(6n, 4n), M(6n, 4n), M(6n, 4n)>:
  T.Linear(6n, 4n, 4n, 0n)

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

def ln4() -> T.LN<4n>:
  T.LN{mat(1n, 4n, 61n), mat(1n, 4n, 67n)}

# =====================================================================================================
# gate: (g, h) to h + Tile(g) * f(h)
# =====================================================================================================

def FIP() -> Type:
  T.Two<Unit, L44>

def FIS() -> Type:
  T.Two<Unit, M(6n, 4n)>

# h to (h, f(h)); the two input gradients are summed
def FI() -> T.Layer<FIP, M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>, FIS>:
  T.Fork(~Unit, ~L44, ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~Unit, ~M(6n, 4n), ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)), ~T.Id(~M(6n, 4n)), ~Fl)

def GX() -> Type:
  T.Two<M(2n, 4n), M(6n, 4n)>

def G1P() -> Type:
  T.Two<Unit, FIP>

def G1S() -> Type:
  T.Two<Unit, FIS>

def G1() -> T.Layer<G1P, GX, X3(6n), G1S>:
  T.Both(~Unit, ~FIP, ~M(2n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~T.Two<M(6n, 4n), M(6n, 4n)>, ~Unit, ~FIS, ~T.Tile(2n, 3n, 4n), ~FI)

def GP() -> Type:
  T.Two<G1P, FmaP>

def GS() -> Type:
  T.Two<G1S, FmaS(6n)>

def Gate() -> T.Layer<GP, GX, M(6n, 4n), GS>:
  T.Seq(~G1P, ~FmaP, ~GX, ~X3(6n), ~M(6n, 4n), ~G1S, ~FmaS(6n), ~G1, ~Fma(~6n))

def gate_p() -> GP:
  T.Two{T.Two{Unit{}, T.Two{Unit{}, lin(4n, 4n, 5n)}}, unitsF}

def gate_dp(g: GP) -> L44:
  match g:
    case T.Two{a, u}:
      match a:
        case T.Two{u2, fi}:
          match fi:
            case T.Two{u3, l}:
              l

def gate_dx(+t: String, d: GX) -> IO(Unit):
  match d:
    case T.Two{dg, dh}:
      do IO<Unit>:
        Unit <- T.Mat.print(2n, 4n, tag(t, "dg"), dg)
        T.Mat.print(6n, 4n, tag(t, "dh"), dh)

def gate_b(r: T.Bwd<GP, GX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- gate_dx("gate", dx)
        show_lin(4n, 4n, "gate.f", gate_dp(g))

def gate_f(r: T.Fwd<GP, M(6n, 4n), GS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "gate.y", y)
        gate_b(T.Layer.bwd(GP, GX, M(6n, 4n), GS, Gate, p, s, mat(6n, 4n, 90n), T.Layer.zeros(GP, GX, M(6n, 4n), GS, Gate)))

def gate() -> IO(Unit):
  gate_f(T.Layer.fwd(GP, GX, M(6n, 4n), GS, Gate, gate_p, T.Two{mat(2n, 4n, 20n), mat2(6n, 4n, 30n)}))

# =====================================================================================================
# mod: (m, (sh, h)) to Tile(sh) + Tile(m) * LayerNorm(h)
# =====================================================================================================

def MX() -> Type:
  T.Two<M(2n, 4n), T.Two<M(2n, 4n), M(6n, 4n)>>

def TLP() -> Type:
  T.Two<Unit, T.LN<4n>>

def TLS() -> Type:
  T.Two<Unit, M(6n, 4n)>

def TL() -> T.Layer<TLP, T.Two<M(2n, 4n), M(6n, 4n)>, T.Two<M(6n, 4n), M(6n, 4n)>, TLS>:
  T.Both(~Unit, ~T.LN<4n>, ~M(2n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~Unit, ~M(6n, 4n), ~T.Tile(2n, 3n, 4n), ~T.LayerNorm(6n, 4n, 0.00001))

def M1P() -> Type:
  T.Two<Unit, TLP>

def M1S() -> Type:
  T.Two<Unit, TLS>

def M1() -> T.Layer<M1P, MX, X3(6n), M1S>:
  T.Both(~Unit, ~TLP, ~M(2n, 4n), ~T.Two<M(2n, 4n), M(6n, 4n)>, ~M(6n, 4n), ~T.Two<M(6n, 4n), M(6n, 4n)>, ~Unit, ~TLS, ~T.Tile(2n, 3n, 4n), ~TL)

def MP() -> Type:
  T.Two<M1P, FmaP>

def MS() -> Type:
  T.Two<M1S, FmaS(6n)>

def Mod() -> T.Layer<MP, MX, M(6n, 4n), MS>:
  T.Seq(~M1P, ~FmaP, ~MX, ~X3(6n), ~M(6n, 4n), ~M1S, ~FmaS(6n), ~M1, ~Fma(~6n))

def mod_p() -> MP:
  T.Two{T.Two{Unit{}, T.Two{Unit{}, ln4}}, unitsF}

def mod_dp(g: MP) -> T.LN<4n>:
  match g:
    case T.Two{a, u}:
      match a:
        case T.Two{u2, tl}:
          match tl:
            case T.Two{u3, l}:
              l

def mod_dx(+t: String, d: MX) -> IO(Unit):
  match d:
    case T.Two{dm, r}:
      match r:
        case T.Two{dsh, dh}:
          do IO<Unit>:
            Unit <- T.Mat.print(2n, 4n, tag(t, "dm"), dm)
            Unit <- T.Mat.print(2n, 4n, tag(t, "dsh"), dsh)
            T.Mat.print(6n, 4n, tag(t, "dh"), dh)

def mod_b(+t: String, r: T.Bwd<MP, MX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- mod_dx(t, dx)
        show_ln(4n, tag(t, "ln"), mod_dp(g))

def mod_f(+t: String, r: T.Fwd<MP, M(6n, 4n), MS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, tag(t, "y"), y)
        mod_b(t, T.Layer.bwd(MP, MX, M(6n, 4n), MS, Mod, p, s, mat(6n, 4n, 95n), T.Layer.zeros(MP, MX, M(6n, 4n), MS, Mod)))

def mod() -> IO(Unit):
  mod_f("mod", T.Layer.fwd(MP, MX, M(6n, 4n), MS, Mod, mod_p, T.Two{mat(2n, 4n, 21n), T.Two{mat(2n, 4n, 25n), mat2(6n, 4n, 31n)}}))

# ---- id: mod with m = 1 and sh = 0, then LayerNorm alone on the same h and dy ----

def LNl() -> T.Layer<T.LN<4n>, M(6n, 4n), M(6n, 4n), M(6n, 4n)>:
  T.LayerNorm(6n, 4n, 0.00001)

def ln_b(r: T.Bwd<T.LN<4n>, M(6n, 4n)>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "ln.dh", dx)
        show_ln(4n, "ln.ln", g)

def ln_f(r: T.Fwd<T.LN<4n>, M(6n, 4n), M(6n, 4n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "ln.y", y)
        ln_b(T.Layer.bwd(T.LN<4n>, M(6n, 4n), M(6n, 4n), M(6n, 4n), LNl, p, s, mat(6n, 4n, 95n), T.Layer.zeros(T.LN<4n>, M(6n, 4n), M(6n, 4n), M(6n, 4n), LNl)))

def id() -> IO(Unit):
  do IO<Unit>:
    Unit <- mod_f("id", T.Layer.fwd(MP, MX, M(6n, 4n), MS, Mod, mod_p, T.Two{TA.Mat.from_list(2n, 4n, ones(8n)), T.Two{TA.Mat.zeros(2n, 4n), mat2(6n, 4n, 31n)}}))
    ln_f(T.Layer.fwd(T.LN<4n>, M(6n, 4n), M(6n, 4n), M(6n, 4n), LNl, ln4, mat2(6n, 4n, 31n)))

# =====================================================================================================
# block: (cond, h) to h + Tile(g) * f(mod(m, sh, h)), with (g, m, sh) three Linear(3, 4) of silu(cond)
# =====================================================================================================

def L34() -> Type:
  T.Lin<3n, 4n>

def Lc() -> T.Layer<L34, M(2n, 3n), M(2n, 4n), M(2n, 3n)>:
  T.Linear(2n, 3n, 4n, 0n)

def C3() -> Type:
  T.Two<M(2n, 4n), T.Two<M(2n, 4n), M(2n, 4n)>>

def dup23(m: M(2n, 3n)) -> T.Two<M(2n, 3n), M(2n, 3n)>:
  T.Mat.dup(2n, 3n, m)

def add23(a: M(2n, 3n), b: M(2n, 3n)) -> M(2n, 3n):
  T.Mat.sum2(2n, 3n, a, b)

# silu(cond) to (m, sh)
def K2() -> T.Layer<T.Two<L34, L34>, M(2n, 3n), T.Two<M(2n, 4n), M(2n, 4n)>, T.Two<M(2n, 3n), M(2n, 3n)>>:
  T.Fork(~L34, ~L34, ~M(2n, 3n), ~M(2n, 4n), ~M(2n, 4n), ~M(2n, 3n), ~M(2n, 3n), ~(m => dup23(m)), ~(a => b => add23(a, b)), ~Lc, ~Lc)

def K3P() -> Type:
  T.Two<L34, T.Two<L34, L34>>

def K3S() -> Type:
  T.Two<M(2n, 3n), T.Two<M(2n, 3n), M(2n, 3n)>>

# silu(cond) to (g, (m, sh))
def K3() -> T.Layer<K3P, M(2n, 3n), C3, K3S>:
  T.Fork(~L34, ~T.Two<L34, L34>, ~M(2n, 3n), ~M(2n, 4n), ~T.Two<M(2n, 4n), M(2n, 4n)>, ~M(2n, 3n), ~T.Two<M(2n, 3n), M(2n, 3n)>, ~(m => dup23(m)), ~(a => b => add23(a, b)), ~Lc, ~K2)

def KP() -> Type:
  T.Two<Unit, K3P>

def KS() -> Type:
  T.Two<M(2n, 3n), K3S>

def K() -> T.Layer<KP, M(2n, 3n), C3, KS>:
  T.Seq(~Unit, ~K3P, ~M(2n, 3n), ~M(2n, 3n), ~C3, ~M(2n, 3n), ~K3S, ~T.SiLU(2n, 3n), ~K3)

# h to (h, h)
def HD() -> T.Layer<T.Two<Unit, Unit>, M(6n, 4n), T.Two<M(6n, 4n), M(6n, 4n)>, T.Two<Unit, Unit>>:
  T.Fork(~Unit, ~Unit, ~M(6n, 4n), ~M(6n, 4n), ~M(6n, 4n), ~Unit, ~Unit, ~(m => T.Mat.dup(6n, 4n, m)), ~(a => b => T.Mat.sum2(6n, 4n, a, b)), ~T.Id(~M(6n, 4n)), ~T.Id(~M(6n, 4n)))

def BX() -> Type:
  T.Two<M(2n, 3n), M(6n, 4n)>

def B1Y() -> Type:
  T.Two<C3, T.Two<M(6n, 4n), M(6n, 4n)>>

def B1P() -> Type:
  T.Two<KP, T.Two<Unit, Unit>>

def B1S() -> Type:
  T.Two<KS, T.Two<Unit, Unit>>

def B1() -> T.Layer<B1P, BX, B1Y, B1S>:
  T.Both(~KP, ~T.Two<Unit, Unit>, ~M(2n, 3n), ~M(6n, 4n), ~C3, ~T.Two<M(6n, 4n), M(6n, 4n)>, ~KS, ~T.Two<Unit, Unit>, ~K, ~HD)

# (g, h2, (m, (sh, h1))): the gated residual's input, with mod's input as its branch's
def B2X() -> Type:
  T.Two<M(2n, 4n), T.Two<M(6n, 4n), MX>>

def rw3(g: M(2n, 4n), m: M(2n, 4n), sh: M(2n, 4n), hh: T.Two<M(6n, 4n), M(6n, 4n)>) -> B2X:
  match hh:
    case T.Two{h1, h2}:
      T.Two{g, T.Two{h2, T.Two{m, T.Two{sh, h1}}}}

def rw2(g: M(2n, 4n), r: T.Two<M(2n, 4n), M(2n, 4n)>, hh: T.Two<M(6n, 4n), M(6n, 4n)>) -> B2X:
  match r:
    case T.Two{m, sh}:
      rw3(g, m, sh, hh)

def rw1(c: C3, hh: T.Two<M(6n, 4n), M(6n, 4n)>) -> B2X:
  match c:
    case T.Two{g, r}:
      rw2(g, r, hh)

# ((g, (m, sh)), (h1, h2)) to (g, (h2, (m, (sh, h1))))
def rw(x: B1Y) -> B2X:
  match x:
    case T.Two{c, hh}:
      rw1(c, hh)

def un3(g: M(2n, 4n), h2: M(6n, 4n), m: M(2n, 4n), r: T.Two<M(2n, 4n), M(6n, 4n)>) -> B1Y:
  match r:
    case T.Two{sh, h1}:
      T.Two{T.Two{g, T.Two{m, sh}}, T.Two{h1, h2}}

def un2(g: M(2n, 4n), h2: M(6n, 4n), r: MX) -> B1Y:
  match r:
    case T.Two{m, r2}:
      un3(g, h2, m, r2)

def un1(g: M(2n, 4n), r: T.Two<M(6n, 4n), MX>) -> B1Y:
  match r:
    case T.Two{h2, r2}:
      un2(g, h2, r2)

# rw's inverse, for the gradients
def un(x: B2X) -> B1Y:
  match x:
    case T.Two{g, r}:
      un1(g, r)

# f(mod(m, sh, h1))
def MFP() -> Type:
  T.Two<MP, L44>

def MFS() -> Type:
  T.Two<MS, M(6n, 4n)>

def MF() -> T.Layer<MFP, MX, M(6n, 4n), MFS>:
  T.Seq(~MP, ~L44, ~MX, ~M(6n, 4n), ~M(6n, 4n), ~MS, ~M(6n, 4n), ~Mod, ~Fl)

def HMP() -> Type:
  T.Two<Unit, MFP>

def HMS() -> Type:
  T.Two<Unit, MFS>

# (h2, mod's input) to (h2, f(mod(...)))
def HM() -> T.Layer<HMP, T.Two<M(6n, 4n), MX>, T.Two<M(6n, 4n), M(6n, 4n)>, HMS>:
  T.Both(~Unit, ~MFP, ~M(6n, 4n), ~MX, ~M(6n, 4n), ~M(6n, 4n), ~Unit, ~MFS, ~T.Id(~M(6n, 4n)), ~MF)

def B2P() -> Type:
  T.Two<Unit, HMP>

def B2S() -> Type:
  T.Two<Unit, HMS>

def B2() -> T.Layer<B2P, B2X, X3(6n), B2S>:
  T.Both(~Unit, ~HMP, ~M(2n, 4n), ~T.Two<M(6n, 4n), MX>, ~M(6n, 4n), ~T.Two<M(6n, 4n), M(6n, 4n)>, ~Unit, ~HMS, ~T.Tile(2n, 3n, 4n), ~HM)

def B3P() -> Type:
  T.Two<B2P, FmaP>

def B3S() -> Type:
  T.Two<B2S, FmaS(6n)>

def B3() -> T.Layer<B3P, B2X, M(6n, 4n), B3S>:
  T.Seq(~B2P, ~FmaP, ~B2X, ~X3(6n), ~M(6n, 4n), ~B2S, ~FmaS(6n), ~B2, ~Fma(~6n))

def B4P() -> Type:
  T.Two<Unit, B3P>

def B4S() -> Type:
  T.Two<Unit, B3S>

def B4() -> T.Layer<B4P, B1Y, M(6n, 4n), B4S>:
  T.Seq(~Unit, ~B3P, ~B1Y, ~B2X, ~M(6n, 4n), ~Unit, ~B3S, ~Rewire(~B1Y, ~B2X, ~(x => rw(x)), ~(x => un(x))), ~B3)

def BP() -> Type:
  T.Two<B1P, B4P>

def BS() -> Type:
  T.Two<B1S, B4S>

def Block() -> T.Layer<BP, BX, M(6n, 4n), BS>:
  T.Seq(~B1P, ~B4P, ~BX, ~B1Y, ~M(6n, 4n), ~B1S, ~B4S, ~B1, ~B4)

def block_p() -> BP:
  T.Two{T.Two{T.Two{Unit{}, T.Two{lin(3n, 4n, 7n), T.Two{lin(3n, 4n, 9n), lin(3n, 4n, 11n)}}}, T.Two{Unit{}, Unit{}}},
    T.Two{Unit{}, T.Two{T.Two{Unit{}, T.Two{Unit{}, T.Two{mod_p, lin(4n, 4n, 5n)}}}, unitsF}}}

def show_k(r: K3P) -> IO(Unit):
  match r:
    case T.Two{lg, r2}:
      match r2:
        case T.Two{lm, lsh}:
          do IO<Unit>:
            Unit <- show_lin(3n, 4n, "block.g", lg)
            Unit <- show_lin(3n, 4n, "block.m", lm)
            show_lin(3n, 4n, "block.sh", lsh)

def show_mf(r: MFP) -> IO(Unit):
  match r:
    case T.Two{mp, lf}:
      do IO<Unit>:
        Unit <- show_ln(4n, "block.ln", mod_dp(mp))
        show_lin(4n, 4n, "block.f", lf)

def mf_of(r: B4P) -> MFP:
  match r:
    case T.Two{u, b3}:
      match b3:
        case T.Two{b2, f}:
          match b2:
            case T.Two{u2, hm}:
              match hm:
                case T.Two{u3, mf}:
                  mf

def k_of(r: B1P) -> K3P:
  match r:
    case T.Two{kp, u}:
      match kp:
        case T.Two{u2, k3}:
          k3

def block_dp(g: BP) -> IO(Unit):
  match g:
    case T.Two{b1, b4}:
      do IO<Unit>:
        Unit <- show_k(k_of(b1))
        show_mf(mf_of(b4))

def block_dx(d: BX) -> IO(Unit):
  match d:
    case T.Two{dc, dh}:
      do IO<Unit>:
        Unit <- T.Mat.print(2n, 3n, "block.dcond", dc)
        T.Mat.print(6n, 4n, "block.dh", dh)

def block_b(r: T.Bwd<BP, BX>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        Unit <- block_dx(dx)
        block_dp(g)

def block_f(r: T.Fwd<BP, M(6n, 4n), BS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        Unit <- T.Mat.print(6n, 4n, "block.y", y)
        block_b(T.Layer.bwd(BP, BX, M(6n, 4n), BS, Block, p, s, mat(6n, 4n, 97n), T.Layer.zeros(BP, BX, M(6n, 4n), BS, Block)))

def block() -> IO(Unit):
  block_f(T.Layer.fwd(BP, BX, M(6n, 4n), BS, Block, block_p, T.Two{mat2(2n, 3n, 23n), mat2(6n, 4n, 33n)}))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- gate()
    Unit <- mod()
    Unit <- id()
    block()
