# Every layer on its own, against reference/test_ops.py: for each one, the forward output, then the backward pass from a
# given dY into zero gradients: dX and the parameter gradients. Inputs are read from the directory given as the first
# argument (<op>.<name>.bin, little-endian F32); every number is printed as the decimal of its 32 bits.
#
#   bend demo/ops_test.bend -o .scratch/ops_test && .scratch/ops_test <dir>

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def ld(+r: Nat, +c: Nat, +dir: String, +name: String) -> IO(TA.Mat<r, c>):
  T.Mat.load(r, c, dir ++ "/" ++ name ++ ".bin", 0)

def show3(~n: Nat, ~c: Nat, ~P: Type, ~pp: @+t: String -> P -> IO(Unit), +tag: String, r: T.Bwd<P, TA.Mat<n, c>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        T.Mat.print(n, c, tag ++ ".dx", dx)
        pp(tag ++ ".g", g)

def show2(~n: Nat, ~c: Nat, ~o: Nat, ~P: Type, ~S: Type, ~l: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, ~pp: @+t: String -> P -> IO(Unit), +tag: String, dy: TA.Mat<n, o>, r: T.Fwd<P, TA.Mat<n, o>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(n, o, tag ++ ".y", y)
        show3(~n, ~c, ~P, ~pp, tag, T.Layer.bwd(P, TA.Mat<n, c>, TA.Mat<n, o>, S, l, p, s, dy, T.Layer.zeros(P, TA.Mat<n, c>, TA.Mat<n, o>, S, l)))

# a layer from n x c to n x o: forward on x, backward from dy, every result printed under `tag`
def check(~n: Nat, ~c: Nat, ~o: Nat, ~P: Type, ~S: Type, ~l: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, ~pp: @+t: String -> P -> IO(Unit), +tag: String, p: P, x: TA.Mat<n, c>, dy: TA.Mat<n, o>) -> IO(Unit):
  show2(~n, ~c, ~o, ~P, ~S, ~l, ~pp, tag, dy, T.Layer.fwd(P, TA.Mat<n, c>, TA.Mat<n, o>, S, l, p, x))

def pp_unit(+tag: String, u: Unit) -> IO(Unit):
  IO.pure(Unit, Unit{})

def pp_unit_d(+tag: String, u: T.DropP) -> IO(Unit):
  IO.pure(Unit, Unit{})

# ---- activations and row layers: 5 x 6 ----

def t_act2(+tag: String, k: Nat, x: TA.Mat<5n, 6n>, dy: TA.Mat<5n, 6n>) -> IO(Unit):
  match k:
    case 0n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.ReLU(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 1n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.Sigmoid(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 2n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.Tanh(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 3n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.SiLU(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 4n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.GELU(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 5n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.Softmax(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case 6n:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.LogSoftmax(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)
    case _:
      check(~5n, ~6n, ~6n, ~Unit, ~TA.Mat<5n, 6n>, ~T.GELUErf(5n, 6n), ~pp_unit, tag, Unit{}, x, dy)

def t_act(+dir: String, +tag: String, +k: Nat) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, tag ++ ".x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, tag ++ ".dy")
    t_act2(tag, k, x, dy)

# erf on its own (GELUErf's building block): 4096 inputs, its 4096 outputs, checked within 1 ulp of float64's erf
def t_erf(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<64n, 64n> <- ld(64n, 64n, dir, "erf.x")
    T.Mat.print(64n, 64n, "erf.y", T.Ew.map(~T.erf, 64n, 64n, x))

def t_rms(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "rmsnorm.x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "rmsnorm.dy")
    g : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "rmsnorm.g")
    check(~5n, ~6n, ~6n, ~TA.Mat<1n, 6n>, ~TA.Mat<5n, 6n>, ~T.RMSNorm(5n, 6n, 0.00001), ~(+t => m => T.Mat.print(1n, 6n, t, m)), "rmsnorm", g, x, dy)

def pp_ln(+tag: String, p: T.LN<6n>) -> IO(Unit):
  match p:
    case T.LN{g, b}:
      do IO<Unit>:
        T.Mat.print(1n, 6n, tag ++ ".g", g)
        T.Mat.print(1n, 6n, tag ++ ".b", b)

def t_ln(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "layernorm.x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "layernorm.dy")
    g : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "layernorm.g")
    b : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "layernorm.b")
    check(~5n, ~6n, ~6n, ~T.LN<6n>, ~TA.Mat<5n, 6n>, ~T.LayerNorm(5n, 6n, 0.00001), ~pp_ln, "layernorm", T.LN{g, b}, x, dy)

# GroupNorm: 2 images of 4 pixels, 3 groups of 2 channels (an 8 x 6 matrix, channels last), so that a group is neither a
# row nor a column
def t_gn(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<8n, 6n> <- ld(8n, 6n, dir, "groupnorm.x")
    dy : TA.Mat<8n, 6n> <- ld(8n, 6n, dir, "groupnorm.dy")
    g : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "groupnorm.g")
    b : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "groupnorm.b")
    check(~8n, ~6n, ~6n, ~T.LN<6n>, ~TA.Mat<8n, 6n>, ~T.GroupNorm(2n, 4n, 3n, 2n, 0.00001), ~pp_ln, "groupnorm", T.LN{g, b}, x, dy)

def t_lnb(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "linear_nb.x")
    dy : TA.Mat<5n, 4n> <- ld(5n, 4n, dir, "linear_nb.dy")
    w : TA.Mat<4n, 6n> <- ld(4n, 6n, dir, "linear_nb.w")
    check(~5n, ~6n, ~4n, ~TA.Mat<4n, 6n>, ~TA.Mat<5n, 6n>, ~T.LinearNB(5n, 6n, 4n, 1n), ~(+t => m => T.Mat.print(4n, 6n, t, m)), "linear_nb", w, x, dy)

# ---- composition: a SwiGLU MLP (down(silu(gate(x)) * up(x))) and a residual RMSNorm, both built from combinators ----

def Gate() -> T.Layer<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<5n, 6n>, TA.Mat<5n, 8n>, T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>>:
  T.Seq(~TA.Mat<8n, 6n>, ~Unit, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 8n>, ~TA.Mat<5n, 8n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 8n>, ~T.LinearNB(5n, 6n, 8n, 0n), ~T.SiLU(5n, 8n))

def GU() -> T.Layer<T.Two<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<8n, 6n>>, TA.Mat<5n, 6n>, T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>, T.Two<T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, TA.Mat<5n, 6n>>>:
  T.Fork(~T.Two<TA.Mat<8n, 6n>, Unit>, ~TA.Mat<8n, 6n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 8n>, ~TA.Mat<5n, 8n>, ~T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, ~TA.Mat<5n, 6n>, ~(m => T.Mat.dup(5n, 6n, m)), ~(a => b => T.Mat.sum2(5n, 6n, a, b)), ~Gate(), ~T.LinearNB(5n, 6n, 8n, 0n))

def GLU() -> T.Layer<T.Two<T.Two<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<8n, 6n>>, Unit>, TA.Mat<5n, 6n>, TA.Mat<5n, 8n>, T.Two<T.Two<T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, TA.Mat<5n, 6n>>, T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>>>:
  T.Seq(~T.Two<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<8n, 6n>>, ~Unit, ~TA.Mat<5n, 6n>, ~T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>, ~TA.Mat<5n, 8n>, ~T.Two<T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, TA.Mat<5n, 6n>>, ~T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>, ~GU(), ~T.Mul2(5n, 8n))

def SwiP() -> Type:
  T.Two<T.Two<T.Two<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<8n, 6n>>, Unit>, TA.Mat<6n, 8n>>

def SwiS() -> Type:
  T.Two<T.Two<T.Two<T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, TA.Mat<5n, 6n>>, T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>>, TA.Mat<5n, 8n>>

def SwiGLU() -> T.Layer<SwiP, TA.Mat<5n, 6n>, TA.Mat<5n, 6n>, SwiS>:
  T.Seq(~T.Two<T.Two<T.Two<TA.Mat<8n, 6n>, Unit>, TA.Mat<8n, 6n>>, Unit>, ~TA.Mat<6n, 8n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 8n>, ~TA.Mat<5n, 6n>, ~T.Two<T.Two<T.Two<TA.Mat<5n, 6n>, TA.Mat<5n, 8n>>, TA.Mat<5n, 6n>>, T.Two<TA.Mat<5n, 8n>, TA.Mat<5n, 8n>>>, ~TA.Mat<5n, 8n>, ~GLU(), ~T.LinearNB(5n, 8n, 6n, 0n))

def pp_swi(+tag: String, p: SwiP) -> IO(Unit):
  match p:
    case T.Two{T.Two{T.Two{T.Two{wg, u1}, wu}, u2}, wd}:
      do IO<Unit>:
        T.Mat.print(8n, 6n, tag ++ ".wg", wg)
        T.Mat.print(8n, 6n, tag ++ ".wu", wu)
        T.Mat.print(6n, 8n, tag ++ ".wd", wd)

def t_swiglu(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "swiglu.x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "swiglu.dy")
    wg : TA.Mat<8n, 6n> <- ld(8n, 6n, dir, "swiglu.wg")
    wu : TA.Mat<8n, 6n> <- ld(8n, 6n, dir, "swiglu.wu")
    wd : TA.Mat<6n, 8n> <- ld(6n, 8n, dir, "swiglu.wd")
    check(~5n, ~6n, ~6n, ~SwiP, ~SwiS, ~SwiGLU(), ~pp_swi, "swiglu", T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}, x, dy)

def ResP() -> Type:
  T.Two<T.Two<Unit, TA.Mat<1n, 6n>>, Unit>

def ResS() -> Type:
  T.Two<T.Two<Unit, TA.Mat<5n, 6n>>, Unit>

def pp_res(+tag: String, p: ResP) -> IO(Unit):
  match p:
    case T.Two{T.Two{u1, g}, u2}:
      T.Mat.print(1n, 6n, tag, g)

def t_residual(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "residual.x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "residual.dy")
    g : TA.Mat<1n, 6n> <- ld(1n, 6n, dir, "residual.g")
    check(~5n, ~6n, ~6n, ~ResP, ~ResS, ~T.Residual(~5n, ~6n, ~TA.Mat<1n, 6n>, ~TA.Mat<5n, 6n>, ~T.RMSNorm(5n, 6n, 0.00001)), ~pp_res, "residual", T.Two{T.Two{Unit{}, g}, Unit{}}, x, dy)

# ---- dropout, p = 0.25, seed 7, at step 0 and at step 3 ----

def t_dropout(+dir: String, +tag: String, +t: U32) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, tag ++ ".x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, tag ++ ".dy")
    check(~5n, ~6n, ~6n, ~T.DropP, ~TA.Mat<5n, 6n>, ~T.Dropout(5n, 6n, 0.25, 7), ~pp_unit_d, tag, T.DropP{t}, x, dy)

# after one optimizer step the counter has moved: the mask of step 4
def stepped(r: T.Step<T.DropP>) -> T.DropP:
  match r:
    case T.Step{p, g, st}:
      p

def t_dropout_stepped(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "dropout4.x")
    dy : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "dropout4.dy")
    check(~5n, ~6n, ~6n, ~T.DropP, ~TA.Mat<5n, 6n>, ~T.Dropout(5n, 6n, 0.25, 7), ~pp_unit_d, "dropout4", stepped(T.Layer.step(T.DropP, TA.Mat<5n, 6n>, TA.Mat<5n, 6n>, TA.Mat<5n, 6n>, T.Dropout(5n, 6n, 0.25, 7), T.SGD{0.1}, T.DropP{3}, T.DropP{0}, T.NoMom{})), x, dy)

# ---- RoPE: 2 sequences of 4 positions, 2 heads of 4 ----

def t_rope(+dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<8n, 8n> <- ld(8n, 8n, dir, "rope.x")
    dy : TA.Mat<8n, 8n> <- ld(8n, 8n, dir, "rope.dy")
    check(~8n, ~8n, ~8n, ~Unit, ~Unit, ~T.RoPE(8n, 8n, 4n, 2n, 2n, 10000.0, {==}), ~pp_unit, "rope", Unit{}, x, dy)

# ---- attention: 2 sequences of 4 positions, 2 key/value heads, 2 query heads each, heads of 4 ----

def AttnX() -> Type:
  T.Two<TA.Mat<8n, 16n>, T.Two<TA.Mat<8n, 8n>, TA.Mat<8n, 8n>>>

def Attn() -> T.Layer<Unit, AttnX, TA.Mat<8n, 16n>, T.AttnS>:
  T.Attention(8n, 16n, 8n, 2n, 4n, 2n, 2n, 4n, 0.5, {==}, {==}, {==})

def attn3(r: T.Bwd<Unit, AttnX>) -> IO(Unit):
  match r:
    case T.Bwd{p, T.Two{dq, T.Two{dk, dv}}, g}:
      do IO<Unit>:
        T.Mat.print(8n, 16n, "attn.dq", dq)
        T.Mat.print(8n, 8n, "attn.dk", dk)
        T.Mat.print(8n, 8n, "attn.dv", dv)

def attn2(dy: TA.Mat<8n, 16n>, r: T.Fwd<Unit, TA.Mat<8n, 16n>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(8n, 16n, "attn.y", y)
        attn3(T.Layer.bwd(Unit, AttnX, TA.Mat<8n, 16n>, T.AttnS, Attn(), p, s, dy, Unit{}))

def t_attn(+dir: String) -> IO(Unit):
  do IO<Unit>:
    q : TA.Mat<8n, 16n> <- ld(8n, 16n, dir, "attn.q")
    k : TA.Mat<8n, 8n> <- ld(8n, 8n, dir, "attn.k")
    v : TA.Mat<8n, 8n> <- ld(8n, 8n, dir, "attn.v")
    dy : TA.Mat<8n, 16n> <- ld(8n, 16n, dir, "attn.dy")
    attn2(dy, T.Layer.fwd(Unit, AttnX, TA.Mat<8n, 16n>, T.AttnS, Attn(), Unit{}, T.Two{q, T.Two{k, v}}))

# ---- embedding: 5 ids into a table of 7 x 4 (ids repeat) ----

def labs(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs(t)

def emb3(r: T.Bwd<TA.Mat<7n, 4n>, List<&2, Nat>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      T.Mat.print(7n, 4n, "embedding.g", g)

def emb2(dy: TA.Mat<5n, 4n>, r: T.Fwd<TA.Mat<7n, 4n>, TA.Mat<5n, 4n>, List<&2, Nat>>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(5n, 4n, "embedding.y", y)
        emb3(T.Layer.bwd(TA.Mat<7n, 4n>, List<&2, Nat>, TA.Mat<5n, 4n>, List<&2, Nat>, T.Embedding(5n, 7n, 4n), p, s, dy, TA.Mat.zeros(7n, 4n)))

def t_emb(+dir: String) -> IO(Unit):
  do IO<Unit>:
    e : TA.Mat<7n, 4n> <- ld(7n, 4n, dir, "embedding.e")
    dy : TA.Mat<5n, 4n> <- ld(5n, 4n, dir, "embedding.dy")
    ids : List<&2, U32> <- T.read_at(dir ++ "/embedding.ids.bin", 0, 5)
    emb2(dy, T.Layer.fwd(TA.Mat<7n, 4n>, List<&2, Nat>, TA.Mat<5n, 4n>, List<&2, Nat>, T.Embedding(5n, 7n, 4n), e, labs(ids)))

# ---- tied embeddings: the same 7 x 4 table looks up the ids and projects back to 7 logits ----

def TiedS() -> Type:
  T.Three<List<&2, Nat>, TA.Mat<5n, 4n>, TA.Mat<5n, 4n>>

def Tied() -> T.Layer<T.Two<TA.Mat<7n, 4n>, TA.Mat<4n, 4n>>, List<&2, Nat>, TA.Mat<5n, 7n>, TiedS>:
  T.Tie(~TA.Mat<7n, 4n>, ~TA.Mat<4n, 4n>, ~List<&2, Nat>, ~TA.Mat<5n, 4n>, ~TA.Mat<5n, 4n>, ~TA.Mat<5n, 7n>, ~List<&2, Nat>, ~TA.Mat<5n, 4n>, ~TA.Mat<5n, 4n>, ~T.Embedding(5n, 7n, 4n), ~T.LinearNB(5n, 4n, 4n, 0n), ~T.LinearNB(5n, 4n, 7n, 0n))

def tied3(r: T.Bwd<T.Two<TA.Mat<7n, 4n>, TA.Mat<4n, 4n>>, List<&2, Nat>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, T.Two{ge, gw}}:
      do IO<Unit>:
        T.Mat.print(7n, 4n, "tied.g.e", ge)
        T.Mat.print(4n, 4n, "tied.g.w", gw)

def tied2(dy: TA.Mat<5n, 7n>, r: T.Fwd<T.Two<TA.Mat<7n, 4n>, TA.Mat<4n, 4n>>, TA.Mat<5n, 7n>, TiedS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(5n, 7n, "tied.y", y)
        tied3(T.Layer.bwd(T.Two<TA.Mat<7n, 4n>, TA.Mat<4n, 4n>>, List<&2, Nat>, TA.Mat<5n, 7n>, TiedS, Tied(), p, s, dy, T.Two{TA.Mat.zeros(7n, 4n), TA.Mat.zeros(4n, 4n)}))

def t_tied(+dir: String) -> IO(Unit):
  do IO<Unit>:
    e : TA.Mat<7n, 4n> <- ld(7n, 4n, dir, "tied.e")
    w : TA.Mat<4n, 4n> <- ld(4n, 4n, dir, "tied.w")
    dy : TA.Mat<5n, 7n> <- ld(5n, 7n, dir, "tied.dy")
    ids : List<&2, U32> <- T.read_at(dir ++ "/embedding.ids.bin", 0, 5)
    tied2(dy, T.Layer.fwd(T.Two<TA.Mat<7n, 4n>, TA.Mat<4n, 4n>>, List<&2, Nat>, TA.Mat<5n, 7n>, TiedS, Tied(), T.Two{e, w}, labs(ids)))

# ---- losses: the loss and its gradient ----

def loss_show(+tag: String, r: T.LossOut<TA.Mat<5n, 6n>>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        T.Mat.print(5n, 6n, tag ++ ".dy", dy)

def t_losses2(+dir: String, +ls: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    lp : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "nll.x")
    loss_show("nll", T.NLL(5n, 6n, lp, ls))
    z : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "ce.x")
    loss_show("ce", T.CrossEntropy(5n, 6n, z, ls))

def t_losses(+dir: String) -> IO(Unit):
  do IO<Unit>:
    y : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "mse.y")
    t : TA.Mat<5n, 6n> <- ld(5n, 6n, dir, "mse.t")
    loss_show("mse", T.MSE(5n, 6n, y, t))
    ls : List<&2, U32> <- T.read_at(dir ++ "/nll.labels.bin", 0, 5)
    t_losses2(dir, labs(ls))

def go(+dir: String) -> IO(Unit):
  do IO<Unit>:
    t_act(dir, "relu", 0n)
    t_act(dir, "sigmoid", 1n)
    t_act(dir, "tanh", 2n)
    t_act(dir, "silu", 3n)
    t_act(dir, "gelu", 4n)
    t_act(dir, "softmax", 5n)
    t_act(dir, "logsoftmax", 6n)
    t_act(dir, "gelu_erf", 7n)
    t_erf(dir)
    t_rms(dir)
    t_ln(dir)
    t_lnb(dir)
    t_swiglu(dir)
    t_residual(dir)
    t_dropout(dir, "dropout0", 0)
    t_dropout(dir, "dropout3", 3)
    t_dropout_stepped(dir)
    t_rope(dir)
    t_attn(dir)
    t_emb(dir)
    t_tied(dir)
    t_losses(dir)
    t_gn(dir)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{dir, Nil{}}}:
      go(dir)
    case _:
      IO.print("usage: ops_test <dir>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
