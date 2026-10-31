# The held-out corpus (bench/corpus/, reference/test_corpus.py): one case a run, every layer, loss and optimizer step of
# torch/main.bend at a shape read at run time. The directory given as the first argument holds case.bin (little-endian
# U32: the operation, then its sizes and scalars, an F32 scalar as its 32 bits) and the tensors (<name>.bin,
# little-endian F32; ids.bin and labels.bin as little-endian U32). Every result is printed as the decimal of its 32
# bits. Operations 30 to 34 are torch/fast.bend's LinearF, LinearFirstF, LinearNBF, SiLU and Mul2 (the layers the demos
# train with) on the same cases as 0, 1, 2, 6 and 19. Operations from 100 up are torch/tensor.bend's, through demo/tensor_test.bend's own reader (op - 100).
#
#   bend demo/corpus_test.bend -o .scratch/corpus_test && .scratch/corpus_test <dir>

import Base
import ../torch/main.bend as T
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import ./tensor_test.bend as TS

def ld(+r: Nat, +c: Nat, +dir: String, +name: String) -> IO(TA.Mat<r, c>):
  T.Mat.load(r, c, dir ++ "/" ++ name ++ ".bin", 0)

def nat_list(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(h) <> nat_list(t)

# n ids or labels, as little-endian U32
def ld_ids(+n: Nat, +dir: String, +name: String) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(dir ++ "/" ++ name ++ ".bin", 0, (U32.from_nat(n) * 4 : U32))
    return nat_list(TS.u32s(bs))

def bad(u: Unit) -> IO(Unit):
  IO.print("bad case")

# ---- the layer protocol: forward on x, backward from dy into zero gradients, every result printed under its name ----

def show3(+n: Nat, +c: Nat, -P: Type, pp: P -> IO(Unit), r: T.Bwd<P, TA.Mat<n, c>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        T.Mat.print(n, c, "dx", dx)
        pp(g)

def show2(+n: Nat, +c: Nat, +o: Nat, -P: Type, -S: Type, lb: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, lz: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, pp: P -> IO(Unit), dy: TA.Mat<n, o>, r: T.Fwd<P, TA.Mat<n, o>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(n, o, "y", y)
        show3(n, c, P, pp, T.Layer.bwd(P, TA.Mat<n, c>, TA.Mat<n, o>, S, lb, p, s, dy, T.Layer.zeros(P, TA.Mat<n, c>, TA.Mat<n, o>, S, lz)))

# the layer comes three times (built afresh each time, its sizes known only at run time): forward, backward, zeros
def check(+n: Nat, +c: Nat, +o: Nat, -P: Type, -S: Type, lf: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, lb: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, lz: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, o>, S>, pp: P -> IO(Unit), p: P, x: TA.Mat<n, c>, dy: TA.Mat<n, o>) -> IO(Unit):
  show2(n, c, o, P, S, lb, lz, pp, dy, T.Layer.fwd(P, TA.Mat<n, c>, TA.Mat<n, o>, S, lf, p, x))

def pp_lin(+i: Nat, +o: Nat, +tag: String, p: T.Lin<i, o>) -> IO(Unit):
  match p:
    case T.Lin{w, b}:
      do IO<Unit>:
        T.Mat.print(o, i, tag ++ ".w", w)
        T.Mat.print(1n, o, tag ++ ".b", b)

def pp_ln(+d: Nat, +tag: String, p: T.LN<d>) -> IO(Unit):
  match p:
    case T.LN{g, b}:
      do IO<Unit>:
        T.Mat.print(1n, d, tag ++ ".g", g)
        T.Mat.print(1n, d, tag ++ ".b", b)

# ---- linear layers (op 0 Linear, 1 LinearFirst, 2 LinearNB): n i o par ----

def t_linear2(first: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>) -> IO(Unit):
  match first:
    case False{}:
      check(n, i, o, T.Lin<i, o>, TA.Mat<n, i>, T.Linear(n, i, o, par), T.Linear(n, i, o, par), T.Linear(n, i, o, par), (m => pp_lin(i, o, "g", m)), p, x, dy)
    case True{}:
      check(n, i, o, T.Lin<i, o>, TA.Mat<n, i>, T.LinearFirst(n, i, o, par), T.LinearFirst(n, i, o, par), T.LinearFirst(n, i, o, par), (m => pp_lin(i, o, "g", m)), p, x, dy)


def t_linear2f(first: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>) -> IO(Unit):
  match first:
    case False{}:
      check(n, i, o, T.Lin<i, o>, TA.Mat<n, i>, F.LinearF(n, i, o, par), F.LinearF(n, i, o, par), F.LinearF(n, i, o, par), (m => pp_lin(i, o, "g", m)), p, x, dy)
    case True{}:
      check(n, i, o, T.Lin<i, o>, TA.Mat<n, i>, F.LinearFirstF(n, i, o, par), F.LinearFirstF(n, i, o, par), F.LinearFirstF(n, i, o, par), (m => pp_lin(i, o, "g", m)), p, x, dy)

def t_linear3(fast: Bool, +first: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, p: T.Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>) -> IO(Unit):
  match fast:
    case False{}:
      t_linear2(first, n, i, o, par, p, x, dy)
    case True{}:
      t_linear2f(first, n, i, o, par, p, x, dy)

def t_linear(+fast: Bool, +first: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, i> <- ld(n, i, dir, "x")
    dy : TA.Mat<n, o> <- ld(n, o, dir, "dy")
    w : TA.Mat<o, i> <- ld(o, i, dir, "w")
    b : TA.Mat<1n, o> <- ld(1n, o, dir, "b")
    t_linear3(fast, first, n, i, o, par, T.Lin{w, b}, x, dy)

def t_linear_nb2(fast: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, w: TA.Mat<o, i>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>) -> IO(Unit):
  match fast:
    case False{}:
      check(n, i, o, TA.Mat<o, i>, TA.Mat<n, i>, T.LinearNB(n, i, o, par), T.LinearNB(n, i, o, par), T.LinearNB(n, i, o, par), (m => T.Mat.print(o, i, "g.w", m)), w, x, dy)
    case True{}:
      check(n, i, o, TA.Mat<o, i>, TA.Mat<n, i>, F.LinearNBF(n, i, o, par), F.LinearNBF(n, i, o, par), F.LinearNBF(n, i, o, par), (m => T.Mat.print(o, i, "g.w", m)), w, x, dy)

def t_linear_nb(+fast: Bool, +n: Nat, +i: Nat, +o: Nat, +par: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, i> <- ld(n, i, dir, "x")
    dy : TA.Mat<n, o> <- ld(n, o, dir, "dy")
    w : TA.Mat<o, i> <- ld(o, i, dir, "w")
    t_linear_nb2(fast, n, i, o, par, w, x, dy)

# ---- activations (ops 3..9: ReLU, Sigmoid, Tanh, SiLU, GELU, Softmax, LogSoftmax): n c ----

def t_act2(k: Nat, +n: Nat, +c: Nat, x: TA.Mat<n, c>, dy: TA.Mat<n, c>) -> IO(Unit):
  match k:
    case 0n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.ReLU(n, c), T.ReLU(n, c), T.ReLU(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 1n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.Sigmoid(n, c), T.Sigmoid(n, c), T.Sigmoid(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 2n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.Tanh(n, c), T.Tanh(n, c), T.Tanh(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 3n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.SiLU(n, c), T.SiLU(n, c), T.SiLU(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 4n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.GELU(n, c), T.GELU(n, c), T.GELU(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 7n:
      check(n, c, c, Unit, TA.Mat<n, c>, F.SiLU(n, c), F.SiLU(n, c), F.SiLU(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 5n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.Softmax(n, c), T.Softmax(n, c), T.Softmax(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case 6n:
      check(n, c, c, Unit, TA.Mat<n, c>, T.LogSoftmax(n, c), T.LogSoftmax(n, c), T.LogSoftmax(n, c), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)
    case _:
      bad(Unit{})

def t_act(+k: Nat, +n: Nat, +c: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, c> <- ld(n, c, dir, "x")
    dy : TA.Mat<n, c> <- ld(n, c, dir, "dy")
    t_act2(k, n, c, x, dy)

# ---- norms (op 10 RMSNorm, 11 LayerNorm): n d eps ----

def t_rms(+n: Nat, +d: Nat, +eps: F32, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, d> <- ld(n, d, dir, "x")
    dy : TA.Mat<n, d> <- ld(n, d, dir, "dy")
    g : TA.Mat<1n, d> <- ld(1n, d, dir, "g")
    check(n, d, d, TA.Mat<1n, d>, TA.Mat<n, d>, T.RMSNorm(n, d, eps), T.RMSNorm(n, d, eps), T.RMSNorm(n, d, eps), (m => T.Mat.print(1n, d, "g", m)), g, x, dy)

def t_ln(+n: Nat, +d: Nat, +eps: F32, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, d> <- ld(n, d, dir, "x")
    dy : TA.Mat<n, d> <- ld(n, d, dir, "dy")
    g : TA.Mat<1n, d> <- ld(1n, d, dir, "g")
    b : TA.Mat<1n, d> <- ld(1n, d, dir, "b")
    check(n, d, d, T.LN<d>, TA.Mat<n, d>, T.LayerNorm(n, d, eps), T.LayerNorm(n, d, eps), T.LayerNorm(n, d, eps), (m => pp_ln(d, "g", m)), T.LN{g, b}, x, dy)

# ---- RoPE (op 12): b t h half base; rows are b sequences of t positions, each row h heads of 2*half ----

def t_rope2(+n: Nat, +w: Nat, +t: Nat, +h: Nat, +half: Nat, +base: F32, -e: {w == Nat.mul(h, Nat.add(half, half)) : Nat}, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, w> <- ld(n, w, dir, "x")
    dy : TA.Mat<n, w> <- ld(n, w, dir, "dy")
    check(n, w, w, Unit, Unit, T.RoPE(n, w, t, h, half, base, e), T.RoPE(n, w, t, h, half, base, e), T.RoPE(n, w, t, h, half, base, e), u => IO.pure(Unit, Unit{}), Unit{}, x, dy)

def AttnX(n: Nat, w: Nat, kvw: Nat) -> Type:
  T.Two<TA.Mat<n, w>, T.Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>

def attn3(+n: Nat, +w: Nat, +kvw: Nat, r: T.Bwd<Unit, AttnX(n, w, kvw)>) -> IO(Unit):
  match r:
    case T.Bwd{p, T.Two{dq, T.Two{dk, dv}}, g}:
      do IO<Unit>:
        T.Mat.print(n, w, "dq", dq)
        T.Mat.print(n, kvw, "dk", dk)
        T.Mat.print(n, kvw, "dv", dv)

def attn2(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, -e1: {n == Nat.mul(b, t) : Nat}, -e2: {w == Nat.mul(Nat.mul(kv, g), hd) : Nat}, -e3: {kvw == Nat.mul(kv, hd) : Nat}, dy: TA.Mat<n, w>, r: T.Fwd<Unit, TA.Mat<n, w>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(n, w, "y", y)
        attn3(n, w, kvw, T.Layer.bwd(Unit, AttnX(n, w, kvw), TA.Mat<n, w>, T.AttnS, T.Attention(n, w, kvw, b, t, kv, g, hd, sc, e1, e2, e3), p, s, dy, Unit{}))

def t_attn3(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, -e1: {n == Nat.mul(b, t) : Nat}, -e2: {w == Nat.mul(Nat.mul(kv, g), hd) : Nat}, -e3: {kvw == Nat.mul(kv, hd) : Nat}, +dir: String) -> IO(Unit):
  do IO<Unit>:
    q : TA.Mat<n, w> <- ld(n, w, dir, "q")
    k : TA.Mat<n, kvw> <- ld(n, kvw, dir, "k")
    v : TA.Mat<n, kvw> <- ld(n, kvw, dir, "v")
    dy : TA.Mat<n, w> <- ld(n, w, dir, "dy")
    attn2(n, w, kvw, b, t, kv, g, hd, sc, e1, e2, e3, dy, T.Layer.fwd(Unit, AttnX(n, w, kvw), TA.Mat<n, w>, T.AttnS, T.Attention(n, w, kvw, b, t, kv, g, hd, sc, e1, e2, e3), Unit{}, T.Two{q, T.Two{k, v}}))

def emb3(+v: Nat, +d: Nat, r: T.Bwd<TA.Mat<v, d>, List<&2, Nat>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      T.Mat.print(v, d, "g", g)

def emb2(+n: Nat, +v: Nat, +d: Nat, dy: TA.Mat<n, d>, r: T.Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(n, d, "y", y)
        emb3(v, d, T.Layer.bwd(TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>, T.Embedding(n, v, d), p, s, dy, TA.Mat.zeros(v, d)))

def t_emb(+n: Nat, +v: Nat, +d: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    e : TA.Mat<v, d> <- ld(v, d, dir, "e")
    dy : TA.Mat<n, d> <- ld(n, d, dir, "dy")
    ids : List<&2, Nat> <- ld_ids(n, dir, "ids")
    emb2(n, v, d, dy, T.Layer.fwd(TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>, T.Embedding(n, v, d), e, ids))

# ---- losses (op 15 MSE, 16 NLL, 17 cross-entropy): n c; the loss and its gradient ----

def loss_show(+n: Nat, +c: Nat, r: T.LossOut<TA.Mat<n, c>>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        T.Mat.print(n, c, "dx", dy)

def t_mse(+n: Nat, +c: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    y : TA.Mat<n, c> <- ld(n, c, dir, "x")
    t : TA.Mat<n, c> <- ld(n, c, dir, "t")
    loss_show(n, c, T.MSE(n, c, y, t))

def t_nll2(ce: Bool, +n: Nat, +c: Nat, x: TA.Mat<n, c>, ls: List<&2, Nat>) -> IO(Unit):
  match ce:
    case False{}:
      loss_show(n, c, T.NLL(n, c, x, ls))
    case True{}:
      loss_show(n, c, T.CrossEntropy(n, c, x, ls))

# ---- dropout (op 18): n c p seed step ----


def t_nll(+ce: Bool, +n: Nat, +c: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, c> <- ld(n, c, dir, "x")
    ls : List<&2, Nat> <- ld_ids(n, dir, "labels")
    t_nll2(ce, n, c, x, ls)

def t_drop(+n: Nat, +c: Nat, +p: F32, +seed: U32, +step: U32, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TA.Mat<n, c> <- ld(n, c, dir, "x")
    dy : TA.Mat<n, c> <- ld(n, c, dir, "dy")
    check(n, c, c, T.DropP, TA.Mat<n, c>, T.Dropout(n, c, p, seed), T.Dropout(n, c, p, seed), T.Dropout(n, c, p, seed), u => IO.pure(Unit, Unit{}), T.DropP{step}, x, dy)

# ---- the product of two inputs (op 19, Mul2): n c ----

def mul3(+n: Nat, +c: Nat, r: T.Bwd<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>) -> IO(Unit):
  match r:
    case T.Bwd{p, T.Two{da, db}, g}:
      do IO<Unit>:
        T.Mat.print(n, c, "dx", da)
        T.Mat.print(n, c, "dy2", db)

def mul2(+n: Nat, +c: Nat, lb: T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>, dy: TA.Mat<n, c>, r: T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      do IO<Unit>:
        T.Mat.print(n, c, "y", y)
        mul3(n, c, T.Layer.bwd(Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, lb, p, s, dy, Unit{}))

def mul_go(-n: Nat, -c: Nat, lf: T.Layer<Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>, x: T.Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> T.Fwd<Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  T.Layer.fwd(Unit, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, lf, Unit{}, x)

def t_mul2(fast: Bool, +n: Nat, +c: Nat, a: TA.Mat<n, c>, b: TA.Mat<n, c>, dy: TA.Mat<n, c>) -> IO(Unit):
  match fast:
    case False{}:
      mul2(n, c, T.Mul2(n, c), dy, mul_go(n, c, T.Mul2(n, c), T.Two{a, b}))
    case True{}:
      mul2(n, c, F.Mul2(n, c), dy, mul_go(n, c, F.Mul2(n, c), T.Two{a, b}))

def t_mul(+fast: Bool, +n: Nat, +c: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    a : TA.Mat<n, c> <- ld(n, c, dir, "x")
    b : TA.Mat<n, c> <- ld(n, c, dir, "x2")
    dy : TA.Mat<n, c> <- ld(n, c, dir, "dy")
    t_mul2(fast, n, c, a, b, dy)

# ---- optimizer steps (op 20): kind (0 SGD, 1 momentum, 2 Adam, 3 AdamW), steps k, r c, lr, mu or b1, b2, eps, wd.
# w.bin holds the initial r x c matrix, g.bin one gradient a step; the matrix is printed after every step, the
# moment buffers after the last. 1 - b1^t and 1 - b2^t are carried as Train.step carries them; the corpus stores the
# betas only, so the complements are 1 - b in F32 here (BendTorch's configs give them as constants). ----

def cfg_of(kind: Nat, +lr: F32, +a: F32, +b2: F32, +eps: F32, +wd: F32) -> T.OptCfg:
  match kind:
    case 0n:
      T.CSGD{lr}
    case 1n:
      T.CMomentum{lr, a}
    case 2n:
      T.CAdam{lr, a, b2, eps, wd, (1.0 - a : F32), (1.0 - b2 : F32)}
    case _:
      T.CAdamW{lr, a, b2, eps, wd, (1.0 - a : F32), (1.0 - b2 : F32)}

def st_new(+r: Nat, +c: Nat, +cfg: T.OptCfg) -> T.OptState<TA.Mat<r, c>>:
  match cfg:
    case T.CSGD{lr}:
      T.NoMom{}
    case T.CMomentum{lr, mu}:
      T.OneMom{TA.Mat.zeros(r, c)}
    case _:
      T.TwoMom{TA.Mat.zeros(r, c), TA.Mat.zeros(r, c)}

def st_show(+r: Nat, +c: Nat, st: T.OptState<TA.Mat<r, c>>) -> IO(Unit):
  match st:
    case T.NoMom{}:
      IO.pure(Unit, Unit{})
    case T.OneMom{m}:
      T.Mat.print(r, c, "m", m)
    case T.TwoMom{m, v}:
      do IO<Unit>:
        T.Mat.print(r, c, "m", m)
        T.Mat.print(r, c, "v", v)
    case T.Packed{t}:
      # packed BF16 moments (T.CAdamWBF16) are not in the frozen corpus; st_new never makes them
      IO.pure(Unit, Unit{})

def mm2_then(-r: Nat, -c: Nat, ww: TA.MM2<r, c>, f: TA.Mat<r, c> -> TA.Mat<r, c> -> IO(Unit)) -> IO(Unit):
  match ww:
    case TA.MM2{a, b}:
      f(a, b)

# the matrix after a step is printed; k more steps follow (the loop goes on inside a closure: no mutual recursion)
def opt_go(k: Nat, +j: U32, +r: Nat, +c: Nat, +cfg: T.OptCfg, +p1: F32, +p2: F32, +dir: String, q: T.MatStep<r, c>) -> IO(Unit):
  match k q:
    case 0n T.MatStep{w, g, st}:
      do IO<Unit>:
        T.Mat.print(r, c, "w", w)
        st_show(r, c, st)
    case 1n+kk T.MatStep{w, g, st}:
      mm2_then(r, c, TA.Mat.clone(r, c, w), w1 => w2 => do IO<Unit>:
        T.Mat.print(r, c, "w", w1)
        g2 : TA.Mat<r, c> <- T.Mat.load(r, c, dir ++ "/g.bin", ((j * U32.from_nat(Nat.mul(r, c)) : U32) * 4 : U32))
        opt_go(kk, (j + 1 : U32), r, c, cfg, (p1 + (T.OptCfg.c1(cfg) * (1.0 - p1 : F32) : F32) : F32), (p2 + (T.OptCfg.c2(cfg) * (1.0 - p2 : F32) : F32) : F32), dir, T.Opt.mat(r, c, T.Opt.at(cfg, (p1 + (T.OptCfg.c1(cfg) * (1.0 - p1 : F32) : F32) : F32), (p2 + (T.OptCfg.c2(cfg) * (1.0 - p2 : F32) : F32) : F32)), w2, g2, st)))

def opt_first(k: Nat, +r: Nat, +c: Nat, +cfg: T.OptCfg, +dir: String, w: TA.Mat<r, c>, g: TA.Mat<r, c>) -> IO(Unit):
  match k:
    case 0n:
      T.Mat.print(r, c, "w", w)
    case 1n+kk:
      opt_go(kk, 1, r, c, cfg, T.OptCfg.c1(cfg), T.OptCfg.c2(cfg), dir, T.Opt.mat(r, c, T.Opt.at(cfg, T.OptCfg.c1(cfg), T.OptCfg.c2(cfg)), w, g, st_new(r, c, cfg)))

def t_opt(+k: Nat, +r: Nat, +c: Nat, +cfg: T.OptCfg, +dir: String) -> IO(Unit):
  do IO<Unit>:
    w : TA.Mat<r, c> <- ld(r, c, dir, "w")
    g : TA.Mat<r, c> <- T.Mat.load(r, c, dir ++ "/g.bin", 0)
    opt_first(k, r, c, cfg, dir, w, g)

# ---- dispatch ----

def n_(+x: U32) -> Nat:
  U32.to_nat(x)

def run(op: Nat, ws: List<&2, U32>, +dir: String) -> IO(Unit):
  match op ws:
    case 0n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear(False{}, False{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 1n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear(False{}, True{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 2n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear_nb(False{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 30n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear(True{}, False{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 31n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear(True{}, True{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 32n Con{+n, Con{+i, Con{+o, Con{+par, r}}}}:
      t_linear_nb(True{}, n_(n), n_(i), n_(o), n_(par), dir)
    case 33n Con{+n, Con{+c, r}}:
      t_act(7n, n_(n), n_(c), dir)
    case 34n Con{+n, Con{+c, r}}:
      t_mul(True{}, n_(n), n_(c), dir)
    case 10n Con{+n, Con{+d, Con{+eps, r}}}:
      t_rms(n_(n), n_(d), T.bitcast(eps), dir)
    case 11n Con{+n, Con{+d, Con{+eps, r}}}:
      t_ln(n_(n), n_(d), T.bitcast(eps), dir)
    case 12n Con{+b, Con{+t, Con{+h, Con{+half, Con{+base, r}}}}}:
      t_rope2(Nat.mul(n_(b), n_(t)), Nat.mul(n_(h), Nat.add(n_(half), n_(half))), n_(t), n_(h), n_(half), T.bitcast(base), {==}, dir)
    case 13n Con{+b, Con{+t, Con{+kv, Con{+g, Con{+hd, Con{+sc, r}}}}}}:
      t_attn3(Nat.mul(n_(b), n_(t)), Nat.mul(Nat.mul(n_(kv), n_(g)), n_(hd)), Nat.mul(n_(kv), n_(hd)), n_(b), n_(t), n_(kv), n_(g), n_(hd), T.bitcast(sc), {==}, {==}, {==}, dir)
    case 14n Con{+n, Con{+v, Con{+d, r}}}:
      t_emb(n_(n), n_(v), n_(d), dir)
    case 15n Con{+n, Con{+c, r}}:
      t_mse(n_(n), n_(c), dir)
    case 16n Con{+n, Con{+c, r}}:
      t_nll(False{}, n_(n), n_(c), dir)
    case 17n Con{+n, Con{+c, r}}:
      t_nll(True{}, n_(n), n_(c), dir)
    case 18n Con{+n, Con{+c, Con{+p, Con{+seed, Con{+step, r}}}}}:
      t_drop(n_(n), n_(c), T.bitcast(p), seed, step, dir)
    case 19n Con{+n, Con{+c, r}}:
      t_mul(False{}, n_(n), n_(c), dir)
    case 20n Con{+kind, Con{+k, Con{+rr, Con{+c, Con{+lr, Con{+a, Con{+b2, Con{+eps, Con{+wd, r}}}}}}}}}:
      t_opt(n_(k), n_(rr), n_(c), cfg_of(n_(kind), T.bitcast(lr), T.bitcast(a), T.bitcast(b2), T.bitcast(eps), T.bitcast(wd)), dir)
    case 3n Con{+n, Con{+c, r}}:
      t_act(0n, n_(n), n_(c), dir)
    case 4n Con{+n, Con{+c, r}}:
      t_act(1n, n_(n), n_(c), dir)
    case 5n Con{+n, Con{+c, r}}:
      t_act(2n, n_(n), n_(c), dir)
    case 6n Con{+n, Con{+c, r}}:
      t_act(3n, n_(n), n_(c), dir)
    case 7n Con{+n, Con{+c, r}}:
      t_act(4n, n_(n), n_(c), dir)
    case 8n Con{+n, Con{+c, r}}:
      t_act(5n, n_(n), n_(c), dir)
    case 9n Con{+n, Con{+c, r}}:
      t_act(6n, n_(n), n_(c), dir)
    case _ _:
      bad(Unit{})

def start2(big: Bool, +op: U32, r: List<&2, U32>, +dir: String) -> IO(Unit):
  match big:
    case True{}:
      TS.start(Con{(op - 100 : U32), r}, dir)
    case False{}:
      run(n_(op), r, dir)


def start(ws: List<&2, U32>, +dir: String) -> IO(Unit):
  match ws:
    case Con{+op, r}:
      start2(U32.is_le(100, op), op, r, dir)
    case Nil{}:
      bad(Unit{})

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+dir, Nil{}}}:
      do IO<Unit>:
        bs : List<&2, U32> <- T.read_at(dir ++ "/case.bin", 0, 4096)
        start(TS.u32s(bs), dir)
    case _:
      IO.print("usage: corpus_test <dir>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
