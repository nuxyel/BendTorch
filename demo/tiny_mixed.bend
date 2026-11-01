# demo/tiny_llama.bend's model with BF16 storage (plan M11), and its int8 inference, against reference/tiny_mixed_ref.py.
#
#   bend demo/tiny_mixed.bend -o .scratch/tiny_mixed
#   .scratch/tiny_mixed train <steps>   # AdamW steps with BF16 storage: the loss before each update, the final weights
#   .scratch/tiny_mixed int8            # the logits on the first batch with F32 weights, then with int8 weights
#
# Mixed precision: every layer's output is rounded to BF16 (D.CastOut: the embedding, each RMSNorm, each projection,
# RoPE, attention, SiLU, the gate product and each residual sum), and every parameter is seen by its layer rounded to
# BF16 (D.CastP), with its gradient rounded to BF16 before AdamW, which updates the F32 master weights. The arithmetic
# inside every layer, the loss and the optimizer are F32. The parameters, their order and the batches are the tiny
# Llama's (the same initial weights, data/tiny/init.bin).
#
# int8: the eight projections of each block and the head quantized by rows (D.Q8), then dequantized into F32 and run
# through the F32 model; the embedding and the norms stay F32.

import Base
import ../torch/main.bend as T
import ../torch/fast.bend as F
import ../torch/dtype.bend as D
import ./tiny_llama.bend as TL
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# a projection without bias: weights seen in BF16, output stored in BF16
def LinBF(~b: Nat, ~i: Nat, ~o: Nat) -> T.Layer<TL.M(o, i), TL.A(b, i), TL.A(b, o), TL.A(b, i)>:
  D.CastOut(~TL.M(o, i), ~TL.A(b, i), ~TL.A(b, i), ~TL.R(b), ~o, ~D.CastP(~TL.A(b, i), ~TL.A(b, o), ~TL.A(b, i), ~o, ~i, ~F.LinearNBF(TL.R(b), i, o, 0n)))

def NormBF(~b: Nat) -> T.Layer<TL.M(1n, 32n), TL.A(b, 32n), TL.A(b, 32n), TL.A(b, 32n)>:
  D.CastOut(~TL.M(1n, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.R(b), ~32n, ~D.CastP(~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~1n, ~32n, ~T.RMSNorm(TL.R(b), 32n, 0.00001)))

def KVL(~b: Nat) -> T.Layer<TL.KVP, TL.A(b, 32n), TL.KVY(b), TL.KVS(b)>:
  T.Fork(~TL.M(16n, 32n), ~TL.M(16n, 32n), ~TL.A(b, 32n), ~TL.A(b, 16n), ~TL.A(b, 16n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~(m => T.Mat.dup(TL.R(b), 32n, m)), ~(x => y => T.Mat.sum2(TL.R(b), 32n, x, y)), ~LinBF(~b, ~32n, ~16n), ~LinBF(~b, ~32n, ~16n))

def QKV(~b: Nat) -> T.Layer<TL.QKVP, TL.A(b, 32n), TL.AX(b), TL.QKVS(b)>:
  T.Fork(~TL.M(32n, 32n), ~TL.KVP, ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.KVY(b), ~TL.A(b, 32n), ~TL.KVS(b), ~(m => T.Mat.dup(TL.R(b), 32n, m)), ~(x => y => T.Mat.sum2(TL.R(b), 32n, x, y)), ~LinBF(~b, ~32n, ~32n), ~KVL(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, TL.KVY(b), TL.KVY(b), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~TL.A(b, 16n), ~TL.A(b, 16n), ~TL.A(b, 16n), ~TL.A(b, 16n), ~Unit, ~Unit, ~D.CastOut(~Unit, ~TL.A(b, 16n), ~Unit, ~TL.R(b), ~16n, ~T.RoPE(TL.R(b), 16n, 16n, 2n, 4n, 10000.0, {==})), ~T.Id(~TL.A(b, 16n)))

def Ropes(~b: Nat) -> T.Layer<TL.RU, TL.AX(b), TL.AX(b), TL.RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~TL.A(b, 32n), ~TL.KVY(b), ~TL.A(b, 32n), ~TL.KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~D.CastOut(~Unit, ~TL.A(b, 32n), ~Unit, ~TL.R(b), ~32n, ~T.RoPE(TL.R(b), 32n, 16n, 4n, 4n, 10000.0, {==})), ~RopeK(~b))

def Core(~b: Nat) -> T.Layer<TL.CoreP, TL.AX(b), TL.A(b, 32n), TL.CoreS(b)>:
  T.Seq(~Unit, ~TL.M(32n, 32n), ~TL.AX(b), ~TL.A(b, 32n), ~TL.A(b, 32n), ~T.AttnS, ~TL.A(b, 32n), ~D.CastOut(~Unit, ~TL.AX(b), ~T.AttnS, ~TL.R(b), ~32n, ~T.Attention(TL.R(b), 32n, 16n, b, 16n, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==})), ~LinBF(~b, ~32n, ~32n))

def RopeCore(~b: Nat) -> T.Layer<TL.RCP, TL.AX(b), TL.A(b, 32n), TL.RCS(b)>:
  T.Seq(~TL.RU, ~TL.CoreP, ~TL.AX(b), ~TL.AX(b), ~TL.A(b, 32n), ~TL.RU, ~TL.CoreS(b), ~Ropes(~b), ~Core(~b))

def Att(~b: Nat) -> T.Layer<TL.AttP, TL.A(b, 32n), TL.A(b, 32n), TL.AttS(b)>:
  T.Seq(~TL.QKVP, ~TL.RCP, ~TL.A(b, 32n), ~TL.AX(b), ~TL.A(b, 32n), ~TL.QKVS(b), ~TL.RCS(b), ~QKV(~b), ~RopeCore(~b))

def NAtt(~b: Nat) -> T.Layer<TL.NAttP, TL.A(b, 32n), TL.A(b, 32n), TL.NAttS(b)>:
  T.Seq(~TL.M(1n, 32n), ~TL.AttP, ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.AttS(b), ~NormBF(~b), ~Att(~b))

def Gate(~b: Nat) -> T.Layer<T.Two<TL.M(64n, 32n), Unit>, TL.A(b, 32n), TL.A(b, 64n), TL.GateS(b)>:
  T.Seq(~TL.M(64n, 32n), ~Unit, ~TL.A(b, 32n), ~TL.A(b, 64n), ~TL.A(b, 64n), ~TL.A(b, 32n), ~TL.A(b, 64n), ~LinBF(~b, ~32n, ~64n), ~D.CastOut(~Unit, ~TL.A(b, 64n), ~TL.A(b, 64n), ~TL.R(b), ~64n, ~T.SiLU(TL.R(b), 64n)))

def GU(~b: Nat) -> T.Layer<TL.GUP, TL.A(b, 32n), TL.GUY(b), TL.GUS(b)>:
  T.Fork(~T.Two<TL.M(64n, 32n), Unit>, ~TL.M(64n, 32n), ~TL.A(b, 32n), ~TL.A(b, 64n), ~TL.A(b, 64n), ~TL.GateS(b), ~TL.A(b, 32n), ~(m => T.Mat.dup(TL.R(b), 32n, m)), ~(x => y => T.Mat.sum2(TL.R(b), 32n, x, y)), ~Gate(~b), ~LinBF(~b, ~32n, ~64n))

def GLU(~b: Nat) -> T.Layer<T.Two<TL.GUP, Unit>, TL.A(b, 32n), TL.A(b, 64n), TL.GLUS(b)>:
  T.Seq(~TL.GUP, ~Unit, ~TL.A(b, 32n), ~TL.GUY(b), ~TL.A(b, 64n), ~TL.GUS(b), ~TL.GUY(b), ~GU(~b), ~D.CastOut(~Unit, ~TL.GUY(b), ~TL.GUY(b), ~TL.R(b), ~64n, ~T.Mul2(TL.R(b), 64n)))

def Mlp(~b: Nat) -> T.Layer<TL.MlpP, TL.A(b, 32n), TL.A(b, 32n), TL.MlpS(b)>:
  T.Seq(~T.Two<TL.GUP, Unit>, ~TL.M(32n, 64n), ~TL.A(b, 32n), ~TL.A(b, 64n), ~TL.A(b, 32n), ~TL.GLUS(b), ~TL.A(b, 64n), ~GLU(~b), ~LinBF(~b, ~64n, ~32n))

def NMlp(~b: Nat) -> T.Layer<TL.NMlpP, TL.A(b, 32n), TL.A(b, 32n), TL.NMlpS(b)>:
  T.Seq(~TL.M(1n, 32n), ~TL.MlpP, ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.MlpS(b), ~NormBF(~b), ~Mlp(~b))

# each residual sum is stored in BF16 too
def Block(~b: Nat) -> T.Layer<TL.BlockP, TL.A(b, 32n), TL.A(b, 32n), TL.BlockS(b)>:
  T.Seq(~T.ResidualType(TL.NAttP), ~T.ResidualType(TL.NMlpP), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~T.ResidualType(TL.NAttS(b)), ~T.ResidualType(TL.NMlpS(b)), ~D.CastOut(~T.ResidualType(TL.NAttP), ~TL.A(b, 32n), ~T.ResidualType(TL.NAttS(b)), ~TL.R(b), ~32n, ~T.Residual(~TL.R(b), ~32n, ~TL.NAttP, ~TL.NAttS(b), ~NAtt(~b))), ~D.CastOut(~T.ResidualType(TL.NMlpP), ~TL.A(b, 32n), ~T.ResidualType(TL.NMlpS(b)), ~TL.R(b), ~32n, ~T.Residual(~TL.R(b), ~32n, ~TL.NMlpP, ~TL.NMlpS(b), ~NMlp(~b))))

def Head(~b: Nat) -> T.Layer<TL.HeadP, TL.A(b, 32n), TL.A(b, 65n), TL.HeadS(b)>:
  T.Seq(~TL.M(1n, 32n), ~TL.M(65n, 32n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 65n), ~TL.A(b, 32n), ~TL.A(b, 32n), ~NormBF(~b), ~LinBF(~b, ~32n, ~65n))

def Top(~b: Nat) -> T.Layer<TL.TopP, TL.A(b, 32n), TL.A(b, 65n), TL.TopS(b)>:
  T.Seq(~TL.BlockP, ~TL.HeadP, ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 65n), ~TL.BlockS(b), ~TL.HeadS(b), ~Block(~b), ~Head(~b))

def Body(~b: Nat) -> T.Layer<TL.BodyP, TL.A(b, 32n), TL.A(b, 65n), TL.BodyS(b)>:
  T.Seq(~TL.BlockP, ~TL.TopP, ~TL.A(b, 32n), ~TL.A(b, 32n), ~TL.A(b, 65n), ~TL.BlockS(b), ~TL.TopS(b), ~Block(~b), ~Top(~b))

def Model(~b: Nat) -> T.Layer<TL.P, TL.X, TL.Y(b), TL.S(b)>:
  T.Seq(~TL.M(65n, 32n), ~TL.BodyP, ~TL.X, ~TL.A(b, 32n), ~TL.Y(b), ~List<&2, Nat>, ~TL.BodyS(b), ~D.CastOut(~TL.M(65n, 32n), ~TL.X, ~List<&2, Nat>, ~TL.R(b), ~32n, ~D.CastP(~TL.X, ~TL.A(b, 32n), ~List<&2, Nat>, ~65n, ~32n, ~T.Embedding(TL.R(b), 65n, 32n))), ~Body(~b))

# ---- training: tiny_llama.bend's loop with this model ----

def step_with(+cfg: T.OptCfg, +ids: List<&2, Nat>, tr: T.Train<TL.P>) -> T.Train<TL.P>:
  T.Train.step(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~List<&2, Nat>, ~Model(~2n), ~TL.ce, cfg, tr, TL.xs_of(ids), TL.ys_of(ids))

def tloop(k: Nat, +step: U32, +cfg: T.OptCfg, tr: T.Train<TL.P>) -> IO(T.Train<TL.P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<TL.P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<TL.P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- TL.load_batch(step)
        tloop(q, (step + 1 : U32), cfg, step_with(cfg, ids, T.Train{p, g, st, p1, p2, l}))

def train(+k: Nat) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(TL.cfg())
    p : TL.P <- TL.load_params()
    t0 : Nat <- IO.now()
    tr : T.Train<TL.P> <- tloop(k, 0, TL.cfg(), T.Train.new(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~Model(~2n), TL.cfg(), p))
    t1 : Nat <- IO.now()
    IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))
    TL.finish(tr)

# ---- int8 inference ----

def dq(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  D.Q8.dequantize(r, c, D.Q8.quantize(r, c, m))

def q8_block(b: TL.BlockP) -> TL.BlockP:
  match b:
    case T.Two{T.Two{T.Two{u1, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{u2, T.Two{u3, u4}}, T.Two{u5, wo}}}}}, u6}, T.Two{T.Two{u7, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, u8}, wu}, u9}, wd}}}, u10}}:
      TL.mk_block(ln1, dq(32n, 32n, wq), dq(16n, 32n, wk), dq(16n, 32n, wv), dq(32n, 32n, wo), ln2, dq(64n, 32n, wg), dq(64n, 32n, wu), dq(32n, 64n, wd))

def q8_params(p: TL.P) -> TL.P:
  match p:
    case T.Two{emb, T.Two{b0, T.Two{b1, T.Two{lnf, head}}}}:
      T.Two{emb, T.Two{q8_block(b0), T.Two{q8_block(b1), T.Two{lnf, dq(65n, 32n, head)}}}}

def logits_of(+tag: String, r: T.Fwd<TL.P, TL.Y(2n), TL.S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      T.Mat.print(32n, 65n, tag, y)

def int8() -> IO(Unit):
  do IO<Unit>:
    ids : List<&2, Nat> <- TL.load_batch(0)
    ids2 : List<&2, Nat> <- TL.load_batch(0)
    p : TL.P <- TL.load_params()
    q : TL.P <- TL.load_params()
    logits_of("logits_f32", T.Layer.fwd(TL.P, TL.X, TL.Y(2n), TL.S(2n), TL.Model(~2n), p, TL.xs_of(ids)))
    logits_of("logits_int8", T.Layer.fwd(TL.P, TL.X, TL.Y(2n), TL.S(2n), TL.Model(~2n), q8_params(q), TL.xs_of(ids2)))

# ---- the command line ----

def usage() -> IO(Unit):
  IO.die(Unit, 2, "usage: tiny_mixed train <steps> | int8")

def train_m(t: Bool, mk: Maybe<&2, Nat>) -> IO(Unit):
  match t mk:
    case True{} Some{k}:
      train(k)
    case _ _:
      usage()

def int8_m(t: Bool) -> IO(Unit):
  match t:
    case True{}:
      int8()
    case False{}:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Nil{}}}:
      int8_m(String.eq(mode, "int8"))
    case Con{prog, Con{+mode, Con{k, Nil{}}}}:
      train_m(String.eq(mode, "train"), Nat.read(k))
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
