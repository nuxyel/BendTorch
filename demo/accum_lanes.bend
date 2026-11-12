# Gradient accumulation (Train.accumulate and Train.flush, BendTorch #36) over the layers of torch/fast.bend and
# torch/gpu.bend. One source for the three lanes: the layers come from the file imported as F, and make check and make
# gpu build copies of this file whose F is torch/main.bend or torch/gpu.bend (the import line changed, as make gpu does
# for mnist, tiny_llama and smol_block). Five models, trained from the same weights with AdamW:
#   lin   LinearFirst(12, 16) -> RMSNorm -> LinearNB(16, 8), MSE
#   mlp   Residual(RMSNorm -> SwiGLU, 16 -> 24 -> 16), MSE
#   attn  Residual(RMSNorm -> attention: 2 query heads and 1 key/value head of 8, RoPE, sequences of 8), MSE
#   gn    GroupNorm (images of 4 pixels, 2 groups of 4 channels) -> SiLU -> Linear(8, 6), MSE
#   emb   Embedding(13 ids, 8) -> LinearNB(8, 13), cross-entropy
# each four ways, two optimizer steps each:
#   acc   k = 4 micro-batches a step: Train.accumulate with each loss divided by 4 (LossOut.div), Train.flush after the
#         fourth
#   big   Train.step on the four micro-batches at once (their rows in order)
#   one   k = 1: one Train.accumulate and a Train.flush a step, on micro-batches 0 and 1
#   step  Train.step on micro-batches 0 and 1
# Each run prints its state as Train.dump gives it, the gradient the record holds and its loss, every number as the
# decimal of its 32 bits. reference/accum_lanes.py compares a lane's output with main.bend's (in ulps), one with step
# (bit for bit) and acc with big (within the tolerance: the sums are added in another order).
#
#   accum_lanes          every run of every model, in the order above
#   accum_lanes <m> <r>  model m (0 = lin .. 4 = emb), run r (0 = acc .. 3 = step) alone

import Base
import ../torch/main.bend as T
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# ---- numbers from a hash of their index, in [-1, 1) (demo/gpu_layers_test.bend's), no specials ----

def val(+i: U32, +seed: U32) -> F32:
  +h = U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), 1000)
  ((U32.to_f32(h) / 500.0 : F32) - 1.0 : F32)

# a[i] = val(off + i, seed) for the k cells from i
def fill(k: Nat, +i: U32, +off: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n: a
    case 1n+q: fill(q, (i + 1 : U32), off, seed, Array.set(F32, a, i, val((off + i : U32), seed)))

def arr(+cnt: Nat, +off: U32, +seed: U32) -> Array<F32>:
  fill(cnt, 0, off, seed, Array.new(F32, TA.cap_depth(cnt), 0.0))

# rows j·r .. j·r + r - 1 of a stream of rows of c numbers: micro-batch j of r rows, or, with r four times as large,
# micro-batches 4j .. 4j + 3 in order
def rows(+r: Nat, +c: Nat, +j: U32, +seed: U32) -> M(r, c):
  TA.Mat{arr(Nat.mul(r, c), (j * U32.from_nat(Nat.mul(r, c)) : U32), seed)}

# k ids below m from position i of a stream
def idl(k: Nat, +i: U32, +m: U32, +seed: U32) -> List<&2, Nat>:
  match k:
    case 0n: Nil{}
    case 1n+q: U32.to_nat(U32.mod(((i * 2654435761 : U32) + (seed * 40503 : U32) : U32), m)) <> idl(q, (i + 1 : U32), m, seed)

def ids(+r: Nat, +j: U32, +m: U32, +seed: U32) -> List<&2, Nat>:
  idl(r, (j * U32.from_nat(r) : U32), m, seed)

# ---- the runs, for any model ----

def init_pa(-P: Type, r: T.PA<P>) -> P:
  match r:
    case T.PA{p, xs, off}: p

# the parameters from val, in the layer's dump order (Layer.load over its zeros)
def init(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +seed: U32) -> P:
  init_pa(P, T.Layer.load(P, X, Y, S, m, T.Layer.zeros(P, X, Y, S, m), arr(U32.to_nat(T.Layer.count(P, X, Y, S, m)), 0, seed), 0))

def new(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +seed: U32) -> T.Train<P>:
  T.Train.new(~P, ~X, ~Y, ~S, ~m, T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001}, init(~P, ~X, ~Y, ~S, ~m, seed))

def flush_if(~P: Type, ~fl: T.Train<P> -> T.Train<P>, last: Bool, tr: T.Train<P>) -> T.Train<P>:
  match last:
    case True{}: fl(tr)
    case False{}: tr

# tr, evaluated, handed on through an effect, so that the next call starts a new evaluation on the main thread: a
# bang that follows a parallel let in the same evaluation runs on the CPU threads (demo/gpu_layers_test.bend), and
# Train.new's zeros are made by parallel lets
def settle(~P: Type, tr: T.Train<P>) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, st, oc, p1, p2, +l}:
      IO.bind(Nat, T.Train<P>, IO.now(), t => IO.pure(T.Train<P>, T.Train{p, g, st, oc, p1, p2, l}))

# left micro-batches more, j the next: acc on its rows, then fl after every fourth
def micro(~P: Type, ~X: Type, ~G: Type, ~acc: T.Train<P> -> X -> G -> T.Train<P>, ~fl: T.Train<P> -> T.Train<P>, ~gx: U32 -> X, ~gt: U32 -> G, left: Nat, +j: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match left:
    case 0n: IO.pure(T.Train<P>, tr)
    case 1n+q: IO.bind(T.Train<P>, T.Train<P>, settle(~P, flush_if(~P, ~fl, U32.is_eq(U32.mod(j, 4), 3), acc(tr, gx(j), gt(j)))), t2 => micro(~P, ~X, ~G, ~acc, ~fl, ~gx, ~gt, q, (j + 1 : U32), t2))

# left calls of f more, on batch j, j + 1, ...
def steps(~P: Type, ~X: Type, ~G: Type, ~f: T.Train<P> -> X -> G -> T.Train<P>, ~gx: U32 -> X, ~gt: U32 -> G, left: Nat, +j: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match left:
    case 0n: IO.pure(T.Train<P>, tr)
    case 1n+q: IO.bind(T.Train<P>, T.Train<P>, settle(~P, f(tr, gx(j), gt(j))), t2 => steps(~P, ~X, ~G, ~f, ~gx, ~gt, q, (j + 1 : U32), t2))

def pl_xs(-P: Type, r: T.PL<P>) -> List<&2, F32>:
  match r:
    case T.PL{p, xs}: xs

def show2(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +tag: String, r: T.TrainPL<P>) -> IO(Unit):
  match r:
    case T.TrainPL{T.Train{p, g, st, c, p1, p2, +l}, xs}:
      do IO<Unit>:
        IO.print(tag ++ ".state" ++ T.words(xs))
        IO.print(tag ++ ".g" ++ T.words(pl_xs(P, T.Layer.dump(P, X, Y, S, m, g))))
        IO.print(tag ++ ".loss" ++ T.words([l]))

# the state (parameters, moments, 1 - b1^t and 1 - b2^t), the gradient and the loss
def show(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +tag: String, tr: T.Train<P>) -> IO(Unit):
  show2(~P, ~X, ~Y, ~S, ~m, tag, T.Train.dump(~P, ~X, ~Y, ~S, ~m, tr))

# from the model's first weights, through loop, then printed
def train(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, ~loop: T.Train<P> -> IO(T.Train<P>), +tag: String, +seed: U32) -> IO(Unit):
  do IO<Unit>:
    t0 : T.Train<P> <- settle(~P, new(~P, ~X, ~Y, ~S, ~m, seed))
    t1 : T.Train<P> <- loop(t0)
    show(~P, ~X, ~Y, ~S, ~m, tag, t1)

# run r of a model: m on micro-batches (data gx, targets gt), mb on four of them at once (gxb, gtb)
def run(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~G: Type, ~XB: Type, ~YB: Type, ~SB: Type, ~GB: Type, ~m: T.Layer<P, X, Y, S>, ~mb: T.Layer<P, XB, YB, SB>, ~loss: Y -> G -> T.LossOut<Y>, ~lossq: Y -> G -> T.LossOut<Y>, ~lossb: YB -> GB -> T.LossOut<YB>, ~gx: U32 -> X, ~gt: U32 -> G, ~gxb: U32 -> XB, ~gtb: U32 -> GB, +tag: String, +seed: U32, r: Nat) -> IO(Unit):
  match r:
    case 0n:
      train(~P, ~X, ~Y, ~S, ~m, ~(t => micro(~P, ~X, ~G, ~(tr => x => g => T.Train.accumulate(~P, ~X, ~Y, ~S, ~G, ~m, ~lossq, tr, x, g)), ~(tr => T.Train.flush(~P, ~X, ~Y, ~S, ~m, tr)), ~gx, ~gt, 8n, 0, t)), tag ++ ".acc", seed)
    case 1n+q:
      match q:
        case 0n:
          train(~P, ~XB, ~YB, ~SB, ~mb, ~(t => steps(~P, ~XB, ~GB, ~(tr => x => g => T.Train.step(~P, ~XB, ~YB, ~SB, ~GB, ~mb, ~lossb, tr, x, g)), ~gxb, ~gtb, 2n, 0, t)), tag ++ ".big", seed)
        case 1n+w:
          match w:
            case 0n:
              train(~P, ~X, ~Y, ~S, ~m, ~(t => steps(~P, ~X, ~G, ~(tr => x => g => T.Train.flush(~P, ~X, ~Y, ~S, ~m, T.Train.accumulate(~P, ~X, ~Y, ~S, ~G, ~m, ~loss, tr, x, g))), ~gx, ~gt, 2n, 0, t)), tag ++ ".one", seed)
            case 1n+z:
              train(~P, ~X, ~Y, ~S, ~m, ~(t => steps(~P, ~X, ~G, ~(tr => x => g => T.Train.step(~P, ~X, ~Y, ~S, ~G, ~m, ~loss, tr, x, g)), ~gx, ~gt, 2n, 0, t)), tag ++ ".step", seed)

# ---- lin: LinearFirst(12, 16) -> RMSNorm -> LinearNB(16, 8) on r rows (8 a micro-batch) ----

def LinP() -> Type:
  T.Two<T.Two<T.Lin<12n, 16n>, M(1n, 16n)>, M(8n, 16n)>

def LinS(r: Nat) -> Type:
  T.Two<T.Two<M(r, 12n), M(r, 16n)>, M(r, 16n)>

def LinH(~r: Nat) -> T.Layer<T.Two<T.Lin<12n, 16n>, M(1n, 16n)>, M(r, 12n), M(r, 16n), T.Two<M(r, 12n), M(r, 16n)>>:
  T.Seq(~T.Lin<12n, 16n>, ~M(1n, 16n), ~M(r, 12n), ~M(r, 16n), ~M(r, 16n), ~M(r, 12n), ~M(r, 16n), ~F.LinearFirst(r, 12n, 16n, 2n), ~F.RMSNorm(r, 16n, 0.00001))

def Lin(~r: Nat) -> T.Layer<LinP, M(r, 12n), M(r, 8n), LinS(r)>:
  T.Seq(~T.Two<T.Lin<12n, 16n>, M(1n, 16n)>, ~M(8n, 16n), ~M(r, 12n), ~M(r, 16n), ~M(r, 8n), ~T.Two<M(r, 12n), M(r, 16n)>, ~M(r, 16n), ~LinH(~r), ~F.LinearNB(r, 16n, 8n, 2n))

def lin(r: Nat) -> IO(Unit):
  run(~LinP, ~M(8n, 12n), ~M(8n, 8n), ~LinS(8n), ~M(8n, 8n), ~M(32n, 12n), ~M(32n, 8n), ~LinS(32n), ~M(32n, 8n), ~Lin(~8n), ~Lin(~32n), ~(y => t => F.MSE(8n, 8n, y, t)), ~(y => t => T.LossOut.div(8n, 8n, 4, F.MSE(8n, 8n, y, t))), ~(y => t => F.MSE(32n, 8n, y, t)), ~(j => rows(8n, 12n, j, 11)), ~(j => rows(8n, 8n, j, 12)), ~(j => rows(32n, 12n, j, 11)), ~(j => rows(32n, 8n, j, 12)), "lin", 13, r)

# ---- mlp: Residual(RMSNorm -> down(silu(gate(x)) * up(x))), width 16, feed-forward 24 (demo/smol_block.bend's) ----

def GateS(r: Nat) -> Type:
  T.Two<M(r, 16n), M(r, 24n)>

def Gate(~r: Nat) -> T.Layer<T.Two<M(24n, 16n), Unit>, M(r, 16n), M(r, 24n), GateS(r)>:
  T.Seq(~M(24n, 16n), ~Unit, ~M(r, 16n), ~M(r, 24n), ~M(r, 24n), ~M(r, 16n), ~M(r, 24n), ~F.LinearNB(r, 16n, 24n, 2n), ~F.SiLU(r, 24n))

def GUP() -> Type:
  T.Two<T.Two<M(24n, 16n), Unit>, M(24n, 16n)>

def GUY(r: Nat) -> Type:
  T.Two<M(r, 24n), M(r, 24n)>

def GUS(r: Nat) -> Type:
  T.Two<GateS(r), M(r, 16n)>

def GU(~r: Nat) -> T.Layer<GUP, M(r, 16n), GUY(r), GUS(r)>:
  T.Fork(~T.Two<M(24n, 16n), Unit>, ~M(24n, 16n), ~M(r, 16n), ~M(r, 24n), ~M(r, 24n), ~GateS(r), ~M(r, 16n), ~(m => F.Mat.dup(r, 16n, m)), ~(x => y => F.Mat.sum2(r, 16n, x, y)), ~Gate(~r), ~F.LinearNB(r, 16n, 24n, 2n))

def GLUS(r: Nat) -> Type:
  T.Two<GUS(r), GUY(r)>

def GLU(~r: Nat) -> T.Layer<T.Two<GUP, Unit>, M(r, 16n), M(r, 24n), GLUS(r)>:
  T.Seq(~GUP, ~Unit, ~M(r, 16n), ~GUY(r), ~M(r, 24n), ~GUS(r), ~GUY(r), ~GU(~r), ~F.Mul2(r, 24n))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(16n, 24n))

def MlpS(r: Nat) -> Type:
  T.Two<GLUS(r), M(r, 24n)>

def Mlp(~r: Nat) -> T.Layer<MlpP, M(r, 16n), M(r, 16n), MlpS(r)>:
  T.Seq(~T.Two<GUP, Unit>, ~M(16n, 24n), ~M(r, 16n), ~M(r, 24n), ~M(r, 16n), ~GLUS(r), ~M(r, 24n), ~GLU(~r), ~F.LinearNB(r, 24n, 16n, 2n))

def NMlpP() -> Type:
  T.Two<M(1n, 16n), MlpP>

def NMlpS(r: Nat) -> Type:
  T.Two<M(r, 16n), MlpS(r)>

def NMlp(~r: Nat) -> T.Layer<NMlpP, M(r, 16n), M(r, 16n), NMlpS(r)>:
  T.Seq(~M(1n, 16n), ~MlpP, ~M(r, 16n), ~M(r, 16n), ~M(r, 16n), ~M(r, 16n), ~MlpS(r), ~F.RMSNorm(r, 16n, 0.00001), ~Mlp(~r))

def MlpR(~r: Nat) -> T.Layer<T.ResidualType(NMlpP), M(r, 16n), M(r, 16n), T.ResidualType(NMlpS(r))>:
  F.Residual(~r, ~16n, ~NMlpP, ~NMlpS(r), ~NMlp(~r))

def mlp(r: Nat) -> IO(Unit):
  run(~T.ResidualType(NMlpP), ~M(8n, 16n), ~M(8n, 16n), ~T.ResidualType(NMlpS(8n)), ~M(8n, 16n), ~M(32n, 16n), ~M(32n, 16n), ~T.ResidualType(NMlpS(32n)), ~M(32n, 16n), ~MlpR(~8n), ~MlpR(~32n), ~(y => t => F.MSE(8n, 16n, y, t)), ~(y => t => T.LossOut.div(8n, 16n, 4, F.MSE(8n, 16n, y, t))), ~(y => t => F.MSE(32n, 16n, y, t)), ~(j => rows(8n, 16n, j, 21)), ~(j => rows(8n, 16n, j, 22)), ~(j => rows(32n, 16n, j, 21)), ~(j => rows(32n, 16n, j, 22)), "mlp", 23, r)

# ---- attn: Residual(RMSNorm -> attention) on b sequences of 8 positions (2 a micro-batch): width 16, 2 query heads
# and 1 key/value head of 8, RoPE base 10000 (demo/smol_block.bend's attention) ----

def R(b: Nat) -> Nat:
  Nat.mul(b, 8n)

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

def KVP() -> Type:
  T.Two<M(8n, 16n), M(8n, 16n)>

def KVY(b: Nat) -> Type:
  T.Two<A(b, 8n), A(b, 8n)>

def KVS(b: Nat) -> Type:
  T.Two<A(b, 16n), A(b, 16n)>

def KVL(~b: Nat) -> T.Layer<KVP, A(b, 16n), KVY(b), KVS(b)>:
  T.Fork(~M(8n, 16n), ~M(8n, 16n), ~A(b, 16n), ~A(b, 8n), ~A(b, 8n), ~A(b, 16n), ~A(b, 16n), ~(m => F.Mat.dup(R(b), 16n, m)), ~(x => y => F.Mat.sum2(R(b), 16n, x, y)), ~F.LinearNB(R(b), 16n, 8n, 2n), ~F.LinearNB(R(b), 16n, 8n, 2n))

def AX(b: Nat) -> Type:
  T.Two<A(b, 16n), KVY(b)>

def QKVP() -> Type:
  T.Two<M(16n, 16n), KVP>

def QKVS(b: Nat) -> Type:
  T.Two<A(b, 16n), KVS(b)>

def QKV(~b: Nat) -> T.Layer<QKVP, A(b, 16n), AX(b), QKVS(b)>:
  T.Fork(~M(16n, 16n), ~KVP, ~A(b, 16n), ~A(b, 16n), ~KVY(b), ~A(b, 16n), ~KVS(b), ~(m => F.Mat.dup(R(b), 16n, m)), ~(x => y => F.Mat.sum2(R(b), 16n, x, y)), ~F.LinearNB(R(b), 16n, 16n, 2n), ~KVL(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, KVY(b), KVY(b), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, 8n), ~A(b, 8n), ~A(b, 8n), ~A(b, 8n), ~Unit, ~Unit, ~F.RoPE(R(b), 8n, 8n, 1n, 4n, 10000.0, {==}), ~T.Id(~A(b, 8n)))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(~b: Nat) -> T.Layer<RU, AX(b), AX(b), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, 16n), ~KVY(b), ~A(b, 16n), ~KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~F.RoPE(R(b), 16n, 8n, 2n, 4n, 10000.0, {==}), ~RopeK(~b))

def CoreP() -> Type:
  T.Two<Unit, M(16n, 16n)>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, 16n)>

def Core(~b: Nat) -> T.Layer<CoreP, AX(b), A(b, 16n), CoreS(b)>:
  T.Seq(~Unit, ~M(16n, 16n), ~AX(b), ~A(b, 16n), ~A(b, 16n), ~T.AttnS, ~A(b, 16n), ~F.Attention(R(b), 16n, 8n, b, 8n, 1n, 2n, 8n, 0.35355339, {==}, {==}, {==}), ~F.LinearNB(R(b), 16n, 16n, 2n))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS(b: Nat) -> Type:
  T.Two<RU, CoreS(b)>

def RopeCore(~b: Nat) -> T.Layer<RCP, AX(b), A(b, 16n), RCS(b)>:
  T.Seq(~RU, ~CoreP, ~AX(b), ~AX(b), ~A(b, 16n), ~RU, ~CoreS(b), ~Ropes(~b), ~Core(~b))

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS(b: Nat) -> Type:
  T.Two<QKVS(b), RCS(b)>

def Att(~b: Nat) -> T.Layer<AttP, A(b, 16n), A(b, 16n), AttS(b)>:
  T.Seq(~QKVP, ~RCP, ~A(b, 16n), ~AX(b), ~A(b, 16n), ~QKVS(b), ~RCS(b), ~QKV(~b), ~RopeCore(~b))

def NAttP() -> Type:
  T.Two<M(1n, 16n), AttP>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, 16n), AttS(b)>

def NAtt(~b: Nat) -> T.Layer<NAttP, A(b, 16n), A(b, 16n), NAttS(b)>:
  T.Seq(~M(1n, 16n), ~AttP, ~A(b, 16n), ~A(b, 16n), ~A(b, 16n), ~A(b, 16n), ~AttS(b), ~F.RMSNorm(R(b), 16n, 0.00001), ~Att(~b))

def AttR(~b: Nat) -> T.Layer<T.ResidualType(NAttP), A(b, 16n), A(b, 16n), T.ResidualType(NAttS(b))>:
  F.Residual(~R(b), ~16n, ~NAttP, ~NAttS(b), ~NAtt(~b))

def attn(r: Nat) -> IO(Unit):
  run(~T.ResidualType(NAttP), ~A(2n, 16n), ~A(2n, 16n), ~T.ResidualType(NAttS(2n)), ~A(2n, 16n), ~A(8n, 16n), ~A(8n, 16n), ~T.ResidualType(NAttS(8n)), ~A(8n, 16n), ~AttR(~2n), ~AttR(~8n), ~(y => t => F.MSE(R(2n), 16n, y, t)), ~(y => t => T.LossOut.div(R(2n), 16n, 4, F.MSE(R(2n), 16n, y, t))), ~(y => t => F.MSE(R(8n), 16n, y, t)), ~(j => rows(R(2n), 16n, j, 31)), ~(j => rows(R(2n), 16n, j, 32)), ~(j => rows(R(8n), 16n, j, 31)), ~(j => rows(R(8n), 16n, j, 32)), "attn", 33, r)

# ---- gn: GroupNorm -> SiLU -> Linear(8, 6) on n images of 4 pixels (2 a micro-batch), 2 groups of 4 channels ----

def GR(n: Nat) -> Nat:
  Nat.mul(n, 4n)

def GnP() -> Type:
  T.Two<T.Two<T.LN<8n>, Unit>, T.Lin<8n, 6n>>

def GnS(n: Nat) -> Type:
  T.Two<T.Two<M(GR(n), 8n), M(GR(n), 8n)>, M(GR(n), 8n)>

def GnH(~n: Nat) -> T.Layer<T.Two<T.LN<8n>, Unit>, M(GR(n), 8n), M(GR(n), 8n), T.Two<M(GR(n), 8n), M(GR(n), 8n)>>:
  T.Seq(~T.LN<8n>, ~Unit, ~M(GR(n), 8n), ~M(GR(n), 8n), ~M(GR(n), 8n), ~M(GR(n), 8n), ~M(GR(n), 8n), ~F.GroupNorm(n, 4n, 2n, 4n, 0.00001), ~F.SiLU(GR(n), 8n))

def Gn(~n: Nat) -> T.Layer<GnP, M(GR(n), 8n), M(GR(n), 6n), GnS(n)>:
  T.Seq(~T.Two<T.LN<8n>, Unit>, ~T.Lin<8n, 6n>, ~M(GR(n), 8n), ~M(GR(n), 8n), ~M(GR(n), 6n), ~T.Two<M(GR(n), 8n), M(GR(n), 8n)>, ~M(GR(n), 8n), ~GnH(~n), ~F.Linear(GR(n), 8n, 6n, 2n))

def gn(r: Nat) -> IO(Unit):
  run(~GnP, ~M(GR(2n), 8n), ~M(GR(2n), 6n), ~GnS(2n), ~M(GR(2n), 6n), ~M(GR(8n), 8n), ~M(GR(8n), 6n), ~GnS(8n), ~M(GR(8n), 6n), ~Gn(~2n), ~Gn(~8n), ~(y => t => F.MSE(GR(2n), 6n, y, t)), ~(y => t => T.LossOut.div(GR(2n), 6n, 4, F.MSE(GR(2n), 6n, y, t))), ~(y => t => F.MSE(GR(8n), 6n, y, t)), ~(j => rows(GR(2n), 8n, j, 41)), ~(j => rows(GR(2n), 6n, j, 42)), ~(j => rows(GR(8n), 8n, j, 41)), ~(j => rows(GR(8n), 6n, j, 42)), "gn", 43, r)

# ---- emb: Embedding(13 ids, 8) -> LinearNB(8, 13) on r ids (8 a micro-batch), cross-entropy over the 13 ----

def EmbP() -> Type:
  T.Two<M(13n, 8n), M(13n, 8n)>

def EmbS(r: Nat) -> Type:
  T.Two<List<&2, Nat>, M(r, 8n)>

def Emb(~r: Nat) -> T.Layer<EmbP, List<&2, Nat>, M(r, 13n), EmbS(r)>:
  T.Seq(~M(13n, 8n), ~M(13n, 8n), ~List<&2, Nat>, ~M(r, 8n), ~M(r, 13n), ~List<&2, Nat>, ~M(r, 8n), ~F.Embedding(r, 13n, 8n), ~F.LinearNB(r, 8n, 13n, 2n))

def emb(r: Nat) -> IO(Unit):
  run(~EmbP, ~List<&2, Nat>, ~M(8n, 13n), ~EmbS(8n), ~List<&2, Nat>, ~List<&2, Nat>, ~M(32n, 13n), ~EmbS(32n), ~List<&2, Nat>, ~Emb(~8n), ~Emb(~32n), ~(z => t => F.CrossEntropy(8n, 13n, z, t)), ~(z => t => T.LossOut.div(8n, 13n, 4, F.CrossEntropy(8n, 13n, z, t))), ~(z => t => F.CrossEntropy(32n, 13n, z, t)), ~(j => ids(8n, j, 13, 51)), ~(j => ids(8n, j, 13, 52)), ~(j => ids(32n, j, 13, 51)), ~(j => ids(32n, j, 13, 52)), "emb", 53, r)

# ---- the program ----

def model(m: Nat, r: Nat) -> IO(Unit):
  match m:
    case 0n: lin(r)
    case 1n+a:
      match a:
        case 0n: mlp(r)
        case 1n+b:
          match b:
            case 0n: attn(r)
            case 1n+c:
              match c:
                case 0n: gn(r)
                case 1n+d: emb(r)

def runs(+m: Nat) -> IO(Unit):
  do IO<Unit>:
    model(m, 0n)
    model(m, 1n)
    model(m, 2n)
    model(m, 3n)

def all() -> IO(Unit):
  do IO<Unit>:
    runs(0n)
    runs(1n)
    runs(2n)
    runs(3n)
    runs(4n)

# an argument other than two numbers is an error: since Bend 2.0.38 a runtime option goes between +RTS and -RTS
def bad_arg(a: String) -> IO(Unit):
  do IO<Unit>:
    IO.print_err("unexpected argument " ++ a ++ ": a model and a run number, or none (runtime options go between +RTS and -RTS)")
    IO.exit(2)

def one_m(m: Maybe<&2, Nat>, r: Maybe<&2, Nat>, a: String) -> IO(Unit):
  match m r:
    case Some{i} Some{j}: model(i, j)
    case _ _: bad_arg(a)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Nil{}: all()
    case Con{prog, Nil{}}: all()
    case Con{prog, Con{+a, Con{+b, Nil{}}}}: one_m(Nat.read(a), Nat.read(b), a ++ " " ++ b)
    case Con{prog, Con{a, t}}: bad_arg(a)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
