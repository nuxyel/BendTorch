# A two-layer character-level Llama trained in Bend on Tiny Shakespeare, against its PyTorch twin
# (reference/tiny_ref.py, which also writes the initial weights and the batches into data/tiny/).
#
#   bend demo/tiny_llama.bend -o .scratch/tiny_llama
#   .scratch/tiny_llama grads                     # the loss and every gradient on the first batch
#   .scratch/tiny_llama train <steps>             # AdamW steps: the loss before each update, then the final weights
#   .scratch/tiny_llama sample <steps> <chars> <prompt>   # train, then continue a prompt greedily, a character at a time
#   .scratch/tiny_llama draw <steps> <chars> <seed> <prompt>  # the same, drawing each character (temperature 0.8)
#   .scratch/tiny_llama save <steps> <file>                # train, then write a checkpoint
#   .scratch/tiny_llama resume <file> <from> <steps>       # read a checkpoint, train on from batch <from>
#
# Every block is composed from the package's layers: Residual(Seq(RMSNorm, attention)) then
# Residual(Seq(RMSNorm, SwiGLU)); attention is Fork(q, Fork(k, v)) -> Both(RoPE, Both(RoPE, Id)) -> Attention -> wo.
# The model is a template over the number of sequences b of 16 positions each (rows R(b) = 16·b): training uses b = 2,
# sampling b = 1, and the parameters have the same type in both. Vocabulary 65, width 32, 4 query heads and 2
# key/value heads of 8, feed-forward 64. Each number is printed as the decimal of its 32 bits.
# The linear layers are torch/fast.bend's (products over bend-ml-tensor-fork, outside the proofs): main.bend's
# layers with the same bits (demo/fast_test.bend), faster where a product splits into parallel blocks.

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# the rows of a batch of b sequences of 16 positions
def R(b: Nat) -> Nat:
  Nat.mul(b, 16n)

# an activation: one row per position
def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

# ---- attention: q, k, v projections, RoPE on q and k, the attention core, the output projection ----

def KVP() -> Type:
  T.Two<M(16n, 32n), M(16n, 32n)>

def KVY(b: Nat) -> Type:
  T.Two<A(b, 16n), A(b, 16n)>

def KVS(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 32n)>

def KVL(~b: Nat) -> T.Layer<KVP, A(b, 32n), KVY(b), KVS(b)>:
  T.Fork(~M(16n, 32n), ~M(16n, 32n), ~A(b, 32n), ~A(b, 16n), ~A(b, 16n), ~A(b, 32n), ~A(b, 32n), ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)), ~F.LinearNB(R(b), 32n, 16n, 0n), ~F.LinearNB(R(b), 32n, 16n, 0n))

def AX(b: Nat) -> Type:
  T.Two<A(b, 32n), KVY(b)>

def QKVP() -> Type:
  T.Two<M(32n, 32n), KVP>

def QKVS(b: Nat) -> Type:
  T.Two<A(b, 32n), KVS(b)>

def QKV(~b: Nat) -> T.Layer<QKVP, A(b, 32n), AX(b), QKVS(b)>:
  T.Fork(~M(32n, 32n), ~KVP, ~A(b, 32n), ~A(b, 32n), ~KVY(b), ~A(b, 32n), ~KVS(b), ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)), ~F.LinearNB(R(b), 32n, 32n, 0n), ~KVL(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, KVY(b), KVY(b), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, 16n), ~A(b, 16n), ~A(b, 16n), ~A(b, 16n), ~Unit, ~Unit, ~T.RoPE(R(b), 16n, 16n, 2n, 4n, 10000.0, {==}), ~T.Id(~A(b, 16n)))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(~b: Nat) -> T.Layer<RU, AX(b), AX(b), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, 32n), ~KVY(b), ~A(b, 32n), ~KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~T.RoPE(R(b), 32n, 16n, 4n, 4n, 10000.0, {==}), ~RopeK(~b))

def CoreP() -> Type:
  T.Two<Unit, M(32n, 32n)>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, 32n)>

def Core(~b: Nat) -> T.Layer<CoreP, AX(b), A(b, 32n), CoreS(b)>:
  T.Seq(~Unit, ~M(32n, 32n), ~AX(b), ~A(b, 32n), ~A(b, 32n), ~T.AttnS, ~A(b, 32n), ~T.Attention(R(b), 32n, 16n, b, 16n, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==}), ~F.LinearNB(R(b), 32n, 32n, 0n))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS(b: Nat) -> Type:
  T.Two<RU, CoreS(b)>

def RopeCore(~b: Nat) -> T.Layer<RCP, AX(b), A(b, 32n), RCS(b)>:
  T.Seq(~RU, ~CoreP, ~AX(b), ~AX(b), ~A(b, 32n), ~RU, ~CoreS(b), ~Ropes(~b), ~Core(~b))

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS(b: Nat) -> Type:
  T.Two<QKVS(b), RCS(b)>

def Att(~b: Nat) -> T.Layer<AttP, A(b, 32n), A(b, 32n), AttS(b)>:
  T.Seq(~QKVP, ~RCP, ~A(b, 32n), ~AX(b), ~A(b, 32n), ~QKVS(b), ~RCS(b), ~QKV(~b), ~RopeCore(~b))

def NAttP() -> Type:
  T.Two<M(1n, 32n), AttP>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, 32n), AttS(b)>

def NAtt(~b: Nat) -> T.Layer<NAttP, A(b, 32n), A(b, 32n), NAttS(b)>:
  T.Seq(~M(1n, 32n), ~AttP, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~AttS(b), ~T.RMSNorm(R(b), 32n, 0.00001), ~Att(~b))

# ---- the feed-forward block: down(silu(gate(x)) * up(x)) ----

def GateS(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 64n)>

def Gate(~b: Nat) -> T.Layer<T.Two<M(64n, 32n), Unit>, A(b, 32n), A(b, 64n), GateS(b)>:
  T.Seq(~M(64n, 32n), ~Unit, ~A(b, 32n), ~A(b, 64n), ~A(b, 64n), ~A(b, 32n), ~A(b, 64n), ~F.LinearNB(R(b), 32n, 64n, 0n), ~T.SiLU(R(b), 64n))

def GUP() -> Type:
  T.Two<T.Two<M(64n, 32n), Unit>, M(64n, 32n)>

def GUY(b: Nat) -> Type:
  T.Two<A(b, 64n), A(b, 64n)>

def GUS(b: Nat) -> Type:
  T.Two<GateS(b), A(b, 32n)>

def GU(~b: Nat) -> T.Layer<GUP, A(b, 32n), GUY(b), GUS(b)>:
  T.Fork(~T.Two<M(64n, 32n), Unit>, ~M(64n, 32n), ~A(b, 32n), ~A(b, 64n), ~A(b, 64n), ~GateS(b), ~A(b, 32n), ~(m => T.Mat.dup(R(b), 32n, m)), ~(x => y => T.Mat.sum2(R(b), 32n, x, y)), ~Gate(~b), ~F.LinearNB(R(b), 32n, 64n, 0n))

def GLUS(b: Nat) -> Type:
  T.Two<GUS(b), GUY(b)>

def GLU(~b: Nat) -> T.Layer<T.Two<GUP, Unit>, A(b, 32n), A(b, 64n), GLUS(b)>:
  T.Seq(~GUP, ~Unit, ~A(b, 32n), ~GUY(b), ~A(b, 64n), ~GUS(b), ~GUY(b), ~GU(~b), ~T.Mul2(R(b), 64n))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(32n, 64n))

def MlpS(b: Nat) -> Type:
  T.Two<GLUS(b), A(b, 64n)>

def Mlp(~b: Nat) -> T.Layer<MlpP, A(b, 32n), A(b, 32n), MlpS(b)>:
  T.Seq(~T.Two<GUP, Unit>, ~M(32n, 64n), ~A(b, 32n), ~A(b, 64n), ~A(b, 32n), ~GLUS(b), ~A(b, 64n), ~GLU(~b), ~F.LinearNB(R(b), 64n, 32n, 0n))

def NMlpP() -> Type:
  T.Two<M(1n, 32n), MlpP>

def NMlpS(b: Nat) -> Type:
  T.Two<A(b, 32n), MlpS(b)>

def NMlp(~b: Nat) -> T.Layer<NMlpP, A(b, 32n), A(b, 32n), NMlpS(b)>:
  T.Seq(~M(1n, 32n), ~MlpP, ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~MlpS(b), ~T.RMSNorm(R(b), 32n, 0.00001), ~Mlp(~b))

# ---- a block, the model ----

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BlockS(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def Block(~b: Nat) -> T.Layer<BlockP, A(b, 32n), A(b, 32n), BlockS(b)>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(b, 32n), ~A(b, 32n), ~A(b, 32n), ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)), ~T.Residual(~R(b), ~32n, ~NAttP, ~NAttS(b), ~NAtt(~b)), ~T.Residual(~R(b), ~32n, ~NMlpP, ~NMlpS(b), ~NMlp(~b)))

def HeadP() -> Type:
  T.Two<M(1n, 32n), M(65n, 32n)>

def HeadS(b: Nat) -> Type:
  T.Two<A(b, 32n), A(b, 32n)>

def Head(~b: Nat) -> T.Layer<HeadP, A(b, 32n), A(b, 65n), HeadS(b)>:
  T.Seq(~M(1n, 32n), ~M(65n, 32n), ~A(b, 32n), ~A(b, 32n), ~A(b, 65n), ~A(b, 32n), ~A(b, 32n), ~T.RMSNorm(R(b), 32n, 0.00001), ~F.LinearNB(R(b), 32n, 65n, 0n))

def TopP() -> Type:
  T.Two<BlockP, HeadP>

def TopS(b: Nat) -> Type:
  T.Two<BlockS(b), HeadS(b)>

def Top(~b: Nat) -> T.Layer<TopP, A(b, 32n), A(b, 65n), TopS(b)>:
  T.Seq(~BlockP, ~HeadP, ~A(b, 32n), ~A(b, 32n), ~A(b, 65n), ~BlockS(b), ~HeadS(b), ~Block(~b), ~Head(~b))

def BodyP() -> Type:
  T.Two<BlockP, TopP>

def BodyS(b: Nat) -> Type:
  T.Two<BlockS(b), TopS(b)>

def Body(~b: Nat) -> T.Layer<BodyP, A(b, 32n), A(b, 65n), BodyS(b)>:
  T.Seq(~BlockP, ~TopP, ~A(b, 32n), ~A(b, 32n), ~A(b, 65n), ~BlockS(b), ~TopS(b), ~Block(~b), ~Top(~b))

def P() -> Type:
  T.Two<M(65n, 32n), BodyP>

def S(b: Nat) -> Type:
  T.Two<List<&2, Nat>, BodyS(b)>

def X() -> Type:
  List<&2, Nat>

def Y(b: Nat) -> Type:
  A(b, 65n)

def Model(~b: Nat) -> T.Layer<P, X, Y(b), S(b)>:
  T.Seq(~M(65n, 32n), ~BodyP, ~X, ~A(b, 32n), ~Y(b), ~List<&2, Nat>, ~BodyS(b), ~T.Embedding(R(b), 65n, 32n), ~Body(~b))

def ce(z: Y(2n), labels: List<&2, Nat>) -> T.LossOut<Y(2n)>:
  T.CrossEntropy(32n, 65n, z, labels)

# ---- weights: data/tiny/init/<name>.bin, in reference/tiny_ref.py's names ----

def mk_block(ln1: M(1n, 32n), wq: M(32n, 32n), wk: M(16n, 32n), wv: M(16n, 32n), wo: M(32n, 32n), ln2: M(1n, 32n), wg: M(64n, 32n), wu: M(64n, 32n), wd: M(32n, 64n)) -> BlockP:
  T.Two{T.Two{T.Two{Unit{}, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}}, T.Two{T.Two{Unit{}, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, Unit{}}, wu}, Unit{}}, wd}}}, Unit{}}}

def load_params() -> IO(P):
  L.weights(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), "data/tiny/init.bin", T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n)))

def print_block(+tag: String, b: BlockP) -> IO(Unit):
  match b:
    case T.Two{T.Two{T.Two{u1, T.Two{ln1, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{u2, T.Two{u3, u4}}, T.Two{u5, wo}}}}}, u6}, T.Two{T.Two{u7, T.Two{ln2, T.Two{T.Two{T.Two{T.Two{wg, u8}, wu}, u9}, wd}}}, u10}}:
      do IO<Unit>:
        T.Mat.print(1n, 32n, tag ++ ".ln1", ln1)
        T.Mat.print(32n, 32n, tag ++ ".wq", wq)
        T.Mat.print(16n, 32n, tag ++ ".wk", wk)
        T.Mat.print(16n, 32n, tag ++ ".wv", wv)
        T.Mat.print(32n, 32n, tag ++ ".wo", wo)
        T.Mat.print(1n, 32n, tag ++ ".ln2", ln2)
        T.Mat.print(64n, 32n, tag ++ ".wg", wg)
        T.Mat.print(64n, 32n, tag ++ ".wu", wu)
        T.Mat.print(32n, 64n, tag ++ ".wd", wd)

def print_params(+tag: String, p: P) -> IO(Unit):
  match p:
    case T.Two{emb, T.Two{b0, T.Two{b1, T.Two{lnf, head}}}}:
      do IO<Unit>:
        T.Mat.print(65n, 32n, tag ++ ".emb", emb)
        print_block(tag ++ ".b0", b0)
        print_block(tag ++ ".b1", b1)
        T.Mat.print(1n, 32n, tag ++ ".lnf", lnf)
        T.Mat.print(65n, 32n, tag ++ ".head", head)

# ---- batches: data/tiny/batches.bin, 2 x 17 ids per step; the inputs are the first 16 of each, the targets the last 16 ----

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> nats(t)

def xs_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, ids, 16n), List.take(&2, Nat, List.drop(&2, Nat, ids, 17n), 16n))

def ys_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, List.drop(&2, Nat, ids, 1n), 16n), List.drop(&2, Nat, ids, 18n))

def load_batch(+step: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at("data/tiny/batches.bin", (step * 34 : U32), 34)
    return nats(bs)

# ---- one forward and backward pass ----

def grads3(r: T.Bwd<P, X>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      print_params("grad", g)

def grads2(p: P, s: S(2n), r: T.LossOut<Y(2n)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(T.Layer.bwd(P, X, Y(2n), S(2n), Model(~2n), p, s, dy, T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n))))

def grads1(+ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      grads2(p, s, ce(y, ys))

def grads(+ids: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    grads1(ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

# ---- training ----

def step_with(+cfg: T.OptCfg, +ids: List<&2, Nat>, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y(2n), ~S(2n), ~List<&2, Nat>, ~Model(~2n), ~ce, cfg, tr, xs_of(ids), ys_of(ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +step: U32, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- load_batch(step)
        tloop(q, (step + 1 : U32), cfg, step_with(cfg, ids, T.Train{p, g, st, p1, p2, l}))

def cfg() -> T.OptCfg:
  T.CAdamW{0.003, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def train_run(+k: Nat, p: P) -> IO(T.Train<P>):
  do IO<T.Train<P>>:
    T.OptCfg.check(cfg())
    tloop(k, 0, cfg(), T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), p))

def finish(tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      print_params("final", p)

def train(+k: Nat) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, p)
    finish(tr)

# ---- checkpoints: stop after k steps, resume from step k in another process ----

def save(+k: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, p)
    tr2 : T.Train<P> <- T.Train.save(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), path, tr)
    IO.print("saved " ++ path)

def resume(+path: String, +from: Nat, +k: Nat) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    tr : T.Train<P> <- L.train(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), path, T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n))))
    tr2 : T.Train<P> <- tloop(k, U32.from_nat(from), cfg(), tr)
    finish(tr2)

# ---- sampling: one sequence of 16 positions (b = 1), the same parameters ----

type Gen is Type:
  Gen{p: P, ids: List<&2, Nat>}

# the most likely next character: the argmax of the last row of the logits (the first one on a tie, as torch.argmax)
def last_argmax(r: TA.MRow<16n, 65n>) -> Nat:
  match r:
    case TA.MRow{m, row}:
      TA.argmax(row)

def fwd_then(r: T.Fwd<P, Y(1n), S(1n)>, k: P -> @+i: Nat -> Gen) -> Gen:
  match r:
    case T.Fwd{p, y, s}:
      k(p, last_argmax(TA.Mat.read_row(16n, 65n, y, 15)))

# k more characters after the window of the last 16; acc holds the generated ids, newest first
def gen(k: Nat, p: P, +win: List<&2, Nat>, +acc: List<&2, Nat>) -> Gen:
  match k:
    case 0n:
      Gen{p, List.reverse(&2, Nat, acc)}
    case 1n+q:
      fwd_then(T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, win), p2 => +id => gen(q, p2, List.append(&2, Nat, List.drop(&2, Nat, win, 1n), [id]), id <> acc))

# ---- drawing: the next character sampled from the softmax of the last row at temperature 0.8, with a uniform number
# from a counter-based hash of (seed, character index), so a run is reproducible. Greedy decoding (above) is the one
# the twin checks; drawing only shows what the trained model writes without falling into a loop ----

def row_max(xs: List<&2, F32>, +m: F32) -> F32:
  match xs:
    case Nil{}:
      m
    case Con{+h, t}:
      row_max(t, F32.max(m, h))

# exp((x - m) / 0.8) for each x
def row_w(xs: List<&2, F32>, +m: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      F32.exp(((h - m : F32) / 0.8 : F32)) <> row_w(t, m)

def row_sum(xs: List<&2, F32>, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      row_sum(t, (acc + h : F32))

# the first index whose running sum passes u (the last one if rounding leaves u at the total)
def pick_at(xs: List<&2, F32>, hit: Bool, +u: F32, +acc: F32, +i: Nat) -> Nat:
  match xs hit:
    case _ True{}:
      Nat.sub(i, 1n)
    case Nil{} False{}:
      Nat.sub(i, 1n)
    case Con{+h, t} False{}:
      pick_at(t, F32.is_gt((acc + h : F32), u), u, (acc + h : F32), 1n+i)

# a uniform number in [0, 1) from the 24 high bits of the hash of (seed, n)
def unif(+seed: U32, +n: U32) -> F32:
  (U32.to_f32(U32.shrn(T.mix32(U32.xor(seed, T.mix32(n))), 8n)) / 16777216.0 : F32)

def draw_row(+xs: List<&2, F32>, +u: F32) -> Nat:
  +w = row_w(xs, row_max(xs, (0.0 - 1000000.0 : F32)))
  pick_at(w, False{}, (u * row_sum(w, 0.0) : F32), 0.0, 0n)

def last_draw(+u: F32, r: TA.MRow<16n, 65n>) -> Nat:
  match r:
    case TA.MRow{m, row}:
      draw_row(row, u)

def fwd_draw(r: T.Fwd<P, Y(1n), S(1n)>, +u: F32, k: P -> @+i: Nat -> Gen) -> Gen:
  match r:
    case T.Fwd{p, y, s}:
      k(p, last_draw(u, TA.Mat.read_row(16n, 65n, y, 15)))

# i counts the characters drawn so far
def gen_d(k: Nat, p: P, +seed: U32, +i: U32, +win: List<&2, Nat>, +acc: List<&2, Nat>) -> Gen:
  match k:
    case 0n:
      Gen{p, List.reverse(&2, Nat, acc)}
    case 1n+q:
      fwd_draw(T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, win), unif(seed, i), p2 => +id => gen_d(q, p2, seed, (i + 1 : U32), List.append(&2, Nat, List.drop(&2, Nat, win, 1n), [id]), id <> acc))

# the position of byte c in the vocabulary (0 if absent)
def index_of(voc: List<&2, U32>, +c: U32, +i: Nat) -> Nat:
  match voc:
    case Nil{}:
      0n
    case Con{+v, t}:
      Bool.pick(Nat, U32.is_eq(v, c), i, index_of(t, c, 1n+i))

def byte_at(voc: List<&2, U32>, id: Nat) -> U32:
  match voc id:
    case Con{v, t} 0n:
      v
    case Con{v, t} 1n+q:
      byte_at(t, q)
    case Nil{} _:
      63

def ids_of(cs: List<&2, Char>, +voc: List<&2, U32>) -> List<&2, Nat>:
  match cs:
    case Nil{}:
      Nil{}
    case Con{Chr{u}, t}:
      index_of(voc, u, 0n) <> ids_of(t, voc)

def chars_of(ids: List<&2, Nat>, +voc: List<&2, U32>) -> List<&2, Char>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{id, t}:
      Chr{byte_at(voc, id)} <> chars_of(t, voc)

def nat_words(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{x, t}:
      " " ++ Nat.show(x) ++ nat_words(t)

def show_gen(+voc: List<&2, U32>, r: Gen) -> IO(Unit):
  match r:
    case Gen{p, +ids}:
      do IO<Unit>:
        IO.print("ids" ++ nat_words(ids))
        IO.print("text " ++ String.from_list(chars_of(ids, voc)))
        print_params("final", p)

def sample2(+n: Nat, +win: List<&2, Nat>, +voc: List<&2, U32>, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      show_gen(voc, gen(n, p, win, Nil{}))

# trains for k steps, then continues the last 16 characters of the prompt by n characters, greedily
def sample_v(+k: Nat, +n: Nat, +prompt: String, +voc: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, p)
    sample2(n, List.drop(&2, Nat, ids_of(String.to_list(prompt), voc), Nat.sub(String.length(prompt), 16n)), voc, tr)

def sample(+k: Nat, +n: Nat, +prompt: String) -> IO(Unit):
  do IO<Unit>:
    voc : List<&2, U32> <- T.read_at("data/tiny/vocab.txt", 0, 65)
    sample_v(k, n, prompt, voc)

def train_m(m: Maybe<&2, Nat>) -> IO(Unit):
  match m:
    case Some{k}:
      train(k)
    case None{}:
      IO.print("usage: tiny_llama train <steps>")

def sample_m(mk: Maybe<&2, Nat>, mn: Maybe<&2, Nat>, +prompt: String) -> IO(Unit):
  match mk mn:
    case Some{k} Some{n}:
      sample(k, n, prompt)
    case _ _:
      IO.print("usage: tiny_llama sample <steps> <chars> <prompt of at least 16 characters>")

def draw2(+n: Nat, +seed: U32, +win: List<&2, Nat>, +voc: List<&2, U32>, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      show_gen(voc, gen_d(n, p, seed, 0, win, Nil{}))

# trains for k steps, then continues the last 16 characters of the prompt by n characters, drawing each one
def draw_v(+k: Nat, +n: Nat, +seed: U32, +prompt: String, +voc: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, p)
    draw2(n, seed, List.drop(&2, Nat, ids_of(String.to_list(prompt), voc), Nat.sub(String.length(prompt), 16n)), voc, tr)

def draw(+k: Nat, +n: Nat, +seed: U32, +prompt: String) -> IO(Unit):
  do IO<Unit>:
    voc : List<&2, U32> <- T.read_at("data/tiny/vocab.txt", 0, 65)
    draw_v(k, n, seed, prompt, voc)

def draw_m(mk: Maybe<&2, Nat>, mn: Maybe<&2, Nat>, ms: Maybe<&2, Nat>, +prompt: String) -> IO(Unit):
  match mk mn ms:
    case Some{k} Some{n} Some{s}:
      draw(k, n, U32.from_nat(s), prompt)
    case _ _ _:
      IO.print("usage: tiny_llama draw <steps> <chars> <seed> <prompt of at least 16 characters>")

def usage() -> IO(Unit):
  IO.print("usage: tiny_llama grads | train <steps> | sample <steps> <chars> <prompt> | draw <steps> <chars> <seed> <prompt> | save <steps> <file> | resume <file> <from step> <steps>")

def dispatch_d(d: Bool, a: String, b: String, c: String, +e: String) -> IO(Unit):
  match d:
    case True{}:
      draw_m(Nat.read(a), Nat.read(b), Nat.read(c), e)
    case False{}:
      usage()

def dispatch_g(g: Bool) -> IO(Unit):
  match g:
    case True{}:
      do IO<Unit>:
        ids : List<&2, Nat> <- load_batch(0)
        grads(ids)
    case False{}:
      usage()

def dispatch_t(t: Bool, k: String) -> IO(Unit):
  match t:
    case True{}:
      train_m(Nat.read(k))
    case False{}:
      usage()

def resume_m(+path: String, mf: Maybe<&2, Nat>, mk: Maybe<&2, Nat>) -> IO(Unit):
  match mf mk:
    case Some{f} Some{k}:
      resume(path, f, k)
    case _ _:
      usage()

def dispatch_s(s: Bool, r: Bool, a: String, b: String, +c: String) -> IO(Unit):
  match s r:
    case True{} _:
      sample_m(Nat.read(a), Nat.read(b), c)
    case False{} True{}:
      resume_m(a, Nat.read(b), Nat.read(c))
    case _ _:
      usage()

def save_m(+path: String, mk: Maybe<&2, Nat>) -> IO(Unit):
  match mk:
    case Some{k}:
      save(k, path)
    case None{}:
      usage()

def dispatch_2(s: Bool, k: String, +path: String) -> IO(Unit):
  match s:
    case True{}:
      save_m(path, Nat.read(k))
    case False{}:
      usage()


def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Nil{}}}:
      dispatch_g(String.eq(mode, "grads"))
    case Con{prog, Con{mode, Con{k, Nil{}}}}:
      dispatch_t(String.eq(mode, "train"), k)
    case Con{prog, Con{mode, Con{k, Con{path, Nil{}}}}}:
      dispatch_2(String.eq(mode, "save"), k, path)
    case Con{prog, Con{+mode, Con{a, Con{b, Con{c, Nil{}}}}}}:
      dispatch_s(String.eq(mode, "sample"), String.eq(mode, "resume"), a, b, c)
    case Con{prog, Con{mode, Con{a, Con{b, Con{c, Con{e, Nil{}}}}}}}:
      dispatch_d(String.eq(mode, "draw"), a, b, c, e)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
