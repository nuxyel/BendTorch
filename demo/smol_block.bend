# One decoder block at SmolLM2-135M's sizes, trained with AdamW against a target under MSE: the workload where the
# row-wise layers and the products are big enough for parallelism to pay, against its PyTorch twin
# (reference/block_ref.py). The weights, the inputs and the targets come from a hash of each number's index, the same
# in both programs, so no files are read.
#
#   bend demo/smol_block.bend -o .scratch/smol_block
#   .scratch/smol_block +RTS --threads 16 -RTS train <steps>   # 1 + steps losses (bits); train_ms times the last <steps>
#
# The block is tiny_llama.bend's: Residual(Seq(RMSNorm, attention)) then Residual(Seq(RMSNorm, SwiGLU)). Width 576, 9
# query heads and 3 key/value heads of 64, RoPE base 100000, feed-forward 1536 (SmolLM2-135M's), on 8 sequences of 64
# positions (512 rows). The linear layers are torch/fast.bend's.

import Base
import ../torch/main.bend as T
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# the rows of a batch of b sequences of 64 positions
def R(b: Nat) -> Nat:
  Nat.mul(b, 64n)

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

# log2 of the most parallel blocks of each product: 256 blocks balance the P- and E-cores better than 16 (16 threads,
# per 3 steps: 763 -> 577 ms at 135M's sizes, 1840 -> 1313 ms at 360M's; NOTES 2026-10-17)
def PAR() -> Nat:
  8n

# the sizes: SmolLM2-135M's (scripts/block_bench.sh writes 360M's and 1.7B's into copies of these five defs)
def D() -> Nat:
  576n

def KVW() -> Nat:
  192n

def FF() -> Nat:
  1536n

def HKV() -> Nat:
  3n

def GRP() -> Nat:
  3n

def HQ() -> Nat:
  Nat.mul(HKV(), GRP())

# ---- attention ----

def KVP() -> Type:
  T.Two<M(KVW(), D()), M(KVW(), D())>

def KVY(b: Nat) -> Type:
  T.Two<A(b, KVW()), A(b, KVW())>

def KVS(b: Nat) -> Type:
  T.Two<A(b, D()), A(b, D())>

def KVL(~b: Nat) -> T.Layer<KVP, A(b, D()), KVY(b), KVS(b)>:
  T.Fork(~M(KVW(), D()), ~M(KVW(), D()), ~A(b, D()), ~A(b, KVW()), ~A(b, KVW()), ~A(b, D()), ~A(b, D()), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~F.LinearNBF(R(b), D(), KVW(), PAR()), ~F.LinearNBF(R(b), D(), KVW(), PAR()))

def AX(b: Nat) -> Type:
  T.Two<A(b, D()), KVY(b)>

def QKVP() -> Type:
  T.Two<M(D(), D()), KVP>

def QKVS(b: Nat) -> Type:
  T.Two<A(b, D()), KVS(b)>

def QKV(~b: Nat) -> T.Layer<QKVP, A(b, D()), AX(b), QKVS(b)>:
  T.Fork(~M(D(), D()), ~KVP, ~A(b, D()), ~A(b, D()), ~KVY(b), ~A(b, D()), ~KVS(b), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~F.LinearNBF(R(b), D(), D(), PAR()), ~KVL(~b))

def RopeK(~b: Nat) -> T.Layer<T.Two<Unit, Unit>, KVY(b), KVY(b), T.Two<Unit, Unit>>:
  T.Both(~Unit, ~Unit, ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~Unit, ~Unit, ~F.RoPE(R(b), KVW(), 64n, HKV(), 32n, 100000.0, {==}), ~T.Id(~A(b, KVW())))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(~b: Nat) -> T.Layer<RU, AX(b), AX(b), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, D()), ~KVY(b), ~A(b, D()), ~KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~F.RoPE(R(b), D(), 64n, HQ(), 32n, 100000.0, {==}), ~RopeK(~b))

def CoreP() -> Type:
  T.Two<Unit, M(D(), D())>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, D())>

def Core(~b: Nat) -> T.Layer<CoreP, AX(b), A(b, D()), CoreS(b)>:
  T.Seq(~Unit, ~M(D(), D()), ~AX(b), ~A(b, D()), ~A(b, D()), ~T.AttnS, ~A(b, D()), ~F.Attention(R(b), D(), KVW(), b, 64n, HKV(), GRP(), 64n, 0.125, {==}, {==}, {==}), ~F.LinearNBF(R(b), D(), D(), PAR()))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS(b: Nat) -> Type:
  T.Two<RU, CoreS(b)>

def RopeCore(~b: Nat) -> T.Layer<RCP, AX(b), A(b, D()), RCS(b)>:
  T.Seq(~RU, ~CoreP, ~AX(b), ~AX(b), ~A(b, D()), ~RU, ~CoreS(b), ~Ropes(~b), ~Core(~b))

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS(b: Nat) -> Type:
  T.Two<QKVS(b), RCS(b)>

def Att(~b: Nat) -> T.Layer<AttP, A(b, D()), A(b, D()), AttS(b)>:
  T.Seq(~QKVP, ~RCP, ~A(b, D()), ~AX(b), ~A(b, D()), ~QKVS(b), ~RCS(b), ~QKV(~b), ~RopeCore(~b))

def NAttP() -> Type:
  T.Two<M(1n, D()), AttP>

def NAttS(b: Nat) -> Type:
  T.Two<A(b, D()), AttS(b)>

def NAtt(~b: Nat) -> T.Layer<NAttP, A(b, D()), A(b, D()), NAttS(b)>:
  T.Seq(~M(1n, D()), ~AttP, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~AttS(b), ~F.RMSNorm(R(b), D(), 0.00001), ~Att(~b))

# ---- the feed-forward block: down(silu(gate(x)) * up(x)) ----

def GateS(b: Nat) -> Type:
  T.Two<A(b, D()), A(b, FF())>

def Gate(~b: Nat) -> T.Layer<T.Two<M(FF(), D()), Unit>, A(b, D()), A(b, FF()), GateS(b)>:
  T.Seq(~M(FF(), D()), ~Unit, ~A(b, D()), ~A(b, FF()), ~A(b, FF()), ~A(b, D()), ~A(b, FF()), ~F.LinearNBF(R(b), D(), FF(), PAR()), ~F.SiLU(R(b), FF()))

def GUP() -> Type:
  T.Two<T.Two<M(FF(), D()), Unit>, M(FF(), D())>

def GUY(b: Nat) -> Type:
  T.Two<A(b, FF()), A(b, FF())>

def GUS(b: Nat) -> Type:
  T.Two<GateS(b), A(b, D())>

def GU(~b: Nat) -> T.Layer<GUP, A(b, D()), GUY(b), GUS(b)>:
  T.Fork(~T.Two<M(FF(), D()), Unit>, ~M(FF(), D()), ~A(b, D()), ~A(b, FF()), ~A(b, FF()), ~GateS(b), ~A(b, D()), ~(m => F.Mat.dup(R(b), D(), m)), ~(x => y => F.Mat.sum2(R(b), D(), x, y)), ~Gate(~b), ~F.LinearNBF(R(b), D(), FF(), PAR()))

def GLUS(b: Nat) -> Type:
  T.Two<GUS(b), GUY(b)>

def GLU(~b: Nat) -> T.Layer<T.Two<GUP, Unit>, A(b, D()), A(b, FF()), GLUS(b)>:
  T.Seq(~GUP, ~Unit, ~A(b, D()), ~GUY(b), ~A(b, FF()), ~GUS(b), ~GUY(b), ~GU(~b), ~F.Mul2(R(b), FF()))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(D(), FF()))

def MlpS(b: Nat) -> Type:
  T.Two<GLUS(b), A(b, FF())>

def Mlp(~b: Nat) -> T.Layer<MlpP, A(b, D()), A(b, D()), MlpS(b)>:
  T.Seq(~T.Two<GUP, Unit>, ~M(D(), FF()), ~A(b, D()), ~A(b, FF()), ~A(b, D()), ~GLUS(b), ~A(b, FF()), ~GLU(~b), ~F.LinearNBF(R(b), FF(), D(), PAR()))

def NMlpP() -> Type:
  T.Two<M(1n, D()), MlpP>

def NMlpS(b: Nat) -> Type:
  T.Two<A(b, D()), MlpS(b)>

def NMlp(~b: Nat) -> T.Layer<NMlpP, A(b, D()), A(b, D()), NMlpS(b)>:
  T.Seq(~M(1n, D()), ~MlpP, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~MlpS(b), ~F.RMSNorm(R(b), D(), 0.00001), ~Mlp(~b))

# ---- the block ----

def P() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def S(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def Block(~b: Nat) -> T.Layer<P, A(b, D()), A(b, D()), S(b)>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)), ~F.Residual(~R(b), ~D(), ~NAttP, ~NAttS(b), ~NAtt(~b)), ~F.Residual(~R(b), ~D(), ~NMlpP, ~NMlpS(b), ~NMlp(~b)))

def mse(y: A(8n, D()), t: A(8n, D())) -> T.LossOut<A(8n, D())>:
  F.MSE(R(8n), D(), y, t)

# ---- numbers from a hash of their index: (h mod 65536 - 32768) · scale, exact in F32 (reference/block_ref.py's gen) ----

def hsh(+i: U32, +s: U32) -> U32:
  +x0 = ((i * 2654435761 : U32) + (s * 40503 : U32) : U32)
  +x1 = U32.xor(x0, U32.div(x0, 32768))
  +x2 = (x1 * 2246822519 : U32)
  U32.xor(x2, U32.div(x2, 8192))

def val(+i: U32, +s: U32, +sc: F32) -> F32:
  ((U32.to_f32(U32.mod(hsh(i, s), 65536)) - 32768.0 : F32) * sc : F32)

def gen_l(k: Nat, +i: U32, +s: U32, +sc: F32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n:
      Array.set(F32, a, i, val(i, s, sc))
    case 1n+q:
      gen_l(q, (i + 1 : U32), s, sc, Array.set(F32, a, i, val(i, s, sc)))

def gen(+r: Nat, +c: Nat, +s: U32, +sc: F32) -> TA.Mat<r, c>:
  TA.Mat{gen_l(Nat.sub(Nat.mul(r, c), 1n), 0, s, sc, T.zeros_a(Nat.mul(r, c)))}

def ones(+c: Nat) -> TA.Mat<1n, c>:
  TA.Mat{Array.new(F32, T.depth(c), 1.0)}

# weights in about ±1/64 (scale 2^-21), inputs and targets in about ±1 (scale 2^-15)
def w(+r: Nat, +c: Nat, +s: U32) -> TA.Mat<r, c>:
  gen(r, c, s, 0.000000476837158203125)

def params() -> P:
  T.Two{T.Two{T.Two{Unit{}, T.Two{ones(D()), T.Two{T.Two{w(D(), D(), 1), T.Two{w(KVW(), D(), 2), w(KVW(), D(), 3)}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, w(D(), D(), 4)}}}}}, Unit{}}, T.Two{T.Two{Unit{}, T.Two{ones(D()), T.Two{T.Two{T.Two{T.Two{w(FF(), D(), 5), Unit{}}, w(FF(), D(), 6)}, Unit{}}, w(D(), FF(), 7)}}}, Unit{}}}

def input(+step: U32) -> A(8n, D()):
  gen(R(8n), D(), (1000 + step : U32), 0.000030517578125)

def target(+step: U32) -> A(8n, D()):
  gen(R(8n), D(), (2000 + step : U32), 0.000030517578125)

# ---- training ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def then(tr: T.Train<P>, k: @+l: F32 -> T.Train<P> -> IO(T.Train<P>)) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, st, p1, p2, +l}:
      k(l, T.Train{p, g, st, p1, p2, l})

# k steps from input `step`, printing each step's loss
def tloop(k: Nat, +step: U32, +opt: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  match k:
    case 0n:
      IO.pure(T.Train<P>, tr)
    case 1n+q:
      then(T.Train.step(~P, ~A(8n, D()), ~A(8n, D()), ~S(8n), ~A(8n, D()), ~Block(~8n), ~mse, opt, tr, input(step), target(step)), +l => tr2 => IO.bind(Unit, T.Train<P>, T.F32.print("loss", l), u => tloop(q, (step + 1 : U32), opt, tr2)))

# one step to warm up (the optimizer's buffers are made then), then k timed steps
def train(+k: Nat, +opt: T.OptCfg) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(opt)
    tr : T.Train<P> <- tloop(1n, 0, opt, T.Train.new(~P, ~A(8n, D()), ~A(8n, D()), ~S(8n), ~Block(~8n), opt, params()))
    t0 : Nat <- IO.now()
    tr2 : T.Train<P> <- tloop(k, 1, opt, tr)
    t1 : Nat <- IO.now()
    IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))

def usage() -> IO(Unit):
  IO.print("usage: smol_block train <steps> [sgd|adamw]")

def train_m(mk: Maybe<&2, Nat>) -> IO(Unit):
  match mk:
    case Some{k}:
      train(k, cfg())
    case None{}:
      usage()

def dispatch_t(t: Bool, k: String) -> IO(Unit):
  match t:
    case True{}:
      train_m(Nat.read(k))
    case False{}:
      usage()

def train_opt(ok: Bool, sgd: Bool, mk: Maybe<&2, Nat>) -> IO(Unit):
  match ok sgd mk:
    case True{} True{} Some{k}: train(k, T.CSGD{0.001})
    case True{} False{} Some{k}: train(k, cfg())
    case _ _ _: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Con{k, Nil{}}}}:
      dispatch_t(String.eq(mode, "train"), k)
    case Con{prog, Con{mode, Con{k, Con{+opt, Nil{}}}}}:
      train_opt(Bool.and(String.eq(mode, "train"), Bool.or(String.eq(opt, "sgd"), String.eq(opt, "adamw"))), String.eq(opt, "sgd"), Nat.read(k))
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
