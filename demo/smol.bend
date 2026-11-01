# SmolLM2-135M in BendTorch's layers, fine-tuned for a few steps against its PyTorch twin (reference/smol_ref.py, which
# also writes the weights and the batches into data/smol135/; scripts/smol.sh 360m writes the 360M's copy).
#
#   bend demo/smol.bend -o .scratch/smol
#   .scratch/smol +RTS --threads 16 -RTS loss              # the loss on the first batch
#   .scratch/smol +RTS --threads 16 -RTS loss <file>       # the same, the weights from Hugging Face's model.safetensors
#   .scratch/smol +RTS --threads 16 -RTS train <steps>     # AdamW steps: the loss before each update; train_ms for the steps
#   .scratch/smol +RTS --threads 16 -RTS train <steps> <file>  # the same from Hugging Face's model.safetensors
#   .scratch/smol +RTS --threads 16 -RTS grads <file>      # the loss on the first batch; every gradient into <file>, dump order
#   .scratch/smol +RTS --threads 16 -RTS finetune <k> <e>  # k steps on ft_train.bin; the held-out loss (ft_eval.bin) every e
#   .scratch/smol +RTS --threads 16 -RTS finetune <k> <e> <out>  # the same, then the weights into <out> (init.bin's layout)
#   .scratch/smol +RTS --threads 16 -RTS finetunest <k> <e> <file>  # finetune from Hugging Face's model.safetensors
#   .scratch/smol +RTS --threads 16 -RTS gen <n> <weights> <ids>  # n tokens after the prompt <ids> (comma-separated), greedy
#   .scratch/smol +RTS --threads 16 -RTS draw <n> <seed> <weights> <ids>  # the same, each token drawn at temperature 0.8
#   .scratch/smol +RTS --threads 16 -RTS genkv <n> <weights> <ids>        # gen with a key/value cache: one position a token
#   .scratch/smol +RTS --threads 16 -RTS drawkv <n> <seed> <weights> <ids>  # draw with the cache
#   .scratch/smol +RTS --threads 16 -RTS kvcheck <n> <weights> <ids>      # both passes in step: their logits compared bit for bit
#
# The model: the token embedding, LAYERS() decoder blocks (smol_block.bend's: Residual(Seq(RMSNorm, attention)) then
# Residual(Seq(RMSNorm, SwiGLU))), the final RMSNorm, and the output head, which shares the embedding's matrix (Tie,
# as SmolLM2 ties them). Batches of 2 sequences of TT() = 128 positions (256 rows). The layers and the loss are
# torch/fast.bend's (its RMSNorm, RoPE, Attention and CrossEntropy are main.bend's, its Embedding main.bend's with the
# step in blocks), so the import line alone switches them to torch/gpu.bend's (make smol-gpu). Weights are in Layer.dump order (reference/smol_layout.txt names each tensor).

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# where the weights and the batches are
def DATA() -> String:
  "data/smol135/"

# positions per sequence
def TT() -> Nat:
  128n

def V() -> Nat:
  49152n

def LAYERS() -> Nat:
  30n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# the rows of a batch of b sequences of TT() positions
def R(b: Nat) -> Nat:
  Nat.mul(b, TT())

def A(b: Nat, c: Nat) -> Type:
  TA.Mat<R(b), c>

# log2 of the most parallel blocks of each product: 256 blocks balance the P- and E-cores better than 16 (16 threads,
# per 3 steps: 763 -> 577 ms at 135M's sizes, 1840 -> 1313 ms at 360M's; NOTES 2026-10-17)
def PAR() -> Nat:
  8n

# the sizes: SmolLM2-135M's (scripts/smol.sh writes 360M's into a copy of these defs and of the stack below)
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
  T.Both(~Unit, ~Unit, ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~A(b, KVW()), ~Unit, ~Unit, ~F.RoPE(R(b), KVW(), TT(), HKV(), 32n, 100000.0, {==}), ~T.Id(~A(b, KVW())))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(~b: Nat) -> T.Layer<RU, AX(b), AX(b), RU>:
  T.Both(~Unit, ~T.Two<Unit, Unit>, ~A(b, D()), ~KVY(b), ~A(b, D()), ~KVY(b), ~Unit, ~T.Two<Unit, Unit>, ~F.RoPE(R(b), D(), TT(), HQ(), 32n, 100000.0, {==}), ~RopeK(~b))

def CoreP() -> Type:
  T.Two<Unit, M(D(), D())>

def CoreS(b: Nat) -> Type:
  T.Two<T.AttnS, A(b, D())>

def Core(~b: Nat) -> T.Layer<CoreP, AX(b), A(b, D()), CoreS(b)>:
  T.Seq(~Unit, ~M(D(), D()), ~AX(b), ~A(b, D()), ~A(b, D()), ~T.AttnS, ~A(b, D()), ~F.Attention(R(b), D(), KVW(), b, TT(), HKV(), GRP(), 64n, 0.125, {==}, {==}, {==}), ~F.LinearNBF(R(b), D(), D(), PAR()))

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

def BP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BS(b: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b)), T.ResidualType(NMlpS(b))>

def Block(~b: Nat) -> T.Layer<BP, A(b, D()), A(b, D()), BS(b)>:
  T.Seq(~T.ResidualType(NAttP), ~T.ResidualType(NMlpP), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~T.ResidualType(NAttS(b)), ~T.ResidualType(NMlpS(b)), ~F.Residual(~R(b), ~D(), ~NAttP, ~NAttS(b), ~NAtt(~b)), ~F.Residual(~R(b), ~D(), ~NMlpP, ~NMlpS(b), ~NMlp(~b)))

# ---- Box: a layer whose parameters and saved values are one boxed word each ----

# Bend lays a value of nested records out flat, a word a field, and a def or a continuation holds at most 247 words
# (bend guide, SHADERS.md): eight blocks' parameters and saved values held across a call are more. A datatype that
# mentions itself is boxed (one word), so Boxed<A> carries A behind one pointer; Chain never occurs.
type Boxed<-A: Type> is Type:
  Boxed{a: A}
  Chain{next: Boxed<A>}

def Boxed.get(-A: Type, b: Boxed<A>) -> A:
  match b:
    case Boxed{a}: a
    case Chain{n}: Boxed.get(A, n)

def Boxed.put(-A: Type, a: A) -> Boxed<A>:
  Boxed{a}

def box_f2(-P: Type, -Y: Type, -S: Type, r: T.Fwd<P, Y, S>) -> T.Fwd<Boxed<P>, Y, Boxed<S>>:
  match r:
    case T.Fwd{p, y, s}: T.Fwd{Boxed{p}, y, Boxed{s}}

def box_b2(-P: Type, -X: Type, r: T.Bwd<P, X>) -> T.Bwd<Boxed<P>, X>:
  match r:
    case T.Bwd{p, dx, g}: T.Bwd{Boxed{p}, dx, Boxed{g}}

def box_s2(-P: Type, r: T.Step<P>) -> T.Step<Boxed<P>>:
  match r:
    case T.Step{p, g, st}: T.Step{Boxed{p}, Boxed{g}, T.OptState.map(P, Boxed<P>, st, x => Boxed{x}, x => Boxed{x})}

def box_d2(-P: Type, r: T.PL<P>) -> T.PL<Boxed<P>>:
  match r:
    case T.PL{p, xs}: T.PL{Boxed{p}, xs}

def box_l2(-P: Type, r: T.PA<P>) -> T.PA<Boxed<P>>:
  match r:
    case T.PA{p, xs, off}: T.PA{Boxed{p}, xs, off}

def box_t2(-P: Type, r: T.PA<P>) -> T.PA<Boxed<P>>:
  box_l2(P, r)

def Box(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~l: T.Layer<P, X, Y, S>) -> T.Layer<Boxed<P>, X, Y, Boxed<S>>:
  T.Layer{p => x => box_f2(P, Y, S, T.Layer.fwd(P, X, Y, S, l, Boxed.get(P, p), x)), p => s => dy => g => box_b2(P, X, T.Layer.bwd(P, X, Y, S, l, Boxed.get(P, p), Boxed.get(S, s), dy, Boxed.get(P, g))), op => p => g => st => box_s2(P, T.Layer.step(P, X, Y, S, l, op, Boxed.get(P, p), Boxed.get(P, g), T.OptState.map(Boxed<P>, P, st, x => Boxed.get(P, x), x => Boxed.get(P, x)))), u => Boxed{T.Layer.zeros(P, X, Y, S, l)}, p => box_d2(P, T.Layer.dump(P, X, Y, S, l, Boxed.get(P, p))), p => xs => off => box_l2(P, T.Layer.load(P, X, Y, S, l, Boxed.get(P, p), xs, off)), p => xs => off => box_t2(P, T.Layer.store(P, X, Y, S, l, Boxed.get(P, p), xs, off)), u => T.Layer.count(P, X, Y, S, l)}

# ---- generation with a key/value cache: one position a step (modes genkv and drawkv) ----
#
# gen and draw run the whole window again for every token. A step here runs one position of one sequence: the
# token's embedding, then each block on that one row, keeping the keys (after RoPE, at the token's position) and the
# values of every position written so far in the block's cache, then the final norm and the head on that row. The
# window starts at position 0 and never slides, so a position's keys and values never change once written.
#
# Every number of a step is the number of the same row in the whole-window pass: the products of one row run
# tensor-array's scalar loop, whose bits are those of its cells of 8 (tensor-array's note on gemm_t8); RMSNorm, RoPE,
# SiLU and the sums work row by row with main.bend's own loops; attention takes main.bend's own row of scores and
# softmax (qrow, prow2) over positions 0..pos, and its sum with the values (pmul) stops at pos where the whole-window
# pass goes on over the probabilities after pos, which are +0 there and add nothing (a sum that starts at +0 never
# becomes -0, and x + 0 is x). kvcheck compares the two passes' logits bit for bit.

# one block's cache: keys and values, TT() rows of KVW() numbers, row p for position p
type KV is Type:
  KV{k: Array<F32>, v: Array<F32>}

# what a step carries through the blocks: the row (1 x D()), the caches of the blocks still ahead (in block order), the
# caches of the blocks already passed (newest first), and RoPE's table of cos and sin
type KS is Type:
  KS{x: TA.Mat<1n, D()>, ins: List<&1, KV>, outs: List<&1, KV>, tb: T.CS}

# a part of the model after a step: its parameters, handed back, and the carried state
type KR<-P: Type> is Type:
  KR{p: P, s: KS}

def M1(c: Nat) -> Type:
  TA.Mat<1n, c>

def kf3(-P: Type, -Y: Type, -S: Type, -Z: Type, r: T.Fwd<P, Y, S>, k: P -> Y -> S -> Z) -> Z:
  match r:
    case T.Fwd{p, y, s}: k(p, y, s)

# a part of the model made of two parts, one after the other (as T.Seq)
def kseq2(-A: Type, -C: Type, a: A, r: KR<C>) -> KR<T.Two<A, C>>:
  match r:
    case KR{c, s}: KR{T.Two{a, c}, s}

def kseq(-A: Type, -C: Type, r: KR<A>, f: KS -> KR<C>) -> KR<T.Two<A, C>>:
  match r:
    case KR{a, s}: kseq2(A, C, a, f(s))

# the layers on one row; the products and RMSNorm hand their input back as the saved value
def klin(+i: Nat, +o: Nat, w: M(o, i), x: M1(i)) -> T.Fwd<M(o, i), M1(o), M1(i)>:
  T.Layer.fwd(M(o, i), M1(i), M1(o), M1(i), F.LinearNBF(1n, i, o, PAR()), w, x)

def krms(g: M(1n, D()), x: M1(D())) -> T.Fwd<M(1n, D()), M1(D()), M1(D())>:
  T.Layer.fwd(M(1n, D()), M1(D()), M1(D()), M1(D()), F.RMSNorm(1n, D(), 0.00001), g, x)

def ksilu(x: M1(FF())) -> T.Fwd<Unit, M1(FF()), M1(FF())>:
  T.Layer.fwd(Unit, M1(FF()), M1(FF()), M1(FF()), F.SiLU(1n, FF()), Unit{}, x)

def kmul(a: M1(FF()), b: M1(FF())) -> T.Fwd<Unit, M1(FF()), T.Two<M1(FF()), M1(FF())>>:
  T.Layer.fwd(Unit, T.Two<M1(FF()), M1(FF())>, M1(FF()), T.Two<M1(FF()), M1(FF())>, F.Mul2(1n, FF()), Unit{}, T.Two{a, b})

# RoPE on one row at position pos: pair e is head e / 32, column c = e mod 32 of it, rotated by table entry pos*32 + c,
# with main.bend's rope_p2, as rope_one rotates row pos of the whole window
def rope_pair(+e: U32, +pos: U32, st: T.RS3) -> T.RS3:
  match st:
    case T.RS3{x, cs, sn}:
      +c = U32.mod(e, 32)
      +i1 = ((U32.div(e, 32) * 64 : U32) + c : U32)
      T.rope_p2(True{}, i1, (i1 + 32 : U32), ((pos * 32 : U32) + c : U32), cs, sn, Array.get(F32, x, i1))

def rope_row(left: Nat, +e: U32, +pos: U32, st: T.RS3) -> T.RS3:
  match left:
    case 0n:
      rope_pair(e, pos, st)
    case 1n+q:
      rope_row(q, (e + 1 : U32), pos, rope_pair(e, pos, st))

# the table of every position of the window: RoPE's own (rope_tab, as rope_go builds it at every call)
def ktable() -> T.CS:
  T.rope_tab(Nat.sub(Nat.mul(TT(), 32n), 1n), 0, 32, 100000.0, T.CS{T.zeros_a(Nat.mul(TT(), 32n)), T.zeros_a(Nat.mul(TT(), 32n))})

def krope_then(-Z: Type, st: T.RS3, k: Array<F32> -> T.CS -> Z) -> Z:
  match st:
    case T.RS3{x, cs, sn}: k(x, T.CS{cs, sn})

# h heads of 64 numbers of one row at position pos
def krope(+h: Nat, +pos: Nat, x: Array<F32>, tb: T.CS) -> T.RS3:
  match tb:
    case T.CS{cs, sn}:
      rope_row(Nat.sub(Nat.mul(h, 32n), 1n), 0, U32.from_nat(pos), T.RS3{x, cs, sn})

# row pos of a cache, from a row of KVW() numbers
def kput(+pos: Nat, row: Array<F32>, c: Array<F32>) -> Array<F32>:
  T.a2_b(T.copy_n(Nat.sub(KVW(), 1n), 0, U32.from_nat(Nat.mul(pos, KVW())), c, Array.get(F32, row, 0)))

# the query row, the caches and the output row (1 x D())
type KA is Type:
  KA{q: Array<F32>, k: Array<F32>, v: Array<F32>, o: Array<F32>}

def ka_fin(q: Array<F32>, k: Array<F32>, r: T.PO) -> KA:
  match r:
    case T.PO{p, v, o}: KA{q, k, v, o}

# head hi's 64 outputs into its columns of o: the probabilities are row pos of the whole window's (prow2 over qrow's
# scores of keys 0..pos, scale 1/8 as Core's), and pmul sums them with values 0..pos
def khead2(+pos: Nat, +hc: U32, +kc: U32, v: Array<F32>, o: Array<F32>, r: T.PR) -> KA:
  match r:
    case T.PR{q, k, p, n}:
      ka_fin(q, k, T.pmul(63n, 0, 64, 1n+pos, U32.from_nat(KVW()), kc, U32.from_nat(D()), hc, o, T.dot0(1n+pos, 0, kc, 1, U32.from_nat(KVW()), p, v)))

# head hi reads its own columns of the query and those of key/value head kv_of(hi, GRP()), as head_cols
def khead(+pos: Nat, +hi: Nat, s: KA) -> KA:
  match s:
    case KA{q, k, v, o}:
      +hc = U32.from_nat(Nat.mul(hi, 64n))
      +kc = U32.from_nat(Nat.mul(T.kv_of(hi, GRP()), 64n))
      khead2(pos, hc, kc, v, o, T.prow2(0.125, 0, T.zeros_a(1n+pos), T.qrow(pos, 64n, 0, hc, U32.from_nat(KVW()), kc, q, k)))

# heads hi.., left of them
def kheads(left: Nat, +hi: Nat, +pos: Nat, s: KA) -> KA:
  match left:
    case 0n:
      s
    case 1n+n:
      kheads(n, 1n+hi, pos, khead(pos, hi, s))

def ka_then(-Z: Type, s: KA, k: Array<F32> -> Array<F32> -> Array<F32> -> Array<F32> -> Z) -> Z:
  match s:
    case KA{q, k2, v, o}: k(q, k2, v, o)

# after the attention half: the output projection, the row with the residual added, the cache, the table
type KO is Type:
  KO{wo: M(D(), D()), y: M1(D()), c: KV, tb: T.CS}

# RoPE on q and k, k and v into row pos of the cache, attention over positions 0..pos, the output projection and the
# residual (x + the output, as Add2)
def katt_core(+pos: Nat, wo: M(D(), D()), x: M1(D()), q: M1(D()), k: M1(KVW()), v: M1(KVW()), c: KV, tb: T.CS) -> KO:
  match q k v c:
    case TA.Mat{qa} TA.Mat{ka} TA.Mat{va} KV{ck, cv}:
      krope_then(KO, krope(HQ(), pos, qa, tb), qr => tb2 => krope_then(KO, krope(HKV(), pos, ka, tb2), kr => tb3 => ka_then(KO, kheads(HQ(), 0n, pos, KA{qr, kput(pos, kr, ck), kput(pos, va, cv), T.zeros_a(D())}), q3 => k3 => v3 => o => kf3(M(D(), D()), M1(D()), M1(D()), KO, klin(D(), D(), wo, TA.Mat{o}), wo2 => y => s => KO{wo2, F.Mat.sum2(1n, D(), x, y), KV{k3, v3}, tb3}))))

def katt_fin(g: M(1n, D()), wq: M(D(), D()), wk: M(KVW(), D()), wv: M(KVW(), D()), ins: List<&1, KV>, outs: List<&1, KV>, r: KO) -> KR<T.ResidualType(NAttP)>:
  match r:
    case KO{wo, y, c, tb}:
      KR{T.Two{T.Two{Unit{}, T.Two{g, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{T.Two{Unit{}, T.Two{Unit{}, Unit{}}}, T.Two{Unit{}, wo}}}}}, Unit{}}, KS{y, ins, c <> outs, tb}}

def katt2(pa: T.ResidualType(NAttP), +pos: Nat, x: M1(D()), c: KV, ins: List<&1, KV>, outs: List<&1, KV>, tb: T.CS) -> KR<T.ResidualType(NAttP)>:
  match pa:
    case T.Two{T.Two{u1, T.Two{g, T.Two{T.Two{wq, T.Two{wk, wv}}, T.Two{ru, T.Two{u2, wo}}}}}, u3}:
      kf3(M(1n, D()), M1(D()), M1(D()), KR<T.ResidualType(NAttP)>, krms(g, x), g2 => h => x2 => kf3(M(D(), D()), M1(D()), M1(D()), KR<T.ResidualType(NAttP)>, klin(D(), D(), wq, h), wq2 => q => h2 => kf3(M(KVW(), D()), M1(KVW()), M1(D()), KR<T.ResidualType(NAttP)>, klin(D(), KVW(), wk, h2), wk2 => k => h3 => kf3(M(KVW(), D()), M1(KVW()), M1(D()), KR<T.ResidualType(NAttP)>, klin(D(), KVW(), wv, h3), wv2 => v => h4 => katt_fin(g2, wq2, wk2, wv2, ins, outs, katt_core(pos, wo, x2, q, k, v, c, tb))))))

# the attention half of a block (its residual included) on the row, with the block's cache (the first of ins)
def katt(pa: T.ResidualType(NAttP), +pos: Nat, s: KS) -> KR<T.ResidualType(NAttP)>:
  match s:
    case KS{x, Con{c, ins}, outs, tb}:
      katt2(pa, pos, x, c, ins, outs, tb)
    case KS{x, Nil{}, outs, tb}:
      KR{pa, KS{x, Nil{}, outs, tb}}

# the feed-forward half (its residual included): x + down(silu(gate(h)) * up(h)), h = RMSNorm(x)
def kmlp(pm: T.ResidualType(NMlpP), s: KS) -> KR<T.ResidualType(NMlpP)>:
  match pm s:
    case T.Two{T.Two{u1, T.Two{g, T.Two{T.Two{T.Two{T.Two{wg, u2}, wu}, u3}, wd}}}, u4} KS{x, ins, outs, tb}:
      kf3(M(1n, D()), M1(D()), M1(D()), KR<T.ResidualType(NMlpP)>, krms(g, x), g2 => h => x2 => kf3(M(FF(), D()), M1(FF()), M1(D()), KR<T.ResidualType(NMlpP)>, klin(D(), FF(), wg, h), wg2 => a => h2 => kf3(M(FF(), D()), M1(FF()), M1(D()), KR<T.ResidualType(NMlpP)>, klin(D(), FF(), wu, h2), wu2 => b => h3 => kf3(Unit, M1(FF()), M1(FF()), KR<T.ResidualType(NMlpP)>, ksilu(a), e1 => a2 => sa => kf3(Unit, M1(FF()), T.Two<M1(FF()), M1(FF())>, KR<T.ResidualType(NMlpP)>, kmul(a2, b), e2 => m => sm => kf3(M(D(), FF()), M1(D()), M1(FF()), KR<T.ResidualType(NMlpP)>, klin(FF(), D(), wd, m), wd2 => y => sy => KR{T.Two{T.Two{Unit{}, T.Two{g2, T.Two{T.Two{T.Two{T.Two{wg2, Unit{}}, wu2}, Unit{}}, wd2}}}, Unit{}}, KS{F.Mat.sum2(1n, D(), x2, y), ins, outs, tb}}))))))

def kblock(p: BP, +pos: Nat, s: KS) -> KR<BP>:
  match p:
    case T.Two{pa, pm}:
      kseq(T.ResidualType(NAttP), T.ResidualType(NMlpP), katt(pa, pos, s), s2 => kmlp(pm, s2))

def kbox(r: KR<BP>) -> KR<Boxed<BP>>:
  match r:
    case KR{p, s}: KR{Boxed{p}, s}

# a block's keys (after RoPE) and values from what its attention saved in a whole-window forward pass (AttnS): row p
# for position p, TT() rows of KVW() numbers, what a step's cache holds. The rows after the prompt hold the padding's
# and are written over before any step reads them.
def kget(s: BS(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{T.Two{T.Two{u1, T.Two{xs, T.Two{qkvs, T.Two{ru, T.Two{T.AttnS{q, k, v}, os}}}}}, u2}, ms}:
      KV{k, v} <> acc

# ---- the safetensors tensors of block i (model.layers.i), in Layer.dump order; LStack reads each block with these ----

def st_layer(+i: Nat) -> List<&2, L.Entry>:
  +pre = "model.layers." ++ Nat.show(i) ++ "."
  +d = U32.from_nat(D())
  +kv = U32.from_nat(KVW())
  +ff = U32.from_nat(FF())
  [L.Entry{pre ++ "input_layernorm.weight", [d]}, L.Entry{pre ++ "self_attn.q_proj.weight", [d, d]}, L.Entry{pre ++ "self_attn.k_proj.weight", [kv, d]}, L.Entry{pre ++ "self_attn.v_proj.weight", [kv, d]}, L.Entry{pre ++ "self_attn.o_proj.weight", [d, d]}, L.Entry{pre ++ "post_attention_layernorm.weight", [d]}, L.Entry{pre ++ "mlp.gate_proj.weight", [ff, d]}, L.Entry{pre ++ "mlp.up_proj.weight", [ff, d]}, L.Entry{pre ++ "mlp.down_proj.weight", [d, ff]}]

def st_layers(k: Nat, +i: Nat) -> List<&2, L.Entry>:
  match k:
    case 0n: Nil{}
    case 1n+q: List.append(&2, L.Entry, st_layer(i), st_layers(q, 1n+i))

# ---- the stack of blocks: a balanced tree of Seq over LAYERS() blocks, read in order (block 0 first) ----

# A template takes only arguments fixed when the program compiles, and Bend instantiates templates 64 levels deep at
# most, so the stack is neither a recursion on its depth nor a chain of 30 Seq: scripts/smol.sh writes one def per
# subtree size (StackN = Seq(StackN/2, StackN - N/2), Stack1 a block), between the two markers.
# ---- stack: begin ----
def StP1() -> Type:
  Boxed<BP>

def StS1(b: Nat) -> Type:
  Boxed<BS(b)>

def Stack1(~b: Nat) -> T.Layer<StP1, A(b, D()), A(b, D()), StS1(b)>:
  Box(~BP, ~A(b, D()), ~A(b, D()), ~BS(b), ~Block(~b))

def KStack1(p: StP1, +pos: Nat, s: KS) -> KR<StP1>:
  kbox(kblock(Boxed.get(BP, p), pos, s))

def KGet1(s: StS1(1n), acc: List<&1, KV>) -> List<&1, KV>:
  kget(Boxed.get(BS(1n), s), acc)

def LStack1(+path: String, +h: L.Header, +i: Nat, p: StP1) -> IO(StP1):
  L.piece(~StP1, ~A(2n, D()), ~A(2n, D()), ~StS1(2n), ~Stack1(~2n), path, h, st_layer(i), p)

def StP2() -> Type:
  T.Two<StP1, StP1>

def StS2(b: Nat) -> Type:
  T.Two<StS1(b), StS1(b)>

def Stack2(~b: Nat) -> T.Layer<StP2, A(b, D()), A(b, D()), StS2(b)>:
  T.Seq(~StP1, ~StP1, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS1(b), ~StS1(b), ~Stack1(~b), ~Stack1(~b))

def KStack2(p: StP2, +pos: Nat, s: KS) -> KR<StP2>:
  match p:
    case T.Two{a, c}:
      kseq(StP1, StP1, KStack1(a, pos, s), s2 => KStack1(c, pos, s2))

def KGet2(s: StS2(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet1(c, KGet1(a, acc))

def LStack2(+path: String, +h: L.Header, +i: Nat, p: StP2) -> IO(StP2):
  L.two(StP1, StP1, p, x => LStack1(path, h, i, x), y => LStack1(path, h, Nat.add(i, 1n), y))

def StP3() -> Type:
  T.Two<StP1, StP2>

def StS3(b: Nat) -> Type:
  T.Two<StS1(b), StS2(b)>

def Stack3(~b: Nat) -> T.Layer<StP3, A(b, D()), A(b, D()), StS3(b)>:
  T.Seq(~StP1, ~StP2, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS1(b), ~StS2(b), ~Stack1(~b), ~Stack2(~b))

def KStack3(p: StP3, +pos: Nat, s: KS) -> KR<StP3>:
  match p:
    case T.Two{a, c}:
      kseq(StP1, StP2, KStack1(a, pos, s), s2 => KStack2(c, pos, s2))

def KGet3(s: StS3(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet2(c, KGet1(a, acc))

def LStack3(+path: String, +h: L.Header, +i: Nat, p: StP3) -> IO(StP3):
  L.two(StP1, StP2, p, x => LStack1(path, h, i, x), y => LStack2(path, h, Nat.add(i, 1n), y))

def StP4() -> Type:
  T.Two<StP2, StP2>

def StS4(b: Nat) -> Type:
  T.Two<StS2(b), StS2(b)>

def Stack4(~b: Nat) -> T.Layer<StP4, A(b, D()), A(b, D()), StS4(b)>:
  T.Seq(~StP2, ~StP2, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS2(b), ~StS2(b), ~Stack2(~b), ~Stack2(~b))

def KStack4(p: StP4, +pos: Nat, s: KS) -> KR<StP4>:
  match p:
    case T.Two{a, c}:
      kseq(StP2, StP2, KStack2(a, pos, s), s2 => KStack2(c, pos, s2))

def KGet4(s: StS4(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet2(c, KGet2(a, acc))

def LStack4(+path: String, +h: L.Header, +i: Nat, p: StP4) -> IO(StP4):
  L.two(StP2, StP2, p, x => LStack2(path, h, i, x), y => LStack2(path, h, Nat.add(i, 2n), y))

def StP7() -> Type:
  T.Two<StP3, StP4>

def StS7(b: Nat) -> Type:
  T.Two<StS3(b), StS4(b)>

def Stack7(~b: Nat) -> T.Layer<StP7, A(b, D()), A(b, D()), StS7(b)>:
  T.Seq(~StP3, ~StP4, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS3(b), ~StS4(b), ~Stack3(~b), ~Stack4(~b))

def KStack7(p: StP7, +pos: Nat, s: KS) -> KR<StP7>:
  match p:
    case T.Two{a, c}:
      kseq(StP3, StP4, KStack3(a, pos, s), s2 => KStack4(c, pos, s2))

def KGet7(s: StS7(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet4(c, KGet3(a, acc))

def LStack7(+path: String, +h: L.Header, +i: Nat, p: StP7) -> IO(StP7):
  L.two(StP3, StP4, p, x => LStack3(path, h, i, x), y => LStack4(path, h, Nat.add(i, 3n), y))

def StP8() -> Type:
  T.Two<StP4, StP4>

def StS8(b: Nat) -> Type:
  T.Two<StS4(b), StS4(b)>

def Stack8(~b: Nat) -> T.Layer<StP8, A(b, D()), A(b, D()), StS8(b)>:
  T.Seq(~StP4, ~StP4, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS4(b), ~StS4(b), ~Stack4(~b), ~Stack4(~b))

def KStack8(p: StP8, +pos: Nat, s: KS) -> KR<StP8>:
  match p:
    case T.Two{a, c}:
      kseq(StP4, StP4, KStack4(a, pos, s), s2 => KStack4(c, pos, s2))

def KGet8(s: StS8(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet4(c, KGet4(a, acc))

def LStack8(+path: String, +h: L.Header, +i: Nat, p: StP8) -> IO(StP8):
  L.two(StP4, StP4, p, x => LStack4(path, h, i, x), y => LStack4(path, h, Nat.add(i, 4n), y))

def StP15() -> Type:
  T.Two<StP7, StP8>

def StS15(b: Nat) -> Type:
  T.Two<StS7(b), StS8(b)>

def Stack15(~b: Nat) -> T.Layer<StP15, A(b, D()), A(b, D()), StS15(b)>:
  T.Seq(~StP7, ~StP8, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS7(b), ~StS8(b), ~Stack7(~b), ~Stack8(~b))

def KStack15(p: StP15, +pos: Nat, s: KS) -> KR<StP15>:
  match p:
    case T.Two{a, c}:
      kseq(StP7, StP8, KStack7(a, pos, s), s2 => KStack8(c, pos, s2))

def KGet15(s: StS15(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet8(c, KGet7(a, acc))

def LStack15(+path: String, +h: L.Header, +i: Nat, p: StP15) -> IO(StP15):
  L.two(StP7, StP8, p, x => LStack7(path, h, i, x), y => LStack8(path, h, Nat.add(i, 7n), y))

def StP30() -> Type:
  T.Two<StP15, StP15>

def StS30(b: Nat) -> Type:
  T.Two<StS15(b), StS15(b)>

def Stack30(~b: Nat) -> T.Layer<StP30, A(b, D()), A(b, D()), StS30(b)>:
  T.Seq(~StP15, ~StP15, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS15(b), ~StS15(b), ~Stack15(~b), ~Stack15(~b))

def KStack30(p: StP30, +pos: Nat, s: KS) -> KR<StP30>:
  match p:
    case T.Two{a, c}:
      kseq(StP15, StP15, KStack15(a, pos, s), s2 => KStack15(c, pos, s2))

def KGet30(s: StS30(1n), acc: List<&1, KV>) -> List<&1, KV>:
  match s:
    case T.Two{a, c}:
      KGet15(c, KGet15(a, acc))

def LStack30(+path: String, +h: L.Header, +i: Nat, p: StP30) -> IO(StP30):
  L.two(StP15, StP15, p, x => LStack15(path, h, i, x), y => LStack15(path, h, Nat.add(i, 15n), y))

def StackP() -> Type:
  StP30

def StackS(b: Nat) -> Type:
  StS30(b)

def Stack(~b: Nat) -> T.Layer<StackP, A(b, D()), A(b, D()), StackS(b)>:
  Stack30(~b)

def KStack(p: StackP, +pos: Nat, s: KS) -> KR<StackP>:
  KStack30(p, pos, s)

def KGet(s: StackS(1n), acc: List<&1, KV>) -> List<&1, KV>:
  KGet30(s, acc)

def LStack(+path: String, +h: L.Header, p: StackP) -> IO(StackP):
  LStack30(path, h, 0n, p)
# ---- stack: end ----

# ---- the model: Tie(embedding, Seq(stack, final norm), head), the head sharing the embedding's matrix ----

def BodyP() -> Type:
  T.Two<StackP, M(1n, D())>

def BodyS(b: Nat) -> Type:
  T.Two<StackS(b), A(b, D())>

def Body(~b: Nat) -> T.Layer<BodyP, A(b, D()), A(b, D()), BodyS(b)>:
  T.Seq(~StackP, ~M(1n, D()), ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StackS(b), ~A(b, D()), ~Stack(~b), ~F.RMSNorm(R(b), D(), 0.00001))

def P() -> Type:
  T.Two<M(V(), D()), BodyP>

def S(b: Nat) -> Type:
  T.Three<List<&2, Nat>, BodyS(b), A(b, D())>

def X() -> Type:
  List<&2, Nat>

def Y(b: Nat) -> Type:
  A(b, V())

def Model(~b: Nat) -> T.Layer<P, X, Y(b), S(b)>:
  T.Tie(~M(V(), D()), ~BodyP, ~X, ~A(b, D()), ~A(b, D()), ~Y(b), ~List<&2, Nat>, ~BodyS(b), ~A(b, D()), ~F.Embedding(R(b), V(), D()), ~Body(~b), ~F.LinearNBF(R(b), D(), V(), PAR()))

def ce(z: Y(2n), labels: List<&2, Nat>) -> T.LossOut<Y(2n)>:
  F.CrossEntropy(R(2n), V(), z, labels)

def load_params() -> IO(P):
  L.weights(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), DATA() ++ "init.bin", T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n)))

# ---- batches: DATA()/batches.bin, little-endian U32 token ids, 2 sequences of TT() + 1 per step; the inputs are the
# first TT() of each, the targets the last TT() ----

def ids_of(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      U32.to_nat((((b0 + (b1 * 256 : U32) : U32) + (b2 * 65536 : U32) : U32) + (b3 * 16777216 : U32) : U32)) <> ids_of(t)
    case _:
      Nil{}

def xs_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, ids, TT()), List.take(&2, Nat, List.drop(&2, Nat, ids, 1n+TT()), TT()))

def ys_of(+ids: List<&2, Nat>) -> List<&2, Nat>:
  List.append(&2, Nat, List.take(&2, Nat, List.drop(&2, Nat, ids, 1n), TT()), List.drop(&2, Nat, ids, 2n+TT()))

def load_from(+name: String, +step: U32) -> IO(List<&2, Nat>):
  +per = U32.from_nat(Nat.mul(2n, 1n+TT()))
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(DATA() ++ name, ((step * per : U32) * 4 : U32), (per * 4 : U32))
    return ids_of(bs)

def load_batch(+step: U32) -> IO(List<&2, Nat>):
  load_from("batches.bin", step)

# ---- the loss on the first batch ----

def loss2(r: T.LossOut<Y(2n)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}: T.F32.print("loss", l)

def loss1(+ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}: loss2(ce(y, ys))

def loss_with(p: P, +ids: List<&2, Nat>) -> IO(Unit):
  loss1(ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def loss() -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    ids : List<&2, Nat> <- load_batch(0)
    loss_with(p, ids)

# ---- the same loss with the weights read from Hugging Face's model.safetensors (BF16, widened exactly: the numbers
# init.bin holds) through the layout below, reference/smol_layout.txt's names and shapes ----

# the embedding, LAYERS() blocks, the final norm; the output head is tied to the embedding and not in the file
def st_layout() -> List<&2, L.Entry>:
  Con{L.Entry{"model.embed_tokens.weight", [U32.from_nat(V()), U32.from_nat(D())]}, List.append(&2, L.Entry, st_layers(LAYERS(), 0n), [L.Entry{"model.norm.weight", [U32.from_nat(D())]}])}

# The same parameters filled piece by piece from one parsed header: the embedding, each block (LStack, written with the
# stack by scripts/smol.sh), the final norm. No array of the whole model is made, which is what lets SmolLM2-1.7B load
# in 32 GB; the numbers are st_layout()'s, in the same order (make smol checks the loss against bench/bits.txt).
def load_st(+path: String) -> IO(P):
  do IO<P>:
    +h : L.Header <- L.header(path)
    L.two(M(V(), D()), BodyP, T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n)), e => L.piece(~M(V(), D()), ~X, ~A(2n, D()), ~List<&2, Nat>, ~F.Embedding(R(2n), V(), D()), path, h, [L.Entry{"model.embed_tokens.weight", [U32.from_nat(V()), U32.from_nat(D())]}], e), b => L.two(StackP, M(1n, D()), b, st => LStack(path, h, st), nm => L.piece(~M(1n, D()), ~A(2n, D()), ~A(2n, D()), ~A(2n, D()), ~F.RMSNorm(R(2n), D(), 0.00001), path, h, [L.Entry{"model.norm.weight", [U32.from_nat(D())]}], nm)))

def loss_st(+path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_st(path)
    ids : List<&2, Nat> <- load_batch(0)
    loss_with(p, ids)

# ---- the loss and every gradient on the first batch, the gradients into a file in dump order ----

def grads3(+path: String, r: T.Bwd<P, X>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        g2 : P <- T.Params.save(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), path, g)
        IO.print("grads " ++ path)

def grads2(+path: String, p: P, s: S(2n), r: T.LossOut<Y(2n)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(path, T.Layer.bwd(P, X, Y(2n), S(2n), Model(~2n), p, s, dy, T.Layer.zeros(P, X, Y(2n), S(2n), Model(~2n))))

def grads1(+path: String, +ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}: grads2(path, p, s, ce(y, ys))

def grads_with(+path: String, p: P, +ids: List<&2, Nat>) -> IO(Unit):
  grads1(path, ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def grads(+path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    ids : List<&2, Nat> <- load_batch(0)
    grads_with(path, p, ids)

# ---- training ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.0001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def step_with(+ids: List<&2, Nat>, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y(2n), ~S(2n), ~List<&2, Nat>, ~Model(~2n), ~ce, cfg(), tr, xs_of(ids), ys_of(ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +step: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- load_batch(step)
        tloop(q, (step + 1 : U32), step_with(ids, T.Train{p, g, st, p1, p2, l}))

# ---- fine-tuning: training on ft_train.bin, the mean loss on ft_eval.bin's held-out batches every few steps ----

def FT_EVAL() -> Nat:
  16n

# the parameters handed back by a forward pass, and its loss
type PE is Type:
  PE{p: P, l: F32}

def ev2(p: P, r: T.LossOut<Y(2n)>) -> PE:
  match r:
    case T.LossOut{l, dy}: PE{p, l}

def ev1(+ys: List<&2, Nat>, r: T.Fwd<P, Y(2n), S(2n)>) -> PE:
  match r:
    case T.Fwd{p, y, s}: ev2(p, ce(y, ys))

def ev_one(p: P, +ids: List<&2, Nat>) -> PE:
  ev1(ys_of(ids), T.Layer.fwd(P, X, Y(2n), S(2n), Model(~2n), p, xs_of(ids)))

def ev_then(r: PE, k: P -> F32 -> IO(PE)) -> IO(PE):
  match r:
    case PE{p, l}: k(p, l)

# the sum of the losses on k held-out batches from batch e, added to acc in order; the parameters come back
def ev_loop(k: Nat, +e: U32, +acc: F32, p: P) -> IO(PE):
  match k:
    case 0n:
      IO.pure(PE, PE{p, acc})
    case 1n+q:
      do IO<PE>:
        ids : List<&2, Nat> <- load_from("ft_eval.bin", e)
        ev_then(ev_one(p, ids), p2 => +l => ev_loop(q, (e + 1 : U32), (acc + l : F32), p2))

def ev_print(r: PE, g: P, st: T.OptState<P>, +p1: F32, +p2: F32, +l: F32, k: T.Train<P> -> IO(T.Train<P>)) -> IO(T.Train<P>):
  match r:
    case PE{p, s}:
      do IO<T.Train<P>>:
        T.F32.print("eval", (s / F32.from_nat(FT_EVAL()) : F32))
        k(T.Train{p, g, st, p1, p2, l})

# after a step: its loss, then the held-out loss when do_eval
def ft_after(do_eval: Bool, tr: T.Train<P>, k: T.Train<P> -> IO(T.Train<P>)) -> IO(T.Train<P>):
  match do_eval tr:
    case True{} T.Train{p, g, st, +p1, +p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        r : PE <- ev_loop(FT_EVAL(), 0, 0.0, p)
        ev_print(r, g, st, p1, p2, l, k)
    case False{} T.Train{p, g, st, +p1, +p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        k(T.Train{p, g, st, p1, p2, l})

def ft_loop(k: Nat, +step: U32, +every: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k:
    case 0n:
      IO.pure(T.Train<P>, tr)
    case 1n+q:
      do IO<T.Train<P>>:
        ids : List<&2, Nat> <- load_from("ft_train.bin", step)
        ft_after(U32.is_eq(U32.mod((step + 1 : U32), every), 0), step_with(ids, tr), tr2 => ft_loop(q, (step + 1 : U32), every, tr2))

# the trained weights into out, in init.bin's layout, unless out is empty
def ft_save(skip: Bool, +out: String, tr: T.Train<P>) -> IO(Unit):
  match skip tr:
    case True{} _:
      IO.pure(Unit, Unit{})
    case False{} T.Train{p, g, st, p1, p2, l}:
      do IO<Unit>:
        p3 : P <- T.Params.save(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), out, p)
        IO.print("saved " ++ out)

def ft_start(+k: Nat, +every: U32, +out: String, r: PE) -> IO(Unit):
  match r:
    case PE{p, s}:
      do IO<Unit>:
        T.F32.print("eval", (s / F32.from_nat(FT_EVAL()) : F32))
        T.OptCfg.check(cfg())
        t0 : Nat <- IO.now()
        tr : T.Train<P> <- ft_loop(k, 0, every, T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), p))
        t1 : Nat <- IO.now()
        IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))
        ft_save(String.eq(out, ""), out, tr)

# the held-out loss before any step, then k steps, each loss, and the held-out loss after every `every` steps; the
# weights into out at the end unless out is empty
def finetune(+k: Nat, +every: U32, +out: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    r : PE <- ev_loop(FT_EVAL(), 0, 0.0, p)
    ft_start(k, every, out, r)

# the same from the weights in a Hugging Face model.safetensors (load_st), with nothing saved
def finetune_st(+k: Nat, +every: U32, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_st(path)
    r : PE <- ev_loop(FT_EVAL(), 0, 0.0, p)
    ft_start(k, every, "", r)

def train_from(+k: Nat, p: P) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    t0 : Nat <- IO.now()
    tr : T.Train<P> <- tloop(k, 0, T.Train.new(~P, ~X, ~Y(2n), ~S(2n), ~Model(~2n), cfg(), p))
    t1 : Nat <- IO.now()
    IO.print("train_ms " ++ Nat.show(Nat.sub(t1, t0)))

def train(+k: Nat) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    train_from(k, p)

# the same steps from the weights in a Hugging Face model.safetensors (load_st)
def train_st(+k: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_st(path)
    train_from(k, p)

# ---- generation: one sequence (b = 1) with the same parameters. The tokens so far sit at the start of the TT()
# window and the rest is padding (token 0): the causal mask keeps the padding out of every written position, so the
# logits at the last written position are the model's prediction for the next token ----

type GenR is Type:
  GenR{p: P, ids: List<&2, Nat>}

# the n tokens so far, padded with 0 to TT()
def window(+ids: List<&2, Nat>, +n: Nat) -> List<&2, Nat>:
  List.append(&2, Nat, ids, List.replicate(Nat, Nat.sub(TT(), n), 0n))

# the most likely next token (the first one on a tie, as torch.argmax)
def last_argmax(r: TA.MRow<R(1n), V()>) -> Nat:
  match r:
    case TA.MRow{m, row}:
      TA.argmax(row)

def gfwd_then(+n: Nat, r: T.Fwd<P, Y(1n), S(1n)>, k: P -> @+i: Nat -> GenR) -> GenR:
  match r:
    case T.Fwd{p, y, s}:
      k(p, last_argmax(TA.Mat.read_row(R(1n), V(), y, U32.from_nat(Nat.sub(n, 1n)))))

# bend-lint: disable-begin bend-lint-rules/performance/append_in_loop -- generation appends one token to a prompt of at most TT() ids (60 tokens by default) not-expected-because: window() already rebuilds that list and a 135M forward runs at each step
# k more tokens after the n in seq, greedily
def gen_g(k: Nat, p: P, +n: Nat, +seq: List<&2, Nat>) -> GenR:
  match k:
    case 0n:
      GenR{p, seq}
    case 1n+q:
      gfwd_then(n, T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, window(seq, n)), p2 => +id => gen_g(q, p2, 1n+n, List.append(&2, Nat, seq, [id])))

# drawing, as tiny_llama.bend's draw mode: the next token sampled from the softmax of the logits at temperature 0.8,
# with a uniform number from a hash of (seed, token index), so a run is reproducible; the twin checks greedy decoding
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

def last_draw(+u: F32, r: TA.MRow<R(1n), V()>) -> Nat:
  match r:
    case TA.MRow{m, row}:
      draw_row(row, u)

def dfwd_then(+n: Nat, +u: F32, r: T.Fwd<P, Y(1n), S(1n)>, k: P -> @+i: Nat -> GenR) -> GenR:
  match r:
    case T.Fwd{p, y, s}:
      k(p, last_draw(u, TA.Mat.read_row(R(1n), V(), y, U32.from_nat(Nat.sub(n, 1n)))))

# k more tokens after the n in seq, drawn; i counts the tokens drawn so far
def gen_d(k: Nat, p: P, +seed: U32, +i: U32, +n: Nat, +seq: List<&2, Nat>) -> GenR:
  match k:
    case 0n:
      GenR{p, seq}
    case 1n+q:
      dfwd_then(n, unif(seed, i), T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, window(seq, n)), p2 => +id => gen_d(q, p2, seed, (i + 1 : U32), 1n+n, List.append(&2, Nat, seq, [id])))

def nat_words(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{x, t}:
      " " ++ Nat.show(x) ++ nat_words(t)

def read_ids(ss: List<&2, String>) -> List<&2, Nat>:
  match ss:
    case Nil{}:
      Nil{}
    case Con{s, t}:
      Maybe.default(&2, Nat, Nat.read(s), 0n) <> read_ids(t)

def gen_print(+t0: Nat, r: GenR) -> IO(Unit):
  match r:
    case GenR{p, +ids}:
      do IO<Unit>:
        t1 : Nat <- IO.now()
        IO.print("ids" ++ nat_words(ids))
        IO.print("gen_ms " ++ Nat.show(Nat.sub(t1, t0)))

def gen_run(greedy: Bool, +k: Nat, +seed: U32, +m: Nat, +ids: List<&2, Nat>, p: P) -> GenR:
  match greedy:
    case True{}:
      gen_g(k, p, m, ids)
    case False{}:
      gen_d(k, p, seed, 0, m, ids)

# n tokens after the prompt (at most TT() in all), greedily when greedy, else drawn with the seed
def gen_with(+greedy: Bool, +n: Nat, +seed: U32, +ids: List<&2, Nat>, p: P) -> IO(Unit):
  +m = List.length(&2, Nat, ids)
  do IO<Unit>:
    t0 : Nat <- IO.now()
    gen_print(t0, gen_run(greedy, Nat.min(n, Nat.sub(TT(), m)), seed, m, ids, p))

def generate(+greedy: Bool, +n: Nat, +seed: U32, +weights: String, +ids: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- L.weights(~P, ~X, ~Y(1n), ~S(1n), ~Model(~1n), weights, T.Layer.zeros(P, X, Y(1n), S(1n), Model(~1n)))
    gen_with(greedy, n, seed, read_ids(String.split(ids, ',')), p)

# ---- generation with the cache (the blocks' cached step is above the stack) ----

# ---- the head over bands of rows (bend-ml-tensor-array's Bands): one row's logits in parallel ----
#
# A product with one row runs tensor-array's scalar loop on one thread, and the head (V() x D()) is the largest one. A
# copy of the embedding's matrix in 2^HB() bands of rows, made once, lets Bands.matvec give each band its own task and
# the same numbers, bit for bit (its own claim: each band runs Mat.matmul_nt's gemm with one row).

def HB() -> Nat:
  4n

type BA<-r: Nat, -c: Nat> is Type:
  BA{b: TA.Bands<r, c>, src: Array<F32>}

def ba_then(-r: Nat, -c: Nat, -Z: Type, q: BA<r, c>, k: TA.Bands<r, c> -> Array<F32> -> Z) -> Z:
  match q:
    case BA{b, src}: k(b, src)

def ba_node(-r: Nat, -c: Nat, bx: TA.Bands<TA.half(r), c>, q: BA<Nat.sub(r, TA.half(r)), c>) -> BA<r, c>:
  match q:
    case BA{by, src}: BA{TA.BNode{bx, by}, src}

def ba_leaf(-r: Nat, -c: Nat, q: T.A2) -> BA<r, c>:
  match q:
    case T.A2{src, d}: BA{TA.BLeaf{TA.Mat{d}}, src}

def bc_leaf(-r: Nat, +c: Nat, +rr: Nat, +r0: Nat, src: Array<F32>, m: TA.Mat<r, c>) -> BA<r, c>:
  match m:
    case TA.Mat{d}:
      ba_leaf(r, c, T.copy_n(Nat.sub(Nat.mul(rr, c), 1n), U32.from_nat(Nat.mul(r0, c)), 0, d, Array.get(F32, src, U32.from_nat(Nat.mul(r0, c)))))

# rows r0.. of src (c wide) into the bands of b, rr rows in all (b's row count at run time, as tensor-array's bfill)
def bcopy(+c: Nat, -r: Nat, b: TA.Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, +r0: Nat, src: Array<F32>) -> BA<r, c>:
  match b:
    case TA.BLeaf{m}:
      bc_leaf(r, c, rr, r0, src, m)
    case TA.BNode{x, y}:
      ba_then(TA.half(r), c, BA<r, c>, bcopy(c, TA.half(r), x, TA.half(rr), TA.eq_half(rr, r, e), r0, src), bx => s2 => ba_node(r, c, bx, bcopy(c, Nat.sub(r, TA.half(r)), y, Nat.sub(rr, TA.half(rr)), TA.eq_rest(rr, r, e), Nat.add(r0, TA.half(rr)), s2)))

# the embedding's matrix, handed back, and its copy in bands
type HE is Type:
  HE{e: M(V(), D()), hb: TA.Bands<V(), D()>}

def he_fin(q: BA<V(), D()>) -> HE:
  match q:
    case BA{b, src}: HE{TA.Mat{src}, b}

def head_bands(e: M(V(), D())) -> HE:
  match e:
    case TA.Mat{ea}:
      he_fin(bcopy(D(), V(), TA.Bands.zeros(HB(), V(), D()), V(), {==}, 0n, ea))

# the model after a step: its parameters, the logits of the row, the caches in block order, RoPE's table, the head's bands
type KStep is Type:
  KStep{p: P, z: List<&2, F32>, cs: List<&1, KV>, tb: T.CS, hb: TA.Bands<V(), D()>}

def kmrow_d(r: TA.MRow<1n, D()>) -> List<&2, F32>:
  match r:
    case TA.MRow{m, row}:
      row

def kstep3(e: M(V(), D()), st: StackP, gf: M(1n, D()), cs: List<&1, KV>, tb: T.CS, r: TA.BL<V(), D()>) -> KStep:
  match r:
    case TA.BL{hb, z}:
      KStep{T.Two{e, T.Two{st, gf}}, z, cs, tb, hb}

def kstep2(e: M(V(), D()), gf: M(1n, D()), hb: TA.Bands<V(), D()>, r: KR<StackP>) -> KStep:
  match r:
    case KR{st, KS{y, ins, outs, tb}}:
      kf3(M(1n, D()), M1(D()), M1(D()), KStep, krms(gf, y), gf2 => h => y2 => kstep3(e, st, gf2, List.reverse(&1, KV, outs), tb, TA.Bands.matvec_l(V(), D(), hb, kmrow_d(TA.Mat.read_row(1n, D(), h, 0)))))

# token id at position pos: its embedding, the blocks with their caches, the final norm, the head (Model's layers)
def kstep(p: P, hb: TA.Bands<V(), D()>, +id: Nat, +pos: Nat, cs: List<&1, KV>, tb: T.CS) -> KStep:
  match p:
    case T.Two{e, T.Two{st, gf}}:
      kf3(M(V(), D()), M1(D()), List<&2, Nat>, KStep, T.Layer.fwd(M(V(), D()), X, M1(D()), List<&2, Nat>, T.Embedding(1n, V(), D()), e, [id]), e2 => x => s => kstep2(e2, gf, hb, KStack(st, pos, KS{x, cs, Nil{}, tb})))

def wrow(r: TA.MRow<R(1n), V()>) -> List<&2, F32>:
  match r:
    case TA.MRow{m, row}:
      row

# after the prompt: the parameters, the logits of its last position, the caches in block order
type KP is Type:
  KP{p: P, row: List<&2, F32>, cs: List<&1, KV>, hb: TA.Bands<V(), D()>}

def kp_fin(+row: List<&2, F32>, cs: List<&1, KV>, body: BodyP, h: HE) -> KP:
  match h:
    case HE{e, hb}: KP{T.Two{e, body}, row, cs, hb}

# the prompt (m tokens) once through the whole window, as gen's first step: the logits at row m - 1 give the first new
# token, and the keys and values each block's attention saved for its backward pass are the caches (KGet)
def kpre(+m: Nat, r: T.Fwd<P, Y(1n), S(1n)>) -> KP:
  match r:
    case T.Fwd{T.Two{e, body}, y, T.Three{xs, T.Two{st, hx}, hy}}:
      kp_fin(wrow(TA.Mat.read_row(R(1n), V(), y, U32.from_nat(Nat.sub(m, 1n)))), List.reverse(&1, KV, KGet(st, Nil{})), body, head_bands(e))

def kprompt(+m: Nat, +ids: List<&2, Nat>, p: P) -> KP:
  kpre(m, T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, window(ids, m)))

def kg_then(r: KStep, k: P -> TA.Bands<V(), D()> -> @+i: Nat -> List<&1, KV> -> T.CS -> GenR) -> GenR:
  match r:
    case KStep{p, z, cs, tb, hb}: k(p, hb, TA.argmax(z), cs, tb)

# k more tokens, the first of them id (chosen, at position pos); each step runs the one before's position and its
# logits give the next token, greedily. The last token needs no step.
def kgen_g(k: Nat, p: P, hb: TA.Bands<V(), D()>, +id: Nat, +pos: Nat, +seq: List<&2, Nat>, cs: List<&1, KV>, tb: T.CS) -> GenR:
  match k:
    case 0n:
      GenR{p, seq}
    case 1n:
      GenR{p, List.append(&2, Nat, seq, [id])}
    case 2n+q:
      kg_then(kstep(p, hb, id, pos, cs, tb), p2 => hb2 => +id2 => cs2 => tb2 => kgen_g(1n+q, p2, hb2, id2, 1n+pos, List.append(&2, Nat, seq, [id]), cs2, tb2))

def kd_then(+u: F32, r: KStep, k: P -> TA.Bands<V(), D()> -> @+i: Nat -> List<&1, KV> -> T.CS -> GenR) -> GenR:
  match r:
    case KStep{p, z, cs, tb, hb}: k(p, hb, draw_row(z, u), cs, tb)

# the same, each token drawn as gen_d draws it (i: the index of the next draw)
def kgen_d(k: Nat, p: P, hb: TA.Bands<V(), D()>, +seed: U32, +i: U32, +id: Nat, +pos: Nat, +seq: List<&2, Nat>, cs: List<&1, KV>, tb: T.CS) -> GenR:
  match k:
    case 0n:
      GenR{p, seq}
    case 1n:
      GenR{p, List.append(&2, Nat, seq, [id])}
    case 2n+q:
      kd_then(unif(seed, i), kstep(p, hb, id, pos, cs, tb), p2 => hb2 => +id2 => cs2 => tb2 => kgen_d(1n+q, p2, hb2, seed, (i + 1 : U32), id2, 1n+pos, List.append(&2, Nat, seq, [id]), cs2, tb2))

def kgen_go(greedy: Bool, +k: Nat, +seed: U32, +m: Nat, +ids: List<&2, Nat>, r: KP) -> GenR:
  match greedy r:
    case True{} KP{p, +row, cs, hb}:
      kgen_g(k, p, hb, TA.argmax(row), m, ids, cs, ktable())
    case False{} KP{p, +row, cs, hb}:
      kgen_d(k, p, hb, seed, 1, draw_row(row, unif(seed, 0)), m, ids, cs, ktable())

# n tokens after the prompt (at most TT() in all), as gen_with, with the cache: the prompt once through the window, then
# one position a token
def kgen_with(+greedy: Bool, +n: Nat, +seed: U32, +ids: List<&2, Nat>, p: P) -> IO(Unit):
  +m = List.length(&2, Nat, ids)
  do IO<Unit>:
    t0 : Nat <- IO.now()
    gen_print(t0, kgen_go(greedy, Nat.min(n, Nat.sub(TT(), m)), seed, m, ids, kprompt(m, ids, p)))

def kgenerate(+greedy: Bool, +n: Nat, +seed: U32, +weights: String, +ids: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- L.weights(~P, ~X, ~Y(1n), ~S(1n), ~Model(~1n), weights, T.Layer.zeros(P, X, Y(1n), S(1n), Model(~1n)))
    kgen_with(greedy, n, seed, read_ids(String.split(ids, ',')), p)

# ---- kvcheck: both passes in step, greedily; for each token after the first (which both take from the prompt's
# window), the logits of the whole window's row and of the cached step compared bit for bit ----

def b01(b: Bool) -> U32:
  match b:
    case True{}: 1
    case False{}: 0

type CD is Type:
  CD{c: U32, n: U32, m: F32}

# how many pairs were compared, how many differ in their bits, and the largest |a - b|
def kdiff(a: List<&2, F32>, b: List<&2, F32>, +c: U32, +n: U32, +m: F32) -> CD:
  match a b:
    case Con{+x, at} Con{+y, bt}:
      kdiff(at, bt, (c + 1 : U32), (n + b01(Bool.not(U32.is_eq(F32.bits(x), F32.bits(y)))) : U32), F32.max(m, F32.abs((x - y : F32))))
    case _ _:
      CD{c, n, m}

def kc_line(+ia: Nat, +ib: Nat, d: CD) -> String:
  match d:
    case CD{+c, +n, +m}:
      "window " ++ Nat.show(ia) ++ " cached " ++ Nat.show(ib) ++ " logits " ++ U32.show(c) ++ " differing " ++ U32.show(n) ++ " max_abs_diff " ++ F32.show(m)

def kc_step2(+a: List<&2, F32>, r: KStep, k: P -> TA.Bands<V(), D()> -> List<&1, KV> -> T.CS -> @+i: Nat -> @+line: String -> String) -> String:
  match r:
    case KStep{p, +b, cs, tb, hb}:
      k(p, hb, cs, tb, TA.argmax(a), kc_line(TA.argmax(a), TA.argmax(b), kdiff(a, b, 0, 0, 0.0)))

# the window's row pos (the window holds pos + 1 tokens), then the cached step of token id at pos
def kc_step(+pos: Nat, +id: Nat, hb: TA.Bands<V(), D()>, cs: List<&1, KV>, tb: T.CS, r: T.Fwd<P, Y(1n), S(1n)>, k: P -> TA.Bands<V(), D()> -> List<&1, KV> -> T.CS -> @+i: Nat -> @+line: String -> String) -> String:
  match r:
    case T.Fwd{p, y, s}:
      kc_step2(wrow(TA.Mat.read_row(R(1n), V(), y, U32.from_nat(pos))), kstep(p, hb, id, pos, cs, tb), k)

# k more tokens, the first of them id (at position n, after the n in seq); the window's choice goes on; a line a step
def kc_loop(k: Nat, p: P, hb: TA.Bands<V(), D()>, +id: Nat, +n: Nat, +seq: List<&2, Nat>, cs: List<&1, KV>, tb: T.CS) -> String:
  match k:
    case 0n:
      "ids" ++ nat_words(seq)
    case 1n:
      "ids" ++ nat_words(List.append(&2, Nat, seq, [id]))
    case 2n+q:
      kc_step(n, id, hb, cs, tb, T.Layer.fwd(P, X, Y(1n), S(1n), Model(~1n), p, window(List.append(&2, Nat, seq, [id]), 1n+n)), p2 => hb2 => cs2 => tb2 => +id2 => +line => line ++ "\n" ++ kc_loop(1n+q, p2, hb2, id2, 1n+n, List.append(&2, Nat, seq, [id]), cs2, tb2))
# bend-lint: disable-end bend-lint-rules/performance/append_in_loop

def kc_go(+k: Nat, +m: Nat, +ids: List<&2, Nat>, r: KP) -> String:
  match r:
    case KP{p, +row, cs, hb}:
      kc_loop(k, p, hb, TA.argmax(row), m, ids, cs, ktable())

def kvcheck(+n: Nat, +weights: String, +ids0: String) -> IO(Unit):
  +ids = read_ids(String.split(ids0, ','))
  +m = List.length(&2, Nat, ids)
  do IO<Unit>:
    p : P <- L.weights(~P, ~X, ~Y(1n), ~S(1n), ~Model(~1n), weights, T.Layer.zeros(P, X, Y(1n), S(1n), Model(~1n)))
    IO.print(kc_go(Nat.min(n, Nat.sub(TT(), m)), m, ids, kprompt(m, ids, p)))

def usage() -> IO(Unit):
  IO.die(Unit, 2, "usage: smol loss [<model.safetensors>] | train <steps> [<model.safetensors>] | grads <file> | finetune <steps> <every> [<out>] | finetunest <steps> <every> <model.safetensors> | gen <n> <weights> <ids> | draw <n> <seed> <weights> <ids> | genkv <n> <weights> <ids> | drawkv <n> <seed> <weights> <ids> | kvcheck <n> <weights> <ids>")

def train_m(mk: Maybe<&2, Nat>) -> IO(Unit):
  match mk:
    case Some{k}: train(k)
    case None{}: usage()

def dispatch_l(is_loss: Bool) -> IO(Unit):
  match is_loss:
    case True{}: loss()
    case False{}: usage()

def dispatch_t(is_train: Bool, is_grads: Bool, is_loss: Bool, +arg: String) -> IO(Unit):
  match is_train is_grads is_loss:
    case True{} _ _: train_m(Nat.read(arg))
    case _ True{} _: grads(arg)
    case _ _ True{}: loss_st(arg)
    case _ _ _: usage()

def finetune_m(mk: Maybe<&2, Nat>, me: Maybe<&2, U32>, +out: String) -> IO(Unit):
  match mk me:
    case Some{k} Some{e}: finetune(k, e, out)
    case _ _: usage()

def train_st_m(mk: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match mk:
    case Some{k}: train_st(k, path)
    case None{}: usage()

def dispatch_f(is_ft: Bool, is_train: Bool, +k: String, +e: String) -> IO(Unit):
  match is_ft is_train:
    case True{} _: finetune_m(Nat.read(k), U32.read(e), "")
    case _ True{}: train_st_m(Nat.read(k), e)
    case _ _: usage()

def gen_m(mn: Maybe<&2, Nat>, +weights: String, +ids: String) -> IO(Unit):
  match mn:
    case Some{n}: generate(True{}, n, 0, weights, ids)
    case None{}: usage()

def genkv_m(mn: Maybe<&2, Nat>, +weights: String, +ids: String) -> IO(Unit):
  match mn:
    case Some{n}: kgenerate(True{}, n, 0, weights, ids)
    case None{}: usage()

def kvcheck_m(mn: Maybe<&2, Nat>, +weights: String, +ids: String) -> IO(Unit):
  match mn:
    case Some{n}: kvcheck(n, weights, ids)
    case None{}: usage()

def finetune_st_m(mk: Maybe<&2, Nat>, me: Maybe<&2, U32>, +path: String) -> IO(Unit):
  match mk me:
    case Some{k} Some{e}: finetune_st(k, e, path)
    case _ _: usage()

def dispatch_k(is_genkv: Bool, is_check: Bool, is_ftst: Bool, +a: String, +b: String, +c: String) -> IO(Unit):
  match is_genkv is_check is_ftst:
    case True{} _ _: genkv_m(Nat.read(a), b, c)
    case _ True{} _: kvcheck_m(Nat.read(a), b, c)
    case _ _ True{}: finetune_st_m(Nat.read(a), U32.read(b), c)
    case _ _ _: usage()

def dispatch_5(is_ft: Bool, is_gen: Bool, +mode: String, +a: String, +b: String, +c: String) -> IO(Unit):
  match is_ft is_gen:
    case True{} _: finetune_m(Nat.read(a), U32.read(b), c)
    case _ True{}: gen_m(Nat.read(a), b, c)
    case _ _: dispatch_k(String.eq(mode, "genkv"), String.eq(mode, "kvcheck"), String.eq(mode, "finetunest"), a, b, c)

def draw_m(mn: Maybe<&2, Nat>, ms: Maybe<&2, U32>, +weights: String, +ids: String) -> IO(Unit):
  match mn ms:
    case Some{n} Some{s}: generate(False{}, n, s, weights, ids)
    case _ _: usage()

def drawkv_m(mn: Maybe<&2, Nat>, ms: Maybe<&2, U32>, +weights: String, +ids: String) -> IO(Unit):
  match mn ms:
    case Some{n} Some{s}: kgenerate(False{}, n, s, weights, ids)
    case _ _: usage()

def dispatch_d(is_draw: Bool, is_drawkv: Bool, +a: String, +b: String, +c: String, +d: String) -> IO(Unit):
  match is_draw is_drawkv:
    case True{} _: draw_m(Nat.read(a), U32.read(b), c, d)
    case _ True{}: drawkv_m(Nat.read(a), U32.read(b), c, d)
    case _ _: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{a, Con{b, Con{c, Con{d, Nil{}}}}}}}:
      dispatch_d(String.eq(mode, "draw"), String.eq(mode, "drawkv"), a, b, c, d)
    case Con{prog, Con{+mode, Con{a, Con{b, Con{c, Nil{}}}}}}:
      dispatch_5(String.eq(mode, "finetune"), String.eq(mode, "gen"), mode, a, b, c)
    case Con{prog, Con{mode, Con{k, Con{e, Nil{}}}}}:
      dispatch_f(String.eq(mode, "finetune"), String.eq(mode, "train"), k, e)
    case Con{prog, Con{mode, Nil{}}}:
      dispatch_l(String.eq(mode, "loss"))
    case Con{prog, Con{+mode, Con{k, Nil{}}}}:
      dispatch_t(String.eq(mode, "train"), String.eq(mode, "grads"), String.eq(mode, "loss"), k)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
