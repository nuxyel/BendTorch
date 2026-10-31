# The tiny Llama of demo/tiny_llama.bend with its batch B and sequence length T read from the command line: the
# program is compiled once and runs for any B and T. Its PyTorch twin is reference/tiny_rt_ref.py.
#
#   bend demo/tiny_rt.bend -o .scratch/tiny_rt
#   .scratch/tiny_rt <B> <T> <batches> grads                          # loss and every gradient on the first batch
#   .scratch/tiny_rt <B> <T> <batches> train <steps>                  # AdamW steps, as tiny_llama's train mode
#   .scratch/tiny_rt <B> <T> <batches> sample <steps> <chars> <prompt>   # train, then continue the last T characters
#
# <batches> is a file of bytes, B sequences of T + 1 character ids a step (data/tiny/batches.bin for B = 2, T = 16;
# reference/tiny_rt_ref.py --prep B T STEPS <file> writes one for other sizes). With B = 2 and T = 16 the output is
# tiny_llama's, bit for bit.
#
# How the shapes stay checked although B and T are unknown when the program is compiled: every layer below is a
# function of b and t, written once for every b and t, like bend-ml's examples/runtime_batch.bend. The checker never
# learns their values; it checks that the shapes meet for all of them at once (the 32 columns of one layer are the 32
# columns the next one takes, the rows are b * t everywhere). tiny_llama passes its sizes as templates (`~b`), which
# Bend substitutes when it compiles, so they must be literals; here they are ordinary arguments, and the composition
# uses the package's owned combinator Seq.at plus run-time forms of Fork, Both, Id and Residual written in this file.
# What the types cannot see (the file holding B x (T + 1) ids a step, every id inside the vocabulary, the prompt
# holding T characters) is checked once, at the boundary, and refused with a message.

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# the rows of a batch of b sequences of t positions
def R(b: Nat, t: Nat) -> Nat:
  Nat.mul(b, t)

# an activation: one row per position
def A(b: Nat, t: Nat, c: Nat) -> Type:
  TA.Mat<R(b, t), c>

# ---- run-time forms of the package's template combinators ----
# T.Fork, T.Both, T.Id and T.Residual take their layers (and Id its type) as templates, and a template argument must be
# closed: it cannot mention b or t, which are local values here. These take the layers as ordinary values instead, as
# T.Seq.at does: each child record is opened once and each of its closures goes to exactly one closure of the composite
# record, so no closure is called twice. They run the same operations in the same order as the templates.

def IdR(-X: Type) -> T.Layer<Unit, X, X, Unit>:
  T.Layer{p => x => T.Fwd{p, x, Unit{}}, p => s => dy => g => T.Bwd{p, dy, g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# runs the second layer on x2 once the first one's forward pass is done
def pair_f3(-P1: Type, -P2: Type, -X2: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, fb: P2 -> X2 -> T.Fwd<P2, Y2, S2>, p2: P2, x2: X2, r: T.Fwd<P1, Y1, S1>) -> T.Fwd<T.Two<P1, P2>, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match r:
    case T.Fwd{p1, y1, s1}:
      T.pair_f4(P1, P2, Y1, Y2, S1, S2, p1, y1, s1, fb(p2, x2))

def fork_f2(-P1: Type, -P2: Type, -X: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, fa: P1 -> X -> T.Fwd<P1, Y1, S1>, fb: P2 -> X -> T.Fwd<P2, Y2, S2>, p1: P1, p2: P2, d: T.Two<X, X>) -> T.Fwd<T.Two<P1, P2>, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match d:
    case T.Two{xa, xb}:
      pair_f3(P1, P2, X, Y1, Y2, S1, S2, fb, p2, xb, fa(p1, xa))

def fork_f(-P1: Type, -P2: Type, -X: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, dup: X -> T.Two<X, X>, fa: P1 -> X -> T.Fwd<P1, Y1, S1>, fb: P2 -> X -> T.Fwd<P2, Y2, S2>, p: T.Two<P1, P2>, x: X) -> T.Fwd<T.Two<P1, P2>, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match p:
    case T.Two{p1, p2}:
      fork_f2(P1, P2, X, Y1, Y2, S1, S2, fa, fb, p1, p2, dup(x))

def fork_b3(-P1: Type, -P2: Type, -X: Type, add: X -> X -> X, p2: P2, g2: P2, dx2: X, r: T.Bwd<P1, X>) -> T.Bwd<T.Two<P1, P2>, X>:
  match r:
    case T.Bwd{p1, dx1, g1}:
      T.Bwd{T.Two{p1, p2}, add(dx1, dx2), T.Two{g1, g2}}

def fork_b2(-P1: Type, -P2: Type, -X: Type, -Y1: Type, -S1: Type, add: X -> X -> X, ba: P1 -> S1 -> Y1 -> P1 -> T.Bwd<P1, X>, p1: P1, s1: S1, d1: Y1, g1: P1, r: T.Bwd<P2, X>) -> T.Bwd<T.Two<P1, P2>, X>:
  match r:
    case T.Bwd{p2, dx2, g2}:
      fork_b3(P1, P2, X, add, p2, g2, dx2, ba(p1, s1, d1, g1))

def fork_b(-P1: Type, -P2: Type, -X: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, add: X -> X -> X, ba: P1 -> S1 -> Y1 -> P1 -> T.Bwd<P1, X>, bb: P2 -> S2 -> Y2 -> P2 -> T.Bwd<P2, X>, p: T.Two<P1, P2>, s: T.Two<S1, S2>, dy: T.Two<Y1, Y2>, g: T.Two<P1, P2>) -> T.Bwd<T.Two<P1, P2>, X>:
  match p s dy g:
    case T.Two{p1, p2} T.Two{s1, s2} T.Two{d1, d2} T.Two{g1, g2}:
      fork_b2(P1, P2, X, Y1, S1, add, ba, p1, s1, d1, g1, bb(p2, s2, d2, g2))

# x goes to both layers (copied by dup); the two input gradients are summed by add
def ForkR(-P1: Type, -P2: Type, -X: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, dup: X -> T.Two<X, X>, add: X -> X -> X, a: T.Layer<P1, X, Y1, S1>, b: T.Layer<P2, X, Y2, S2>) -> T.Layer<T.Two<P1, P2>, X, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match a b:
    case T.Layer{fa, ba, sa, za, da, la, ta, na} T.Layer{fb, bb, sb, zb, db, lb, tb, nb}:
      T.Layer{p => x => fork_f(P1, P2, X, Y1, Y2, S1, S2, dup, fa, fb, p, x), p => s => dy => g => fork_b(P1, P2, X, Y1, Y2, S1, S2, add, ba, bb, p, s, dy, g), op => p => g => st => T.seq_at_step(P1, P2, sa, sb, op, p, g, T.OptState.split(P1, P2, st)), u => T.Two{za(Unit{}), zb(Unit{})}, p => T.seq_at_dump(P1, P2, da, db, p), p => xs => off => T.seq_at_load(P1, P2, la, lb, p, xs, off), p => xs => off => T.seq_at_store(P1, P2, ta, tb, p, xs, off), u => (na(Unit{}) + nb(Unit{}) : U32)}

def both_f(-P1: Type, -P2: Type, -X1: Type, -X2: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, fa: P1 -> X1 -> T.Fwd<P1, Y1, S1>, fb: P2 -> X2 -> T.Fwd<P2, Y2, S2>, p: T.Two<P1, P2>, x: T.Two<X1, X2>) -> T.Fwd<T.Two<P1, P2>, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match p x:
    case T.Two{p1, p2} T.Two{x1, x2}:
      pair_f3(P1, P2, X2, Y1, Y2, S1, S2, fb, p2, x2, fa(p1, x1))

def both_b2(-P1: Type, -P2: Type, -X1: Type, -X2: Type, -Y1: Type, -S1: Type, ba: P1 -> S1 -> Y1 -> P1 -> T.Bwd<P1, X1>, p1: P1, s1: S1, d1: Y1, g1: P1, r: T.Bwd<P2, X2>) -> T.Bwd<T.Two<P1, P2>, T.Two<X1, X2>>:
  match r:
    case T.Bwd{p2, dx2, g2}:
      T.both_b3(P1, P2, X1, X2, p2, g2, dx2, ba(p1, s1, d1, g1))

def both_b(-P1: Type, -P2: Type, -X1: Type, -X2: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, ba: P1 -> S1 -> Y1 -> P1 -> T.Bwd<P1, X1>, bb: P2 -> S2 -> Y2 -> P2 -> T.Bwd<P2, X2>, p: T.Two<P1, P2>, s: T.Two<S1, S2>, dy: T.Two<Y1, Y2>, g: T.Two<P1, P2>) -> T.Bwd<T.Two<P1, P2>, T.Two<X1, X2>>:
  match p s dy g:
    case T.Two{p1, p2} T.Two{s1, s2} T.Two{d1, d2} T.Two{g1, g2}:
      both_b2(P1, P2, X1, X2, Y1, S1, ba, p1, s1, d1, g1, bb(p2, s2, d2, g2))

# the first layer on the first of a pair, the second layer on the second
def BothR(-P1: Type, -P2: Type, -X1: Type, -X2: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, a: T.Layer<P1, X1, Y1, S1>, b: T.Layer<P2, X2, Y2, S2>) -> T.Layer<T.Two<P1, P2>, T.Two<X1, X2>, T.Two<Y1, Y2>, T.Two<S1, S2>>:
  match a b:
    case T.Layer{fa, ba, sa, za, da, la, ta, na} T.Layer{fb, bb, sb, zb, db, lb, tb, nb}:
      T.Layer{p => x => both_f(P1, P2, X1, X2, Y1, Y2, S1, S2, fa, fb, p, x), p => s => dy => g => both_b(P1, P2, X1, X2, Y1, Y2, S1, S2, ba, bb, p, s, dy, g), op => p => g => st => T.seq_at_step(P1, P2, sa, sb, op, p, g, T.OptState.split(P1, P2, st)), u => T.Two{za(Unit{}), zb(Unit{})}, p => T.seq_at_dump(P1, P2, da, db, p), p => xs => off => T.seq_at_load(P1, P2, la, lb, p, xs, off), p => xs => off => T.seq_at_store(P1, P2, ta, tb, p, xs, off), u => (na(Unit{}) + nb(Unit{}) : U32)}

# y = x + f(x) for any layer f from n x c to n x c, n known only at run time (T.Residual with the run-time combinators)
def ResidualR(+n: Nat, +c: Nat, -P: Type, -S: Type, f: T.Layer<P, TA.Mat<n, c>, TA.Mat<n, c>, S>) -> T.Layer<T.Two<T.Two<Unit, P>, Unit>, TA.Mat<n, c>, TA.Mat<n, c>, T.Two<T.Two<Unit, S>, Unit>>:
  T.Seq.at(T.Two<Unit, P>, Unit, TA.Mat<n, c>, T.Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, T.Two<Unit, S>, Unit, ForkR(Unit, P, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>, Unit, S, m => T.Mat.dup(n, c, m), x => y => T.Mat.sum2(n, c, x, y), IdR(TA.Mat<n, c>), f), T.Add2(n, c))

# ---- attention: q, k, v projections, RoPE on q and k, the attention core, the output projection ----

def KVP() -> Type:
  T.Two<M(16n, 32n), M(16n, 32n)>

def KVY(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 16n), A(b, t, 16n)>

def KVS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), A(b, t, 32n)>

def KVL(+b: Nat, +t: Nat) -> T.Layer<KVP, A(b, t, 32n), KVY(b, t), KVS(b, t)>:
  ForkR(M(16n, 32n), M(16n, 32n), A(b, t, 32n), A(b, t, 16n), A(b, t, 16n), A(b, t, 32n), A(b, t, 32n), m => T.Mat.dup(R(b, t), 32n, m), x => y => T.Mat.sum2(R(b, t), 32n, x, y), F.LinearNBF(R(b, t), 32n, 16n, 0n), F.LinearNBF(R(b, t), 32n, 16n, 0n))

def AX(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), KVY(b, t)>

def QKVP() -> Type:
  T.Two<M(32n, 32n), KVP>

def QKVS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), KVS(b, t)>

def QKV(+b: Nat, +t: Nat) -> T.Layer<QKVP, A(b, t, 32n), AX(b, t), QKVS(b, t)>:
  ForkR(M(32n, 32n), KVP, A(b, t, 32n), A(b, t, 32n), KVY(b, t), A(b, t, 32n), KVS(b, t), m => T.Mat.dup(R(b, t), 32n, m), x => y => T.Mat.sum2(R(b, t), 32n, x, y), F.LinearNBF(R(b, t), 32n, 32n, 0n), KVL(b, t))

# RoPE's position of row r is r mod t: the sequence length is an ordinary argument of the layer
def RopeK(+b: Nat, +t: Nat) -> T.Layer<T.Two<Unit, Unit>, KVY(b, t), KVY(b, t), T.Two<Unit, Unit>>:
  BothR(Unit, Unit, A(b, t, 16n), A(b, t, 16n), A(b, t, 16n), A(b, t, 16n), Unit, Unit, T.RoPE(R(b, t), 16n, t, 2n, 4n, 10000.0, {==}), IdR(A(b, t, 16n)))

def RU() -> Type:
  T.Two<Unit, T.Two<Unit, Unit>>

def Ropes(+b: Nat, +t: Nat) -> T.Layer<RU, AX(b, t), AX(b, t), RU>:
  BothR(Unit, T.Two<Unit, Unit>, A(b, t, 32n), KVY(b, t), A(b, t, 32n), KVY(b, t), Unit, T.Two<Unit, Unit>, T.RoPE(R(b, t), 32n, t, 4n, 4n, 10000.0, {==}), RopeK(b, t))

def CoreP() -> Type:
  T.Two<Unit, M(32n, 32n)>

def CoreS(b: Nat, t: Nat) -> Type:
  T.Two<T.AttnS, A(b, t, 32n)>

# Attention asks for a proof that its rows are b sequences of t positions (n == b * t): with n = R(b, t) it is
# true by definition for every b and t, so the checker accepts {==} without knowing them
def Core(+b: Nat, +t: Nat) -> T.Layer<CoreP, AX(b, t), A(b, t, 32n), CoreS(b, t)>:
  T.Seq.at(Unit, M(32n, 32n), AX(b, t), A(b, t, 32n), A(b, t, 32n), T.AttnS, A(b, t, 32n), T.Attention(R(b, t), 32n, 16n, b, t, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==}), F.LinearNBF(R(b, t), 32n, 32n, 0n))

def RCP() -> Type:
  T.Two<RU, CoreP>

def RCS(b: Nat, t: Nat) -> Type:
  T.Two<RU, CoreS(b, t)>

def RopeCore(+b: Nat, +t: Nat) -> T.Layer<RCP, AX(b, t), A(b, t, 32n), RCS(b, t)>:
  T.Seq.at(RU, CoreP, AX(b, t), AX(b, t), A(b, t, 32n), RU, CoreS(b, t), Ropes(b, t), Core(b, t))

def AttP() -> Type:
  T.Two<QKVP, RCP>

def AttS(b: Nat, t: Nat) -> Type:
  T.Two<QKVS(b, t), RCS(b, t)>

def Att(+b: Nat, +t: Nat) -> T.Layer<AttP, A(b, t, 32n), A(b, t, 32n), AttS(b, t)>:
  T.Seq.at(QKVP, RCP, A(b, t, 32n), AX(b, t), A(b, t, 32n), QKVS(b, t), RCS(b, t), QKV(b, t), RopeCore(b, t))

def NAttP() -> Type:
  T.Two<M(1n, 32n), AttP>

def NAttS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), AttS(b, t)>

def NAtt(+b: Nat, +t: Nat) -> T.Layer<NAttP, A(b, t, 32n), A(b, t, 32n), NAttS(b, t)>:
  T.Seq.at(M(1n, 32n), AttP, A(b, t, 32n), A(b, t, 32n), A(b, t, 32n), A(b, t, 32n), AttS(b, t), T.RMSNorm(R(b, t), 32n, 0.00001), Att(b, t))

# ---- the feed-forward block: down(silu(gate(x)) * up(x)) ----

def GateS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), A(b, t, 64n)>

def Gate(+b: Nat, +t: Nat) -> T.Layer<T.Two<M(64n, 32n), Unit>, A(b, t, 32n), A(b, t, 64n), GateS(b, t)>:
  T.Seq.at(M(64n, 32n), Unit, A(b, t, 32n), A(b, t, 64n), A(b, t, 64n), A(b, t, 32n), A(b, t, 64n), F.LinearNBF(R(b, t), 32n, 64n, 0n), T.SiLU(R(b, t), 64n))

def GUP() -> Type:
  T.Two<T.Two<M(64n, 32n), Unit>, M(64n, 32n)>

def GUY(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 64n), A(b, t, 64n)>

def GUS(b: Nat, t: Nat) -> Type:
  T.Two<GateS(b, t), A(b, t, 32n)>

def GU(+b: Nat, +t: Nat) -> T.Layer<GUP, A(b, t, 32n), GUY(b, t), GUS(b, t)>:
  ForkR(T.Two<M(64n, 32n), Unit>, M(64n, 32n), A(b, t, 32n), A(b, t, 64n), A(b, t, 64n), GateS(b, t), A(b, t, 32n), m => T.Mat.dup(R(b, t), 32n, m), x => y => T.Mat.sum2(R(b, t), 32n, x, y), Gate(b, t), F.LinearNBF(R(b, t), 32n, 64n, 0n))

def GLUS(b: Nat, t: Nat) -> Type:
  T.Two<GUS(b, t), GUY(b, t)>

def GLU(+b: Nat, +t: Nat) -> T.Layer<T.Two<GUP, Unit>, A(b, t, 32n), A(b, t, 64n), GLUS(b, t)>:
  T.Seq.at(GUP, Unit, A(b, t, 32n), GUY(b, t), A(b, t, 64n), GUS(b, t), GUY(b, t), GU(b, t), T.Mul2(R(b, t), 64n))

def MlpP() -> Type:
  T.Pair3(GUP, Unit, M(32n, 64n))

def MlpS(b: Nat, t: Nat) -> Type:
  T.Two<GLUS(b, t), A(b, t, 64n)>

def Mlp(+b: Nat, +t: Nat) -> T.Layer<MlpP, A(b, t, 32n), A(b, t, 32n), MlpS(b, t)>:
  T.Seq.at(T.Two<GUP, Unit>, M(32n, 64n), A(b, t, 32n), A(b, t, 64n), A(b, t, 32n), GLUS(b, t), A(b, t, 64n), GLU(b, t), F.LinearNBF(R(b, t), 64n, 32n, 0n))

def NMlpP() -> Type:
  T.Two<M(1n, 32n), MlpP>

def NMlpS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), MlpS(b, t)>

def NMlp(+b: Nat, +t: Nat) -> T.Layer<NMlpP, A(b, t, 32n), A(b, t, 32n), NMlpS(b, t)>:
  T.Seq.at(M(1n, 32n), MlpP, A(b, t, 32n), A(b, t, 32n), A(b, t, 32n), A(b, t, 32n), MlpS(b, t), T.RMSNorm(R(b, t), 32n, 0.00001), Mlp(b, t))

# ---- a block, the model ----

def BlockP() -> Type:
  T.Two<T.ResidualType(NAttP), T.ResidualType(NMlpP)>

def BlockS(b: Nat, t: Nat) -> Type:
  T.Two<T.ResidualType(NAttS(b, t)), T.ResidualType(NMlpS(b, t))>

def Block(+b: Nat, +t: Nat) -> T.Layer<BlockP, A(b, t, 32n), A(b, t, 32n), BlockS(b, t)>:
  T.Seq.at(T.ResidualType(NAttP), T.ResidualType(NMlpP), A(b, t, 32n), A(b, t, 32n), A(b, t, 32n), T.ResidualType(NAttS(b, t)), T.ResidualType(NMlpS(b, t)), ResidualR(R(b, t), 32n, NAttP, NAttS(b, t), NAtt(b, t)), ResidualR(R(b, t), 32n, NMlpP, NMlpS(b, t), NMlp(b, t)))

def HeadP() -> Type:
  T.Two<M(1n, 32n), M(65n, 32n)>

def HeadS(b: Nat, t: Nat) -> Type:
  T.Two<A(b, t, 32n), A(b, t, 32n)>

def Head(+b: Nat, +t: Nat) -> T.Layer<HeadP, A(b, t, 32n), A(b, t, 65n), HeadS(b, t)>:
  T.Seq.at(M(1n, 32n), M(65n, 32n), A(b, t, 32n), A(b, t, 32n), A(b, t, 65n), A(b, t, 32n), A(b, t, 32n), T.RMSNorm(R(b, t), 32n, 0.00001), F.LinearNBF(R(b, t), 32n, 65n, 0n))

def TopP() -> Type:
  T.Two<BlockP, HeadP>

def TopS(b: Nat, t: Nat) -> Type:
  T.Two<BlockS(b, t), HeadS(b, t)>

def Top(+b: Nat, +t: Nat) -> T.Layer<TopP, A(b, t, 32n), A(b, t, 65n), TopS(b, t)>:
  T.Seq.at(BlockP, HeadP, A(b, t, 32n), A(b, t, 32n), A(b, t, 65n), BlockS(b, t), HeadS(b, t), Block(b, t), Head(b, t))

def BodyP() -> Type:
  T.Two<BlockP, TopP>

def BodyS(b: Nat, t: Nat) -> Type:
  T.Two<BlockS(b, t), TopS(b, t)>

def Body(+b: Nat, +t: Nat) -> T.Layer<BodyP, A(b, t, 32n), A(b, t, 65n), BodyS(b, t)>:
  T.Seq.at(BlockP, TopP, A(b, t, 32n), A(b, t, 32n), A(b, t, 65n), BlockS(b, t), TopS(b, t), Block(b, t), Top(b, t))

# the parameters do not mention b or t: one P for every size, so weights and the training state are shared by all
def P() -> Type:
  T.Two<M(65n, 32n), BodyP>

def S(b: Nat, t: Nat) -> Type:
  T.Two<List<&2, Nat>, BodyS(b, t)>

def X() -> Type:
  List<&2, Nat>

def Y(b: Nat, t: Nat) -> Type:
  A(b, t, 65n)

# The model for b sequences of t positions. A call builds a fresh record of closures (each closure may run once), so the
# program calls Model again for each forward pass, training step or generated character.
def Model(+b: Nat, +t: Nat) -> T.Layer<P, X, Y(b, t), S(b, t)>:
  T.Seq.at(M(65n, 32n), BodyP, X, A(b, t, 32n), Y(b, t), List<&2, Nat>, BodyS(b, t), T.Embedding(R(b, t), 65n, 32n), Body(b, t))

def ce(+b: Nat, +t: Nat, z: Y(b, t), labels: List<&2, Nat>) -> T.LossOut<Y(b, t)>:
  T.CrossEntropy(R(b, t), 65n, z, labels)

# ---- weights: data/tiny/init.bin, in reference/tiny_ref.py's names ----

# L.weights and T.Train.new take the model as a template, so they need a model with literal sizes. P has no size in
# it, so any one serves: 1 sequence of 1 position, whose record only counts, loads and zeroes the parameters.
def load_params() -> IO(P):
  L.weights(~P, ~X, ~Y(1n, 1n), ~S(1n, 1n), ~Model(1n, 1n), "data/tiny/init.bin", T.Layer.zeros(P, X, Y(1n, 1n), S(1n, 1n), Model(1n, 1n)))

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

# ---- batches: b sequences of t + 1 ids a step; the inputs are the first t of each, the targets the last t ----

def nats(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> nats(t)

# the first t ids of each of k sequences of t + 1
def xs_of(k: Nat, +t: Nat, +ids: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case 0n:
      Nil{}
    case 1n+q:
      List.append(&2, Nat, List.take(&2, Nat, ids, t), xs_of(q, t, List.drop(&2, Nat, ids, 1n+t)))

# the last t ids of each of k sequences of t + 1
def ys_of(k: Nat, +t: Nat, +ids: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case 0n:
      Nil{}
    case 1n+q:
      List.append(&2, Nat, List.take(&2, Nat, List.drop(&2, Nat, ids, 1n), t), ys_of(q, t, List.drop(&2, Nat, ids, 1n+t)))

# every id inside the vocabulary of 65 (Embedding would read past its table otherwise)
def ids_ok(bs: List<&2, U32>, ok: Bool) -> Bool:
  match bs ok:
    case _ False{}:
      False{}
    case Nil{} True{}:
      True{}
    case Con{+h, t} True{}:
      ids_ok(t, U32.is_lt(h, 65))

def batch_ids(+path: String, +step: U32, +per: U32, +bs: List<&2, U32>) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    L.require(U32.is_eq(U32.from_nat(List.length(&2, U32, bs)), per), path ++ ": batch " ++ U32.show(step) ++ " is short")
    L.require(ids_ok(bs, True{}), path ++ ": batch " ++ U32.show(step) ++ " holds an id past the vocabulary of 65")
    return nats(bs)

def per_step(+b: Nat, +t: Nat) -> U32:
  U32.from_nat(Nat.mul(b, 1n+t))

def load_batch(+b: Nat, +t: Nat, +path: String, +step: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(path, (step * per_step(b, t) : U32), per_step(b, t))
    batch_ids(path, step, per_step(b, t), bs)

# ---- one forward and backward pass ----

def grads3(r: T.Bwd<P, X>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      print_params("grad", g)

def grads2(+b: Nat, +t: Nat, p: P, s: S(b, t), r: T.LossOut<Y(b, t)>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(T.Layer.bwd(P, X, Y(b, t), S(b, t), Model(b, t), p, s, dy, T.Layer.zeros(P, X, Y(b, t), S(b, t), Model(b, t))))

def grads1(+b: Nat, +t: Nat, +ys: List<&2, Nat>, r: T.Fwd<P, Y(b, t), S(b, t)>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      grads2(b, t, p, s, ce(b, t, y, ys))

def grads(+b: Nat, +t: Nat, +ids: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    grads1(b, t, ys_of(b, t, ids), T.Layer.fwd(P, X, Y(b, t), S(b, t), Model(b, t), p, xs_of(b, t, ids)))

# ---- training ----

def step_with(+b: Nat, +t: Nat, +cfg: T.OptCfg, +ids: List<&2, Nat>, tr: T.Train<P>) -> T.Train<P>:
  T.Train.step_at(P, X, Y(b, t), S(b, t), List<&2, Nat>, Model(b, t), z => ys => ce(b, t, z, ys), cfg, tr, xs_of(b, t, ids), ys_of(b, t, ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +b: Nat, +t: Nat, +path: String, +step: U32, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- load_batch(b, t, path, step)
        tloop(q, b, t, path, (step + 1 : U32), cfg, step_with(b, t, cfg, ids, T.Train{p, g, st, p1, p2, l}))

def cfg() -> T.OptCfg:
  T.CAdamW{0.003, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def train_run(+k: Nat, +b: Nat, +t: Nat, +path: String, p: P) -> IO(T.Train<P>):
  do IO<T.Train<P>>:
    T.OptCfg.check(cfg())
    tloop(k, b, t, path, 0, cfg(), T.Train.new(~P, ~X, ~Y(1n, 1n), ~S(1n, 1n), ~Model(1n, 1n), cfg(), p))

def finish(tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      print_params("final", p)

def train(+k: Nat, +b: Nat, +t: Nat, +path: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, b, t, path, p)
    finish(tr)

# ---- sampling: one sequence of t positions (b = 1), the same parameters ----

type Gen is Type:
  Gen{p: P, ids: List<&2, Nat>}

# the most likely next character: the argmax of the last row of the logits (the first one on a tie, as torch.argmax)
def last_argmax(-r: Nat, m: TA.MRow<r, 65n>) -> Nat:
  match m:
    case TA.MRow{x, row}:
      TA.argmax(row)

def fwd_then(+t: Nat, r: T.Fwd<P, Y(1n, t), S(1n, t)>, k: P -> @+i: Nat -> Gen) -> Gen:
  match r:
    case T.Fwd{p, y, s}:
      k(p, last_argmax(R(1n, t), TA.Mat.read_row(R(1n, t), 65n, y, U32.from_nat(Nat.sub(t, 1n)))))

# k more characters after the window of the last t; acc holds the generated ids, newest first
def gen(k: Nat, +t: Nat, p: P, +win: List<&2, Nat>, +acc: List<&2, Nat>) -> Gen:
  match k:
    case 0n:
      Gen{p, List.reverse(&2, Nat, acc)}
    case 1n+q:
      fwd_then(t, T.Layer.fwd(P, X, Y(1n, t), S(1n, t), Model(1n, t), p, win), p2 => +id => gen(q, t, p2, List.append(&2, Nat, List.drop(&2, Nat, win, 1n), [id]), id <> acc))

# the position of byte c in the vocabulary (0 if absent)
def index_of(voc: List<&2, U32>, +c: U32, +i: Nat) -> Nat:
  match voc:
    case Nil{}:
      0n
    case Con{+v, t}:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a 65-character vocabulary, read when the corpus is encoded not-expected-because: not in the training step
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

def sample2(+n: Nat, +t: Nat, +win: List<&2, Nat>, +voc: List<&2, U32>, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      show_gen(voc, gen(n, t, p, win, Nil{}))

# trains for k steps, then continues the last t characters of the prompt by n characters, greedily
def sample_v(+k: Nat, +n: Nat, +b: Nat, +t: Nat, +path: String, +prompt: String, +voc: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params()
    tr : T.Train<P> <- train_run(k, b, t, path, p)
    sample2(n, t, List.drop(&2, Nat, ids_of(String.to_list(prompt), voc), Nat.sub(String.length(prompt), t)), voc, tr)

def sample(+k: Nat, +n: Nat, +b: Nat, +t: Nat, +path: String, +prompt: String) -> IO(Unit):
  do IO<Unit>:
    voc : List<&2, U32> <- T.read_at("data/tiny/vocab.txt", 0, 65)
    sample_v(k, n, b, t, path, prompt, voc)

# ---- the boundary: B, T and the batch file checked once, then everything is typed ----

def usage() -> IO(Unit):
  IO.print("usage: tiny_rt <B> <T> <batches file> grads | train <steps> | sample <steps> <chars> <prompt of at least T characters>")

def size_done(r: File & Result<&1, &1, U32 & String, U32>) -> IO(U32):
  match r:
    case Tuple{f, res}:
      do IO<U32>:
        +size : U32 <- IO.pass(U32, res)
        File.close(f)
        return size

def file_size(+path: String) -> IO(U32):
  do IO<U32>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    size_done(rr)

def steps_ok(+k: Nat, +nb: U32, +path: String) -> IO(Unit):
  L.require(Nat.is_le(k, U32.to_nat(nb)), path ++ " holds " ++ U32.show(nb) ++ " batches of these sizes, fewer than the " ++ Nat.show(k) ++ " steps asked for")

def mode_g(ok: Bool, +b: Nat, +t: Nat, +path: String) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        ids : List<&2, Nat> <- load_batch(b, t, path, 0)
        grads(b, t, ids)
    case False{}:
      usage()

def mode_t(ok: Bool, +b: Nat, +t: Nat, +path: String, +nb: U32, mk: Maybe<&2, Nat>) -> IO(Unit):
  match ok mk:
    case True{} Some{+k}:
      do IO<Unit>:
        steps_ok(k, nb, path)
        train(k, b, t, path)
    case _ _:
      usage()

def mode_s(ok: Bool, +b: Nat, +t: Nat, +path: String, +nb: U32, mk: Maybe<&2, Nat>, mn: Maybe<&2, Nat>, +prompt: String) -> IO(Unit):
  match ok mk mn:
    case True{} Some{+k} Some{n}:
      do IO<Unit>:
        steps_ok(k, nb, path)
        L.require(Nat.is_le(t, String.length(prompt)), "the prompt has " ++ Nat.show(String.length(prompt)) ++ " characters, fewer than T = " ++ Nat.show(t))
        sample(k, n, b, t, path, prompt)
    case _ _ _:
      usage()

def run_mode(+b: Nat, +t: Nat, +path: String, +nb: U32, rest: List<String>) -> IO(Unit):
  match rest:
    case Con{mode, Nil{}}:
      mode_g(String.eq(mode, "grads"), b, t, path)
    case Con{mode, Con{k, Nil{}}}:
      mode_t(String.eq(mode, "train"), b, t, path, nb, Nat.read(k))
    case Con{mode, Con{k, Con{n, Con{prompt, Nil{}}}}}:
      mode_s(String.eq(mode, "sample"), b, t, path, nb, Nat.read(k), Nat.read(n), prompt)
    case _:
      usage()

# B and T at least 1, B * T rows at most 65,536 (the logits, 65 numbers a row, stay well inside a U32 index), and the
# file a whole number of steps of B sequences of T + 1 ids: a file written for other sizes is refused here
def start(+b: Nat, +t: Nat, +path: String, rest: List<String>) -> IO(Unit):
  do IO<Unit>:
    L.require(Nat.is_le(1n, b), "B must be at least 1")
    L.require(Nat.is_le(1n, t), "T must be at least 1")
    L.require(Nat.is_le(Nat.mul(b, t), 65536n), "B x T must be at most 65536 rows")
    +size : U32 <- file_size(path)
    L.require(U32.is_eq(U32.mod(size, per_step(b, t)), 0), path ++ " holds " ++ U32.show(size) ++ " ids, not a whole number of steps of B x (T + 1) = " ++ U32.show(per_step(b, t)))
    IO.print_err("sizes read at run time: B = " ++ Nat.show(b) ++ ", T = " ++ Nat.show(t) ++ ", " ++ Nat.show(R(b, t)) ++ " rows a step, " ++ U32.show(U32.div(size, per_step(b, t))) ++ " steps in " ++ path)
    run_mode(b, t, path, U32.div(size, per_step(b, t)), rest)

def sized(mb: Maybe<&2, Nat>, mt: Maybe<&2, Nat>, +path: String, rest: List<String>) -> IO(Unit):
  match mb mt:
    case Some{+b} Some{+t}:
      start(b, t, path, rest)
    case _ _:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{bs, Con{ts, Con{path, rest}}}}:
      sized(Nat.read(bs), Nat.read(ts), path, rest)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
