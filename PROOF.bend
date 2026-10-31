# The proofs of LAWS.bend. Every proof is plain Bend: no @unsafe, no ?TODO, no axioms. The arithmetic lemmas come
# from bend-ml-nat-lemmas; the division lemmas behind gqa_head are llama.bend's (MIT, same author).

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec
import ./torch/tensor.bend as TT
import ./LAWS.bend as Laws
import bend-ml-nat-lemmas@0.1.2.0/main.bend as NL
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# =====================================================================================================
# Dot products
# =====================================================================================================

# dot(a + b, v) == dot(a, v) + dot(b, v): induction on the three vectors; at each entry (x + y)·z splits into x·z + y·z
# (distributivity) and add4 regroups the four terms
def dot_vadd(a: List<&2, Nat>, b: List<&2, Nat>, v: List<&2, Nat>) -> {Spec.dotn(Spec.vadd(a, b), v) == Nat.add(Spec.dotn(a, v), Spec.dotn(b, v)) : Nat}:
  match a b v:
    case Nil{} _ _:
      {==}
    case Con{+x, +xt} Nil{} +v:
      Equal.sym(Nat, Nat.add(Spec.dotn(Con{x, xt}, v), 0n), Spec.dotn(Con{x, xt}, v), NL.add_zero(Spec.dotn(Con{x, xt}, v)))
    case Con{x, xt} Con{y, yt} Nil{}:
      {==}
    case Con{+x, +xt} Con{+y, +yt} Con{+z, +zt}:
      +xz = Nat.mul(x, z)
      +yz = Nat.mul(y, z)
      +pa = Spec.dotn(xt, zt)
      +pb = Spec.dotn(yt, zt)
      Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(x, y), z), Spec.dotn(Spec.vadd(xt, yt), zt)), Nat.add(Nat.mul(Nat.add(x, y), z), Nat.add(pa, pb)), Nat.add(Nat.add(xz, pa), Nat.add(yz, pb)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(Nat.add(x, y), z), s), Spec.dotn(Spec.vadd(xt, yt), zt), Nat.add(pa, pb), dot_vadd(xt, yt, zt)),
        Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(x, y), z), Nat.add(pa, pb)), Nat.add(Nat.add(xz, yz), Nat.add(pa, pb)), Nat.add(Nat.add(xz, pa), Nat.add(yz, pb)),
          Equal.cong(Nat, Nat, s => Nat.add(s, Nat.add(pa, pb)), Nat.mul(Nat.add(x, y), z), Nat.add(xz, yz), Equal.sym(Nat, Nat.add(xz, yz), Nat.mul(Nat.add(x, y), z), NL.mul_dist(x, y, z))),
          NL.add4(xz, yz, pa, pb)))

# dot(k·r, v) == k · dot(r, v): at each entry (k·x)·z == k·(x·z) (associativity), then k factors out of the sum
def dot_scale(+k: Nat, r: List<&2, Nat>, v: List<&2, Nat>) -> {Spec.dotn(Spec.vscale(k, r), v) == Nat.mul(k, Spec.dotn(r, v)) : Nat}:
  match r v:
    case Nil{} _:
      Equal.sym(Nat, Nat.mul(k, 0n), 0n, NL.mul_zero(k))
    case Con{x, xt} Nil{}:
      Equal.sym(Nat, Nat.mul(k, 0n), 0n, NL.mul_zero(k))
    case Con{+x, +xt} Con{+z, +zt}:
      +d = Spec.dotn(xt, zt)
      Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(k, x), z), Spec.dotn(Spec.vscale(k, xt), zt)), Nat.add(Nat.mul(Nat.mul(k, x), z), Nat.mul(k, d)), Nat.mul(k, Nat.add(Nat.mul(x, z), d)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(Nat.mul(k, x), z), s), Spec.dotn(Spec.vscale(k, xt), zt), Nat.mul(k, d), dot_scale(k, xt, zt)),
        Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(k, x), z), Nat.mul(k, d)), Nat.add(Nat.mul(k, Nat.mul(x, z)), Nat.mul(k, d)), Nat.mul(k, Nat.add(Nat.mul(x, z), d)),
          Equal.cong(Nat, Nat, s => Nat.add(s, Nat.mul(k, d)), Nat.mul(Nat.mul(k, x), z), Nat.mul(k, Nat.mul(x, z)), Equal.sym(Nat, Nat.mul(k, Nat.mul(x, z)), Nat.mul(Nat.mul(k, x), z), NL.mul_assoc(k, x, z))),
          Equal.sym(Nat, Nat.mul(k, Nat.add(Nat.mul(x, z), d)), Nat.add(Nat.mul(k, Nat.mul(x, z)), Nat.mul(k, d)), NL.mul_dist_l(k, Nat.mul(x, z), d))))

# dot(a, b) == dot(b, a)
def dot_comm(a: List<&2, Nat>, b: List<&2, Nat>) -> {Spec.dotn(a, b) == Spec.dotn(b, a) : Nat}:
  match a b:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{y, yt}:
      {==}
    case Con{x, xt} Nil{}:
      {==}
    case Con{+x, +xt} Con{+y, +yt}:
      Equal.trans(Nat, Nat.add(Nat.mul(x, y), Spec.dotn(xt, yt)), Nat.add(Nat.mul(y, x), Spec.dotn(xt, yt)), Nat.add(Nat.mul(y, x), Spec.dotn(yt, xt)),
        Equal.cong(Nat, Nat, s => Nat.add(s, Spec.dotn(xt, yt)), Nat.mul(x, y), Nat.mul(y, x), NL.mul_comm(x, y)),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(y, x), s), Spec.dotn(xt, yt), Spec.dotn(yt, xt), dot_comm(xt, yt)))

# =====================================================================================================
# The linear laws
# =====================================================================================================

# PROOF (adjoint): induction on the rows of W and the entries of dy. Row r with weight g_i contributes g_i·(r·x) on the
# left and (g_i·r)·x on the right (dot_scale); the right side's sum of weighted rows splits by dot_vadd.
def Laws.adjoint(w, g, v):
  match w g:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{gi, gt}:
      {==}
    case Con{r, rt} Nil{}:
      {==}
    case Con{+r, +rt} Con{+gi, +gt}:
      +a = Spec.dotn(Spec.vscale(gi, r), v)
      +m = Spec.mtv(rt, gt)
      Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(gt, Spec.mv(rt, v))), Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(m, v)), Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(gi, Spec.dotn(r, v)), s), Spec.dotn(gt, Spec.mv(rt, v)), Spec.dotn(m, v), Laws.adjoint(rt, gt, v)),
        Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(m, v)), Nat.add(a, Spec.dotn(m, v)), Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v),
          Equal.cong(Nat, Nat, s => Nat.add(s, Spec.dotn(m, v)), Nat.mul(gi, Spec.dotn(r, v)), a, Equal.sym(Nat, a, Nat.mul(gi, Spec.dotn(r, v)), dot_scale(gi, r, v))),
          Equal.sym(Nat, Spec.dotn(Spec.vadd(Spec.vscale(gi, r), m), v), Nat.add(a, Spec.dotn(m, v)), dot_vadd(Spec.vscale(gi, r), m, v))))

# PROOF (weight_adjoint): row by row, g_i·(r·x) == (g_i·x)·r, by dot_comm and dot_scale.
def Laws.weight_adjoint(w, g, v):
  match w g:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{gi, gt}:
      {==}
    case Con{r, rt} Nil{}:
      {==}
    case Con{+r, +rt} Con{+gi, +gt}:
      +f = Spec.frob(Spec.outer(gt, v), rt)
      Equal.trans(Nat, Nat.add(Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(gt, Spec.mv(rt, v))), Nat.add(Nat.mul(gi, Spec.dotn(r, v)), f), Nat.add(Spec.dotn(Spec.vscale(gi, v), r), f),
        Equal.cong(Nat, Nat, s => Nat.add(Nat.mul(gi, Spec.dotn(r, v)), s), Spec.dotn(gt, Spec.mv(rt, v)), f, Laws.weight_adjoint(rt, gt, v)),
        Equal.cong(Nat, Nat, s => Nat.add(s, f), Nat.mul(gi, Spec.dotn(r, v)), Spec.dotn(Spec.vscale(gi, v), r),
          Equal.trans(Nat, Nat.mul(gi, Spec.dotn(r, v)), Nat.mul(gi, Spec.dotn(v, r)), Spec.dotn(Spec.vscale(gi, v), r),
            Equal.cong(Nat, Nat, s => Nat.mul(gi, s), Spec.dotn(r, v), Spec.dotn(v, r), dot_comm(r, v)),
            Equal.sym(Nat, Spec.dotn(Spec.vscale(gi, v), r), Nat.mul(gi, Spec.dotn(v, r)), dot_scale(gi, v, r)))))

# PROOF (adjoint_seq): adjoint once for W2 (with W1·x as the input), then once for W1.
def Laws.adjoint_seq(w1, w2, g, v):
  Equal.trans(Nat, Spec.dotn(g, Spec.mv(w2, Spec.mv(w1, v))), Spec.dotn(Spec.mtv(w2, g), Spec.mv(w1, v)), Spec.dotn(Spec.mtv(w1, Spec.mtv(w2, g)), v), Laws.adjoint(w2, g, Spec.mv(w1, v)), Laws.adjoint(w1, Spec.mtv(w2, g), v))

# =====================================================================================================
# Causal attention
# =====================================================================================================

# PROOF (seq_chain): Seq's forward pass computes W2·(W1·x) and saves x and W1·x; its backward pass computes
# W1ᵀ·(W2ᵀ·dy). Every step of the combinator reduces, so this is adjoint_seq.
def Laws.seq_chain(w1, w2, x, dy, g1, g2):
  Laws.adjoint_seq(w1, w2, dy, x)

# PROOF (fork_sum): Fork's backward pass is W1ᵀ·dy1 + W2ᵀ·dy2 (vadd, the add it was given); the dot of a sum is the sum
# of the dots (dot_vadd), and each is its layer's adjoint law.
def Laws.fork_sum(w1, w2, x, dy1, dy2, g1, g2):
  Equal.trans(Nat, Nat.add(Spec.dotn(dy1, Spec.mv(w1, x)), Spec.dotn(dy2, Spec.mv(w2, x))), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x), Spec.dotn(Spec.mtv(w2, dy2), x)), Spec.dotn(Spec.vadd(Spec.mtv(w1, dy1), Spec.mtv(w2, dy2)), x),
    Equal.trans(Nat, Nat.add(Spec.dotn(dy1, Spec.mv(w1, x)), Spec.dotn(dy2, Spec.mv(w2, x))), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x), Spec.dotn(dy2, Spec.mv(w2, x))), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x), Spec.dotn(Spec.mtv(w2, dy2), x)),
      Equal.cong(Nat, Nat, z => Nat.add(z, Spec.dotn(dy2, Spec.mv(w2, x))), Spec.dotn(dy1, Spec.mv(w1, x)), Spec.dotn(Spec.mtv(w1, dy1), x), Laws.adjoint(w1, dy1, x)),
      Equal.cong(Nat, Nat, z => Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x), z), Spec.dotn(dy2, Spec.mv(w2, x)), Spec.dotn(Spec.mtv(w2, dy2), x), Laws.adjoint(w2, dy2, x))),
    Equal.sym(Nat, Spec.dotn(Spec.vadd(Spec.mtv(w1, dy1), Spec.mtv(w2, dy2)), x), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x), Spec.dotn(Spec.mtv(w2, dy2), x)), dot_vadd(Spec.mtv(w1, dy1), Spec.mtv(w2, dy2), x)))

# PROOF (both_sum): Both runs each layer on its own side, so both sides are pairs of linear passes; each half of the sum
# is its layer's adjoint law.
def Laws.both_sum(w1, w2, x1, x2, dy1, dy2, g1, g2):
  Equal.trans(Nat, Nat.add(Spec.dotn(dy1, Spec.mv(w1, x1)), Spec.dotn(dy2, Spec.mv(w2, x2))), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x1), Spec.dotn(dy2, Spec.mv(w2, x2))), Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x1), Spec.dotn(Spec.mtv(w2, dy2), x2)),
    Equal.cong(Nat, Nat, z => Nat.add(z, Spec.dotn(dy2, Spec.mv(w2, x2))), Spec.dotn(dy1, Spec.mv(w1, x1)), Spec.dotn(Spec.mtv(w1, dy1), x1), Laws.adjoint(w1, dy1, x1)),
    Equal.cong(Nat, Nat, z => Nat.add(Spec.dotn(Spec.mtv(w1, dy1), x1), z), Spec.dotn(dy2, Spec.mv(w2, x2)), Spec.dotn(Spec.mtv(w2, dy2), x2), Laws.adjoint(w2, dy2, x2)))

# PROOF (residual): the forward pass gives x + W·x and the backward pass dy + Wᵀ·dy (Id hands dy back, the linear layer
# Wᵀ·dy, Fork adds them). Then dy·(x + W·x) = x·dy + (W·x)·dy (dot_comm, dot_vadd) = dy·x + dy·(W·x) (dot_comm twice)
# = dy·x + (Wᵀ·dy)·x (adjoint) = (dy + Wᵀ·dy)·x (dot_vadd).
def Laws.residual(w, x, dy, g):
  +wx = Spec.mv(w, x)
  +wd = Spec.mtv(w, dy)
  Equal.trans(Nat, Spec.dotn(dy, Spec.vadd(x, wx)), Spec.dotn(Spec.vadd(x, wx), dy), Spec.dotn(Spec.vadd(dy, wd), x),
    dot_comm(dy, Spec.vadd(x, wx)),
    Equal.trans(Nat, Spec.dotn(Spec.vadd(x, wx), dy), Nat.add(Spec.dotn(x, dy), Spec.dotn(wx, dy)), Spec.dotn(Spec.vadd(dy, wd), x),
      dot_vadd(x, wx, dy),
      Equal.trans(Nat, Nat.add(Spec.dotn(x, dy), Spec.dotn(wx, dy)), Nat.add(Spec.dotn(dy, x), Spec.dotn(wx, dy)), Spec.dotn(Spec.vadd(dy, wd), x),
        Equal.cong(Nat, Nat, z => Nat.add(z, Spec.dotn(wx, dy)), Spec.dotn(x, dy), Spec.dotn(dy, x), dot_comm(x, dy)),
        Equal.trans(Nat, Nat.add(Spec.dotn(dy, x), Spec.dotn(wx, dy)), Nat.add(Spec.dotn(dy, x), Spec.dotn(wd, x)), Spec.dotn(Spec.vadd(dy, wd), x),
          Equal.cong(Nat, Nat, z => Nat.add(Spec.dotn(dy, x), z), Spec.dotn(wx, dy), Spec.dotn(wd, x),
            Equal.trans(Nat, Spec.dotn(wx, dy), Spec.dotn(dy, wx), Spec.dotn(wd, x), dot_comm(wx, dy), Laws.adjoint(w, dy, x))),
          Equal.sym(Nat, Spec.dotn(Spec.vadd(dy, wd), x), Nat.add(Spec.dotn(dy, x), Spec.dotn(wd, x)), dot_vadd(dy, wd, x))))))

def disc(b: Bool) -> Type:
  match b:
    case True{}:
      Unit
    case False{}:
      Empty

# False == True is impossible: rewriting the goal Empty = disc(False) with it gives disc(True) = Unit
def false_true(e: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool, False{}, True{}, e) : disc(_)
  Unit{}

# xs reversed onto acc has the length of both
def len_rcat(+xs: List<&2, F32>, +acc: List<&2, F32>) -> {List.length(&2, F32, T.rcat(xs, acc)) == Nat.add(List.length(&2, F32, xs), List.length(&2, F32, acc)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      Equal.trans(Nat, List.length(&2, F32, T.rcat(t, x <> acc)), Nat.add(List.length(&2, F32, t), 1n+List.length(&2, F32, acc)), 1n+Nat.add(List.length(&2, F32, t), List.length(&2, F32, acc)), len_rcat(t, x <> acc), Equal.sym(Nat, 1n+Nat.add(List.length(&2, F32, t), List.length(&2, F32, acc)), Nat.add(List.length(&2, F32, t), 1n+List.length(&2, F32, acc)), NL.add_succ(List.length(&2, F32, t), List.length(&2, F32, acc))))

# drow's row has the current dot, the m after it, and acc: 1 + m + |acc| numbers
def len_drow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, +acc: List<&2, F32>, r: T.D2) -> {List.length(&2, F32, T.dr_row(T.drow(m, hd, ia, ib, bs, acc, r))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}:
  match m r:
    case 0n T.D2{a, b, +s}:
      Equal.trans(Nat, List.length(&2, F32, T.rcat(s <> acc, Nil{})), Nat.add(1n+List.length(&2, F32, acc), 0n), 1n+List.length(&2, F32, acc), len_rcat(s <> acc, Nil{}), NL.add_zero(1n+List.length(&2, F32, acc)))
    case 1n++p T.D2{a, b, +s}:
      Equal.trans(Nat, List.length(&2, F32, T.dr_row(T.drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)))), Nat.add(1n+p, 1n+List.length(&2, F32, acc)), 2n+Nat.add(p, List.length(&2, F32, acc)), len_drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 1n+List.length(&2, F32, acc)), 1n+Nat.add(p, List.length(&2, F32, acc)), Equal.sym(Nat, 1n+Nat.add(p, List.length(&2, F32, acc)), Nat.add(p, 1n+List.length(&2, F32, acc)), NL.add_succ(p, List.length(&2, F32, acc)))))

# PROOF (causal_keys): qrow is drow(i, ..) from an empty accumulator, so its row has 1 + i + 0 numbers (len_drow).
def Laws.causal_keys(i, hd, qs, qc, ks, kc, q, k):
  Equal.trans(Nat, List.length(&2, F32, T.dr_row(T.qrow(i, hd, qs, qc, ks, kc, q, k))), Nat.add(1n+i, 0n), 1n+i, len_drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks, Nil{}, T.dot0(hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, 1, 1, q, k)), NL.add_zero(1n+i))

# n - 0 is n
def sub0(+n: Nat) -> {Nat.sub(n, 0n) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

# the division loop of a softmax counts the 1 + k numbers it divides onto c (pr_of hands the count on as the row's)
def cnt_div(q: Array<F32>, k: Array<F32>, kk: Nat, +idx: U32, +s: F32, +c: Nat, r: Array<F32> & F32) -> {T.pr_n(T.pr_of(q, k, T.pdiv(kk, idx, s, c, r))) == Nat.add(1n+kk, c) : Nat}:
  match kk r:
    case 0n Tuple{a, +x}:
      {==}
    case 1n++j Tuple{a, +x}:
      Equal.trans(Nat, T.pr_n(T.pr_of(q, k, T.pdiv(j, (idx + 1 : U32), s, 1n+c, Array.get(F32, Array.set(F32, a, idx, (x / s : F32)), (idx + 1 : U32))))), Nat.add(1n+j, 1n+c), 2n+Nat.add(j, c), cnt_div(q, k, j, (idx + 1 : U32), s, 1n+c, Array.get(F32, Array.set(F32, a, idx, (x / s : F32)), (idx + 1 : U32))), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(j, 1n+c), 1n+Nat.add(j, c), Equal.sym(Nat, 1n+Nat.add(j, c), Nat.add(j, 1n+c), NL.add_succ(j, c))))

# the whole softmax over 1 + kk numbers reports 1 + kk, whatever max and exp found (they are matched, not computed)
def cnt_psm3(q: Array<F32>, k: Array<F32>, +kk: Nat, +off: U32, r: T.AS) -> {T.pr_n(T.pr_of(q, k, T.psm3(kk, off, r))) == 1n+kk : Nat}:
  match r:
    case T.AS{a, +s}:
      Equal.trans(Nat, T.pr_n(T.pr_of(q, k, T.pdiv(kk, off, s, 0n, Array.get(F32, a, off)))), Nat.add(1n+kk, 0n), 1n+kk, cnt_div(q, k, kk, off, s, 0n, Array.get(F32, a, off)), NL.add_zero(1n+kk))

def cnt_psm2(q: Array<F32>, k: Array<F32>, +kk: Nat, +off: U32, r: T.AS) -> {T.pr_n(T.pr_of(q, k, T.psm2(kk, off, r))) == 1n+kk : Nat}:
  match r:
    case T.AS{a, +m}:
      cnt_psm3(q, k, kk, off, T.pexp(kk, off, m, 0.0, Array.get(F32, a, off)))

# pw writes xs and counts them from n; the softmax prow3 runs on them reports 1 + (|xs| + n - 1)
def cnt_pw(+xs: List<&2, F32>, +idx: U32, +sc: F32, +n: Nat, +off: U32, q: Array<F32>, k: Array<F32>, p: Array<F32>) -> {T.pr_n(T.prow3(off, q, k, T.pw(xs, idx, sc, n, p))) == 1n+Nat.sub(Nat.add(List.length(&2, F32, xs), n), 1n) : Nat}:
  match xs:
    case Nil{}:
      cnt_psm2(q, k, Nat.sub(n, 1n), off, T.pmax0(Nat.sub(n, 1n), off, Array.get(F32, p, off)))
    case Con{+x, +t}:
      Equal.trans(Nat, T.pr_n(T.prow3(off, q, k, T.pw(t, (idx + 1 : U32), sc, 1n+n, Array.set(F32, p, idx, (x * sc : F32))))), 1n+Nat.sub(Nat.add(List.length(&2, F32, t), 1n+n), 1n), 1n+Nat.sub(1n+Nat.add(List.length(&2, F32, t), n), 1n), cnt_pw(t, (idx + 1 : U32), sc, 1n+n, off, q, k, Array.set(F32, p, idx, (x * sc : F32))), Equal.cong(Nat, Nat, z => 1n+Nat.sub(z, 1n), Nat.add(List.length(&2, F32, t), 1n+n), 1n+Nat.add(List.length(&2, F32, t), n), Equal.sym(Nat, 1n+Nat.add(List.length(&2, F32, t), n), Nat.add(List.length(&2, F32, t), 1n+n), NL.add_succ(List.length(&2, F32, t), n))))

# the count prow2 hands on for drow's row: 1 + m + |acc|, as len_drow, through cnt_pw
def cnt_drow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, +acc: List<&2, F32>, r: T.D2, +sc: F32, +off: U32, pa: Array<F32>) -> {T.pr_n(T.prow2(sc, off, pa, T.drow(m, hd, ia, ib, bs, acc, r))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}:
  match m r:
    case 0n T.D2{a, b, +s}:
      +rw = T.rcat(s <> acc, Nil{})
      +la = List.length(&2, F32, acc)
      Equal.trans(Nat, T.pr_n(T.prow3(off, a, b, T.pw(rw, off, sc, 0n, pa))), 1n+Nat.sub(Nat.add(List.length(&2, F32, rw), 0n), 1n), 1n+la, cnt_pw(rw, off, sc, 0n, off, a, b, pa),
        Equal.cong(Nat, Nat, z => 1n+z, Nat.sub(Nat.add(List.length(&2, F32, rw), 0n), 1n), la,
          Equal.trans(Nat, Nat.sub(Nat.add(List.length(&2, F32, rw), 0n), 1n), Nat.sub(1n+la, 1n), la,
            Equal.cong(Nat, Nat, z => Nat.sub(z, 1n), Nat.add(List.length(&2, F32, rw), 0n), 1n+la,
              Equal.trans(Nat, Nat.add(List.length(&2, F32, rw), 0n), List.length(&2, F32, rw), 1n+la, NL.add_zero(List.length(&2, F32, rw)),
                Equal.trans(Nat, List.length(&2, F32, rw), Nat.add(1n+la, 0n), 1n+la, len_rcat(s <> acc, Nil{}), NL.add_zero(1n+la)))),
            sub0(la))))
    case 1n++p T.D2{a, b, +s}:
      Equal.trans(Nat, T.pr_n(T.prow2(sc, off, pa, T.drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)))), Nat.add(1n+p, 1n+List.length(&2, F32, acc)), 2n+Nat.add(p, List.length(&2, F32, acc)), cnt_drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b), sc, off, pa), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 1n+List.length(&2, F32, acc)), 1n+Nat.add(p, List.length(&2, F32, acc)), Equal.sym(Nat, 1n+Nat.add(p, List.length(&2, F32, acc)), Nat.add(p, 1n+List.length(&2, F32, acc)), NL.add_succ(p, List.length(&2, F32, acc)))))

# PROOF (causal): prow runs prow2 on qrow's row, drow(i, ..) from an empty accumulator, so the count it hands the
# softmax is 1 + i + 0 (cnt_drow).
def Laws.causal(i, t, hd, sc, qs, qc, ks, kc, q, k, p):
  Equal.trans(Nat, T.pr_n(T.prow(i, t, hd, sc, qs, qc, ks, kc, q, k, p)), Nat.add(1n+i, 0n), 1n+i, cnt_drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks, Nil{}, T.dot0(hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, 1, 1, q, k), sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p), NL.add_zero(1n+i))

# gw writes xs and counts them from n: gr3 hands on |xs| + n (as cnt_pw for the forward pass)
def cnt_gw(+xs: List<&2, F32>, +idx: U32, +n: Nat, +sc: F32, +off: U32, d: Array<F32>, v: Array<F32>, p: Array<F32>, g: Array<F32>) -> {T.gr_n(T.gr3(sc, off, d, v, p, T.gw(xs, idx, n, g))) == Nat.add(List.length(&2, F32, xs), n) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{+x, +t}:
      Equal.trans(Nat, T.gr_n(T.gr3(sc, off, d, v, p, T.gw(t, (idx + 1 : U32), 1n+n, Array.set(F32, g, idx, x)))), Nat.add(List.length(&2, F32, t), 1n+n), 1n+Nat.add(List.length(&2, F32, t), n), cnt_gw(t, (idx + 1 : U32), 1n+n, sc, off, d, v, p, Array.set(F32, g, idx, x)), Equal.sym(Nat, 1n+Nat.add(List.length(&2, F32, t), n), Nat.add(List.length(&2, F32, t), 1n+n), NL.add_succ(List.length(&2, F32, t), n)))

# the count gr2 hands on for drow's row: 1 + m + |acc|, as len_drow, through cnt_gw (cnt_drow's twin)
def cnt_gdrow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, +acc: List<&2, F32>, r: T.D2, +sc: F32, +off: U32, pa: Array<F32>, ga: Array<F32>) -> {T.gr_n(T.gr2(sc, off, pa, ga, T.drow(m, hd, ia, ib, bs, acc, r))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}:
  match m r:
    case 0n T.D2{a, b, +s}:
      +rw = T.rcat(s <> acc, Nil{})
      Equal.trans(Nat, T.gr_n(T.gr3(sc, off, a, b, pa, T.gw(rw, off, 0n, ga))), Nat.add(List.length(&2, F32, rw), 0n), 1n+List.length(&2, F32, acc), cnt_gw(rw, off, 0n, sc, off, a, b, pa, ga),
        Equal.trans(Nat, Nat.add(List.length(&2, F32, rw), 0n), List.length(&2, F32, rw), 1n+List.length(&2, F32, acc), NL.add_zero(List.length(&2, F32, rw)),
          Equal.trans(Nat, List.length(&2, F32, rw), Nat.add(1n+List.length(&2, F32, acc), 0n), 1n+List.length(&2, F32, acc), len_rcat(s <> acc, Nil{}), NL.add_zero(1n+List.length(&2, F32, acc)))))
    case 1n++p T.D2{a, b, +s}:
      Equal.trans(Nat, T.gr_n(T.gr2(sc, off, pa, ga, T.drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b)))), Nat.add(1n+p, 1n+List.length(&2, F32, acc)), 2n+Nat.add(p, List.length(&2, F32, acc)), cnt_gdrow(p, hd, ia, (ib + bs : U32), bs, s <> acc, T.dot0(hd, ia, (ib + bs : U32), 1, 1, a, b), sc, off, pa, ga), Equal.cong(Nat, Nat, z => 1n+z, Nat.add(p, 1n+List.length(&2, F32, acc)), 1n+Nat.add(p, List.length(&2, F32, acc)), Equal.sym(Nat, 1n+Nat.add(p, List.length(&2, F32, acc)), Nat.add(p, 1n+List.length(&2, F32, acc)), NL.add_succ(p, List.length(&2, F32, acc)))))

# PROOF (causal_grad): grow runs gr2 on drow(i, ..) from an empty accumulator, so the count is 1 + i + 0 (cnt_gdrow).
def Laws.causal_grad(i, t, hd, sc, ds, dc, vs, vc, d, v, p, g):
  Equal.trans(Nat, T.gr_n(T.grow(i, t, hd, sc, ds, dc, vs, vc, d, v, p, g)), Nat.add(1n+i, 0n), 1n+i, cnt_gdrow(i, hd, ((U32.from_nat(i) * ds : U32) + dc : U32), vc, vs, Nil{}, T.dot0(hd, ((U32.from_nat(i) * ds : U32) + dc : U32), vc, 1, 1, d, v), sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g), NL.add_zero(1n+i))

# =====================================================================================================
# Attention's trees of sequences: parallel = sequential
# =====================================================================================================

# a ++ b == a2 ++ b2 from a == a2 and b == b2
def app2(-A: Type, -a: List<&1, A>, -a2: List<&1, A>, -b: List<&1, A>, -b2: List<&1, A>, ea: {a == a2 : List<&1, A>}, eb: {b == b2 : List<&1, A>}) -> {List.append(&1, A, a, b) == List.append(&1, A, a2, b2) : List<&1, A>}:
  Equal.trans(List<&1, A>, List.append(&1, A, a, b), List.append(&1, A, a2, b), List.append(&1, A, a2, b2), Equal.cong(List<&1, A>, List<&1, A>, z => List.append(&1, A, z, b), a, a2, ea), Equal.cong(List<&1, A>, List<&1, A>, z => List.append(&1, A, a2, z), b, b2, eb))

# ---- the sequential map the tree is compared with: f(x) = fs_o(fheads(h, 0, dm, x)) for each x, in order ----

def fmap(+h: Nat, +dm: T.Dm, xs: List<&1, T.FS>) -> List<&1, Array<F32>>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      T.fs_o(T.fheads(h, 0n, dm, x)) <> fmap(h, dm, t)

# ---- fleaf (the code's loop with an accumulator) is the map pushed onto the accumulator ----

type L1R<-h: Nat, -dm: T.Dm, -xs: List<&1, T.FS>, -acc: List<&1, Array<F32>>> is Type:
  L1R{ys: List<&1, T.FS>, same: {ys == xs : List<&1, T.FS>}, fact: {T.fleaf(xs, h, dm, acc) == List.reverse.go(&1, Array<F32>, acc, fmap(h, dm, xs)) : List<&1, Array<F32>>}}

def l1_con(x: T.FS, -t: List<&1, T.FS>, +h: Nat, +dm: T.Dm, -acc: List<&1, Array<F32>>, r: L1R<h, dm, t, T.fs_o(T.fheads(h, 0n, dm, x)) <> acc>) -> L1R<h, dm, x <> t, acc>:
  match r:
    case L1R{ys, same, fact}:
      L1R{x <> ys, Equal.cong(List<&1, T.FS>, List<&1, T.FS>, z => x <> z, ys, t, same), fact}

def l1(xs: List<&1, T.FS>, +h: Nat, +dm: T.Dm, -acc: List<&1, Array<F32>>) -> L1R<h, dm, xs, acc>:
  match xs:
    case Nil{}:
      L1R{Nil{}, {==}, {==}}
    case Con{x, t}:
      l1_con(x, t, h, dm, acc, l1(t, h, dm, T.fs_o(T.fheads(h, 0n, dm, x)) <> acc))

# ---- the map of two lists appended is the two maps appended ----

type MAR<-h: Nat, -dm: T.Dm, -a: List<&1, T.FS>, -b: List<&1, T.FS>> is Type:
  MAR{ys: List<&1, T.FS>, same: {ys == a : List<&1, T.FS>}, fact: {fmap(h, dm, List.append(&1, T.FS, a, b)) == List.append(&1, Array<F32>, fmap(h, dm, a), fmap(h, dm, b)) : List<&1, Array<F32>>}}

def ma_con(x: T.FS, -t: List<&1, T.FS>, +h: Nat, +dm: T.Dm, -b: List<&1, T.FS>, r: MAR<h, dm, t, b>) -> MAR<h, dm, x <> t, b>:
  match r:
    case MAR{ys, same, fact}:
      MAR{x <> ys, Equal.cong(List<&1, T.FS>, List<&1, T.FS>, z => x <> z, ys, t, same), Equal.cong(List<&1, Array<F32>>, List<&1, Array<F32>>, z => T.fs_o(T.fheads(h, 0n, dm, x)) <> z, fmap(h, dm, List.append(&1, T.FS, t, b)), List.append(&1, Array<F32>, fmap(h, dm, t), fmap(h, dm, b)), fact)}

def map_app(a: List<&1, T.FS>, +h: Nat, +dm: T.Dm, -b: List<&1, T.FS>) -> MAR<h, dm, a, b>:
  match a:
    case Nil{}:
      MAR{Nil{}, {==}, {==}}
    case Con{x, t}:
      ma_con(x, t, h, dm, b, map_app(t, h, dm, b))

# ---- a reversed onto b, then c appended: a reversed onto b ++ c ----

type GAR<-a: List<&1, T.FS>, -b: List<&1, T.FS>, -c: List<&1, T.FS>> is Type:
  GAR{ys: List<&1, T.FS>, same: {ys == a : List<&1, T.FS>}, fact: {List.append(&1, T.FS, List.reverse.go(&1, T.FS, a, b), c) == List.reverse.go(&1, T.FS, a, List.append(&1, T.FS, b, c)) : List<&1, T.FS>}}

def ga_con(x: T.FS, -t: List<&1, T.FS>, -b: List<&1, T.FS>, -c: List<&1, T.FS>, r: GAR<t, x <> b, c>) -> GAR<x <> t, b, c>:
  match r:
    case GAR{ys, same, fact}:
      GAR{x <> ys, Equal.cong(List<&1, T.FS>, List<&1, T.FS>, z => x <> z, ys, t, same), fact}

def go_app(a: List<&1, T.FS>, -b: List<&1, T.FS>, -c: List<&1, T.FS>) -> GAR<a, b, c>:
  match a:
    case Nil{}:
      GAR{Nil{}, {==}, {==}}
    case Con{x, t}:
      ga_con(x, t, b, c, go_app(t, x <> b, c))

# ---- cut gives two halves that, appended, are acc reversed onto xs ----

type CGR<-k: Nat, -xs: List<&1, T.FS>, -acc: List<&1, T.FS>> is Type:
  CGR{hh: T.Halves<T.FS>, same: {hh == T.cut(T.FS, k, xs, acc) : T.Halves<T.FS>}, cat: {T.hcat(T.FS, hh) == List.reverse.go(&1, T.FS, acc, xs) : List<&1, T.FS>}}

def cg_base(-k: Nat, xs: List<&1, T.FS>, -acc: List<&1, T.FS>, e: {T.cut(T.FS, k, xs, acc) == T.Halves{List.reverse(&1, T.FS, acc), xs} : T.Halves<T.FS>}, r: GAR<acc, Nil{}, xs>) -> CGR<k, xs, acc>:
  match r:
    case GAR{ys, +same, fact}:
      CGR{T.Halves{List.reverse(&1, T.FS, ys), xs},
        Equal.trans(T.Halves<T.FS>, T.Halves{List.reverse(&1, T.FS, ys), xs}, T.Halves{List.reverse(&1, T.FS, acc), xs}, T.cut(T.FS, k, xs, acc), Equal.cong(List<&1, T.FS>, T.Halves<T.FS>, z => T.Halves{List.reverse(&1, T.FS, z), xs}, ys, acc, same), Equal.sym(T.Halves<T.FS>, T.cut(T.FS, k, xs, acc), T.Halves{List.reverse(&1, T.FS, acc), xs}, e)),
        Equal.trans(List<&1, T.FS>, List.append(&1, T.FS, List.reverse.go(&1, T.FS, ys, Nil{}), xs), List.append(&1, T.FS, List.reverse.go(&1, T.FS, acc, Nil{}), xs), List.reverse.go(&1, T.FS, acc, xs), Equal.cong(List<&1, T.FS>, List<&1, T.FS>, z => List.append(&1, T.FS, List.reverse.go(&1, T.FS, z, Nil{}), xs), ys, acc, same), fact)}

def cg_con(-m: Nat, -x: T.FS, -t: List<&1, T.FS>, -acc: List<&1, T.FS>, r: CGR<m, t, x <> acc>) -> CGR<1n+m, x <> t, acc>:
  match r:
    case CGR{hh, same, cat}:
      CGR{hh, same, cat}

def cg(k: Nat, xs: List<&1, T.FS>, acc: List<&1, T.FS>) -> CGR<k, xs, acc>:
  match k xs:
    case 1n+m Con{x, t}:
      cg_con(m, x, t, acc, cg(m, t, x <> acc))
    case 0n ys:
      cg_base(0n, ys, acc, {==}, go_app(acc, Nil{}, ys))
    case 1n+m Nil{}:
      cg_base(1n+m, Nil{}, acc, {==}, go_app(acc, Nil{}, Nil{}))

# ---- the tree: fpar at depth d on two halves is the map of the halves appended ----

type MR<-d: Nat, -gs: T.Halves<T.FS>, -n: Nat, -h: Nat, -dm: T.Dm> is Type:
  MR{ys: List<&1, T.FS>, same: {ys == T.hcat(T.FS, gs) : List<&1, T.FS>}, fact: {T.fpar(d, gs, n, h, dm) == fmap(h, dm, T.hcat(T.FS, gs)) : List<&1, Array<F32>>}}

# depth 0: the two leaves' loops, appended
def m0c(-l: List<&1, T.FS>, -r: List<&1, T.FS>, -n: Nat, +h: Nat, +dm: T.Dm, -l1y: List<&1, T.FS>, +sl: {l1y == l : List<&1, T.FS>}, fl: {T.fleaf(l, h, dm, Nil{}) == fmap(h, dm, l) : List<&1, Array<F32>>}, r1: List<&1, T.FS>, sr: {r1 == r : List<&1, T.FS>}, fr: {T.fleaf(r, h, dm, Nil{}) == fmap(h, dm, r) : List<&1, Array<F32>>}, c: MAR<h, dm, l1y, r>) -> MR<0n, T.Halves{l, r}, n, h, dm>:
  match c:
    case MAR{l2, s2, fa}:
      MR{List.append(&1, T.FS, l2, r1), app2(T.FS, l2, l, r1, r, Equal.trans(List<&1, T.FS>, l2, l1y, l, s2, sl), sr),
        Equal.trans(List<&1, Array<F32>>, List.append(&1, Array<F32>, T.fleaf(l, h, dm, Nil{}), T.fleaf(r, h, dm, Nil{})), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), fmap(h, dm, List.append(&1, T.FS, l, r)), app2(Array<F32>, T.fleaf(l, h, dm, Nil{}), fmap(h, dm, l), T.fleaf(r, h, dm, Nil{}), fmap(h, dm, r), fl, fr),
          Equal.sym(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, l, r)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)),
            Equal.trans(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, l, r)), fmap(h, dm, List.append(&1, T.FS, l1y, r)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, List.append(&1, T.FS, z, r)), l, l1y, Equal.sym(List<&1, T.FS>, l1y, l, sl)),
              Equal.trans(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, l1y, r)), List.append(&1, Array<F32>, fmap(h, dm, l1y), fmap(h, dm, r)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), fa, Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => List.append(&1, Array<F32>, fmap(h, dm, z), fmap(h, dm, r)), l1y, l, sl)))))}

def m0b(-l: List<&1, T.FS>, -r: List<&1, T.FS>, -n: Nat, +h: Nat, +dm: T.Dm, l1y: List<&1, T.FS>, +sl: {l1y == l : List<&1, T.FS>}, fl: {T.fleaf(l, h, dm, Nil{}) == fmap(h, dm, l) : List<&1, Array<F32>>}, b: L1R<h, dm, r, Nil{}>) -> MR<0n, T.Halves{l, r}, n, h, dm>:
  match b:
    case L1R{r1, sr, fr}:
      m0c(l, r, n, h, dm, l1y, sl, fl, r1, sr, fr, map_app(l1y, h, dm, r))

def m0a(-l: List<&1, T.FS>, r: List<&1, T.FS>, -n: Nat, +h: Nat, +dm: T.Dm, a: L1R<h, dm, l, Nil{}>) -> MR<0n, T.Halves{l, r}, n, h, dm>:
  match a:
    case L1R{l1y, sl, fl}:
      m0b(l, r, n, h, dm, l1y, sl, fl, l1(r, h, dm, Nil{}))

# depth 1 + e: each half cut in two and its tree (the closures run main at depth e), then the two maps appended
def m1c(-e: Nat, -l: List<&1, T.FS>, -r: List<&1, T.FS>, +n: Nat, +h: Nat, +dm: T.Dm, -hh1: T.Halves<T.FS>, s1: {hh1 == T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}) : T.Halves<T.FS>}, +cat1: {T.hcat(T.FS, hh1) == l : List<&1, T.FS>}, -hh2: T.Halves<T.FS>, s2: {hh2 == T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}) : T.Halves<T.FS>}, +cat2: {T.hcat(T.FS, hh2) == r : List<&1, T.FS>}, -y1: List<&1, T.FS>, sy1: {y1 == T.hcat(T.FS, hh1) : List<&1, T.FS>}, f1: {T.fpar(e, hh1, TA.half(n), h, dm) == fmap(h, dm, T.hcat(T.FS, hh1)) : List<&1, Array<F32>>}, y2: List<&1, T.FS>, sy2: {y2 == T.hcat(T.FS, hh2) : List<&1, T.FS>}, f2: {T.fpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm) == fmap(h, dm, T.hcat(T.FS, hh2)) : List<&1, Array<F32>>}, ma: MAR<h, dm, y1, y2>) -> MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match ma:
    case MAR{y1b, sb, fa}:
      +ly1 = Equal.trans(List<&1, T.FS>, y1, T.hcat(T.FS, hh1), l, sy1, cat1)
      +ly2 = Equal.trans(List<&1, T.FS>, y2, T.hcat(T.FS, hh2), r, sy2, cat2)
      +q1 = Equal.trans(List<&1, Array<F32>>, T.fpar(e, T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), T.fpar(e, hh1, TA.half(n), h, dm), fmap(h, dm, l), Equal.cong(T.Halves<T.FS>, List<&1, Array<F32>>, z => T.fpar(e, z, TA.half(n), h, dm), T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}), hh1, Equal.sym(T.Halves<T.FS>, hh1, T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}), s1)), Equal.trans(List<&1, Array<F32>>, T.fpar(e, hh1, TA.half(n), h, dm), fmap(h, dm, T.hcat(T.FS, hh1)), fmap(h, dm, l), f1, Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), T.hcat(T.FS, hh1), l, cat1)))
      +q2 = Equal.trans(List<&1, Array<F32>>, T.fpar(e, T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm), T.fpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm), fmap(h, dm, r), Equal.cong(T.Halves<T.FS>, List<&1, Array<F32>>, z => T.fpar(e, z, Nat.sub(n, TA.half(n)), h, dm), T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), hh2, Equal.sym(T.Halves<T.FS>, hh2, T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), s2)), Equal.trans(List<&1, Array<F32>>, T.fpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm), fmap(h, dm, T.hcat(T.FS, hh2)), fmap(h, dm, r), f2, Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), T.hcat(T.FS, hh2), r, cat2)))
      +bb = Equal.trans(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, l, r)), fmap(h, dm, List.append(&1, T.FS, y1, y2)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), List.append(&1, T.FS, l, r), List.append(&1, T.FS, y1, y2), app2(T.FS, l, y1, r, y2, Equal.sym(List<&1, T.FS>, y1, l, ly1), Equal.sym(List<&1, T.FS>, y2, r, ly2))), Equal.trans(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, y1, y2)), List.append(&1, Array<F32>, fmap(h, dm, y1), fmap(h, dm, y2)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), fa, app2(Array<F32>, fmap(h, dm, y1), fmap(h, dm, l), fmap(h, dm, y2), fmap(h, dm, r), Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), y1, l, ly1), Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), y2, r, ly2))))
      MR{List.append(&1, T.FS, y1b, y2), app2(T.FS, y1b, l, y2, r, Equal.trans(List<&1, T.FS>, y1b, y1, l, sb, ly1), ly2),
        Equal.trans(List<&1, Array<F32>>, List.append(&1, Array<F32>, T.fpar(e, T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), T.fpar(e, T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), fmap(h, dm, List.append(&1, T.FS, l, r)), app2(Array<F32>, T.fpar(e, T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), fmap(h, dm, l), T.fpar(e, T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm), fmap(h, dm, r), q1, q2), Equal.sym(List<&1, Array<F32>>, fmap(h, dm, List.append(&1, T.FS, l, r)), List.append(&1, Array<F32>, fmap(h, dm, l), fmap(h, dm, r)), bb))}

def m1b(-e: Nat, -l: List<&1, T.FS>, -r: List<&1, T.FS>, +n: Nat, +h: Nat, +dm: T.Dm, -hh1: T.Halves<T.FS>, s1: {hh1 == T.cut(T.FS, TA.half(TA.half(n)), l, Nil{}) : T.Halves<T.FS>}, cat1: {T.hcat(T.FS, hh1) == l : List<&1, T.FS>}, -hh2: T.Halves<T.FS>, s2: {hh2 == T.cut(T.FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}) : T.Halves<T.FS>}, cat2: {T.hcat(T.FS, hh2) == r : List<&1, T.FS>}, r1: MR<e, hh1, TA.half(n), h, dm>, r2: MR<e, hh2, Nat.sub(n, TA.half(n)), h, dm>) -> MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match r1 r2:
    case MR{y1, sy1, f1} MR{y2, sy2, f2}:
      m1c(e, l, r, n, h, dm, hh1, s1, cat1, hh2, s2, cat2, y1, sy1, f1, y2, sy2, f2, map_app(y1, h, dm, y2))

def m1a(-e: Nat, -l: List<&1, T.FS>, -r: List<&1, T.FS>, +n: Nat, +h: Nat, +dm: T.Dm, c1: CGR<TA.half(TA.half(n)), l, Nil{}>, c2: CGR<TA.half(Nat.sub(n, TA.half(n))), r, Nil{}>, k1: @g: T.Halves<T.FS> -> MR<e, g, TA.half(n), h, dm>, k2: @g: T.Halves<T.FS> -> MR<e, g, Nat.sub(n, TA.half(n)), h, dm>) -> MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match c1 c2:
    case CGR{hh1, s1, cat1} CGR{hh2, s2, cat2}:
      m1b(e, l, r, n, h, dm, hh1, s1, cat1, hh2, s2, cat2, k1(hh1), k2(hh2))

def tree(d: Nat, gs: T.Halves<T.FS>, +n: Nat, +h: Nat, +dm: T.Dm) -> MR<d, gs, n, h, dm>:
  match d gs:
    case 0n T.Halves{l, r}:
      m0a(l, r, n, h, dm, l1(l, h, dm, Nil{}))
    case 1n++e T.Halves{l, r}:
      m1a(e, l, r, n, h, dm, cg(TA.half(TA.half(n)), l, Nil{}), cg(TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), g1 => tree(e, g1, TA.half(n), h, dm), g2 => tree(e, g2, Nat.sub(n, TA.half(n)), h, dm))

# the law: the tree gives what fleaf, the loop over the sequences one after another, gives
def par_fin(-d: Nat, -gs: T.Halves<T.FS>, -n: Nat, +h: Nat, +dm: T.Dm, -ys: List<&1, T.FS>, +sy: {ys == T.hcat(T.FS, gs) : List<&1, T.FS>}, f: {T.fpar(d, gs, n, h, dm) == fmap(h, dm, T.hcat(T.FS, gs)) : List<&1, Array<F32>>}, r: L1R<h, dm, ys, Nil{}>) -> {T.fpar(d, gs, n, h, dm) == T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}) : List<&1, Array<F32>>}:
  match r:
    case L1R{ys2, s2, fl}:
      Equal.trans(List<&1, Array<F32>>, T.fpar(d, gs, n, h, dm), fmap(h, dm, T.hcat(T.FS, gs)), T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}), f,
        Equal.sym(List<&1, Array<F32>>, T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}), fmap(h, dm, T.hcat(T.FS, gs)),
          Equal.trans(List<&1, Array<F32>>, T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}), T.fleaf(ys, h, dm, Nil{}), fmap(h, dm, T.hcat(T.FS, gs)), Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => T.fleaf(z, h, dm, Nil{}), T.hcat(T.FS, gs), ys, Equal.sym(List<&1, T.FS>, ys, T.hcat(T.FS, gs), sy)),
            Equal.trans(List<&1, Array<F32>>, T.fleaf(ys, h, dm, Nil{}), fmap(h, dm, ys), fmap(h, dm, T.hcat(T.FS, gs)), fl, Equal.cong(List<&1, T.FS>, List<&1, Array<F32>>, z => fmap(h, dm, z), ys, T.hcat(T.FS, gs), sy)))))

def par_mid(-d: Nat, -gs: T.Halves<T.FS>, -n: Nat, +h: Nat, +dm: T.Dm, r: MR<d, gs, n, h, dm>) -> {T.fpar(d, gs, n, h, dm) == T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}) : List<&1, Array<F32>>}:
  match r:
    case MR{ys, sy, f}:
      par_fin(d, gs, n, h, dm, ys, sy, f, l1(ys, h, dm, Nil{}))

# PROOF (attn_par_fwd): by induction on the depth, each fact handing its list back (the lists hold arrays, so each is
# used once): a leaf pair is fleaf twice, and fleaf is the map pushed onto its accumulator (l1); the map of two lists
# appended is the two maps appended (map_app); cut's halves, appended, are the list it cut (cg); a deeper tree is two
# trees on cut's halves (the closures).
def Laws.attn_par_fwd(d, gs, n, h, dm):
  par_mid(d, gs, n, h, dm, tree(d, gs, n, h, dm))

# ---- the sequential map the b_tree is compared with: f(x) = fs_o(fheads(h, 0, dm, x)) for each x, in order ----

def b_fmap(+h: Nat, +dm: T.Dm, xs: List<&1, T.BG>) -> List<&1, T.DB>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      T.db_of(T.bgroups(h, 0n, dm, x)) <> b_fmap(h, dm, t)

# ---- fleaf (the code's loop with an accumulator) is the map pushed onto the accumulator ----

type b_L1R<-h: Nat, -dm: T.Dm, -xs: List<&1, T.BG>, -acc: List<&1, T.DB>> is Type:
  b_L1R{ys: List<&1, T.BG>, same: {ys == xs : List<&1, T.BG>}, fact: {T.bleaf(xs, h, dm, acc) == List.reverse.go(&1, T.DB, acc, b_fmap(h, dm, xs)) : List<&1, T.DB>}}

def b_l1_con(x: T.BG, -t: List<&1, T.BG>, +h: Nat, +dm: T.Dm, -acc: List<&1, T.DB>, r: b_L1R<h, dm, t, T.db_of(T.bgroups(h, 0n, dm, x)) <> acc>) -> b_L1R<h, dm, x <> t, acc>:
  match r:
    case b_L1R{ys, same, fact}:
      b_L1R{x <> ys, Equal.cong(List<&1, T.BG>, List<&1, T.BG>, z => x <> z, ys, t, same), fact}

def b_l1(xs: List<&1, T.BG>, +h: Nat, +dm: T.Dm, -acc: List<&1, T.DB>) -> b_L1R<h, dm, xs, acc>:
  match xs:
    case Nil{}:
      b_L1R{Nil{}, {==}, {==}}
    case Con{x, t}:
      b_l1_con(x, t, h, dm, acc, b_l1(t, h, dm, T.db_of(T.bgroups(h, 0n, dm, x)) <> acc))

# ---- the map of two lists appended is the two maps appended ----

type b_MAR<-h: Nat, -dm: T.Dm, -a: List<&1, T.BG>, -b: List<&1, T.BG>> is Type:
  b_MAR{ys: List<&1, T.BG>, same: {ys == a : List<&1, T.BG>}, fact: {b_fmap(h, dm, List.append(&1, T.BG, a, b)) == List.append(&1, T.DB, b_fmap(h, dm, a), b_fmap(h, dm, b)) : List<&1, T.DB>}}

def b_ma_con(x: T.BG, -t: List<&1, T.BG>, +h: Nat, +dm: T.Dm, -b: List<&1, T.BG>, r: b_MAR<h, dm, t, b>) -> b_MAR<h, dm, x <> t, b>:
  match r:
    case b_MAR{ys, same, fact}:
      b_MAR{x <> ys, Equal.cong(List<&1, T.BG>, List<&1, T.BG>, z => x <> z, ys, t, same), Equal.cong(List<&1, T.DB>, List<&1, T.DB>, z => T.db_of(T.bgroups(h, 0n, dm, x)) <> z, b_fmap(h, dm, List.append(&1, T.BG, t, b)), List.append(&1, T.DB, b_fmap(h, dm, t), b_fmap(h, dm, b)), fact)}

def b_map_app(a: List<&1, T.BG>, +h: Nat, +dm: T.Dm, -b: List<&1, T.BG>) -> b_MAR<h, dm, a, b>:
  match a:
    case Nil{}:
      b_MAR{Nil{}, {==}, {==}}
    case Con{x, t}:
      b_ma_con(x, t, h, dm, b, b_map_app(t, h, dm, b))

# ---- a reversed onto b, then c appended: a reversed onto b ++ c ----

type b_GAR<-a: List<&1, T.BG>, -b: List<&1, T.BG>, -c: List<&1, T.BG>> is Type:
  b_GAR{ys: List<&1, T.BG>, same: {ys == a : List<&1, T.BG>}, fact: {List.append(&1, T.BG, List.reverse.go(&1, T.BG, a, b), c) == List.reverse.go(&1, T.BG, a, List.append(&1, T.BG, b, c)) : List<&1, T.BG>}}

def b_ga_con(x: T.BG, -t: List<&1, T.BG>, -b: List<&1, T.BG>, -c: List<&1, T.BG>, r: b_GAR<t, x <> b, c>) -> b_GAR<x <> t, b, c>:
  match r:
    case b_GAR{ys, same, fact}:
      b_GAR{x <> ys, Equal.cong(List<&1, T.BG>, List<&1, T.BG>, z => x <> z, ys, t, same), fact}

def b_go_app(a: List<&1, T.BG>, -b: List<&1, T.BG>, -c: List<&1, T.BG>) -> b_GAR<a, b, c>:
  match a:
    case Nil{}:
      b_GAR{Nil{}, {==}, {==}}
    case Con{x, t}:
      b_ga_con(x, t, b, c, b_go_app(t, x <> b, c))

# ---- cut gives two halves that, appended, are acc reversed onto xs ----

type b_CGR<-k: Nat, -xs: List<&1, T.BG>, -acc: List<&1, T.BG>> is Type:
  b_CGR{hh: T.Halves<T.BG>, same: {hh == T.cut(T.BG, k, xs, acc) : T.Halves<T.BG>}, cat: {T.hcat(T.BG, hh) == List.reverse.go(&1, T.BG, acc, xs) : List<&1, T.BG>}}

def b_cg_base(-k: Nat, xs: List<&1, T.BG>, -acc: List<&1, T.BG>, e: {T.cut(T.BG, k, xs, acc) == T.Halves{List.reverse(&1, T.BG, acc), xs} : T.Halves<T.BG>}, r: b_GAR<acc, Nil{}, xs>) -> b_CGR<k, xs, acc>:
  match r:
    case b_GAR{ys, +same, fact}:
      b_CGR{T.Halves{List.reverse(&1, T.BG, ys), xs},
        Equal.trans(T.Halves<T.BG>, T.Halves{List.reverse(&1, T.BG, ys), xs}, T.Halves{List.reverse(&1, T.BG, acc), xs}, T.cut(T.BG, k, xs, acc), Equal.cong(List<&1, T.BG>, T.Halves<T.BG>, z => T.Halves{List.reverse(&1, T.BG, z), xs}, ys, acc, same), Equal.sym(T.Halves<T.BG>, T.cut(T.BG, k, xs, acc), T.Halves{List.reverse(&1, T.BG, acc), xs}, e)),
        Equal.trans(List<&1, T.BG>, List.append(&1, T.BG, List.reverse.go(&1, T.BG, ys, Nil{}), xs), List.append(&1, T.BG, List.reverse.go(&1, T.BG, acc, Nil{}), xs), List.reverse.go(&1, T.BG, acc, xs), Equal.cong(List<&1, T.BG>, List<&1, T.BG>, z => List.append(&1, T.BG, List.reverse.go(&1, T.BG, z, Nil{}), xs), ys, acc, same), fact)}

def b_cg_con(-m: Nat, -x: T.BG, -t: List<&1, T.BG>, -acc: List<&1, T.BG>, r: b_CGR<m, t, x <> acc>) -> b_CGR<1n+m, x <> t, acc>:
  match r:
    case b_CGR{hh, same, cat}:
      b_CGR{hh, same, cat}

def b_cg(k: Nat, xs: List<&1, T.BG>, acc: List<&1, T.BG>) -> b_CGR<k, xs, acc>:
  match k xs:
    case 1n+m Con{x, t}:
      b_cg_con(m, x, t, acc, b_cg(m, t, x <> acc))
    case 0n ys:
      b_cg_base(0n, ys, acc, {==}, b_go_app(acc, Nil{}, ys))
    case 1n+m Nil{}:
      b_cg_base(1n+m, Nil{}, acc, {==}, b_go_app(acc, Nil{}, Nil{}))

# ---- the b_tree: fpar at depth d on two halves is the map of the halves appended ----

type b_MR<-d: Nat, -gs: T.Halves<T.BG>, -n: Nat, -h: Nat, -dm: T.Dm> is Type:
  b_MR{ys: List<&1, T.BG>, same: {ys == T.hcat(T.BG, gs) : List<&1, T.BG>}, fact: {T.bpar(d, gs, n, h, dm) == b_fmap(h, dm, T.hcat(T.BG, gs)) : List<&1, T.DB>}}

# depth 0: the two leaves' loops, appended
def b_m0c(-l: List<&1, T.BG>, -r: List<&1, T.BG>, -n: Nat, +h: Nat, +dm: T.Dm, -l1y: List<&1, T.BG>, +sl: {l1y == l : List<&1, T.BG>}, fl: {T.bleaf(l, h, dm, Nil{}) == b_fmap(h, dm, l) : List<&1, T.DB>}, r1: List<&1, T.BG>, sr: {r1 == r : List<&1, T.BG>}, fr: {T.bleaf(r, h, dm, Nil{}) == b_fmap(h, dm, r) : List<&1, T.DB>}, c: b_MAR<h, dm, l1y, r>) -> b_MR<0n, T.Halves{l, r}, n, h, dm>:
  match c:
    case b_MAR{l2, s2, fa}:
      b_MR{List.append(&1, T.BG, l2, r1), app2(T.BG, l2, l, r1, r, Equal.trans(List<&1, T.BG>, l2, l1y, l, s2, sl), sr),
        Equal.trans(List<&1, T.DB>, List.append(&1, T.DB, T.bleaf(l, h, dm, Nil{}), T.bleaf(r, h, dm, Nil{})), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), b_fmap(h, dm, List.append(&1, T.BG, l, r)), app2(T.DB, T.bleaf(l, h, dm, Nil{}), b_fmap(h, dm, l), T.bleaf(r, h, dm, Nil{}), b_fmap(h, dm, r), fl, fr),
          Equal.sym(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, l, r)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)),
            Equal.trans(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, l, r)), b_fmap(h, dm, List.append(&1, T.BG, l1y, r)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, List.append(&1, T.BG, z, r)), l, l1y, Equal.sym(List<&1, T.BG>, l1y, l, sl)),
              Equal.trans(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, l1y, r)), List.append(&1, T.DB, b_fmap(h, dm, l1y), b_fmap(h, dm, r)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), fa, Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => List.append(&1, T.DB, b_fmap(h, dm, z), b_fmap(h, dm, r)), l1y, l, sl)))))}

def b_m0b(-l: List<&1, T.BG>, -r: List<&1, T.BG>, -n: Nat, +h: Nat, +dm: T.Dm, l1y: List<&1, T.BG>, +sl: {l1y == l : List<&1, T.BG>}, fl: {T.bleaf(l, h, dm, Nil{}) == b_fmap(h, dm, l) : List<&1, T.DB>}, b: b_L1R<h, dm, r, Nil{}>) -> b_MR<0n, T.Halves{l, r}, n, h, dm>:
  match b:
    case b_L1R{r1, sr, fr}:
      b_m0c(l, r, n, h, dm, l1y, sl, fl, r1, sr, fr, b_map_app(l1y, h, dm, r))

def b_m0a(-l: List<&1, T.BG>, r: List<&1, T.BG>, -n: Nat, +h: Nat, +dm: T.Dm, a: b_L1R<h, dm, l, Nil{}>) -> b_MR<0n, T.Halves{l, r}, n, h, dm>:
  match a:
    case b_L1R{l1y, sl, fl}:
      b_m0b(l, r, n, h, dm, l1y, sl, fl, b_l1(r, h, dm, Nil{}))

# depth 1 + e: each half cut in two and its b_tree (the closures run main at depth e), then the two maps appended
def b_m1c(-e: Nat, -l: List<&1, T.BG>, -r: List<&1, T.BG>, +n: Nat, +h: Nat, +dm: T.Dm, -hh1: T.Halves<T.BG>, s1: {hh1 == T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}) : T.Halves<T.BG>}, +cat1: {T.hcat(T.BG, hh1) == l : List<&1, T.BG>}, -hh2: T.Halves<T.BG>, s2: {hh2 == T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}) : T.Halves<T.BG>}, +cat2: {T.hcat(T.BG, hh2) == r : List<&1, T.BG>}, -y1: List<&1, T.BG>, sy1: {y1 == T.hcat(T.BG, hh1) : List<&1, T.BG>}, f1: {T.bpar(e, hh1, TA.half(n), h, dm) == b_fmap(h, dm, T.hcat(T.BG, hh1)) : List<&1, T.DB>}, y2: List<&1, T.BG>, sy2: {y2 == T.hcat(T.BG, hh2) : List<&1, T.BG>}, f2: {T.bpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm) == b_fmap(h, dm, T.hcat(T.BG, hh2)) : List<&1, T.DB>}, ma: b_MAR<h, dm, y1, y2>) -> b_MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match ma:
    case b_MAR{y1b, sb, fa}:
      +ly1 = Equal.trans(List<&1, T.BG>, y1, T.hcat(T.BG, hh1), l, sy1, cat1)
      +ly2 = Equal.trans(List<&1, T.BG>, y2, T.hcat(T.BG, hh2), r, sy2, cat2)
      +q1 = Equal.trans(List<&1, T.DB>, T.bpar(e, T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), T.bpar(e, hh1, TA.half(n), h, dm), b_fmap(h, dm, l), Equal.cong(T.Halves<T.BG>, List<&1, T.DB>, z => T.bpar(e, z, TA.half(n), h, dm), T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}), hh1, Equal.sym(T.Halves<T.BG>, hh1, T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}), s1)), Equal.trans(List<&1, T.DB>, T.bpar(e, hh1, TA.half(n), h, dm), b_fmap(h, dm, T.hcat(T.BG, hh1)), b_fmap(h, dm, l), f1, Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), T.hcat(T.BG, hh1), l, cat1)))
      +q2 = Equal.trans(List<&1, T.DB>, T.bpar(e, T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm), T.bpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm), b_fmap(h, dm, r), Equal.cong(T.Halves<T.BG>, List<&1, T.DB>, z => T.bpar(e, z, Nat.sub(n, TA.half(n)), h, dm), T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), hh2, Equal.sym(T.Halves<T.BG>, hh2, T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), s2)), Equal.trans(List<&1, T.DB>, T.bpar(e, hh2, Nat.sub(n, TA.half(n)), h, dm), b_fmap(h, dm, T.hcat(T.BG, hh2)), b_fmap(h, dm, r), f2, Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), T.hcat(T.BG, hh2), r, cat2)))
      +bb = Equal.trans(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, l, r)), b_fmap(h, dm, List.append(&1, T.BG, y1, y2)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), List.append(&1, T.BG, l, r), List.append(&1, T.BG, y1, y2), app2(T.BG, l, y1, r, y2, Equal.sym(List<&1, T.BG>, y1, l, ly1), Equal.sym(List<&1, T.BG>, y2, r, ly2))), Equal.trans(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, y1, y2)), List.append(&1, T.DB, b_fmap(h, dm, y1), b_fmap(h, dm, y2)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), fa, app2(T.DB, b_fmap(h, dm, y1), b_fmap(h, dm, l), b_fmap(h, dm, y2), b_fmap(h, dm, r), Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), y1, l, ly1), Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), y2, r, ly2))))
      b_MR{List.append(&1, T.BG, y1b, y2), app2(T.BG, y1b, l, y2, r, Equal.trans(List<&1, T.BG>, y1b, y1, l, sb, ly1), ly2),
        Equal.trans(List<&1, T.DB>, List.append(&1, T.DB, T.bpar(e, T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), T.bpar(e, T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), b_fmap(h, dm, List.append(&1, T.BG, l, r)), app2(T.DB, T.bpar(e, T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm), b_fmap(h, dm, l), T.bpar(e, T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm), b_fmap(h, dm, r), q1, q2), Equal.sym(List<&1, T.DB>, b_fmap(h, dm, List.append(&1, T.BG, l, r)), List.append(&1, T.DB, b_fmap(h, dm, l), b_fmap(h, dm, r)), bb))}

def b_m1b(-e: Nat, -l: List<&1, T.BG>, -r: List<&1, T.BG>, +n: Nat, +h: Nat, +dm: T.Dm, -hh1: T.Halves<T.BG>, s1: {hh1 == T.cut(T.BG, TA.half(TA.half(n)), l, Nil{}) : T.Halves<T.BG>}, cat1: {T.hcat(T.BG, hh1) == l : List<&1, T.BG>}, -hh2: T.Halves<T.BG>, s2: {hh2 == T.cut(T.BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}) : T.Halves<T.BG>}, cat2: {T.hcat(T.BG, hh2) == r : List<&1, T.BG>}, r1: b_MR<e, hh1, TA.half(n), h, dm>, r2: b_MR<e, hh2, Nat.sub(n, TA.half(n)), h, dm>) -> b_MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match r1 r2:
    case b_MR{y1, sy1, f1} b_MR{y2, sy2, f2}:
      b_m1c(e, l, r, n, h, dm, hh1, s1, cat1, hh2, s2, cat2, y1, sy1, f1, y2, sy2, f2, b_map_app(y1, h, dm, y2))

def b_m1a(-e: Nat, -l: List<&1, T.BG>, -r: List<&1, T.BG>, +n: Nat, +h: Nat, +dm: T.Dm, c1: b_CGR<TA.half(TA.half(n)), l, Nil{}>, c2: b_CGR<TA.half(Nat.sub(n, TA.half(n))), r, Nil{}>, k1: @g: T.Halves<T.BG> -> b_MR<e, g, TA.half(n), h, dm>, k2: @g: T.Halves<T.BG> -> b_MR<e, g, Nat.sub(n, TA.half(n)), h, dm>) -> b_MR<1n+e, T.Halves{l, r}, n, h, dm>:
  match c1 c2:
    case b_CGR{hh1, s1, cat1} b_CGR{hh2, s2, cat2}:
      b_m1b(e, l, r, n, h, dm, hh1, s1, cat1, hh2, s2, cat2, k1(hh1), k2(hh2))

def b_tree(d: Nat, gs: T.Halves<T.BG>, +n: Nat, +h: Nat, +dm: T.Dm) -> b_MR<d, gs, n, h, dm>:
  match d gs:
    case 0n T.Halves{l, r}:
      b_m0a(l, r, n, h, dm, b_l1(l, h, dm, Nil{}))
    case 1n++e T.Halves{l, r}:
      b_m1a(e, l, r, n, h, dm, b_cg(TA.half(TA.half(n)), l, Nil{}), b_cg(TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), g1 => b_tree(e, g1, TA.half(n), h, dm), g2 => b_tree(e, g2, Nat.sub(n, TA.half(n)), h, dm))

# the law: the b_tree gives what fleaf, the loop over the sequences one after another, gives
def b_par_fin(-d: Nat, -gs: T.Halves<T.BG>, -n: Nat, +h: Nat, +dm: T.Dm, -ys: List<&1, T.BG>, +sy: {ys == T.hcat(T.BG, gs) : List<&1, T.BG>}, f: {T.bpar(d, gs, n, h, dm) == b_fmap(h, dm, T.hcat(T.BG, gs)) : List<&1, T.DB>}, r: b_L1R<h, dm, ys, Nil{}>) -> {T.bpar(d, gs, n, h, dm) == T.bleaf(T.hcat(T.BG, gs), h, dm, Nil{}) : List<&1, T.DB>}:
  match r:
    case b_L1R{ys2, s2, fl}:
      Equal.trans(List<&1, T.DB>, T.bpar(d, gs, n, h, dm), b_fmap(h, dm, T.hcat(T.BG, gs)), T.bleaf(T.hcat(T.BG, gs), h, dm, Nil{}), f,
        Equal.sym(List<&1, T.DB>, T.bleaf(T.hcat(T.BG, gs), h, dm, Nil{}), b_fmap(h, dm, T.hcat(T.BG, gs)),
          Equal.trans(List<&1, T.DB>, T.bleaf(T.hcat(T.BG, gs), h, dm, Nil{}), T.bleaf(ys, h, dm, Nil{}), b_fmap(h, dm, T.hcat(T.BG, gs)), Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => T.bleaf(z, h, dm, Nil{}), T.hcat(T.BG, gs), ys, Equal.sym(List<&1, T.BG>, ys, T.hcat(T.BG, gs), sy)),
            Equal.trans(List<&1, T.DB>, T.bleaf(ys, h, dm, Nil{}), b_fmap(h, dm, ys), b_fmap(h, dm, T.hcat(T.BG, gs)), fl, Equal.cong(List<&1, T.BG>, List<&1, T.DB>, z => b_fmap(h, dm, z), ys, T.hcat(T.BG, gs), sy)))))

def b_par_mid(-d: Nat, -gs: T.Halves<T.BG>, -n: Nat, +h: Nat, +dm: T.Dm, r: b_MR<d, gs, n, h, dm>) -> {T.bpar(d, gs, n, h, dm) == T.bleaf(T.hcat(T.BG, gs), h, dm, Nil{}) : List<&1, T.DB>}:
  match r:
    case b_MR{ys, sy, f}:
      b_par_fin(d, gs, n, h, dm, ys, sy, f, b_l1(ys, h, dm, Nil{}))

# PROOF (attn_par_bwd): attn_par_fwd's proof, for BG and bleaf.
def Laws.attn_par_bwd(d, gs, n, h, dm):
  b_par_mid(d, gs, n, h, dm, b_tree(d, gs, n, h, dm))

# =====================================================================================================
# Grouped-query heads (llama.bend's proof)
# =====================================================================================================

# one block of 1 + m numbers adds 1 to the quotient (Nat.divmod.go counts down the remainder from m)
def div_block(+m: Nat, -n: Nat, -d: Nat, -r: Nat) -> {Nat.divmod.go(Nat.add(1n+m, n), m, d, r) == Nat.divmod.go(n, Nat.add(m, r), 1n+d, 0n) : Nat & Nat}:
  match m:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, 1n+Nat.add(p, r), Nat.add(p, 1n+r), NL.add_succ(p, r)) : {Nat.divmod.go(Nat.add(1n+p, n), p, d, 1n+r) == Nat.divmod.go(n, _, 1n+d, 0n) : Nat & Nat}
      div_block(p, n, d, 1n+r)

# j whole blocks of 1 + s numbers add j to the quotient
def div_blocks(+j: Nat, +s: Nat, +n: Nat, +d: Nat) -> {Nat.divmod.go(Nat.add(Nat.mul(j, 1n+s), n), s, d, 0n) == Nat.divmod.go(n, s, Nat.add(j, d), 0n) : Nat & Nat}:
  match j:
    case 0n:
      {==}
    case 1n+jp:
      %NL.add_assoc(1n+s, Nat.mul(jp, 1n+s), n) : {Nat.divmod.go(_, s, d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(1n+s, Nat.add(Nat.mul(jp, 1n+s), n)), s, d, 0n), Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), Nat.add(s, 0n), 1n+d, 0n), div_block(s, Nat.add(Nat.mul(jp, 1n+s), n), d, 0n)) : {_ == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(s, 0n), s, NL.add_zero(s)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), _, 1n+d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, 1n+Nat.add(jp, d), Nat.add(jp, 1n+d), NL.add_succ(jp, d)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), s, 1n+d, 0n) == Nat.divmod.go(n, s, _, 0n) : Nat & Nat}
      div_blocks(jp, s, n, 1n+d)

# fewer than 1 + s numbers after the last whole block leave the quotient alone
def div_rest(+i: Nat, -k: Nat, -d: Nat, -r: Nat) -> {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), d, r)) == d : Nat}:
  match i:
    case 0n:
      {==}
    case 1n+p:
      div_rest(p, k, d, 1n+r)

# PROOF (gqa_head): Nat.div counts how many whole blocks of g numbers fit. j·g is j whole blocks (div_blocks), and
# i < g numbers after them do not complete another one (div_rest).
def Laws.gqa_head(j, i, k):
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), Nat.add(i, k), 0n, 0n), Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n), div_blocks(j, Nat.add(i, k), i, 0n)) : {Pair.fst(Nat, Nat, _) == j : Nat}
  %NL.add_zero(j) : {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n)) == _ : Nat}
  div_rest(i, k, Nat.add(j, 0n), 0n)

# PROOF (kv_of_group): kv_of is Nat.div, so this is gqa_head.
def Laws.kv_of_group(j, i, k):
  Laws.gqa_head(j, i, k)

# PROOF (kv_cols): head_cols' key column is kv_of(head, g)·hd; kv_of of a head of group j is j (kv_of_group).
def Laws.kv_cols(t, hd, w, kvw, sc, j, i, k):
  Equal.cong(Nat, U32, n => U32.from_nat(Nat.mul(n, hd)), T.kv_of(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)), j, Laws.kv_of_group(j, i, k))

# =====================================================================================================
# Tensors: shapes
# =====================================================================================================

# a || b == b || a, and a && b == b && a: every case computes
def or_comm(+a: Bool, +b: Bool) -> {Bool.or(a, b) == Bool.or(b, a) : Bool}:
  match a b:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

def and_comm(+a: Bool, +b: Bool) -> {Bool.and(a, b) == Bool.and(b, a) : Bool}:
  match a b:
    case True{} True{}:
      {==}
    case True{} False{}:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      {==}

# equal operands give equal conjunctions and disjunctions
def and_cong(+p: Bool, +p2: Bool, +q: Bool, +q2: Bool, e: {p == p2 : Bool}, f: {q == q2 : Bool}) -> {Bool.and(p, q) == Bool.and(p2, q2) : Bool}:
  Equal.trans(Bool, Bool.and(p, q), Bool.and(p2, q), Bool.and(p2, q2), Equal.cong(Bool, Bool, x => Bool.and(x, q), p, p2, e), Equal.cong(Bool, Bool, x => Bool.and(p2, x), q, q2, f))

def or_cong(+p: Bool, +p2: Bool, +q: Bool, +q2: Bool, e: {p == p2 : Bool}, f: {q == q2 : Bool}) -> {Bool.or(p, q) == Bool.or(p2, q2) : Bool}:
  Equal.trans(Bool, Bool.or(p, q), Bool.or(p2, q), Bool.or(p2, q2), Equal.cong(Bool, Bool, x => Bool.or(x, q), p, p2, e), Equal.cong(Bool, Bool, x => Bool.or(p2, x), q, q2, f))

# one axis: bc_dim is written as pair(x, y) || pair(y, x), so swapping x and y only swaps the two sides of the ||
def bc_dim_sym(+x: Nat, +y: Nat, +z: Nat) -> {TT.bc_dim(x, y, z) == TT.bc_dim(y, x, z) : Bool}:
  or_comm(TT.bc_pair(x, y, z), TT.bc_pair(y, x, z))

# axis by axis: the same on each axis (bc_dim_sym), and by induction on the rest; shapes of different ranks are False
# either way round
def bc_eq_sym(a: List<&2, Nat>, b: List<&2, Nat>, o: List<&2, Nat>) -> {TT.bc_eq(a, b, o) == TT.bc_eq(b, a, o) : Bool}:
  match a b o:
    case Nil{} Nil{} Nil{}:
      {==}
    case Nil{} Nil{} Con{z, ro}:
      {==}
    case Nil{} Con{y, rb} Nil{}:
      {==}
    case Nil{} Con{y, rb} Con{z, ro}:
      {==}
    case Con{x, ra} Nil{} Nil{}:
      {==}
    case Con{x, ra} Nil{} Con{z, ro}:
      {==}
    case Con{x, ra} Con{y, rb} Nil{}:
      {==}
    case Con{+x, +ra} Con{+y, +rb} Con{+z, +ro}:
      and_cong(TT.bc_dim(x, y, z), TT.bc_dim(y, x, z), TT.bc_eq(ra, rb, ro), TT.bc_eq(rb, ra, ro), bc_dim_sym(x, y, z), bc_eq_sym(ra, rb, ro))

# the ranks: the same two comparisons and the same two equalities, in the other order
def bc_rank_sym(+la: Nat, +lb: Nat, +lo: Nat) -> {TT.bc_rank(la, lb, lo) == TT.bc_rank(lb, la, lo) : Bool}:
  and_cong(Bool.and(Nat.is_le(la, lo), Nat.is_le(lb, lo)), Bool.and(Nat.is_le(lb, lo), Nat.is_le(la, lo)), Bool.or(Nat.is_eq(la, lo), Nat.is_eq(lb, lo)), Bool.or(Nat.is_eq(lb, lo), Nat.is_eq(la, lo)),
    and_comm(Nat.is_le(la, lo), Nat.is_le(lb, lo)), or_comm(Nat.is_eq(la, lo), Nat.is_eq(lb, lo)))

def Laws.bc_sym(a, b, o):
  and_cong(TT.bc_rank(List.length(&2, Nat, a), List.length(&2, Nat, b), List.length(&2, Nat, o)), TT.bc_rank(List.length(&2, Nat, b), List.length(&2, Nat, a), List.length(&2, Nat, o)),
    TT.bc_eq(TT.pad(a, List.length(&2, Nat, o)), TT.pad(b, List.length(&2, Nat, o)), o), TT.bc_eq(TT.pad(b, List.length(&2, Nat, o)), TT.pad(a, List.length(&2, Nat, o)), o),
    bc_rank_sym(List.length(&2, Nat, a), List.length(&2, Nat, b), List.length(&2, Nat, o)), bc_eq_sym(TT.pad(a, List.length(&2, Nat, o)), TT.pad(b, List.length(&2, Nat, o)), o))

# h * (a * r) == a * (h * r): associativity, then commutativity of the first two factors, then associativity back
def mul_swap(+h: Nat, +a: Nat, +r: Nat) -> {Nat.mul(h, Nat.mul(a, r)) == Nat.mul(a, Nat.mul(h, r)) : Nat}:
  Equal.trans(Nat, Nat.mul(h, Nat.mul(a, r)), Nat.mul(Nat.mul(h, a), r), Nat.mul(a, Nat.mul(h, r)), NL.mul_assoc(h, a, r),
    Equal.trans(Nat, Nat.mul(Nat.mul(h, a), r), Nat.mul(Nat.mul(a, h), r), Nat.mul(a, Nat.mul(h, r)),
      Equal.cong(Nat, Nat, x => Nat.mul(x, r), Nat.mul(h, a), Nat.mul(a, h), NL.mul_comm(h, a)),
      Equal.sym(Nat, Nat.mul(a, Nat.mul(h, r)), Nat.mul(Nat.mul(a, h), r), NL.mul_assoc(a, h, r))))

# by induction on the shape and the axis: axis 0 computes; past it, the first size h stays in front on both sides, and
# mul_swap moves it past size(k)
def Laws.sum_numel(s, k):
  match s k:
    case Nil{} _:
      {==}
    case Con{h, t} 0n:
      {==}
    case Con{+h, +t} 1n+p:
      Equal.trans(Nat, Nat.mul(h, NL.product(t)), Nat.mul(h, Nat.mul(TT.at1(t, p), NL.product(TT.remove(t, p)))), Nat.mul(TT.at1(t, p), Nat.mul(h, NL.product(TT.remove(t, p)))),
        Equal.cong(Nat, Nat, x => Nat.mul(h, x), NL.product(t), Nat.mul(TT.at1(t, p), NL.product(TT.remove(t, p))), Laws.sum_numel(t, p)),
        mul_swap(h, TT.at1(t, p), NL.product(TT.remove(t, p))))

# =====================================================================================================
# Tensors: narrow, cat, of_mat, permute
# =====================================================================================================

# by induction: axis 0 is replaced and then taken out; past it, the first size stays in front on both sides
def Laws.narrow_rest(s, d, l):
  match s d:
    case Nil{} _:
      {==}
    case Con{h, t} 0n:
      {==}
    case Con{+h, +t} 1n+p:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- the binder of Equal.cong's motive, used in its body not-expected-because: a law, never run
      Equal.cong(List<&2, Nat>, List<&2, Nat>, x => h <> x, TT.remove(TT.setd(t, p, l), p), TT.remove(t, p), Laws.narrow_rest(t, p, l))

# a shape with no axis d contradicts the premise (d < rank is False); axis 0 is l; past it, the induction
def Laws.narrow_axis(s, d, l, e):
  match s d:
    case Nil{} 0n:
      Empty.absurd({TT.at1(TT.setd([], 0n, l), 0n) == l : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Nil{} 1n+p:
      Empty.absurd({TT.at1(TT.setd([], 1n+p, l), 1n+p) == l : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Con{h, t} 0n:
      {==}
    case Con{h, +t} 1n+p:
      Laws.narrow_axis(t, p, l, e)

# axis 0: (x + y)·n == x·n + y·n (distributivity); past it, the induction and k·(a + b) == k·a + k·b
def Laws.cat_numel(s, d, x, y, e):
  match s d:
    case Nil{} 0n:
      Empty.absurd({TT.numel(TT.setd([], 0n, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd([], 0n, x)), TT.numel(TT.setd([], 0n, y))) : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Nil{} 1n+p:
      Empty.absurd({TT.numel(TT.setd([], 1n+p, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd([], 1n+p, x)), TT.numel(TT.setd([], 1n+p, y))) : Nat}, false_true(Equal.sym(Bool, True{}, False{}, e)))
    case Con{h, +t} 0n:
      Equal.sym(Nat, Nat.add(Nat.mul(x, NL.product(t)), Nat.mul(y, NL.product(t))), Nat.mul(Nat.add(x, y), NL.product(t)), NL.mul_dist(x, y, NL.product(t)))
    case Con{+h, +t} 1n+p:
      Equal.trans(Nat, Nat.mul(h, NL.product(TT.setd(t, p, Nat.add(x, y)))), Nat.mul(h, Nat.add(NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y)))), Nat.add(Nat.mul(h, NL.product(TT.setd(t, p, x))), Nat.mul(h, NL.product(TT.setd(t, p, y)))),
        Equal.cong(Nat, Nat, v => Nat.mul(h, v), NL.product(TT.setd(t, p, Nat.add(x, y))), Nat.add(NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y))), Laws.cat_numel(t, p, x, y, e)),
        NL.mul_dist_l(h, NL.product(TT.setd(t, p, x)), NL.product(TT.setd(t, p, y))))

# r·(c·1) == r·c: c·1 == c
def Laws.mat_numel(r, c):
  Equal.cong(Nat, Nat, v => Nat.mul(r, v), Nat.mul(c, 1n), c, NL.mul_one_r(c))

# one axis of the result per entry of p
def Laws.permute_rank(s, p):
  match p:
    case Nil{}:
      {==}
    case Con{k, r}:
      Equal.cong(Nat, Nat, v => 1n+v, TT.rank(TT.permuted(s, r)), List.length(&2, Nat, r), Laws.permute_rank(s, r))

# =====================================================================================================
# The order of a row's additions (block_sum)
# =====================================================================================================

# one pass over 2k blocks of 2^p cells gives the k blocks of 2^(p+1) cells: the pass adds blocks 2j and 2j+1, the two
# halves bal puts under one node. Induction on k; add_succ turns k + (k + 1) into 2 + (k + k), and add_assoc moves the
# next pair's start from (lo + 2^p) + 2^p to lo + 2^(p+1)
def pass_pair(+p: Nat, +k: Nat, +lo: Nat) -> {Spec.spass(Spec.blocks(p, Nat.add(k, k), lo)) == Spec.blocks(1n+p, k, lo) : List<&2, Spec.Sum>}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      +a = Spec.pw(p)
      +nx = Nat.add(Nat.add(lo, a), a)
      %NL.add_succ(j, j) : {Spec.spass(Spec.blocks(p, 1n+_, lo)) == Spec.blocks(1n+p, 1n+j, lo) : List<&2, Spec.Sum>}
      %Equal.sym(Nat, Nat.add(lo, Nat.add(a, a)), nx, NL.add_assoc(lo, a, a)) : {Spec.spass(Spec.blocks(p, 1n+1n+Nat.add(j, j), lo)) == Spec.SAdd{Spec.bal(p, lo), Spec.bal(p, Nat.add(lo, a))} <> Spec.blocks(1n+p, j, _) : List<&2, Spec.Sum>}
      %pass_pair(p, j, nx) : {Spec.spass(Spec.blocks(p, 1n+1n+Nat.add(j, j), lo)) == Spec.SAdd{Spec.bal(p, lo), Spec.bal(p, Nat.add(lo, a))} <> _ : List<&2, Spec.Sum>}
      {==}

# 2^(p+1) · k cells are 2^p · 2k: (a + a)·k == a·k + a·k == a·(k + k)
def cnt2(+p: Nat, +k: Nat) -> {Nat.mul(Spec.pw(1n+p), k) == Nat.mul(Spec.pw(p), Nat.add(k, k)) : Nat}:
  +a = Spec.pw(p)
  Equal.trans(Nat, Nat.mul(Nat.add(a, a), k), Nat.add(Nat.mul(a, k), Nat.mul(a, k)), Nat.mul(a, Nat.add(k, k)),
    Equal.sym(Nat, Nat.add(Nat.mul(a, k), Nat.mul(a, k)), Nat.mul(Nat.add(a, a), k), NL.mul_dist(a, a, k)),
    Equal.sym(Nat, Nat.mul(a, Nat.add(k, k)), Nat.add(Nat.mul(a, k), Nat.mul(a, k)), NL.mul_dist_l(a, k, k)))

# the first p passes over a row of 2^p · k cells leave its k blocks, each summed as bal sums it. Induction on p: p
# passes over 2^(p+1)·k cells (= 2^p·2k, cnt2) leave 2k blocks of 2^p, and one more pass pairs them (pass_pair)
def blocks_passes(+p: Nat, +k: Nat, +lo: Nat) -> {Spec.passes(p, Spec.blocks(0n, Nat.mul(Spec.pw(p), k), lo)) == Spec.blocks(p, k, lo) : List<&2, Spec.Sum>}:
  match p:
    case 0n:
      %NL.mul_one_l(k) : {Spec.blocks(0n, Nat.mul(1n, k), lo) == Spec.blocks(0n, _, lo) : List<&2, Spec.Sum>}
      {==}
    case 1n+q:
      +kk = Nat.add(k, k)
      %Equal.sym(Nat, Nat.mul(Spec.pw(1n+q), k), Nat.mul(Spec.pw(q), kk), cnt2(q, k)) : {Spec.spass(Spec.passes(q, Spec.blocks(0n, _, lo))) == Spec.blocks(1n+q, k, lo) : List<&2, Spec.Sum>}
      %Equal.sym(List<&2, Spec.Sum>, Spec.passes(q, Spec.blocks(0n, Nat.mul(Spec.pw(q), kk), lo)), Spec.blocks(q, kk, lo), blocks_passes(q, kk, lo)) : {Spec.spass(_) == Spec.blocks(1n+q, k, lo) : List<&2, Spec.Sum>}
      pass_pair(q, k, lo)

# r + p passes are p passes, then r more
def passes_add(r: Nat, +p: Nat, +xs: List<&2, Spec.Sum>) -> {Spec.passes(Nat.add(r, p), xs) == Spec.passes(r, Spec.passes(p, xs)) : List<&2, Spec.Sum>}:
  match r:
    case 0n:
      {==}
    case 1n+q:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- the binder of Equal.cong's motive, used in its body not-expected-because: a law, never run
      Equal.cong(List<&2, Spec.Sum>, List<&2, Spec.Sum>, z => Spec.spass(z), Spec.passes(Nat.add(q, p), xs), Spec.passes(q, Spec.passes(p, xs)), passes_add(q, p, xs))

# block_sum: r + p passes over the row are r passes over what its first p passes leave (passes_add), which is its k
# blocks summed as bal sums them (blocks_passes)
def Laws.block_sum(p, k, r):
  +row = Spec.blocks(0n, Nat.mul(Spec.pw(p), k), 0n)
  Equal.trans(List<&2, Spec.Sum>, Spec.passes(Nat.add(r, p), row), Spec.passes(r, Spec.passes(p, row)), Spec.passes(r, Spec.blocks(p, k, 0n)),
    passes_add(r, p, row),
    Equal.cong(List<&2, Spec.Sum>, List<&2, Spec.Sum>, z => Spec.passes(r, z), Spec.passes(p, row), Spec.blocks(p, k, 0n), blocks_passes(p, k, 0n)))

def main() -> Unit:
  Unit{}
