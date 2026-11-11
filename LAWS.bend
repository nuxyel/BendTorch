# BendTorch's laws. Each is a claim about every input, not a test case; PROOF.bend proves them, and
# `bend PROOF.bend --verdict` checks the proofs again with Bend's kernel proved in Lean. Plain-language explanations
# are in torch/README.md. The laws about linear maps speak about the exact model in torch/spec.bend; the others speak
# about the package's own functions.

import Base
import ./torch/main.bend as T
import ./torch/spec.bend as Spec
import ./torch/tensor.bend as TT

# LAW (adjoint): the backward pass of a linear layer, dx = Wᵀ·dy, is the adjoint of its forward pass y = W·x: for any
# W, dy and x, dy·(W·x) == (Wᵀ·dy)·x. This is what makes dx the gradient of the loss with respect to x.
law adjoint:
  for w: List<&2, List<&2, Nat>>
  for g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w, v)) == Spec.dotn(Spec.mtv(w, g), v) : Nat}

# LAW (weight_adjoint): the weight gradient dW = dy ⊗ x is the adjoint in W: dy·(W·x) == <dy ⊗ x, W>.
law weight_adjoint:
  for w: List<&2, List<&2, Nat>>
  for g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w, v)) == Spec.frob(Spec.outer(g, v), w) : Nat}

# LAW (adjoint_seq): Seq runs the second layer's backward pass, then the first's. For two linear layers that is
# W1ᵀ·(W2ᵀ·dy), and it is the adjoint of the composed forward pass W2·(W1·x): the chain rule, for linear maps.
law adjoint_seq:
  for +w1: List<&2, List<&2, Nat>>
  for +w2: List<&2, List<&2, Nat>>
  for +g: List<&2, Nat>
  for +v: List<&2, Nat>
  {Spec.dotn(g, Spec.mv(w2, Spec.mv(w1, v))) == Spec.dotn(Spec.mtv(w1, Spec.mtv(w2, g)), v) : Nat}

# LAW (seq_chain): T.Seq's own code, running two linear layers of the exact model one after the other, gives a backward
# pass that is the adjoint of its forward pass: dy·y == dx·x, for the y the forward pass returns and the dx its backward
# pass returns from the state the forward pass saved. adjoint_seq says it of the formulas; this says it of the
# combinator that composes the layers: Seq feeds the second layer's input gradient to the first, the chain rule.
law seq_chain:
  for +w1: List<&2, List<&2, Nat>>
  for +w2: List<&2, List<&2, Nat>>
  for +x: List<&2, Nat>
  for +dy: List<&2, Nat>
  for g1: List<&2, List<&2, Nat>>
  for g2: List<&2, List<&2, Nat>>
  {Spec.dotn(dy, Spec.fy(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.SeqL(), T.Two{w1, w2}, x))) == Spec.dotn(Spec.bdx(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Layer.bwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.SeqL(), T.Two{w1, w2}, Spec.fsv(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.SeqL(), T.Two{w1, w2}, x)), dy, T.Two{g1, g2})), x) : Nat}

# LAW (fork_sum): T.Fork's own code, giving one input to two linear layers, gives as input gradient the sum of the two
# layers' input gradients, and that sum is the adjoint of the pair of outputs: dy1·y1 + dy2·y2 == dx·x. A residual
# block (Fork of the identity and a layer) and attention's q, k, v projections rely on it.
law fork_sum:
  for +w1: List<&2, List<&2, Nat>>
  for +w2: List<&2, List<&2, Nat>>
  for +x: List<&2, Nat>
  for +dy1: List<&2, Nat>
  for +dy2: List<&2, Nat>
  for g1: List<&2, List<&2, Nat>>
  for g2: List<&2, List<&2, Nat>>
  {Spec.pdot(T.Two{dy1, dy2}, Spec.fy(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.ForkL(), T.Two{w1, w2}, x))) == Spec.dotn(Spec.bdx(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Layer.bwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.ForkL(), T.Two{w1, w2}, Spec.fsv(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.ForkL(), T.Two{w1, w2}, x)), T.Two{dy1, dy2}, T.Two{g1, g2})), x) : Nat}

# LAW (both_sum): T.Both's own code, running two linear layers side by side on the two inputs of a pair, gives a backward
# pass that is the adjoint of its forward pass: dy1·y1 + dy2·y2 == dx1·x1 + dx2·x2. Each layer gets its own input's
# gradient and nothing crosses between the two sides. Attention applies RoPE to q and k this way (Both(RoPE, Both(RoPE,
# Id))).
law both_sum:
  for +w1: List<&2, List<&2, Nat>>
  for +w2: List<&2, List<&2, Nat>>
  for +x1: List<&2, Nat>
  for +x2: List<&2, Nat>
  for +dy1: List<&2, Nat>
  for +dy2: List<&2, Nat>
  for g1: List<&2, List<&2, Nat>>
  for g2: List<&2, List<&2, Nat>>
  {Spec.pdot(T.Two{dy1, dy2}, Spec.fy(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.BothL(), T.Two{w1, w2}, T.Two{x1, x2}))) == Spec.pdot(Spec.bdx(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.bwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.BothL(), T.Two{w1, w2}, Spec.fsv(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Layer.fwd(T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, Spec.BothL(), T.Two{w1, w2}, T.Two{x1, x2})), T.Two{dy1, dy2}, T.Two{g1, g2})), T.Two{x1, x2}) : Nat}

# LAW (residual): T.Residual's own code (T.ResidualOf: Seq(Fork(Id, f), Add2)), with a linear layer as f, gives a
# backward pass that is the adjoint of its forward pass: for y = x + W·x and its dx, dy·y == dx·x. The backward pass
# hands dy to both branches and adds what comes back, dx = dy + Wᵀ·dy: the gradient goes through the skip connection
# untouched and through the layer, the rule every block of the Llama models relies on twice. (Over matrices the join
# is Add2; here it is its twin over vectors, a + b forward and dy to both backward.)
law residual:
  for +w: List<&2, List<&2, Nat>>
  for +x: List<&2, Nat>
  for +dy: List<&2, Nat>
  for g: List<&2, List<&2, Nat>>
  {Spec.dotn(dy, Spec.fy(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>, T.Layer.fwd(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>, Spec.ResL(), T.Two{T.Two{Unit{}, w}, Unit{}}, x))) == Spec.dotn(Spec.bdx(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, T.Layer.bwd(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>, Spec.ResL(), T.Two{T.Two{Unit{}, w}, Unit{}}, Spec.fsv(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>, T.Layer.fwd(T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>, Spec.ResL(), T.Two{T.Two{Unit{}, w}, Unit{}}, x)), dy, T.Two{T.Two{Unit{}, g}, Unit{}})), x) : Nat}

# LAW (causal_keys): T.Attention reads keys only through T.qrow, and the scores it builds for the query at position i
# come from keys 0..i, one per key: exactly i + 1 of them. So no key after position i is read, and T.causal_row (law
# causal) gets the i + 1 scores its condition asks for: past them, the row of probabilities is zeros.
law causal_keys:
  for +i: Nat
  for +hd: Nat
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  {List.length(&2, F32, T.dr_row(T.qrow(i, hd, qs, qc, ks, kc, q, k))) == 1n+i : Nat}

# LAW (causal): in T.Attention, row i of a head's probabilities is a softmax over exactly i + 1 positions. T.prow
# writes the query's scores from the start of its row and runs the softmax over as many positions as it wrote; the count
# is the one the softmax's own division loop reports (T.psmc), and this law says it is i + 1, the keys 0..i of
# causal_keys. P starts as zeros and nothing else writes the row, so its other t - (i + 1) positions stay 0: no
# probability lands on a later position.
law causal:
  for +i: Nat
  for +t: Nat
  for +hd: Nat
  for +sc: F32
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  for p: Array<F32>
  {T.pr_n(T.prow(i, t, hd, sc, qs, qc, ks, kc, q, k, p)) == 1n+i : Nat}

# ---- the same three facts with every key visible: a row reads all t keys, its softmax runs over all t, and the
# backward's G row sums over all t. The full functions take m = t - 1 (the last key's index) so that a row's count is
# 1 + m and no subtraction appears in a statement. ----

# LAW (full_keys): FullAttention's frow reads every key 0..m, so a row's scores are 1 + m long, not 1 + i.
law full_keys:
  for +i: Nat
  for +m: Nat
  for +hd: Nat
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  {List.length(&2, F32, T.dr_row(T.frow(i, m, hd, qs, qc, ks, kc, q, k))) == 1n+m : Nat}

# LAW (full): row i of a head's probabilities in FullAttention is a softmax over all t positions.
law full:
  for +i: Nat
  for +m: Nat
  for +hd: Nat
  for +sc: F32
  for +qs: U32
  for +qc: U32
  for +ks: U32
  for +kc: U32
  for q: Array<F32>
  for k: Array<F32>
  for pa: Array<F32>
  {T.pr_n(T.fprow(i, m, hd, sc, qs, qc, ks, kc, q, k, pa)) == 1n+m : Nat}

# LAW (full_grad): the backward's G row for row i sums over all t keys, not 1 + i.
law full_grad:
  for +i: Nat
  for +m: Nat
  for +hd: Nat
  for +sc: F32
  for +ds: U32
  for +dc: U32
  for +vs: U32
  for +vc: U32
  for d: Array<F32>
  for v: Array<F32>
  for pa: Array<F32>
  for ga: Array<F32>
  {T.gr_n(T.fgrow(i, m, hd, sc, ds, dc, vs, vc, d, v, pa, ga)) == 1n+m : Nat}

# LAW (causal_grad): in T.Attention's backward pass, row i of a head's G (the gradient of its scores) is computed over
# exactly i + 1 positions. T.grow reads dP's row against values 0..i only (as qrow reads keys 0..i), writes it from the
# start of its row and runs the softmax's backward pass over as many positions as it wrote; this law says that count is
# i + 1. G starts as zeros and nothing else writes the row, so its other t - (i + 1) positions stay 0: no gradient
# reaches a key or a value after the query's position.
law causal_grad:
  for +i: Nat
  for +t: Nat
  for +hd: Nat
  for +sc: F32
  for +ds: U32
  for +dc: U32
  for +vs: U32
  for +vc: U32
  for d: Array<F32>
  for v: Array<F32>
  for p: Array<F32>
  for g: Array<F32>
  {T.gr_n(T.grow(i, t, hd, sc, ds, dc, vs, vc, d, v, p, g)) == 1n+i : Nat}

# LAW (causal_pv): in T.Attention's P·V, and in its backward pass's dQ = G·K, every cell of row i is a dot of exactly
# i + 1 products, keys 0..i. T.trow0 gives each of the row's dots that length (a dot of length k sums k products,
# T.sdot) and hands it on as the row's count; this law says the count is i + 1. The keys after i, where P and G are 0
# (laws causal, causal_grad), are not read: summing them added only 0·x.
law causal_pv:
  for +i: Nat
  for +c: Nat
  for +t: Nat
  for +bs: U32
  for +bc: U32
  for +ow: U32
  for +oc: U32
  for a: Array<F32>
  for b: Array<F32>
  for o: Array<F32>
  {T.pn_n(T.trow0(i, c, t, bs, bc, ow, oc, a, b, o)) == 1n+i : Nat}

# LAW (causal_kv): in T.Attention's backward pass, every cell of key j's row of dK += Gᵀ·Q and of dV += Pᵀ·dO is a dot
# of exactly 1 + r products, rows j..t-1, where r is the number of keys after j (T.tacc walks the keys with
# t = 1 + j + r). T.tcol0 gives each dot that length and hands it on; this law says the count is 1 + r, so t - j. The
# rows before j, where G and P are 0, are not read.
law causal_kv:
  for +j: Nat
  for +r: Nat
  for +c: Nat
  for +t: Nat
  for +bs: U32
  for +bc: U32
  for a: Array<F32>
  for b: Array<F32>
  for o: Array<F32>
  {T.pn_n(T.tcol0(j, r, c, t, bs, bc, a, b, o)) == 1n+r : Nat}

# LAW (attn_par_fwd): T.Attention runs its sequences in a tree of parallel lets (T.fpar: the sequences cut in halves,
# each half cut again, d levels deep, the leaves' outputs appended). Whatever the depth and however the sequences are
# split between the two halves, the tree gives exactly what T.fleaf gives running them one after another, in order: the
# parallel forward pass is the sequential one, and its output does not depend on d or on the thread count.
law attn_par_fwd:
  for d: Nat
  for gs: T.Halves<T.FS>
  for +n: Nat
  for +h: Nat
  for +dm: T.Dm
  {T.fpar(d, gs, n, h, dm) == T.fleaf(T.hcat(T.FS, gs), h, dm, Nil{}) : List<&1, Array<F32>>}

# LAW (attn_par_fwd_full): the same tree law for FullAttention's forward leaf: fpar_full at any depth gives exactly
# what fleaf_full gives running the sequences one after another, so a full-attention forward does not depend on the
# depth or the thread count either.
law attn_par_fwd_full:
  for d: Nat
  for gs: T.Halves<T.FS>
  for +n: Nat
  for +h: Nat
  for +dm: T.Dm
  {T.fpar_full(d, gs, n, h, dm) == T.fleaf_full(T.hcat(T.FS, gs), h, dm, Nil{}) : List<&1, Array<F32>>}

# LAW (attn_par_bwd): the same for the backward pass: T.bpar's tree gives each sequence's dQ, dK and dV exactly as
# T.bleaf, one sequence after another.
law attn_par_bwd:
  for d: Nat
  for gs: T.Halves<T.BG>
  for +n: Nat
  for +kv: Nat
  for +dm: T.Dm
  {T.bpar(d, gs, n, kv, dm) == T.bleaf(T.hcat(T.BG, gs), kv, dm, Nil{}) : List<&1, T.DB>}

# LAW (attn_par_bwd_full): the tree law for FullAttention's backward leaf, as attn_par_bwd is for Attention's.
law attn_par_bwd_full:
  for d: Nat
  for gs: T.Halves<T.BG>
  for +n: Nat
  for +kv: Nat
  for +dm: T.Dm
  {T.bpar_full(d, gs, n, kv, dm) == T.bleaf_full(T.hcat(T.BG, gs), kv, dm, Nil{}) : List<&1, T.DB>}

# LAW (kv_of_group): T.kv_of, the function both passes of T.Attention use to pick the key/value head of query head hi,
# gives j for every head of group j: query heads j·g + 0 .. j·g + g - 1 (g = 1 + i + k, i < g) all read key/value head j.
law kv_of_group:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {T.kv_of(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

# LAW (gqa_head): with g = 1 + i + k query heads per key/value head, query head j·g + i reads key/value head
# (j·g + i) / g = j, the head of its own group: T.Attention's forward pass picks head hi / g, and its backward pass
# sums the gradients of heads j·g + 0 .. j·g + g - 1 into key/value head j, so both see the same groups.
law gqa_head:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {Nat.div(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

# LAW (kv_cols): the attention code takes the columns each query head reads from T.head_cols, in both passes (the head
# loops pass no head index further, so they cannot compute other columns). With g = 1 + i + k query heads per key/value
# head, query head j·g + i reads the key and value columns of head j: j·hd onwards.
law kv_cols:
  for +t: Nat
  for +hd: Nat
  for +w: Nat
  for +kvw: Nat
  for +sc: F32
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {T.hc_kc(T.head_cols(T.Dm{t, 1n+Nat.add(i, k), hd, w, kvw, sc}, Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i))) == U32.from_nat(Nat.mul(j, hd)) : U32}

# LAW (bc_sym): broadcasting does not depend on the order of the operands: a and b broadcast to o exactly when b and a
# do, so x + y and y + x (and x * y, y * x) accept the same shapes.
law bc_sym:
  for +a: List<&2, Nat>
  for +b: List<&2, Nat>
  for +o: List<&2, Nat>
  {TT.bc_ok(a, b, o) == TT.bc_ok(b, a, o) : Bool}

# LAW (sum_numel): summing over axis k partitions the elements: a tensor of shape s has exactly size(k) times as many
# numbers as its sum over axis k, for every shape and every k (past the last axis the size counts as 1).
law sum_numel:
  for +s: List<&2, Nat>
  for +k: Nat
  {TT.numel(s) == Nat.mul(TT.at1(s, k), TT.numel(TT.remove(s, k))) : Nat}

# LAW (narrow_rest): narrow (and setting any axis' size) changes only its own axis: with axis d taken out, the shape is
# what it was.
law narrow_rest:
  for +s: List<&2, Nat>
  for +d: Nat
  for +l: Nat
  {TT.remove(TT.setd(s, d, l), d) == TT.remove(s, d) : List<&2, Nat>}

# LAW (narrow_axis): narrowing axis d of a tensor that has it gives that axis exactly the length asked for.
law narrow_axis:
  for +s: List<&2, Nat>
  for +d: Nat
  for +l: Nat
  for e: {True{} == Nat.is_lt(d, TT.rank(s)) : Bool}
  {TT.at1(TT.setd(s, d, l), d) == l : Nat}

# LAW (cat_numel): concatenating along an axis the tensor has adds the counts: a tensor whose axis d is x + y long holds
# as many numbers as the two with x and y.
law cat_numel:
  for +s: List<&2, Nat>
  for +d: Nat
  for +x: Nat
  for +y: Nat
  for e: {True{} == Nat.is_lt(d, TT.rank(s)) : Bool}
  {TT.numel(TT.setd(s, d, Nat.add(x, y))) == Nat.add(TT.numel(TT.setd(s, d, x)), TT.numel(TT.setd(s, d, y))) : Nat}

# LAW (mat_numel): a Tensor<[r, c]> holds the r·c numbers of a Mat<r, c>, so of_mat and to_mat reuse the array as it is.
law mat_numel:
  for +r: Nat
  for +c: Nat
  {TT.numel([r, c]) == Nat.mul(r, c) : Nat}

# LAW (permute_rank): permuting by p gives as many axes as p has, so a permutation of the axes keeps the rank.
law permute_rank:
  for +s: List<&2, Nat>
  for p: List<&2, Nat>
  {TT.rank(TT.permuted(s, p)) == List.length(&2, Nat, p) : Nat}

# LAW (block_sum): summing a row in blocks gives the row's own sum, bit for bit. tree_sum adds a row's cells in pairs,
# pass after pass. The GPU's loss kernels instead give each block of 2^p cells to one leaf, which sums it as a balanced
# tree of pairs, and then run tree_sum's remaining passes over the block sums. When 2^p divides the row (a row of 2^p·k
# cells), both build the same tree of additions (torch/spec.bend), for every p, k and number of passes r after the
# first p: the same pairs added in the same order, so the same bits in F32. Any block size that divides the row gives
# the row's sum, which is why the mutant that shrinks the cross-entropy's blocks to one cell survives.
law block_sum:
  for +p: Nat
  for +k: Nat
  for +r: Nat
  {Spec.passes(Nat.add(r, p), Spec.blocks(0n, Nat.mul(Spec.pw(p), k), 0n)) == Spec.passes(r, Spec.blocks(p, k, 0n)) : List<&2, Spec.Sum>}

# LAW (guard_nomom .. guard_packed): an optimizer step runs exactly when its configuration is the kind of state the
# Train holds (OptCfg.kind: SGD's none, momentum's one buffer, Adam's or AdamW's two, AdamW's two packed in BF16) and
# passes OptCfg.ok; for every other configuration Opt.guard hands the step the refused optimizer of the state's own
# kind, every number NaN, so the parameters and every later loss are NaN (BendTorch #24: before, a configuration of
# another kind could leave the parameters unchanged with no error). One law per kind of state, for every configuration.
law guard_nomom:
  for +cfg: T.OptCfg
  for +q1: F32
  for +q2: F32
  {T.Guarded.opt(Unit, T.Opt.guard(Unit, cfg, q1, q2, T.NoMom{})) == T.Opt.when(Bool.and(U32.is_eq(T.OptCfg.kind(cfg), 0), T.OptCfg.ok(cfg)), cfg, q1, q2, T.SGD{T.nan()}) : T.Opt}

law guard_onemom:
  for -P: Type
  for +cfg: T.OptCfg
  for +q1: F32
  for +q2: F32
  for m: P
  {T.Guarded.opt(P, T.Opt.guard(P, cfg, q1, q2, T.OneMom{m})) == T.Opt.when(Bool.and(U32.is_eq(T.OptCfg.kind(cfg), 1), T.OptCfg.ok(cfg)), cfg, q1, q2, T.nan_mom()) : T.Opt}

law guard_twomom:
  for -P: Type
  for +cfg: T.OptCfg
  for +q1: F32
  for +q2: F32
  for m: P
  for v: P
  {T.Guarded.opt(P, T.Opt.guard(P, cfg, q1, q2, T.TwoMom{m, v})) == T.Opt.when(Bool.and(U32.is_eq(T.OptCfg.kind(cfg), 2), T.OptCfg.ok(cfg)), cfg, q1, q2, T.nan_adam()) : T.Opt}

law guard_packed:
  for -P: Type
  for +cfg: T.OptCfg
  for +q1: F32
  for +q2: F32
  for t: T.PTree
  {T.Guarded.opt(P, T.Opt.guard(P, cfg, q1, q2, T.Packed{t})) == T.Opt.when(Bool.and(U32.is_eq(T.OptCfg.kind(cfg), 3), T.OptCfg.ok(cfg)), cfg, q1, q2, T.nan_adam()) : T.Opt}

# LAW (im2col_rows): im2col's gathered matrix has n·oh·ow rows and c·k·k columns, so its flat list holds n·oh·ow·c·k·k
# numbers, every one of them the value of an image cell or 0 outside a padded window. The Mat's type carries the two
# factors; this law says the list behind it is that long, for every input, kernel, padding and stride.
law im2col_rows:
  for +n: Nat
  for +h: Nat
  for +w: Nat
  for +c: Nat
  for +k: Nat
  for +p: Nat
  for +s: Nat
  for d: Array<F32>
  {List.length(&2, F32, T.im2col_n(n, h, w, c, k, p, s, d)) == Nat.mul(Nat.mul(n, Nat.mul(T.ocount(h, p, k, s), T.ocount(w, p, k, s))), Nat.mul(c, Nat.mul(k, k))) : Nat}

# LAW (conv_roundtrip_ks): the round trip at stride k. With a stride equal to the kernel size and no padding, one
# gathered cell of a patch row is one image cell: slot (ch, ky, kx) of output position (oy, ox) is image cell
# (oy*k + ky, ox*k + kx), channel ch. im2col reads that cell (p_src) and col2im writes the gradient back into it
# (its scatter is p_src itself, in c2_go), so for k = s and p = 0 the two passes address the same number cell for
# cell. `ky_lt` and `kx_lt` write ky < k and kx < k as k = 1 + ky + a and k = 1 + kx + b, which is how the decode
# gets the bounds its division needs. The identity at the centre is the decode:
# i = (p_ch(i,k)*k + p_ky(i,k))*k + p_kx(i,k), proved from the division algorithm's own block step.
law conv_roundtrip_ks:
  for +n: Nat
  for +oy: Nat
  for +ox: Nat
  for +ch: Nat
  for +ky: Nat
  for +kx: Nat
  for +h: Nat
  for +w: Nat
  for +c: Nat
  for +k: Nat
  for +a: Nat
  for +b: Nat
  for +ky_lt: {k == Nat.add(1n, Nat.add(ky, a)) : Nat}
  for +kx_lt: {k == Nat.add(1n, Nat.add(kx, b)) : Nat}
  {T.p_src(Nat.add(Nat.mul(Nat.add(Nat.mul(ch, k), ky), k), kx), n, oy, ox, h, w, c, k, 0n, k) ==
    U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(n, h), Nat.add(Nat.mul(oy, k), ky)), w),
      Nat.add(Nat.mul(ox, k), kx)), c), ch)) : U32}

# LAW (conv_roundtrip_window): the round trip at any stride and padding, per cell. Slot (ch, ky, kx) of output
# position (oy, ox) reads image cell (oy*s + ky - p, ox*s + kx - p), channel ch: im2col reads that cell and col2im
# writes the gradient back into it, because both address it with p_src. The window hypotheses write the two bounds:
# ky < k and kx < k as k = 1 + ky + a and k = 1 + kx + b, and "the window's row is inside the image" as
# oy*s + ky = p + ty, ox*s + kx = p + tx (without them the subtraction p_iy/p_ix truncates and the cell is not read
# at all: p_ok is false there, and the gather uses 0).
#
# The decode does not care about the stride or the padding: p_ch is i/k^2, p_ky is (i mod k^2)/k and p_kx is i mod k,
# whatever s and p are, so this is conv_roundtrip_ks' proof with two subtraction facts instead of sub(x, 0) = x. It
# covers the geometry the layers use (the UNet's k = 3, s = 2, p = 1 among them), and the two narrower laws,
# conv_roundtrip (k = 1, s = 1, p = 0) and conv_roundtrip_ks (s = k, p = 0), are its instances.
law conv_roundtrip_window:
  for +n: Nat
  for +oy: Nat
  for +ox: Nat
  for +ch: Nat
  for +ky: Nat
  for +kx: Nat
  for +h: Nat
  for +w: Nat
  for +c: Nat
  for +k: Nat
  for +p: Nat
  for +s: Nat
  for +a: Nat
  for +b: Nat
  for +ty: Nat
  for +tx: Nat
  for +ky_lt: {k == Nat.add(1n, Nat.add(ky, a)) : Nat}
  for +kx_lt: {k == Nat.add(1n, Nat.add(kx, b)) : Nat}
  for +row_in: {Nat.add(Nat.mul(oy, s), ky) == Nat.add(p, ty) : Nat}
  for +col_in: {Nat.add(Nat.mul(ox, s), kx) == Nat.add(p, tx) : Nat}
  {T.p_src(Nat.add(Nat.mul(Nat.add(Nat.mul(ch, k), ky), k), kx), n, oy, ox, h, w, c, k, p, s) ==
    U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(n, h), ty), w), tx), c), ch)) : U32}

# LAW (tile_row): the t output rows of image row i are all image row i, in order. With t = 1 + s, output row i*t + j
# (j <= s) reads input row i: Nat.div(i*t + j, t) = i. It is the division fact under the whole layer — the forward's
# broadcast and the backward's sum both address through it — and it is stated over tile_row_of, the def the code calls.
law tile_row:
  for +i: Nat
  for +s: Nat
  for +j: Nat
  for +a: Nat
  for +es: {s == Nat.add(j, a) : Nat}
  {T.tile_row_of(Nat.add(1n, s), Nat.add(Nat.mul(i, Nat.add(1n, s)), j)) == i : Nat}

# LAW (tile_sum): every one of the t rows of image i reads (forward) and adds into (backward) the same cell of image
# row i, column k — tile_src(c, t, i*t + j, k) = i*c + k for every j <= s. That is what makes the backward a sum of
# the t rows rather than the last of them; an implementation that offset the row would break it.
law tile_sum:
  for +c: Nat
  for +i: Nat
  for +s: Nat
  for +j: Nat
  for +a: Nat
  for +k: Nat
  for +es: {s == Nat.add(j, a) : Nat}
  {T.tile_src(c, Nat.add(1n, s), Nat.add(Nat.mul(i, Nat.add(1n, s)), j), k) ==
     Nat.add(Nat.mul(i, c), k) : Nat}

# LAW (rope2_partition): a head's 2*half_y + 2*half_x numbers are tiled by the two runs of pairs with nothing missed
# and nothing twice. The y run's pair at j is (j, j + half_y); the x run's is (2*half_y + j, 2*half_y + j + half_x),
# and the law says the x run starts exactly one half_y past the y pair at j = half_y — i.e. the two runs meet at the
# boundary without a gap or an overlap. Each pair is used once because each run walks its own j from 0 to its half.
law rope2_partition:
  for +w: Nat
  for +hy: Nat
  for +hx: Nat
  for +r: Nat
  for +head: Nat
  for +j: Nat
  {T.rope2_xaddr(w, hy, hx, r, head, j) ==
     Nat.add(T.rope2_yaddr(w, hy, hx, r, head, Nat.add(hy, j)), hy) : Nat}

# LAW (rope2_is_rope): at tw = 1 and half_x = 0 the y run is RoPE. RoPE rotates the pair (c, c + half) of head `head`
# in row r at address r*w + head*(half + half) + c, and the y run at half_y = half, half_x = 0 addresses the same pair
# the same way — so the 2D layer is RoPE's general case, not a second implementation of the same rotation.
law rope2_is_rope:
  for +w: Nat
  for +half: Nat
  for +r: Nat
  for +head: Nat
  for +j: Nat
  {T.rope2_yaddr(w, half, 0n, r, head, j) ==
     Nat.add(Nat.add(Nat.mul(r, w), Nat.mul(head, Nat.add(half, half))), j) : Nat}

# LAW (rope2_is_rope_tab): the same for the table: at tw = 1 the y entry (r mod (th*1))/1 * half + j is RoPE's
# (r mod th)*half + j, so both layers read the same angles for the same positions.
law rope2_is_rope_tab:
  for +half: Nat
  for +r: Nat
  for +j: Nat
  for +th: Nat
  {T.rope2_yentry(Nat.mul(th, 1n), 1n, half, r, j) == Nat.add(Nat.mul(Nat.mod(r, th), half), j) : Nat}

# LAW (conv_roundtrip): with a 1 x 1 kernel, no padding and a stride of 1, a patch is one cell: im2col reads the image
# cell at the patch's own coordinates, (b, oy, ox, ch), and col2im writes that number back into it (its scatter is p_src
# itself, in c2_go). So for k = 1, s = 1 and p = 0 the round trip col2im(im2col(x)) = x is the identity, cell for cell,
# into the cell it came from. The kernel sizes that are covered against PyTorch are the ones in
# reference/im2col_check.py and reference/conv2d_check.py.
law conv_roundtrip:
  for +b: Nat
  for +iy: Nat
  for +ix: Nat
  for +ch: Nat
  for +h: Nat
  for +w: Nat
  for +c: Nat
  {T.p_src(ch, b, iy, ix, h, w, c, 1n, 0n, 1n) == U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(b, h), iy), w), ix), c), ch)) : U32}

# LAW (mp_count): with a surplus h - k = q·s + r and r < s (s = 1 + r + a), the MaxPool output count
# (h - k)/s + 1 is q + 1: q whole windows fit and the next start is past the surplus.
law mp_count:
  for +q: Nat
  for +r: Nat
  for +a: Nat
  for +s: Nat
  for +sr: {s == Nat.add(1n, Nat.add(r, a)) : Nat}
  {Nat.add(Nat.div(Nat.add(Nat.mul(q, s), r), s), 1n) == Nat.add(q, 1n) : Nat}

# LAW (mp_scatter): the backward's write for output (row, col) at window offset (ky, kx) is the cell
# the forward read there — the image's base plus that window's plane index, not another output's
# cell. ky < k and kx < k (written as k = 1 + ky + a) keep the offset inside the kernel.
law mp_scatter:
  for +c: Nat
  for +h: Nat
  for +w: Nat
  for +ch: Nat
  for +image: Nat
  for +row: Nat
  for +col: Nat
  for +ky: Nat
  for +kx: Nat
  for +k: Nat
  for +a: Nat
  for +b: Nat
  for +ky_lt: {k == Nat.add(1n, Nat.add(ky, a)) : Nat}
  for +kx_lt: {k == Nat.add(1n, Nat.add(kx, b)) : Nat}
  {T.mp_flat(c, h, w, ch, image, Nat.add(row, ky), Nat.add(col, kx)) ==
   Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.mul(image, h), w), c), ch),
     Nat.mul(Nat.add(Nat.mul(Nat.add(row, ky), w), Nat.add(col, kx)), c)) : Nat}
