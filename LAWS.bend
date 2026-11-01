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
