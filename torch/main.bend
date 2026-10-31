# BendTorch: layers with typed forward and backward passes, composed generically, trained in Bend 2.
#
#   import ../torch/main.bend as T
#
# A layer is a record of four functions over its parameters P, its input X, its output Y and what its forward pass
# saves for the backward pass, S:
#
#   fwd   : P -> X -> Fwd<P, Y, S>               the output, the parameters handed back, the saved values
#   bwd   : P -> S -> Y -> P -> Bwd<P, X>        from dY (typed Y) to dX (typed X), adding dP into a gradient of type P
#   step  : Opt -> P -> P -> OptState<P> -> Step<P>  one optimizer step on the parameters; the gradient comes back zeroed
#   zeros : Unit -> P                            a zero P (gradients and optimizer state start from it)
#
# The gradient of the input has the input's type and the gradient of the parameters has the parameters' type, so a
# backward pass that returns a transposed or a wrongly sized gradient does not compile. Layers are passed as templates
# (`~l`), so `Seq` composes any two layers whose types meet, with no closure called twice and no runtime cost.
#
# Matrices are bend-ml's `Mat<r, c>` (bend-ml-tensor-array 0.1.14.0): r*c F32 numbers in one flat Array, row by row,
# the dimensions erased type parameters. An Array has one owner, so every operation hands back what it reads.
#
# What is proved and what is tested: the shapes are in the types (checked); the laws at the end of this file are
# checked by Bend's kernel; every F32 result is tested against PyTorch by the scripts in reference/, never proved.

import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import bend-ml-nat-lemmas@0.1.2.0/main.bend as NL

# =====================================================================================================
# The layer protocol
# =====================================================================================================

# what a forward pass returns: the parameters, the output, what the backward pass will need
type Fwd<-P: Type, -Y: Type, -S: Type> is Type:
  Fwd{p: P, y: Y, s: S}

# what a backward pass returns: the parameters, the gradient of the input, the accumulated gradient of the parameters
type Bwd<-P: Type, -X: Type> is Type:
  Bwd{p: P, dx: X, g: P}

# a pair of owned values (parameters of two layers or their saved values)
type Two<-A: Type, -B: Type> is Type:
  Two{a: A, b: B}

# Type helpers for the left-associated pairs used by Seq/Fork and for Residual's two parameter-free layers.
def Pair3(A: Type, B: Type, C: Type) -> Type:
  Two<Two<A, B>, C>

def ResidualType(A: Type) -> Type:
  Pair3(Unit, A, Unit)

# AdamW's moments stored as packed BF16 (CAdamWBF16): a tree that follows the parameters' pairs as the step walks them,
# with the buffers of each matrix at its leaf. A packed buffer has half the words of its matrix (two BF16 values a
# U32, element 2k in the low half of word k, torch/dtype.bend's layout), so it cannot be a value of the parameters'
# type P: the tree holds the words itself. A leaf is allocated on the matrix's first step (PNew marks a part not yet
# stepped, and stays at the parameter-free layers), so the full-size buffers never exist. v16 picks v packed as well;
# otherwise v is F32 (PLeafV). A leaf keeps its matrix's count n, for checkpoints.
type PTree is Type:
  PNew{v16: Bool}
  PLeaf{n: U32, m: Array<U32>, v: Array<U32>}
  PLeafV{n: U32, m: Array<U32>, v: Array<F32>}
  PNode{a: PTree, b: PTree}

# Owned optimizer buffers: SGD has none, momentum one, Adam/AdamW two (Packed: two, in packed BF16, see PTree).
type OptState<-P: Type> is Type:
  NoMom{}
  OneMom{m: P}
  TwoMom{m: P, v: P}
  Packed{t: PTree}

# Two affine maps are supplied separately: each is called at most once. A packed tree is the same under any relabelling
# of the parameters (Lin <-> Two), so it passes unchanged.
def OptState.map(-P: Type, -Q: Type, st: OptState<P>, f: P -> Q, h: P -> Q) -> OptState<Q>:
  match st:
    case NoMom{}: NoMom{}
    case OneMom{m}: OneMom{f(m)}
    case TwoMom{m, v}: TwoMom{f(m), h(v)}
    case Packed{t}: Packed{t}

# A packed tree splits at its node; a part not yet stepped splits into two such parts. A leaf belongs to one matrix and
# is never split (it goes to the left, the right side starts empty).
def OptState.split(-A: Type, -B: Type, st: OptState<Two<A, B>>) -> Two<OptState<A>, OptState<B>>:
  match st:
    case NoMom{}: Two{NoMom{}, NoMom{}}
    case OneMom{Two{a, b}}: Two{OneMom{a}, OneMom{b}}
    case TwoMom{Two{a, b}, Two{c, d}}: Two{TwoMom{a, c}, TwoMom{b, d}}
    case Packed{PNode{a, b}}: Two{Packed{a}, Packed{b}}
    case Packed{PNew{+f}}: Two{Packed{PNew{f}}, Packed{PNew{f}}}
    case Packed{t}: Two{Packed{t}, Packed{PNew{False{}}}}

# Pair the buffers present on both sides. Internal callers split one state, then step both children with the same
# optimizer, so the variants agree. No buffer is ever allocated to fill an absent side.
def OptState.join(-A: Type, -B: Type, a: OptState<A>, b: OptState<B>) -> OptState<Two<A, B>>:
  match a b:
    case OneMom{x} OneMom{y}: OneMom{Two{x, y}}
    case TwoMom{x, u} TwoMom{y, v}: TwoMom{Two{x, y}, Two{u, v}}
    case Packed{x} Packed{y}: Packed{PNode{x, y}}
    case _ _: NoMom{}

# what an optimizer step returns: the new parameters, the zeroed gradient, the optimizer state
type Step<-P: Type> is Type:
  Step{p: P, g: P, st: OptState<P>}

# The optimizer, with the scalars of the current step. Adam's `step` is lr / (1 - b1^t) and `bc2` is
# sqrt(1 - b2^t), as PyTorch computes them; `Opt.at` fills them from an `OptCfg` and the step count.
# `dec` picks AdamW's decoupled weight decay (w *= 1 - lr*wd) over Adam's (g += wd*w).
type Opt is Data:
  SGD{lr: F32}
  Momentum{lr: F32, mu: F32}
  Adam{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32, dec: Bool, step: F32, bc2: F32, c1: F32, c2: F32}

# Parameters and their flat dump in a fixed order. Loading has a separate array cursor (PA).
type PL<-P: Type> is Type:
  PL{p: P, xs: List<&2, F32>}

# Loading cursor: the shared source array and next element offset, handed back without a list or a clone.
# The IO boundary checks the source length against Layer.count before calling the pure loaders.
type PA<-P: Type> is Type:
  PA{p: P, xs: Array<F32>, off: U32}

type Layer<-P: Type, -X: Type, -Y: Type, -S: Type> is Type:
  Layer{fwd: P -> X -> Fwd<P, Y, S>, bwd: P -> S -> Y -> P -> Bwd<P, X>, step: Opt -> P -> P -> OptState<P> -> Step<P>, zeros: Unit -> P, dump: P -> PL<P>, load: P -> Array<F32> -> U32 -> PA<P>, store: P -> Array<F32> -> U32 -> PA<P>, count: Unit -> U32}

def Layer.fwd(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, x: X) -> Fwd<P, Y, S>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}:
      fwd(p, x)

def Layer.bwd(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, s: S, dy: Y, g: P) -> Bwd<P, X>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}:
      bwd(p, s, dy, g)

def Layer.step(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, +o: Opt, p: P, g: P, st: OptState<P>) -> Step<P>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}:
      step(o, p, g, st)

def Layer.zeros(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>) -> P:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}:
      zeros(Unit{})

# every number of the parameters, in the layer's fixed order (the parameters handed back)
def Layer.dump(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P) -> PL<P>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}:
      dump(p)

# Parameters overwritten from xs at an element offset, in dump order; xs and the next offset come back.
def Layer.load(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, xs: Array<F32>, off: U32) -> PA<P>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}: load(p, xs, off)

# The parameters copied into xs from an element offset, in dump order (load's inverse); xs and the next offset come back.
def Layer.store(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>, p: P, xs: Array<F32>, off: U32) -> PA<P>:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}: store(p, xs, off)

def Layer.count(-P: Type, -X: Type, -Y: Type, -S: Type, l: Layer<P, X, Y, S>) -> U32:
  match l:
    case Layer{fwd, bwd, step, zeros, dump, load, store, count}: count(Unit{})

# =====================================================================================================
# Element-wise kernels (templates: one copy per function, no closure at run time)
# =====================================================================================================

# a[t] = f(a[t]) for the n numbers from t; ra holds the array and a[t], already read
def ew1_l(~f: @+a: F32 -> F32, nleft: Nat, +t: U32, ra: Array<F32> & F32) -> Array<F32>:
  match nleft ra:
    case 0n Tuple{a, v}:
      a
    case 1n+p Tuple{a, +v}:
      ew1_l(~f, p, (t + 1 : U32), Array.get(F32, Array.set(F32, a, t, f(v)), (t + 1 : U32)))

# a[t] = f(a[t], b[t]): the result replaces a; b comes back unchanged
def ew2_l(~f: @+a: F32 -> @+b: F32 -> F32, nleft: Nat, +t: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> TA.P2:
  match nleft ra rb:
    case 0n Tuple{a, v} Tuple{b, w}:
      TA.P2{a, b}
    case 1n+p Tuple{a, +v} Tuple{b, +w}:
      ew2_l(~f, p, (t + 1 : U32), Array.get(F32, Array.set(F32, a, t, f(v, w)), (t + 1 : U32)), Array.get(F32, b, (t + 1 : U32)))

def Ew.map(~f: @+a: F32 -> F32, +r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{d}:
      TA.Mat{ew1_l(~f, Nat.mul(r, c), 0, Array.get(F32, d, 0))}

def ew2_fin(-r: Nat, -c: Nat, p: TA.P2) -> TA.MPair<r, c>:
  match p:
    case TA.P2{a, b}:
      TA.MPair{TA.Mat{a}, TA.Mat{b}}

# (f(a, b), b), element by element
def Ew.zip(~f: @+a: F32 -> @+b: F32 -> F32, +r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.MPair<r, c>:
  match a b:
    case TA.Mat{x} TA.Mat{y}:
      ew2_fin(r, c, ew2_l(~f, Nat.mul(r, c), 0, Array.get(F32, x, 0), Array.get(F32, y, 0)))

def f32_add(+a: F32, +b: F32) -> F32:
  (a + b : F32)

def f32_mul(+a: F32, +b: F32) -> F32:
  (a * b : F32)

# (a + b, b)
def Mat.add(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.MPair<r, c>:
  Ew.zip(~f32_add, r, c, a, b)

def zero_of(+x: F32) -> F32:
  0.0

# the smallest d with 2^d >= n (TA.cap_depth's answer), by doubling a U32: log2(n) cheap steps, where cap_depth
# compares Nats one unit at a time at each doubling; f is fuel (32 doublings cover every U32)
def depth_go(f: Nat, ok: Bool, +n: U32, +p: U32, d: Nat) -> Nat:
  match f ok:
    case _ True{}:
      d
    case 0n False{}:
      d
    case 1n+g False{}:
      depth_go(g, U32.is_le(n, (p + p : U32)), n, (p + p : U32), 1n+d)

def depth(+n: Nat) -> Nat:
  depth_go(32n, U32.is_le(U32.from_nat(n), 1), U32.from_nat(n), 1, 0n)

# =====================================================================================================
# Optimizers: one pass over a parameter, its gradient and the two moment buffers; the gradient is zeroed
# =====================================================================================================

type Q3 is Type:
  Q3{w: Array<F32>, g: Array<F32>, m: Array<F32>}

type Q4 is Type:
  Q4{w: Array<F32>, g: Array<F32>, m: Array<F32>, v: Array<F32>}

type MatStep<-r: Nat, -c: Nat> is Type:
  MatStep{w: TA.Mat<r, c>, g: TA.Mat<r, c>, st: OptState<TA.Mat<r, c>>}

# w <- w + (-lr)*g, as PyTorch's p.add_(g, alpha=-lr)
def sgd_l(nleft: Nat, +t: U32, +lr: F32, rw: Array<F32> & F32, rg: Array<F32> & F32) -> TA.P2:
  match nleft rw rg:
    case 0n Tuple{w, x} Tuple{g, y}:
      TA.P2{w, g}
    case 1n+p Tuple{w, +x} Tuple{g, +y}:
      sgd_l(p, (t + 1 : U32), lr, Array.get(F32, Array.set(F32, w, t, (x + (F32.neg(lr) * y : F32) : F32)), (t + 1 : U32)), Array.get(F32, Array.set(F32, g, t, 0.0), (t + 1 : U32)))

# buf <- buf*mu + g; w <- w + (-lr)*buf, as PyTorch's SGD with momentum (no dampening, not Nesterov)
def mom_l(nleft: Nat, +t: U32, +lr: F32, +mu: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32) -> Q3:
  match nleft rw rg rm:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, b}:
      Q3{w, g, m}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +b}:
      +nb = ((b * mu : F32) + y : F32)
      +u = (t + 1 : U32)
      mom_l(p, u, lr, mu, Array.get(F32, Array.set(F32, w, t, (x + (F32.neg(lr) * nb : F32) : F32)), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(F32, Array.set(F32, m, t, nb), u))

# the Adam family, in PyTorch's order of operations (torch.optim.adam._single_tensor_adam):
#   g' = g + wd*w (Adam) or w' = w*(1 - lr*wd) (AdamW); m = m + (1-b1)*(g' - m) (lerp); v = v*b2 + (1-b2)*g'*g';
#   w = w' + (-step*m) / (sqrt(v)/bc2 + eps)
def adam_grad(dec: Bool, +wd: F32, +w: F32, +g: F32) -> F32:
  match dec:
    case True{}:
      g
    case False{}:
      (g + (w * wd : F32) : F32)

def adam_decay(dec: Bool, +lr: F32, +wd: F32, +w: F32) -> F32:
  match dec:
    case True{}:
      (w * (1.0 - (lr * wd : F32) : F32) : F32)
    case False{}:
      w

# c1 = 1 - b1 and c2 = 1 - b2 are given as their own constants (0.1, 0.001), as PyTorch computes 1 - beta in float64
# before rounding: 1 - b2 taken in F32 from b2 = 0.999 (0.99900001287 in F32) is 1.3e-5 too small, and over 100 steps
# that alone put the tiny Llama 21x further from float64 than PyTorch's float32 (NOTES, 2026-10-18)
def adam_m(+c1: F32, +m: F32, +g: F32) -> F32:
  (m + (c1 * (g - m : F32) : F32) : F32)

def adam_v(+b2: F32, +c2: F32, +v: F32, +g: F32) -> F32:
  ((v * b2 : F32) + ((c2 * g : F32) * g : F32) : F32)

def adam_w(+step: F32, +bc2: F32, +eps: F32, +w: F32, +m: F32, +v: F32) -> F32:
  (w + ((F32.neg(step) * m : F32) / ((F32.sqrt(v) / bc2 : F32) + eps : F32) : F32) : F32)

def adam_l(nleft: Nat, +t: U32, +lr: F32, +c1: F32, +b2: F32, +c2: F32, +eps: F32, +wd: F32, +dec: Bool, +step: F32, +bc2: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<F32> & F32, rv: Array<F32> & F32) -> Q4:
  match nleft rw rg rm rv:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, a} Tuple{v, b}:
      Q4{w, g, m, v}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}:
      +gg = adam_grad(dec, wd, x, y)
      +nm = adam_m(c1, a, gg)
      +nv = adam_v(b2, c2, b, gg)
      +nw = adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), nm, nv)
      +u = (t + 1 : U32)
      adam_l(p, u, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, Array.set(F32, w, t, nw), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(F32, Array.set(F32, m, t, nm), u), Array.get(F32, Array.set(F32, v, t, nv), u))

# ---- AdamW over packed BF16 moments (CAdamWBF16) ----
# The same F32 arithmetic as adam_l, element by element: each moment is read from its half of a word (BF16 -> F32 is
# exact), updated in F32, rounded to BF16 (to nearest, ties to even; a NaN becomes 0xFFFF, as torch/dtype.bend and
# PyTorch's CPU kernel do) and written back into its half; the weight update reads the stored, rounded moments, as
# PyTorch's AdamW does with BF16 state. The weights and the gradient stay F32.

def ph_f2u(x: F32) -> U32:
  match x:
    case F32{w}:
      U32{w}

def ph_u2f(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def ph_pick(b: Bool, +x: U32, +y: U32) -> U32:
  match b:
    case True{}:
      x
    case False{}:
      y

# F32 -> BF16 bits (dtype.bend's BF16.of_f32): add 0x7FFF plus the lowest kept bit, keep the top half; NaN -> 0xFFFF
def ph_of_bits(+u: U32) -> U32:
  ph_pick(U32.is_gt((u .&. 2147483647 : U32), 2139095040), 65535, U32.shrn(((u + 32767 : U32) + (U32.shrn(u, 16n) .&. 1 : U32) : U32), 16n))

def ph_of(x: F32) -> U32:
  ph_of_bits(ph_f2u(x))

# BF16 bits -> F32, exact
def ph_to(+h: U32) -> F32:
  ph_u2f(U32.shln(h, 16n))

# the half of word w that element t uses (odd t: the high half), and w with that half replaced by h
def ph_half(odd: Bool, +w: U32) -> U32:
  match odd:
    case True{}:
      U32.shrn(w, 16n)
    case False{}:
      (w .&. 65535 : U32)

def ph_put(odd: Bool, +w: U32, +h: U32) -> U32:
  match odd:
    case True{}:
      ((w .&. 65535 : U32) .|. U32.shln(h, 16n) : U32)
    case False{}:
      ((w .&. 4294901760 : U32) .|. h : U32)

type PQ is Type:
  PQ{w: Array<F32>, g: Array<F32>, m: Array<U32>, v: Array<U32>}

type PQV is Type:
  PQV{w: Array<F32>, g: Array<F32>, m: Array<U32>, v: Array<F32>}

# m and v packed. rm and rv hold the word of element t (index t >> 1), read after the previous element's write.
def adamh_l(nleft: Nat, +t: U32, +lr: F32, +c1: F32, +b2: F32, +c2: F32, +eps: F32, +wd: F32, +dec: Bool, +step: F32, +bc2: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<U32> & U32, rv: Array<U32> & U32) -> PQ:
  match nleft rw rg rm rv:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, a} Tuple{v, b}:
      PQ{w, g, m, v}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}:
      +odd = U32.is_eq((t .&. 1 : U32), 1)
      +gg = adam_grad(dec, wd, x, y)
      +hm = ph_of(adam_m(c1, ph_to(ph_half(odd, a)), gg))
      +hv = ph_of(adam_v(b2, c2, ph_to(ph_half(odd, b)), gg))
      +nw = adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), ph_to(hm), ph_to(hv))
      +u = (t + 1 : U32)
      +k = U32.shrn(t, 1n)
      +ku = U32.shrn(u, 1n)
      adamh_l(p, u, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, Array.set(F32, w, t, nw), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(U32, Array.set(U32, m, k, ph_put(odd, a, hm)), ku), Array.get(U32, Array.set(U32, v, k, ph_put(odd, b, hv)), ku))

# m packed, v in F32 (one cell an element, as adam_l's)
def adamhv_l(nleft: Nat, +t: U32, +lr: F32, +c1: F32, +b2: F32, +c2: F32, +eps: F32, +wd: F32, +dec: Bool, +step: F32, +bc2: F32, rw: Array<F32> & F32, rg: Array<F32> & F32, rm: Array<U32> & U32, rv: Array<F32> & F32) -> PQV:
  match nleft rw rg rm rv:
    case 0n Tuple{w, x} Tuple{g, y} Tuple{m, a} Tuple{v, b}:
      PQV{w, g, m, v}
    case 1n+p Tuple{w, +x} Tuple{g, +y} Tuple{m, +a} Tuple{v, +b}:
      +odd = U32.is_eq((t .&. 1 : U32), 1)
      +gg = adam_grad(dec, wd, x, y)
      +hm = ph_of(adam_m(c1, ph_to(ph_half(odd, a)), gg))
      +nv = adam_v(b2, c2, b, gg)
      +nw = adam_w(step, bc2, eps, adam_decay(dec, lr, wd, x), ph_to(hm), nv)
      +u = (t + 1 : U32)
      adamhv_l(p, u, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, Array.set(F32, w, t, nw), u), Array.get(F32, Array.set(F32, g, t, 0.0), u), Array.get(U32, Array.set(U32, m, U32.shrn(t, 1n), ph_put(odd, a, hm)), U32.shrn(u, 1n)), Array.get(F32, Array.set(F32, v, t, nv), u))

# the words of n packed values: ceil(n / 2), in a block of the next power of two
def ph_words(+n: Nat) -> Array<U32>:
  Array.new(U32, depth(Nat.div(Nat.add(n, 1n), 2n)), 0)

def ph_fin(-r: Nat, -c: Nat, +n: U32, q: PQ) -> MatStep<r, c>:
  match q:
    case PQ{w, g, m, v}: MatStep{TA.Mat{w}, TA.Mat{g}, Packed{PLeaf{n, m, v}}}

def phv_fin(-r: Nat, -c: Nat, +n: U32, q: PQV) -> MatStep<r, c>:
  match q:
    case PQV{w, g, m, v}: MatStep{TA.Mat{w}, TA.Mat{g}, Packed{PLeafV{n, m, v}}}

# one matrix's step on its leaf; anything else leaves the values untouched, as Opt.mat's fallback
def adamh_leaf(+r: Nat, +c: Nat, +o: Opt, wa: Array<F32>, ga: Array<F32>, t: PTree) -> MatStep<r, c>:
  match o t:
    case Adam{lr, b1, b2, eps, wd, dec, step, bc2, c1, c2} PLeaf{+n, ma, va}:
      ph_fin(r, c, n, adamh_l(Nat.mul(r, c), 0, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, wa, 0), Array.get(F32, ga, 0), Array.get(U32, ma, 0), Array.get(U32, va, 0)))
    case Adam{lr, b1, b2, eps, wd, dec, step, bc2, c1, c2} PLeafV{+n, ma, va}:
      phv_fin(r, c, n, adamhv_l(Nat.mul(r, c), 0, lr, c1, b2, c2, eps, wd, dec, step, bc2, Array.get(F32, wa, 0), Array.get(F32, ga, 0), Array.get(U32, ma, 0), Array.get(F32, va, 0)))
    case _ t:
      MatStep{TA.Mat{wa}, TA.Mat{ga}, Packed{t}}

# the leaf of a matrix on its first step: zeros (BF16 0x0000 is +0.0)
def ph_new(+n: Nat, v16: Bool) -> PTree:
  match v16:
    case True{}:
      PLeaf{U32.from_nat(n), ph_words(n), ph_words(n)}
    case False{}:
      PLeafV{U32.from_nat(n), ph_words(n), Array.new(F32, depth(n), 0.0)}

def adamh_mat(+r: Nat, +c: Nat, +o: Opt, wa: Array<F32>, ga: Array<F32>, t: PTree) -> MatStep<r, c>:
  match t:
    case PNew{v16}:
      adamh_leaf(r, c, o, wa, ga, ph_new(Nat.mul(r, c), v16))
    case t:
      adamh_leaf(r, c, o, wa, ga, t)

# One matrix step, with only the buffers this optimizer reads.
def Opt.sgd_fin(-r: Nat, -c: Nat, q: TA.P2) -> MatStep<r, c>:
  match q:
    case TA.P2{w, g}: MatStep{TA.Mat{w}, TA.Mat{g}, NoMom{}}

def Opt.mom_fin(-r: Nat, -c: Nat, q: Q3) -> MatStep<r, c>:
  match q:
    case Q3{w, g, m}: MatStep{TA.Mat{w}, TA.Mat{g}, OneMom{TA.Mat{m}}}

def Opt.fin(-r: Nat, -c: Nat, q: Q4) -> MatStep<r, c>:
  match q:
    case Q4{w, g, m, v}: MatStep{TA.Mat{w}, TA.Mat{g}, TwoMom{TA.Mat{m}, TA.Mat{v}}}

# A mismatched state is initialized only at the training boundary; Opt.mat requires the matching variant.
# The fallback leaves the values untouched, so it cannot fabricate a moment buffer.
def Opt.mat(+r: Nat, +c: Nat, +o: Opt, w: TA.Mat<r, c>, g: TA.Mat<r, c>, st: OptState<TA.Mat<r, c>>) -> MatStep<r, c>:
  match o w g st:
    case SGD{lr} TA.Mat{wd} TA.Mat{gd} _:
      Opt.sgd_fin(r, c, sgd_l(Nat.mul(r, c), 0, lr, Array.get(F32, wd, 0), Array.get(F32, gd, 0)))
    case Momentum{lr, mu} TA.Mat{wd} TA.Mat{gd} OneMom{TA.Mat{md}}:
      Opt.mom_fin(r, c, mom_l(Nat.mul(r, c), 0, lr, mu, Array.get(F32, wd, 0), Array.get(F32, gd, 0), Array.get(F32, md, 0)))
    case Adam{lr, b1, b2, eps, wd2, dec, step, bc2, c1, c2} TA.Mat{wd} TA.Mat{gd} TwoMom{TA.Mat{md}, TA.Mat{vd}}:
      Opt.fin(r, c, adam_l(Nat.mul(r, c), 0, lr, c1, b2, c2, eps, wd2, dec, step, bc2, Array.get(F32, wd, 0), Array.get(F32, gd, 0), Array.get(F32, md, 0), Array.get(F32, vd, 0)))
    case Adam{lr, b1, b2, eps, wd2, dec, step, bc2, c1, c2} TA.Mat{wd} TA.Mat{gd} Packed{t}:
      adamh_mat(r, c, o, wd, gd, t)
    case _ w g st:
      MatStep{w, g, st}

# What the user picks; `Opt.at` turns it into the step's `Opt` from q1 = 1 - b1^t and q2 = 1 - b2^t (t = 1 on the
# first step). Adam's c1 and c2 are 1 - b1 and 1 - b2, written as their own constants (adam_m above). CAdamWBF16 is
# AdamW with its first moment (and its second, when v16) stored as packed BF16 (PTree); weights and gradient stay F32.
type OptCfg is Data:
  CSGD{lr: F32}
  CMomentum{lr: F32, mu: F32}
  CAdam{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32, c1: F32, c2: F32}
  CAdamW{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32, c1: F32, c2: F32}
  CAdamWBF16{lr: F32, b1: F32, b2: F32, eps: F32, wd: F32, c1: F32, c2: F32, v16: Bool}

def Opt.at(+cfg: OptCfg, +q1: F32, +q2: F32) -> Opt:
  match cfg:
    case CSGD{lr}:
      SGD{lr}
    case CMomentum{lr, mu}:
      Momentum{lr, mu}
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      Adam{lr, b1, b2, eps, wd, False{}, (lr / q1 : F32), F32.sqrt(q2), c1, c2}
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      Adam{lr, b1, b2, eps, wd, True{}, (lr / q1 : F32), F32.sqrt(q2), c1, c2}
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      Adam{lr, b1, b2, eps, wd, True{}, (lr / q1 : F32), F32.sqrt(q2), c1, c2}

def OptCfg.b1(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      b1
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      b1
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      b1
    case _:
      0.0

def OptCfg.b2(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      b2
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      b2
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      b2
    case _:
      0.0

def OptCfg.c1(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      c1
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      c1
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      c1
    case _:
      0.0

def OptCfg.c2(+cfg: OptCfg) -> F32:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      c2
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      c2
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      c2
    case _:
      0.0

# c = 1 - b up to F32's rounding, with 0 < c <= 1 (PyTorch's 0 <= beta < 1). For b in [0, 1), 1 - b taken in F32 is
# within half an ulp of the exact value and c's own rounding adds at most as much, so a c written as 1 - b is always
# under 2^-23 from it, while a typo (0.01 for 0.001, c1 and c2 swapped) is far above. A NaN fails every comparison.
def beta_ok(+b: F32, +c: F32) -> Bool:
  +d = ((1.0 - b : F32) - c : F32)
  Bool.and(Bool.and((c > 0.0 : F32), (c <= 1.0 : F32)), ((d * d : F32) <= 0.0000000000000142108547152020037174224853515625 : F32))

# True unless an Adam configuration's c1, c2 disagree with its b1, b2. Train.new poisons the state of a configuration
# that fails (every loss after the first is NaN); a program checks it first with OptCfg.check to stop with a message.
def OptCfg.ok(+cfg: OptCfg) -> Bool:
  match cfg:
    case CAdam{lr, b1, b2, eps, wd, c1, c2}:
      Bool.and(beta_ok(b1, c1), beta_ok(b2, c2))
    case CAdamW{lr, b1, b2, eps, wd, c1, c2}:
      Bool.and(beta_ok(b1, c1), beta_ok(b2, c2))
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}:
      Bool.and(beta_ok(b1, c1), beta_ok(b2, c2))
    case _:
      True{}

def cfg_stop(ok: Bool) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, "CAdam/CAdamW: c1, c2 do not match b1, b2 (c1 = 1 - b1, c2 = 1 - b2, e.g. 0.9, 0.999 with 0.1, 0.001)")

def OptCfg.check(+cfg: OptCfg) -> IO(Unit):
  cfg_stop(OptCfg.ok(cfg))

# dump and load for a matrix: its r*c numbers, row by row
def mat_dump_fin(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> PL<TA.Mat<r, c>>:
  match ml:
    case TA.ML{m, xs}:
      PL{m, xs}

def Mat.dump(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> PL<TA.Mat<r, c>>:
  mat_dump_fin(r, c, TA.Mat.to_list(r, c, m))

def load_cells(left: Nat, +i: U32, +off: U32, rx: Array<F32> & F32, dst: Array<F32>) -> TA.P2:
  match left rx:
    case 0n Tuple{xs, v}: TA.P2{xs, dst}
    case 1n+q Tuple{xs, v}:
      load_cells(q, (i + 1 : U32), (off + 1 : U32), Array.get(F32, xs, (off + 1 : U32)), Array.set(F32, dst, i, v))

def mat_restore_fin(-r: Nat, -c: Nat, +off: U32, q: TA.P2) -> PA<TA.Mat<r, c>>:
  match q:
    case TA.P2{xs, dst}: PA{TA.Mat{dst}, xs, off}

def Mat.restore(+r: Nat, +c: Nat, m: TA.Mat<r, c>, xs: Array<F32>, +off: U32) -> PA<TA.Mat<r, c>>:
  match m:
    case TA.Mat{dst}:
      mat_restore_fin(r, c, (off + U32.from_nat(Nat.mul(r, c)) : U32), load_cells(Nat.mul(r, c), 0, off, Array.get(F32, xs, off), dst))

# load_cells' inverse: copies the matrix's cells i.. into xs from `off`; both arrays come back (the matrix's first)
def store_cells(left: Nat, +i: U32, +off: U32, rm: Array<F32> & F32, xs: Array<F32>) -> TA.P2:
  match left rm:
    case 0n Tuple{ma, v}: TA.P2{ma, xs}
    case 1n+q Tuple{ma, v}:
      store_cells(q, (i + 1 : U32), (off + 1 : U32), Array.get(F32, ma, (i + 1 : U32)), Array.set(F32, xs, off, v))

def mat_store_fin(-r: Nat, -c: Nat, +off: U32, q: TA.P2) -> PA<TA.Mat<r, c>>:
  match q:
    case TA.P2{ma, xs}: PA{TA.Mat{ma}, xs, off}

def Mat.store(+r: Nat, +c: Nat, m: TA.Mat<r, c>, xs: Array<F32>, +off: U32) -> PA<TA.Mat<r, c>>:
  match m:
    case TA.Mat{ma}:
      mat_store_fin(r, c, (off + U32.from_nat(Nat.mul(r, c)) : U32), store_cells(Nat.mul(r, c), 0, off, Array.get(F32, ma, 0), xs))

def pa_second(-A: Type, -B: Type, a: A, rb: PA<B>) -> PA<Two<A, B>>:
  match rb:
    case PA{b, xs, off}: PA{Two{a, b}, xs, off}

def pl_pair(-A: Type, -B: Type, ra: PL<A>, rb: PL<B>) -> PL<Two<A, B>>:
  match ra rb:
    case PL{a, xa} PL{b, xb}:
      PL{Two{a, b}, List.append(&2, F32, xa, xb)}

def pl_second(-A: Type, -B: Type, a: A, rb: PL<B>) -> PL<Two<A, B>>:
  match rb:
    case PL{b, rest}:
      PL{Two{a, b}, rest}

# =====================================================================================================
# Linear: y = x·Wᵀ + b, with W stored out x in (PyTorch's nn.Linear layout); saves its input
# =====================================================================================================

type Lin<-i: Nat, -o: Nat> is Type:
  Lin{w: TA.Mat<o, i>, b: TA.Mat<1n, o>}

def lin_f3(-n: Nat, -i: Nat, -o: Nat, w: TA.Mat<o, i>, x: TA.Mat<n, i>, r: TA.MBias<n, o>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match r:
    case TA.MBias{y, b}:
      Fwd{Lin{w, b}, y, x}

def lin_f2(+n: Nat, -i: Nat, +o: Nat, b: TA.Mat<1n, o>, r: TA.MMulNT<n, i, o>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulNT{x, w, z}:
      lin_f3(n, i, o, w, x, TA.Mat.add_row(n, o, z, b))

def lin_fwd(+n: Nat, +i: Nat, +o: Nat, par: Nat, p: Lin<i, o>, x: TA.Mat<n, i>) -> Fwd<Lin<i, o>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match p:
    case Lin{w, b}:
      lin_f2(n, i, o, b, TA.Mat.matmul_nt(n, i, o, par, x, w))

def lin_b4(-n: Nat, -i: Nat, -o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, dx: TA.Mat<n, i>, gw: TA.Mat<o, i>, r: TA.MColSum<n, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match r:
    case TA.MColSum{dy, gb}:
      Bwd{Lin{w, b}, dx, Lin{gw, gb}}

def lin_b3(+n: Nat, -i: Nat, +o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, dx: TA.Mat<n, i>, gb: TA.Mat<1n, o>, r: TA.MMulTN<o, n, i>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulTN{dy, x, gw}:
      lin_b4(n, i, o, w, b, dx, gw, TA.Mat.col_sums_acc(n, o, dy, gb))

def lin_b2(+n: Nat, +i: Nat, +o: Nat, par: Nat, b: TA.Mat<1n, o>, x: TA.Mat<n, i>, g: Lin<i, o>, r: TA.MMul<n, o, i>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match g r:
    case Lin{gw, gb} TA.MMul{dy, w, dx}:
      lin_b3(n, i, o, w, b, dx, gb, TA.Mat.matmul_tn_acc(o, n, i, par, dy, x, gw))

# dX = dY·W, dW += dYᵀ·X, db += the column sums of dY; the two sums are added straight into the gradient (the
# accumulating products of bend-ml-tensor-array 0.1.10.0), with the bits of computing them apart and adding them
def lin_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: Lin<i, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match p:
    case Lin{w, b}:
      lin_b2(n, i, o, par, b, x, g, TA.Mat.matmul_precise(n, o, i, par, dy, w))

def pair_s3(-P1: Type, -P2: Type, r1: Step<P1>, r2: Step<P2>) -> Step<Two<P1, P2>>:
  match r1 r2:
    case Step{p1, g1, s1} Step{p2, g2, s2}:
      Step{Two{p1, p2}, Two{g1, g2}, OptState.join(P1, P2, s1, s2)}

def lin_pair(-i: Nat, -o: Nat, p: Lin<i, o>) -> Two<TA.Mat<o, i>, TA.Mat<1n, o>>:
  match p:
    case Lin{w, b}: Two{w, b}

def lin_unpair(-i: Nat, -o: Nat, p: Two<TA.Mat<o, i>, TA.Mat<1n, o>>) -> Lin<i, o>:
  match p:
    case Two{w, b}: Lin{w, b}

def lin_s2(-i: Nat, -o: Nat, a: MatStep<o, i>, c: MatStep<1n, o>) -> Step<Lin<i, o>>:
  match a c:
    case MatStep{w, gw, sw} MatStep{b, gb, sb}:
      Step{Lin{w, b}, Lin{gw, gb}, OptState.map(Two<TA.Mat<o, i>, TA.Mat<1n, o>>, Lin<i, o>, OptState.join(TA.Mat<o, i>, TA.Mat<1n, o>, sw, sb), x => lin_unpair(i, o, x), x => lin_unpair(i, o, x))}

def lin_step2(+i: Nat, +o: Nat, +op: Opt, p: Lin<i, o>, g: Lin<i, o>, st: Two<OptState<TA.Mat<o, i>>, OptState<TA.Mat<1n, o>>>) -> Step<Lin<i, o>>:
  match p g st:
    case Lin{w, b} Lin{gw, gb} Two{sw, sb}:
      lin_s2(i, o, Opt.mat(o, i, op, w, gw, sw), Opt.mat(1n, o, op, b, gb, sb))

def lin_step(+i: Nat, +o: Nat, +op: Opt, p: Lin<i, o>, g: Lin<i, o>, st: OptState<Lin<i, o>>) -> Step<Lin<i, o>>:
  lin_step2(i, o, op, p, g, OptState.split(TA.Mat<o, i>, TA.Mat<1n, o>, OptState.map(Lin<i, o>, Two<TA.Mat<o, i>, TA.Mat<1n, o>>, st, x => lin_pair(i, o, x), x => lin_pair(i, o, x))))

def lin_dump2(-i: Nat, -o: Nat, r: PL<Two<TA.Mat<o, i>, TA.Mat<1n, o>>>) -> PL<Lin<i, o>>:
  match r:
    case PL{Two{w, b}, xs}:
      PL{Lin{w, b}, xs}

def lin_dump(+i: Nat, +o: Nat, p: Lin<i, o>) -> PL<Lin<i, o>>:
  match p:
    case Lin{w, b}:
      lin_dump2(i, o, pl_pair(TA.Mat<o, i>, TA.Mat<1n, o>, Mat.dump(o, i, w), Mat.dump(1n, o, b)))

def lin_load3(-i: Nat, -o: Nat, r: PA<Two<TA.Mat<o, i>, TA.Mat<1n, o>>>) -> PA<Lin<i, o>>:
  match r:
    case PA{Two{w, b}, xs, off}: PA{Lin{w, b}, xs, off}

def lin_load2(+i: Nat, +o: Nat, b: TA.Mat<1n, o>, r: PA<TA.Mat<o, i>>) -> PA<Lin<i, o>>:
  match r:
    case PA{w, xs, off}:
      lin_load3(i, o, pa_second(TA.Mat<o, i>, TA.Mat<1n, o>, w, Mat.restore(1n, o, b, xs, off)))

def lin_load(+i: Nat, +o: Nat, p: Lin<i, o>, xs: Array<F32>, off: U32) -> PA<Lin<i, o>>:
  match p:
    case Lin{w, b}: lin_load2(i, o, b, Mat.restore(o, i, w, xs, off))

def lin_store2(+i: Nat, +o: Nat, b: TA.Mat<1n, o>, r: PA<TA.Mat<o, i>>) -> PA<Lin<i, o>>:
  match r:
    case PA{w, xs, off}:
      lin_load3(i, o, pa_second(TA.Mat<o, i>, TA.Mat<1n, o>, w, Mat.store(1n, o, b, xs, off)))

def lin_store(+i: Nat, +o: Nat, p: Lin<i, o>, xs: Array<F32>, off: U32) -> PA<Lin<i, o>>:
  match p:
    case Lin{w, b}: lin_store2(i, o, b, Mat.store(o, i, w, xs, off))

def lin_zeros(+i: Nat, +o: Nat) -> Lin<i, o>:
  Lin{TA.Mat.zeros(o, i), TA.Mat.zeros(1n, o)}

# a linear layer from i to o features on batches of n rows; its products run in 2^par blocks of rows
def Linear(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> Layer<Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => lin_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => lin_zeros(i, o), p => lin_dump(i, o, p), p => xs => off => lin_load(i, o, p, xs, off), p => xs => off => lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# the first layer of a model, whose input (the data) needs no gradient, as an input without requires_grad in PyTorch:
# the backward pass computes dW and db only, skipping dX = dY·W (a third of a linear layer's work), and returns zeros
# as dX. Use it only where nothing before the layer has parameters.
def linf_b3(+n: Nat, +i: Nat, +o: Nat, w: TA.Mat<o, i>, b: TA.Mat<1n, o>, gb: TA.Mat<1n, o>, r: TA.MMulTN<o, n, i>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulTN{dy, x, gw}:
      lin_b4(n, i, o, w, b, TA.Mat.zeros(n, i), gw, TA.Mat.col_sums_acc(n, o, dy, gb))

def linf_bwd(+n: Nat, +i: Nat, +o: Nat, +par: Nat, p: Lin<i, o>, x: TA.Mat<n, i>, dy: TA.Mat<n, o>, g: Lin<i, o>) -> Bwd<Lin<i, o>, TA.Mat<n, i>>:
  match p g:
    case Lin{w, b} Lin{gw, gb}:
      linf_b3(n, i, o, w, b, gb, TA.Mat.matmul_tn_acc(o, n, i, par, dy, x, gw))

def LinearFirst(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> Layer<Lin<i, o>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  Layer{p => x => lin_fwd(n, i, o, par, p, x), p => s => dy => g => linf_bwd(n, i, o, par, p, s, dy, g), op => p => g => st => lin_step(i, o, op, p, g, st), u => lin_zeros(i, o), p => lin_dump(i, o, p), p => xs => off => lin_load(i, o, p, xs, off), p => xs => off => lin_store(i, o, p, xs, off), u => U32.from_nat(Nat.mul(o, 1n+i))}

# =====================================================================================================
# Layers without parameters
# =====================================================================================================

def pick(b: Bool, +x: F32, +y: F32) -> F32:
  match b:
    case True{}:
      x
    case False{}:
      y

def no_step(-P: Type, +op: Opt, p: P, g: P, st: OptState<P>) -> Step<P>:
  Step{p, g, st}

# an element-wise activation that saves a copy of its output (relu, sigmoid, tanh: their derivative is a function of
# the output). `f` is the activation, `d(dy, y)` the gradient of its input.
def act_f2(-n: Nat, -c: Nat, r: TA.MM2<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match r:
    case TA.MM2{y, s}:
      Fwd{Unit{}, y, s}

def act_b2(-n: Nat, -c: Nat, r: TA.MPair<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match r:
    case TA.MPair{dx, y}:
      Bwd{Unit{}, dx, Unit{}}

def ActOut(~f: @+a: F32 -> F32, ~d: @+a: F32 -> @+b: F32 -> F32, +n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => act_f2(n, c, TA.Mat.clone(n, c, Ew.map(~f, n, c, x))), p => s => dy => g => act_b2(n, c, Ew.zip(~d, n, c, dy, s)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# relu(x) = x unless x < 0 (so relu(NaN) = NaN and relu(-0) = -0); its gradient is +0 where the output is <= 0 and
# passes everywhere else, NaN included (PyTorch's threshold_backward)
def relu_v(+x: F32) -> F32:
  pick(F32.is_lt(x, 0.0), 0.0, x)

def relu_d(+dy: F32, +y: F32) -> F32:
  pick(F32.is_le(y, 0.0), 0.0, dy)

def ReLU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~relu_v, ~relu_d, n, c)

# =====================================================================================================
# Seq: la then lb. Parameters, gradients, saved values and optimizer state are pairs of the two layers'.
# =====================================================================================================

def seq_f3(-P1: Type, -P2: Type, -Z: Type, -S1: Type, -S2: Type, p1: P1, s1: S1, r: Fwd<P2, Z, S2>) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match r:
    case Fwd{p2, z, s2}:
      Fwd{Two{p1, p2}, z, Two{s1, s2}}

def seq_f2(~P2: Type, ~Y: Type, ~Z: Type, ~S2: Type, ~lb: Layer<P2, Y, Z, S2>, -P1: Type, -S1: Type, p2: P2, r: Fwd<P1, Y, S1>) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match r:
    case Fwd{p1, y, s1}:
      seq_f3(P1, P2, Z, S1, S2, p1, s1, Layer.fwd(P2, Y, Z, S2, lb, p2, y))

def seq_fwd(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, p: Two<P1, P2>, x: X) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match p:
    case Two{p1, p2}:
      seq_f2(~P2, ~Y, ~Z, ~S2, ~lb, P1, S1, p2, Layer.fwd(P1, X, Y, S1, la, p1, x))

def seq_b3(-P1: Type, -P2: Type, -X: Type, p2: P2, g2: P2, r: Bwd<P1, X>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p1, dx, g1}:
      Bwd{Two{p1, p2}, dx, Two{g1, g2}}

def seq_b2(~P1: Type, ~X: Type, ~Y: Type, ~S1: Type, ~la: Layer<P1, X, Y, S1>, -P2: Type, p1: P1, s1: S1, g1: P1, r: Bwd<P2, Y>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p2, dy, g2}:
      seq_b3(P1, P2, X, p2, g2, Layer.bwd(P1, X, Y, S1, la, p1, s1, dy, g1))

def seq_bwd(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, p: Two<P1, P2>, s: Two<S1, S2>, dz: Z, g: Two<P1, P2>) -> Bwd<Two<P1, P2>, X>:
  match p s g:
    case Two{p1, p2} Two{s1, s2} Two{g1, g2}:
      seq_b2(~P1, ~X, ~Y, ~S1, ~la, P2, p1, s1, g1, Layer.bwd(P2, Y, Z, S2, lb, p2, s2, dz, g2))

def seq_s3(-P1: Type, -P2: Type, r1: Step<P1>, r2: Step<P2>) -> Step<Two<P1, P2>>:
  pair_s3(P1, P2, r1, r2)

def seq_s2(~P2: Type, ~Y: Type, ~Z: Type, ~S2: Type, ~lb: Layer<P2, Y, Z, S2>, -P1: Type, +op: Opt, p2: P2, g2: P2, s2: OptState<P2>, r1: Step<P1>) -> Step<Two<P1, P2>>:
  seq_s3(P1, P2, r1, Layer.step(P2, Y, Z, S2, lb, op, p2, g2, s2))

def seq_step2(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: Two<OptState<P1>, OptState<P2>>) -> Step<Two<P1, P2>>:
  match p g st:
    case Two{p1, p2} Two{g1, g2} Two{s1, s2}:
      seq_s2(~P2, ~Y, ~Z, ~S2, ~lb, P1, op, p2, g2, s2, Layer.step(P1, X, Y, S1, la, op, p1, g1, s1))

def seq_step(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: OptState<Two<P1, P2>>) -> Step<Two<P1, P2>>:
  seq_step2(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, op, p, g, OptState.split(P1, P2, st))

# dump and load for a pair of layers: the first layer's numbers, then the second's
# a pair's zeros, the two halves in a parallel let: a model's gradient and optimizer state are many arrays, and making
# them one after another left the training state's zeroing (page faults, mostly) on one thread
def pair_zeros(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>) -> Two<P1, P2>:
  a b = Layer.zeros(P1, X1, Y1, S1, la) Layer.zeros(P2, X2, Y2, S2, lb)
  Two{a, b}

def pair_dump(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, p: Two<P1, P2>) -> PL<Two<P1, P2>>:
  match p:
    case Two{p1, p2}:
      pl_pair(P1, P2, Layer.dump(P1, X1, Y1, S1, la, p1), Layer.dump(P2, X2, Y2, S2, lb, p2))

def pair_load2(~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~lb: Layer<P2, X2, Y2, S2>, -P1: Type, p2: P2, r: PA<P1>) -> PA<Two<P1, P2>>:
  match r:
    case PA{p1, xs, off}:
      pa_second(P1, P2, p1, Layer.load(P2, X2, Y2, S2, lb, p2, xs, off))

def pair_load(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, p: Two<P1, P2>, xs: Array<F32>, off: U32) -> PA<Two<P1, P2>>:
  match p:
    case Two{p1, p2}:
      pair_load2(~P2, ~X2, ~Y2, ~S2, ~lb, P1, p2, Layer.load(P1, X1, Y1, S1, la, p1, xs, off))

def pair_store2(~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~lb: Layer<P2, X2, Y2, S2>, -P1: Type, p2: P2, r: PA<P1>) -> PA<Two<P1, P2>>:
  match r:
    case PA{p1, xs, off}:
      pa_second(P1, P2, p1, Layer.store(P2, X2, Y2, S2, lb, p2, xs, off))

def pair_store(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, p: Two<P1, P2>, xs: Array<F32>, off: U32) -> PA<Two<P1, P2>>:
  match p:
    case Two{p1, p2}:
      pair_store2(~P2, ~X2, ~Y2, ~S2, ~lb, P1, p2, Layer.store(P1, X1, Y1, S1, la, p1, xs, off))

def Seq(~P1: Type, ~P2: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y, S1>, ~lb: Layer<P2, Y, Z, S2>) -> Layer<Two<P1, P2>, X, Z, Two<S1, S2>>:
  # bend-lint: disable-next bend-lint-rules/performance/gpu_fit -- zeros are made once per run (Train.new, OptState.new), and pair_zeros already forks not-expected-because: the combinators nest, so a bang here would run inside a bang
  Layer{p => x => seq_fwd(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, p, x), p => s => dz => g => seq_bwd(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, p, s, dz, g), op => p => g => st => seq_step(~P1, ~P2, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, op, p, g, st), u => pair_zeros(~P1, ~X, ~Y, ~S1, ~P2, ~Y, ~Z, ~S2, ~la, ~lb), p => pair_dump(~P1, ~X, ~Y, ~S1, ~P2, ~Y, ~Z, ~S2, ~la, ~lb, p), p => xs => off => pair_load(~P1, ~X, ~Y, ~S1, ~P2, ~Y, ~Z, ~S2, ~la, ~lb, p, xs, off), p => xs => off => pair_store(~P1, ~X, ~Y, ~S1, ~P2, ~Y, ~Z, ~S2, ~la, ~lb, p, xs, off), u => (Layer.count(P1, X, Y, S1, la) + Layer.count(P2, Y, Z, S2, lb) : U32)}

# Owned layer composition for dimensions read at run time. Each child record is opened once; each closure goes to
# exactly one corresponding closure of the composite layer. The template-based Seq above keeps its existing laws.
def seq_at_f2(-P1: Type, -P2: Type, -Y: Type, -Z: Type, -S1: Type, -S2: Type, p2: P2, fb: P2 -> Y -> Fwd<P2, Z, S2>, r: Fwd<P1, Y, S1>) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match r:
    case Fwd{p1, y, s1}: seq_f3(P1, P2, Z, S1, S2, p1, s1, fb(p2, y))

def seq_at_f(-P1: Type, -P2: Type, -X: Type, -Y: Type, -Z: Type, -S1: Type, -S2: Type, fa: P1 -> X -> Fwd<P1, Y, S1>, fb: P2 -> Y -> Fwd<P2, Z, S2>, p: Two<P1, P2>, x: X) -> Fwd<Two<P1, P2>, Z, Two<S1, S2>>:
  match p:
    case Two{p1, p2}: seq_at_f2(P1, P2, Y, Z, S1, S2, p2, fb, fa(p1, x))

def seq_at_b2(-P1: Type, -P2: Type, -X: Type, -Y: Type, -S1: Type, ba: P1 -> S1 -> Y -> P1 -> Bwd<P1, X>, p1: P1, s1: S1, g1: P1, r: Bwd<P2, Y>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p2, dy, g2}: seq_b3(P1, P2, X, p2, g2, ba(p1, s1, dy, g1))

def seq_at_b(-P1: Type, -P2: Type, -X: Type, -Y: Type, -Z: Type, -S1: Type, -S2: Type, ba: P1 -> S1 -> Y -> P1 -> Bwd<P1, X>, bb: P2 -> S2 -> Z -> P2 -> Bwd<P2, Y>, p: Two<P1, P2>, s: Two<S1, S2>, dz: Z, g: Two<P1, P2>) -> Bwd<Two<P1, P2>, X>:
  match p s g:
    case Two{p1, p2} Two{s1, s2} Two{g1, g2}: seq_at_b2(P1, P2, X, Y, S1, ba, p1, s1, g1, bb(p2, s2, dz, g2))

def seq_at_step(-P1: Type, -P2: Type, sa: Opt -> P1 -> P1 -> OptState<P1> -> Step<P1>, sb: Opt -> P2 -> P2 -> OptState<P2> -> Step<P2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: Two<OptState<P1>, OptState<P2>>) -> Step<Two<P1, P2>>:
  match p g st:
    case Two{p1, p2} Two{g1, g2} Two{s1, s2}: pair_s3(P1, P2, sa(op, p1, g1, s1), sb(op, p2, g2, s2))

def seq_at_dump(-P1: Type, -P2: Type, da: P1 -> PL<P1>, db: P2 -> PL<P2>, p: Two<P1, P2>) -> PL<Two<P1, P2>>:
  match p:
    case Two{p1, p2}: pl_pair(P1, P2, da(p1), db(p2))

def seq_at_load2(-P1: Type, -P2: Type, lb: P2 -> Array<F32> -> U32 -> PA<P2>, p2: P2, r: PA<P1>) -> PA<Two<P1, P2>>:
  match r:
    case PA{p1, xs, off}: pa_second(P1, P2, p1, lb(p2, xs, off))

def seq_at_load(-P1: Type, -P2: Type, la: P1 -> Array<F32> -> U32 -> PA<P1>, lb: P2 -> Array<F32> -> U32 -> PA<P2>, p: Two<P1, P2>, xs: Array<F32>, off: U32) -> PA<Two<P1, P2>>:
  match p:
    case Two{p1, p2}: seq_at_load2(P1, P2, lb, p2, la(p1, xs, off))

def seq_at_store(-P1: Type, -P2: Type, la: P1 -> Array<F32> -> U32 -> PA<P1>, lb: P2 -> Array<F32> -> U32 -> PA<P2>, p: Two<P1, P2>, xs: Array<F32>, off: U32) -> PA<Two<P1, P2>>:
  match p:
    case Two{p1, p2}: seq_at_load2(P1, P2, lb, p2, la(p1, xs, off))

def Seq.at(-P1: Type, -P2: Type, -X: Type, -Y: Type, -Z: Type, -S1: Type, -S2: Type, a: Layer<P1, X, Y, S1>, b: Layer<P2, Y, Z, S2>) -> Layer<Two<P1, P2>, X, Z, Two<S1, S2>>:
  match a b:
    case Layer{fa, ba, sa, za, da, la, ta, na} Layer{fb, bb, sb, zb, db, lb, tb, nb}:
      Layer{p => x => seq_at_f(P1, P2, X, Y, Z, S1, S2, fa, fb, p, x), p => s => dz => g => seq_at_b(P1, P2, X, Y, Z, S1, S2, ba, bb, p, s, dz, g), op => p => g => st => seq_at_step(P1, P2, sa, sb, op, p, g, OptState.split(P1, P2, st)), u => Two{za(Unit{}), zb(Unit{})}, p => seq_at_dump(P1, P2, da, db, p), p => xs => off => seq_at_load(P1, P2, la, lb, p, xs, off), p => xs => off => seq_at_store(P1, P2, ta, tb, p, xs, off), u => (na(Unit{}) + nb(Unit{}) : U32)}

# =====================================================================================================
# Losses: the mean loss over the batch, and its gradient with respect to the output
# =====================================================================================================

type LossOut<-Y: Type> is Type:
  LossOut{loss: F32, dy: Y}

# =====================================================================================================
# Training: one step of any layer under any loss
# =====================================================================================================

# Parameters, gradient (zero between steps), optimizer-specific buffers, Adam's 1 - b1^t and 1 - b2^t, last loss.
type Train<-P: Type> is Type:
  Train{p: P, g: P, st: OptState<P>, p1: F32, p2: F32, loss: F32}

def OptState.new(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, cfg: OptCfg) -> OptState<P>:
  match cfg:
    case CSGD{lr}: NoMom{}
    case CMomentum{lr, mu}: OneMom{Layer.zeros(P, X, Y, S, m)}
    case CAdamWBF16{lr, b1, b2, eps, wd, c1, c2, v16}: Packed{PNew{v16}}
    case _: TwoMom{Layer.zeros(P, X, Y, S, m), Layer.zeros(P, X, Y, S, m)}

# 1 - b^t starts at 0; a configuration OptCfg.ok rejects starts at NaN, so every loss after the first is NaN.
def q_start(ok: Bool) -> F32:
  match ok:
    case True{}:
      0.0
    case False{}:
      (0.0 / 0.0 : F32)

def Train.new(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +cfg: OptCfg, p: P) -> Train<P>:
  +q = q_start(OptCfg.ok(cfg))
  Train{p, Layer.zeros(P, X, Y, S, m), OptState.new(~P, ~X, ~Y, ~S, ~m, cfg), q, q, 0.0}

def ts5(-P: Type, +l: F32, +p1: F32, +p2: F32, r: Step<P>) -> Train<P>:
  match r:
    case Step{p, g, st}:
      Train{p, g, st, p1, p2, l}

def ts4(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +cfg: OptCfg, st: OptState<P>, +p1: F32, +p2: F32, +l: F32, r: Bwd<P, X>) -> Train<P>:
  match r:
    case Bwd{p, dx, g}:
      +q1 = (p1 + (OptCfg.c1(cfg) * (1.0 - p1 : F32) : F32) : F32)
      +q2 = (p2 + (OptCfg.c2(cfg) * (1.0 - p2 : F32) : F32) : F32)
      ts5(P, l, q1, q2, Layer.step(P, X, Y, S, m, Opt.at(cfg, q1, q2), p, g, st))

def ts3(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +cfg: OptCfg, p: P, s: S, g: P, st: OptState<P>, +p1: F32, +p2: F32, r: LossOut<Y>) -> Train<P>:
  match r:
    case LossOut{l, dy}:
      ts4(~P, ~X, ~Y, ~S, ~m, cfg, st, p1, p2, l, Layer.bwd(P, X, Y, S, m, p, s, dy, g))

def ts2(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~T: Type, ~m: Layer<P, X, Y, S>, ~loss: Y -> T -> LossOut<Y>, +cfg: OptCfg, g: P, st: OptState<P>, +p1: F32, +p2: F32, t: T, r: Fwd<P, Y, S>) -> Train<P>:
  match r:
    case Fwd{p, y, s}:
      ts3(~P, ~X, ~Y, ~S, ~m, cfg, p, s, g, st, p1, p2, loss(y, t))

# forward, loss, backward and one optimizer step: the same code for every model
def Train.step(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~T: Type, ~m: Layer<P, X, Y, S>, ~loss: Y -> T -> LossOut<Y>, +cfg: OptCfg, tr: Train<P>, x: X, t: T) -> Train<P>:
  match tr:
    case Train{p, g, st, p1, p2, l}:
      ts2(~P, ~X, ~Y, ~S, ~T, ~m, ~loss, cfg, g, st, p1, p2, t, Layer.fwd(P, X, Y, S, m, p, x))

# One owned model, including dimensions captured at run time. Its fwd, bwd and step closures each run once.
def train_at_b(-P: Type, -X: Type, sf: Opt -> P -> P -> OptState<P> -> Step<P>, +cfg: OptCfg, st: OptState<P>, +p1: F32, +p2: F32, +l: F32, r: Bwd<P, X>) -> Train<P>:
  match r:
    case Bwd{p, dx, g}:
      +q1 = (p1 + (OptCfg.c1(cfg) * (1.0 - p1 : F32) : F32) : F32)
      +q2 = (p2 + (OptCfg.c2(cfg) * (1.0 - p2 : F32) : F32) : F32)
      ts5(P, l, q1, q2, sf(Opt.at(cfg, q1, q2), p, g, st))

def train_at_loss(-P: Type, -X: Type, -Y: Type, -S: Type, bf: P -> S -> Y -> P -> Bwd<P, X>, sf: Opt -> P -> P -> OptState<P> -> Step<P>, +cfg: OptCfg, p: P, s: S, g: P, st: OptState<P>, +p1: F32, +p2: F32, r: LossOut<Y>) -> Train<P>:
  match r:
    case LossOut{l, dy}: train_at_b(P, X, sf, cfg, st, p1, p2, l, bf(p, s, dy, g))

def train_at_f(-P: Type, -X: Type, -Y: Type, -S: Type, -Target: Type, bf: P -> S -> Y -> P -> Bwd<P, X>, sf: Opt -> P -> P -> OptState<P> -> Step<P>, loss: Y -> Target -> LossOut<Y>, +cfg: OptCfg, g: P, st: OptState<P>, +p1: F32, +p2: F32, t: Target, r: Fwd<P, Y, S>) -> Train<P>:
  match r:
    case Fwd{p, y, s}: train_at_loss(P, X, Y, S, bf, sf, cfg, p, s, g, st, p1, p2, loss(y, t))

def Train.step_at(-P: Type, -X: Type, -Y: Type, -S: Type, -Target: Type, m: Layer<P, X, Y, S>, loss: Y -> Target -> LossOut<Y>, +cfg: OptCfg, tr: Train<P>, x: X, t: Target) -> Train<P>:
  match m tr:
    case Layer{ff, bf, sf, zeros, dump, load, store, count} Train{p, g, st, p1, p2, l}:
      train_at_f(P, X, Y, S, Target, bf, sf, loss, cfg, g, st, p1, p2, t, ff(p, x))

# =====================================================================================================
# Reading and printing numbers (bend-ml's and llama.bend's loaders, MIT)
# =====================================================================================================

# U32 and F32 hold the same Word(32n): these are bitcasts
def bitcast(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def bits(x: F32) -> U32:
  match x:
    case F32{w}:
      U32{w}

def f32_le(+b0: U32, +b1: U32, +b2: U32, +b3: U32) -> F32:
  bitcast((b0 + (b1 * 256 : U32) + (b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32))

def f32s(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
      f32_le(b0, b1, b2, b3) <> f32s(t)
    case _:
      Nil{}

def rd_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  match m:
    case Tuple{f, r}:
      do IO<List<&2, U32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        File.close(f)
        return bs

# n bytes from offset off
def read_at(path: String, +off: U32, +n: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, off, n)
    rd_done(m)

def mat_or_die(-r: Nat, -c: Nat, +path: String, m: Maybe<&1, TA.Mat<r, c>>) -> IO(TA.Mat<r, c>):
  match m:
    case None{}:
      IO.die(TA.Mat<r, c>, 1, path ++ ": wrong number of values for this shape")
    case Some{x}:
      IO.pure(TA.Mat<r, c>, x)

# an r x c matrix stored as r*c little-endian F32 from byte off of a file
def Mat.load(+r: Nat, +c: Nat, +path: String, +off: U32) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    bs : List<&2, U32> <- read_at(path, off, (U32.from_nat(Nat.mul(r, c)) * 4 : U32))
    mat_or_die(r, c, path, TA.Mat.of_list(r, c, f32s(bs)))

def words(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ U32.show(bits(h)) ++ words(t)

def ml_words(-r: Nat, -c: Nat, m: TA.ML<r, c>) -> String:
  match m:
    case TA.ML{d, xs}:
      words(xs)

# one line: the name, then each number as the decimal of its 32 bits (exact, for the tests in reference/)
def Mat.print(+r: Nat, +c: Nat, +name: String, m: TA.Mat<r, c>) -> IO(Unit):
  IO.print(name ++ ml_words(r, c, TA.Mat.to_list(r, c, m)))

def F32.print(+name: String, x: F32) -> IO(Unit):
  IO.print(name ++ " " ++ U32.show(bits(x)))

# n zeros, in an array with room for them
def zeros_a(+n: Nat) -> Array<F32>:
  Array.new(F32, depth(n), 0.0)

# =====================================================================================================
# More element-wise activations
# =====================================================================================================

# an activation whose derivative needs its input (silu, gelu): saves a copy of the input
def acti_f3(~f: @+a: F32 -> F32, +n: Nat, +c: Nat, m: TA.MM2<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match m:
    case TA.MM2{x, s}:
      Fwd{Unit{}, Ew.map(~f, n, c, x), s}

def ActIn(~f: @+a: F32 -> F32, ~d: @+a: F32 -> @+b: F32 -> F32, +n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => acti_f3(~f, n, c, TA.Mat.clone(n, c, x)), p => s => dy => g => act_b2(n, c, Ew.zip(~d, n, c, dy, s)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

def sigmoid_v(+x: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

# PyTorch's sigmoid_backward: dy * (1 - y) * y
def sigmoid_d(+dy: F32, +y: F32) -> F32:
  ((dy * (1.0 - y : F32) : F32) * y : F32)

def Sigmoid(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~sigmoid_v, ~sigmoid_d, n, c)

def tanh_v(+x: F32) -> F32:
  F32.tanh(x)

def tanh_d(+dy: F32, +y: F32) -> F32:
  (dy * (1.0 - (y * y : F32) : F32) : F32)

def Tanh(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActOut(~tanh_v, ~tanh_d, n, c)

# silu(x) = x * sigmoid(x), computed as PyTorch's CPU kernel: x / (1 + exp(-x));
# its gradient dy * s * (1 + x * (1 - s)) with s = sigmoid(x)
def silu_v(+x: F32) -> F32:
  (x / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

def silu_d(+dy: F32, +x: F32) -> F32:
  +s = sigmoid_v(x)
  ((dy * s : F32) * (1.0 + (x * (1.0 - s : F32) : F32) : F32) : F32)

def SiLU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~silu_v, ~silu_d, n, c)

# GELU in its tanh form (PyTorch's approximate="tanh"): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
# The exact form, with erf, is GELUErf below.
def gelu_inner(+x: F32) -> F32:
  (0.7978845608028654 * (x + (0.044715 * ((x * x : F32) * x : F32) : F32) : F32) : F32)

def gelu_v(+x: F32) -> F32:
  (((0.5 * x : F32) * (1.0 + F32.tanh(gelu_inner(x)) : F32)) : F32)

def gelu_d(+dy: F32, +x: F32) -> F32:
  +th = F32.tanh(gelu_inner(x))
  +left = (0.5 * x : F32)
  +dleft = (0.5 * (1.0 + th : F32) : F32)
  +dright = ((left * (1.0 - (th * th : F32) : F32) : F32) * (0.7978845608028654 * (1.0 + ((3.0 * 0.044715 : F32) * (x * x : F32) : F32) : F32) : F32) : F32)
  (dy * (dleft + dright : F32) : F32)

def GELU(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~gelu_v, ~gelu_d, n, c)

# =====================================================================================================
# erf in plain Bend, and GELU in its exact form
# =====================================================================================================

# erf(x) = 2/sqrt(pi) times the integral of exp(-t^2) from 0 to x. Base has no erf, so it is written here with F32
# additions, multiplications and comparisons only: no libm call, so every target runs the same operations (the emitted C
# turns fused multiply-adds off); the C and JavaScript programs give the same bits (CUDA is not checked yet). erf is
# odd, erf(-x) = -erf(x), so the work is done on a = |x| and the sign put back, in three kinds of pieces:
#   - a < 0.84375: erf(x) = x + x P(x^2), with P of degree 6 (fdlibm's erff has this x + x y shape there); it takes
#     x itself, so erf(-0) = -0, and NaN reaches it (every comparison with NaN is false) and comes out NaN;
#   - 0.84375 <= a < 3.92, in six intervals around c = 1, 1.5, ..., 3.5: erf(a) = K + S(a - c), where K is erf(c)
#     rounded to F32 and S a polynomial of degree 7 or 8 whose value is small next to K, so the last addition rounds
#     once and carries almost all of the error (a - c is exact: a and c are within a factor of 2);
#   - a >= 3.92, infinity included: 1, since erf(a) rounds to 1 in F32 from a = 3.9192 on.
# The coefficients are least-squares fits against a 50-digit erf on Chebyshev points, rounded to F32. Measured on all
# 2^32 inputs against glibc's double erf: at most 0.93 ulp off, the correctly rounded F32 on 97.1% of the inputs, and 1
# ulp from it on the rest; NaN for every NaN.

# one Horner step, c + t s; erf_hm is the step for a negative coefficient -c: t s - c gives the bits of -c + t s
def erf_hp(+c: F32, +t: F32, +s: F32) -> F32:
  (c + (t * s : F32) : F32)

def erf_hm(+c: F32, +t: F32, +s: F32) -> F32:
  ((t * s : F32) - c : F32)

def erf_small(+x: F32) -> F32:
  +z = (x * x : F32)
  (x + (x * erf_hp(0.12837916612625122, z, erf_hm(0.37612634897232056, z, erf_hp(0.11283732205629349, z, erf_hm(0.026861993595957756, z, erf_hp(0.005208558402955532, z, erf_hm(0.000823960464913398, z, 8.829222497297451e-05)))))) : F32) : F32)

# [0.84375, 1.25): t = a - 1.0
def erf_p0(+t: F32) -> F32:
  (0.8427007794380188 + erf_hp(1.3508899776581984e-08, t, erf_hp(0.41510748863220215, t, erf_hm(0.41510751843452454, t, erf_hp(0.13837093114852905, t, erf_hp(0.06918490678071976, t, erf_hm(0.06929104775190353, t, erf_hp(0.004778923466801643, t, erf_hp(0.017144912853837013, t, F32.neg(0.010648881085216999))))))))) : F32)

# [1.25, 1.75): t = a - 1.5
def erf_p1(+t: F32) -> F32:
  (0.9661051630973816 + erf_hm(1.6622220044837377e-08, t, erf_hp(0.11893028765916824, t, erf_hm(0.17839543521404266, t, erf_hp(0.1387522667646408, t, erf_hm(0.04459875077009201, t, erf_hm(0.014879402704536915, t, erf_hp(0.019323812797665596, t, erf_hm(0.004488371312618256, t, F32.neg(0.0023522074334323406))))))))) : F32)

# [1.75, 2.25): t = a - 2.0
def erf_p2(+t: F32) -> F32:
  (0.9953222870826721 + erf_hm(2.2064206461891445e-08, t, erf_hp(0.02066698484122753, t, erf_hm(0.04133396968245506, t, erf_hp(0.04822301119565964, t, erf_hm(0.03444507718086243, t, erf_hp(0.013088003732264042, t, erf_hp(0.00046277992078103125, t, erf_hm(0.0033736424520611763, t, 0.0015402256976813078)))))))) : F32)

# [2.25, 2.75): t = a - 2.5
def erf_p3(+t: F32) -> F32:
  (0.9995930194854736 + erf_hp(2.846470792405853e-08, t, erf_hp(0.002178284339606762, t, erf_hm(0.005445694085210562, t, erf_hp(0.008350063115358353, t, erf_hm(0.008623690344393253, t, erf_hp(0.00611893879249692, t, erf_hm(0.002765072975307703, t, 0.0005077228415757418))))))) : F32)

# [2.75, 3.25): t = a - 3.0
def erf_p4(+t: F32) -> F32:
  (0.9999778866767883 + erf_hp(2.2842012725732275e-08, t, erf_hp(0.00013925303937867284, t, erf_hm(0.0004177672672085464, t, erf_hp(0.0007891026325523853, t, erf_hm(0.0010437482269480824, t, erf_hp(0.001016448251903057, t, erf_hm(0.0007547215791419148, t, 0.0003925197524949908))))))) : F32)

# [3.25, 3.92): t = a - 3.5
def erf_p5(+t: F32) -> F32:
  (0.9999992847442627 + erf_hm(2.7855115547481546e-08, t, erf_hp(5.400151167123113e-06, t, erf_hm(1.8894650565925986e-05, t, erf_hp(4.221926792524755e-05, t, erf_hm(6.778862734790891e-05, t, erf_hp(8.422814426012337e-05, t, erf_hm(7.95367595856078e-05, t, 4.151461325818673e-05))))))) : F32)

def erf_sign(+x: F32, +e: F32) -> F32:
  pick(F32.is_lt(x, 0.0), F32.neg(e), e)

# the pieces are chosen from the top down, so that only one polynomial is computed (each test is a match on a Bool
# parameter); b is "a is at least the piece's lower end"
def erf_i0(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p0((a - 1.0 : F32)))
    case False{}: erf_small(x)

def erf_i1(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p1((a - 1.5 : F32)))
    case False{}: erf_i0(F32.is_ge(a, 0.84375), x, a)

def erf_i2(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p2((a - 2.0 : F32)))
    case False{}: erf_i1(F32.is_ge(a, 1.25), x, a)

def erf_i3(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p3((a - 2.5 : F32)))
    case False{}: erf_i2(F32.is_ge(a, 1.75), x, a)

def erf_i4(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p4((a - 3.0 : F32)))
    case False{}: erf_i3(F32.is_ge(a, 2.25), x, a)

def erf_i5(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, erf_p5((a - 3.5 : F32)))
    case False{}: erf_i4(F32.is_ge(a, 2.75), x, a)

def erf_top(b: Bool, +x: F32, +a: F32) -> F32:
  match b:
    case True{}: erf_sign(x, 1.0)
    case False{}: erf_i5(F32.is_ge(a, 3.25), x, a)

def erf(+x: F32) -> F32:
  +a = F32.abs(x)
  erf_top(F32.is_ge(a, 3.92), x, a)

# GELU in its exact form, PyTorch's default (approximate="none"): gelu(x) = x/2 (1 + erf(x / sqrt 2)), in PyTorch's CPU
# order, (x * 0.5) * (1 + erf(x * 0.70710677)). Its gradient is cdf + x pdf, with cdf = 0.5 (1 + erf(x / sqrt 2)) and
# pdf = exp(-x^2 / 2) / sqrt(2 pi), computed as PyTorch does: 0.5 (1 + erf(x * 0.70710677)) and
# 0.3989423 * exp((x * x) * -0.5). The exponential is F32.exp, as in SiLU (so on a GPU the gradient would take CUDA's
# float exp).
def gelu_erf_cdf(+x: F32) -> F32:
  (0.5 * (1.0 + erf((x * 0.7071067811865476 : F32)) : F32) : F32)

def gelu_erf_v(+x: F32) -> F32:
  ((x * 0.5 : F32) * (1.0 + erf((x * 0.7071067811865476 : F32)) : F32) : F32)

def gelu_erf_d(+dy: F32, +x: F32) -> F32:
  +pdf = (0.3989422804014327 * F32.exp(((x * x : F32) * F32.neg(0.5) : F32)) : F32)
  (dy * (gelu_erf_cdf(x) + (x * pdf : F32) : F32) : F32)

def GELUErf(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  ActIn(~gelu_erf_v, ~gelu_erf_d, n, c)

# =====================================================================================================
# Dropout: each number is kept with probability 1 - p and scaled by 1 / (1 - p), or set to 0. The mask comes from a
# counter-based hash of (seed, step, index), so a run is reproducible and a twin can compute the same mask. The step
# counter is the layer's "parameter": the optimizer step only increments it, and checkpoints keep it.
# =====================================================================================================

type DropP is Data:
  DropP{t: U32}

# lowbias32 (Chris Wellons): an invertible mix of the 32 bits
def mix32(+x: U32) -> U32:
  +a = U32.xor(x, U32.shrn(x, 16n))
  +b = (a * 2146121005 : U32)
  +c = U32.xor(b, U32.shrn(b, 15n))
  +d = (c * 2221713035 : U32)
  U32.xor(d, U32.shrn(d, 16n))

# the factor of number i: 1 / (1 - p) if its uniform u = (h >> 8) / 2^24 is >= p, else 0
def drop_k(+key: U32, +i: U32, +p: F32, +scale: F32) -> F32:
  pick(F32.is_lt((U32.to_f32(U32.shrn(mix32((key + i : U32)), 8n)) / 16777216.0 : F32), p), 0.0, scale)

def drop_l(nleft: Nat, +i: U32, +key: U32, +p: F32, +scale: F32, a: Array<F32>) -> Array<F32>:
  match nleft:
    case 0n:
      a
    case 1n+q:
      drop_l(q, (i + 1 : U32), key, p, scale, Array.set(F32, a, i, drop_k(key, i, p, scale)))

def drop_mask(+n: Nat, +c: Nat, +seed: U32, +t: U32, +p: F32) -> TA.Mat<n, c>:
  TA.Mat{drop_l(Nat.mul(n, c), 0, mix32(U32.xor(seed, mix32(t))), p, (1.0 / (1.0 - p : F32) : F32), zeros_a(Nat.mul(n, c)))}

def drop_f2(-n: Nat, -c: Nat, +t: U32, r: TA.MPair<n, c>) -> Fwd<DropP, TA.Mat<n, c>, TA.Mat<n, c>>:
  match r:
    case TA.MPair{y, k}:
      Fwd{DropP{t}, y, k}

def drop_fwd(+n: Nat, +c: Nat, +seed: U32, +p: F32, d: DropP, x: TA.Mat<n, c>) -> Fwd<DropP, TA.Mat<n, c>, TA.Mat<n, c>>:
  match d:
    case DropP{+t}:
      drop_f2(n, c, t, Ew.zip(~f32_mul, n, c, x, drop_mask(n, c, seed, t, p)))

def drop_b2(-n: Nat, -c: Nat, d: DropP, g: DropP, r: TA.MPair<n, c>) -> Bwd<DropP, TA.Mat<n, c>>:
  match r:
    case TA.MPair{dx, k}:
      Bwd{d, dx, g}

def drop_step(+op: Opt, d: DropP, g: DropP, st: OptState<DropP>) -> Step<DropP>:
  match d:
    case DropP{t}:
      Step{DropP{(t + 1 : U32)}, g, st}

def drop_dump(d: DropP) -> PL<DropP>:
  match d:
    case DropP{+t}:
      PL{DropP{t}, [bitcast(t)]}

def drop_load_fin(+off: U32, r: Array<F32> & F32) -> PA<DropP>:
  match r:
    case Tuple{a, v}: PA{DropP{bits(v)}, a, (off + 1 : U32)}

def drop_load(d: DropP, xs: Array<F32>, +off: U32) -> PA<DropP>:
  drop_load_fin(off, Array.get(F32, xs, off))

# the counter, bitcast to one F32 as dump writes it
def drop_store(d: DropP, xs: Array<F32>, +off: U32) -> PA<DropP>:
  match d:
    case DropP{+c}: PA{DropP{c}, Array.set(F32, xs, off, bitcast(c)), (off + 1 : U32)}

def Dropout(+n: Nat, +c: Nat, +p: F32, +seed: U32) -> Layer<DropP, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{d => x => drop_fwd(n, c, seed, p, d, x), d => s => dy => g => drop_b2(n, c, d, g, Ew.zip(~f32_mul, n, c, dy, s)), op => d => g => st => drop_step(op, d, g, st), u => DropP{0}, d => drop_dump(d), d => xs => off => drop_load(d, xs, off), d => xs => off => drop_store(d, xs, off), u => 1}

# =====================================================================================================
# Row kernels over arrays: loops over a row of a flat array by index. Every sum runs left to right from +0.0 and every
# expression keeps its order of operations, the ones the row-wise layers had when they worked on lists of rows, so
# they give the same bits (NOTES.md, M6)
# =====================================================================================================

type D2 is Type:
  D2{a: Array<F32>, b: Array<F32>, s: F32}

type A2 is Type:
  A2{a: Array<F32>, b: Array<F32>}

# a[ia]·b[ib] + a[ia + sa]·b[ib + sb] + ... (k terms) added to acc left to right; ra and rb hold the
# first pair, already read
def sdot(k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> D2:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      D2{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      sdot(p, (ia + sa : U32), (ib + sb : U32), sa, sb, (acc + (x * y : F32) : F32), Array.get(F32, a, (ia + sa : U32)), Array.get(F32, b, (ib + sb : U32)))

def dot0(+k: Nat, +ia: U32, +ib: U32, +sa: U32, +sb: U32, a: Array<F32>, b: Array<F32>) -> D2:
  sdot(k, ia, ib, sa, sb, 0.0, Array.get(F32, a, ia), Array.get(F32, b, ib))


type AS is Type:
  AS{a: Array<F32>, s: F32}

# the largest of a[idx] (held in r) and the k numbers after it: max(max(a0, a1), a2) ..., from the first
def pmax(k: Nat, +idx: U32, cur: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, F32.max(cur, x)}
    case 1n+q Tuple{a, +x}:
      pmax(q, (idx + 1 : U32), F32.max(cur, x), Array.get(F32, a, (idx + 1 : U32)))

def pmax0(k: Nat, +idx: U32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, x}
    case 1n+q Tuple{a, +x}:
      pmax(q, (idx + 1 : U32), x, Array.get(F32, a, (idx + 1 : U32)))

# a[j] = exp(a[j] - m) for j = idx..idx + k, their sum added to acc in order
def pexp(k: Nat, +idx: U32, +m: F32, acc: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      AS{Array.set(F32, a, idx, e), (acc + e : F32)}
    case 1n+q Tuple{a, +x}:
      +e = F32.exp((x - m : F32))
      pexp(q, (idx + 1 : U32), m, (acc + e : F32), Array.get(F32, Array.set(F32, a, idx, e), (idx + 1 : U32)))

# the array and how many numbers the last loop of a softmax divided
type PC is Type:
  PC{a: Array<F32>, c: Nat}

def pc_a(r: PC) -> Array<F32>:
  match r:
    case PC{a, c}:
      a

# a[j] = a[j] / s for j = idx..idx + k, counted onto c
def pdiv(k: Nat, +idx: U32, +s: F32, c: Nat, r: Array<F32> & F32) -> PC:
  match k r:
    case 0n Tuple{a, +x}:
      PC{Array.set(F32, a, idx, (x / s : F32)), 1n+c}
    case 1n+q Tuple{a, +x}:
      pdiv(q, (idx + 1 : U32), s, 1n+c, Array.get(F32, Array.set(F32, a, idx, (x / s : F32)), (idx + 1 : U32)))

def psm3(k: Nat, +off: U32, r: AS) -> PC:
  match r:
    case AS{a, +s}:
      pdiv(k, off, s, 0n, Array.get(F32, a, off))

def psm2(+k: Nat, +off: U32, r: AS) -> PC:
  match r:
    case AS{a, +m}:
      psm3(k, off, pexp(k, off, m, 0.0, Array.get(F32, a, off)))

# softmax in place over the n numbers of p from off (n >= 1), as PyTorch: exp(x - max) / sum; with the count of the
# numbers it divided (the law causal reads it)
def psmc(+n: Nat, +off: U32, p: Array<F32>) -> PC:
  psm2(Nat.sub(n, 1n), off, pmax0(Nat.sub(n, 1n), off, Array.get(F32, p, off)))

def psm(+n: Nat, +off: U32, p: Array<F32>) -> Array<F32>:
  pc_a(psmc(n, off, p))

# dst[di..] = src[si..], cnt + 1 numbers; r holds src[si], already read
def copy_n(cnt: Nat, +si: U32, +di: U32, o: Array<F32>, r: Array<F32> & F32) -> A2:
  match cnt r:
    case 0n Tuple{s, +x}:
      A2{s, Array.set(F32, o, di, x)}
    case 1n+n Tuple{s, +x}:
      copy_n(n, (si + 1 : U32), (di + 1 : U32), Array.set(F32, o, di, x), Array.get(F32, s, (si + 1 : U32)))

# =====================================================================================================
# Row-wise layers: softmax, log-softmax, RMSNorm, LayerNorm (each row of an n x c matrix on its own)
# =====================================================================================================

# rows i.. of x (c wide, left more after i), each softmaxed in place (psm)
def smf_rows(left: Nat, +i: U32, +c: Nat, x: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      psm(c, (i * U32.from_nat(c) : U32), x)
    case 1n+q:
      smf_rows(q, (i + 1 : U32), c, psm(c, (i * U32.from_nat(c) : U32), x))

# the output, and a copy of it saved for the backward pass
def out_saved(-n: Nat, -c: Nat, r: TA.MM2<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match r:
    case TA.MM2{a, b}:
      Fwd{Unit{}, a, b}

def sm_fwd(+n: Nat, +c: Nat, x: TA.Mat<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match x:
    case TA.Mat{a}:
      out_saved(n, c, TA.Mat.clone(n, c, TA.Mat{smf_rows(Nat.sub(n, 1n), 0, c, a)}))

# dy[j] = y[j] * (dy[j] - s), j = idx..idx + k; rd and ry hold dy[idx] and y[idx]
def smb_fix(k: Nat, +idx: U32, +s: F32, rd: Array<F32> & F32, ry: Array<F32> & F32) -> A2:
  match k rd ry:
    case 0n Tuple{d, +g} Tuple{y, +v}:
      A2{Array.set(F32, d, idx, (v * (g - s : F32) : F32)), y}
    case 1n+q Tuple{d, +g} Tuple{y, +v}:
      smb_fix(q, (idx + 1 : U32), s, Array.get(F32, Array.set(F32, d, idx, (v * (g - s : F32) : F32)), (idx + 1 : U32)), Array.get(F32, y, (idx + 1 : U32)))

def smb_row(+c: Nat, +off: U32, r: D2) -> A2:
  match r:
    case D2{d, y, +s}:
      smb_fix(Nat.sub(c, 1n), off, s, Array.get(F32, d, off), Array.get(F32, y, off))

# softmax backward, as PyTorch's: dx = y * (dy - sum(dy * y)), row after row, in place over dy
def smb_rows(left: Nat, +i: U32, +c: Nat, st: A2) -> A2:
  match left st:
    case 0n A2{d, y}:
      smb_row(c, (i * U32.from_nat(c) : U32), dot0(c, (i * U32.from_nat(c) : U32), (i * U32.from_nat(c) : U32), 1, 1, d, y))
    case 1n+q A2{d, y}:
      smb_rows(q, (i + 1 : U32), c, smb_row(c, (i * U32.from_nat(c) : U32), dot0(c, (i * U32.from_nat(c) : U32), (i * U32.from_nat(c) : U32), 1, 1, d, y)))

def dx_fin(-n: Nat, -c: Nat, r: A2) -> Bwd<Unit, TA.Mat<n, c>>:
  match r:
    case A2{d, y}:
      Bwd{Unit{}, TA.Mat{d}, Unit{}}

def sm_bwd(+n: Nat, +c: Nat, s: TA.Mat<n, c>, dy: TA.Mat<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match s dy:
    case TA.Mat{y} TA.Mat{d}:
      dx_fin(n, c, smb_rows(Nat.sub(n, 1n), 0, c, A2{d, y}))

# softmax over each row; saves its output
def Softmax(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => sm_fwd(n, c, x), p => s => dy => g => sm_bwd(n, c, s, dy), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# log_softmax(x) = (x - max) - log(sum(exp(x - max))), as PyTorch's CPU kernel

# acc + exp(a[j] - m) for j = idx..idx + k, in order
def lsum(k: Nat, +idx: U32, +m: F32, acc: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, (acc + F32.exp((x - m : F32)) : F32)}
    case 1n+q Tuple{a, +x}:
      lsum(q, (idx + 1 : U32), m, (acc + F32.exp((x - m : F32)) : F32), Array.get(F32, a, (idx + 1 : U32)))

# a[j] = (a[j] - m) - l for j = idx..idx + k
def lw(k: Nat, +idx: U32, +m: F32, +l: F32, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{a, +x}:
      Array.set(F32, a, idx, ((x - m : F32) - l : F32))
    case 1n+q Tuple{a, +x}:
      lw(q, (idx + 1 : U32), m, l, Array.get(F32, Array.set(F32, a, idx, ((x - m : F32) - l : F32)), (idx + 1 : U32)))

def lsm_r3(+c: Nat, +off: U32, +m: F32, r: AS) -> Array<F32>:
  match r:
    case AS{a, +s}:
      lw(Nat.sub(c, 1n), off, m, F32.log(s), Array.get(F32, a, off))

def lsm_r2(+c: Nat, +off: U32, r: AS) -> Array<F32>:
  match r:
    case AS{a, +m}:
      lsm_r3(c, off, m, lsum(Nat.sub(c, 1n), off, m, 0.0, Array.get(F32, a, off)))

def lsm_row(+c: Nat, +off: U32, x: Array<F32>) -> Array<F32>:
  lsm_r2(c, off, pmax0(Nat.sub(c, 1n), off, Array.get(F32, x, off)))

def lsmf_rows(left: Nat, +i: U32, +c: Nat, x: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      lsm_row(c, (i * U32.from_nat(c) : U32), x)
    case 1n+q:
      lsmf_rows(q, (i + 1 : U32), c, lsm_row(c, (i * U32.from_nat(c) : U32), x))

def lsm_fwd(+n: Nat, +c: Nat, x: TA.Mat<n, c>) -> Fwd<Unit, TA.Mat<n, c>, TA.Mat<n, c>>:
  match x:
    case TA.Mat{a}:
      out_saved(n, c, TA.Mat.clone(n, c, TA.Mat{lsmf_rows(Nat.sub(n, 1n), 0, c, a)}))

# acc + a[j] for j = idx..idx + k, in order
def asum(k: Nat, +idx: U32, acc: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, (acc + x : F32)}
    case 1n+q Tuple{a, +x}:
      asum(q, (idx + 1 : U32), (acc + x : F32), Array.get(F32, a, (idx + 1 : U32)))

# dy[j] = dy[j] - exp(y[j]) * s, j = idx..idx + k
def lsmb_fix(k: Nat, +idx: U32, +s: F32, rd: Array<F32> & F32, ry: Array<F32> & F32) -> A2:
  match k rd ry:
    case 0n Tuple{d, +g} Tuple{y, +v}:
      A2{Array.set(F32, d, idx, (g - (F32.exp(v) * s : F32) : F32)), y}
    case 1n+q Tuple{d, +g} Tuple{y, +v}:
      lsmb_fix(q, (idx + 1 : U32), s, Array.get(F32, Array.set(F32, d, idx, (g - (F32.exp(v) * s : F32) : F32)), (idx + 1 : U32)), Array.get(F32, y, (idx + 1 : U32)))

def lsmb_row(+c: Nat, +off: U32, y: Array<F32>, r: AS) -> A2:
  match r:
    case AS{d, +s}:
      lsmb_fix(Nat.sub(c, 1n), off, s, Array.get(F32, d, off), Array.get(F32, y, off))

# dx = dy - exp(y) * sum(dy), row after row, in place over dy
def lsmb_rows(left: Nat, +i: U32, +c: Nat, st: A2) -> A2:
  match left st:
    case 0n A2{d, y}:
      lsmb_row(c, (i * U32.from_nat(c) : U32), y, asum(Nat.sub(c, 1n), (i * U32.from_nat(c) : U32), 0.0, Array.get(F32, d, (i * U32.from_nat(c) : U32))))
    case 1n+q A2{d, y}:
      lsmb_rows(q, (i + 1 : U32), c, lsmb_row(c, (i * U32.from_nat(c) : U32), y, asum(Nat.sub(c, 1n), (i * U32.from_nat(c) : U32), 0.0, Array.get(F32, d, (i * U32.from_nat(c) : U32)))))

def lsm_bwd(+n: Nat, +c: Nat, s: TA.Mat<n, c>, dy: TA.Mat<n, c>) -> Bwd<Unit, TA.Mat<n, c>>:
  match s dy:
    case TA.Mat{y} TA.Mat{d}:
      dx_fin(n, c, lsmb_rows(Nat.sub(n, 1n), 0, c, A2{d, y}))

def LogSoftmax(+n: Nat, +c: Nat) -> Layer<Unit, TA.Mat<n, c>, TA.Mat<n, c>, TA.Mat<n, c>>:
  Layer{p => x => lsm_fwd(n, c, x), p => s => dy => g => lsm_bwd(n, c, s, dy), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# ---- RMSNorm: y = g * (x / sqrt(mean(x^2) + eps)); its parameter is the gain g (1 x d) ----

# one optimizer step on a layer whose parameter is a single matrix
def mat_step_fin(-r: Nat, -c: Nat, q: MatStep<r, c>) -> Step<TA.Mat<r, c>>:
  match q:
    case MatStep{w, g, st}: Step{w, g, st}

def mat_step(+r: Nat, +c: Nat, +op: Opt, p: TA.Mat<r, c>, g: TA.Mat<r, c>, st: OptState<TA.Mat<r, c>>) -> Step<TA.Mat<r, c>>:
  mat_step_fin(r, c, Opt.mat(r, c, op, p, g, st))

def g_add_fin(-r: Nat, -c: Nat, r2: TA.MPair<r, c>) -> TA.Mat<r, c>:
  match r2:
    case TA.MPair{a, b}:
      a

# a + b, dropping b
def Mat.sum2(+r: Nat, +c: Nat, a: TA.Mat<r, c>, b: TA.Mat<r, c>) -> TA.Mat<r, c>:
  g_add_fin(r, c, Mat.add(r, c, a, b))

def dup_fin(-r: Nat, -c: Nat, m: TA.MM2<r, c>) -> Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  match m:
    case TA.MM2{a, b}:
      Two{a, b}

# two copies of a matrix
def Mat.dup(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> Two<TA.Mat<r, c>, TA.Mat<r, c>>:
  dup_fin(r, c, TA.Mat.clone(r, c, m))

def ml_list(-r: Nat, -c: Nat, ml: TA.ML<r, c>) -> List<&2, F32>:
  match ml:
    case TA.ML{m, xs}:
      xs

# ---- RMSNorm layer, a row at a time over the arrays: x (n x d), the gain g (1 x d), the output y. Each sum is added
# left to right from +0.0, each product in the order the list version multiplied, so the bits are the list version's.
# ----

# acc + a[i]·a[i] + ... over k + 1 numbers from i, in order; r holds a[i], already read
def ssq(k: Nat, +i: U32, acc: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, (acc + (x * x : F32) : F32)}
    case 1n+p Tuple{a, +x}:
      ssq(p, (i + 1 : U32), (acc + (x * x : F32) : F32), Array.get(F32, a, (i + 1 : U32)))

# 1 / rms of a row whose sum of squares is s
def rms_inv(+s: F32, +df: F32, +eps: F32) -> F32:
  (1.0 / F32.sqrt(((s / df : F32) + eps : F32)) : F32)

type R3 is Type:
  R3{x: Array<F32>, g: Array<F32>, y: Array<F32>}

# y[off + j] = g[j] * (x[off + j] * r), j = jj..jj + k; rx and rg hold x[off + jj] and g[jj]
def rms_w(k: Nat, +j: U32, +off: U32, +r: F32, y: Array<F32>, rx: Array<F32> & F32, rg: Array<F32> & F32) -> R3:
  match k rx rg:
    case 0n Tuple{x, +a} Tuple{g, +b}:
      R3{x, g, Array.set(F32, y, (off + j : U32), (b * (a * r : F32) : F32))}
    case 1n+p Tuple{x, +a} Tuple{g, +b}:
      rms_w(p, (j + 1 : U32), off, r, Array.set(F32, y, (off + j : U32), (b * (a * r : F32) : F32)), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def rmsf_row2(+d: Nat, +off: U32, +df: F32, +eps: F32, g: Array<F32>, y: Array<F32>, r: AS) -> R3:
  match r:
    case AS{x, +s}:
      rms_w(Nat.sub(d, 1n), 0, off, rms_inv(s, df, eps), y, Array.get(F32, x, off), Array.get(F32, g, 0))

def rmsf_row(+d: Nat, +off: U32, +df: F32, +eps: F32, s: R3) -> R3:
  match s:
    case R3{x, g, y}:
      rmsf_row2(d, off, df, eps, g, y, ssq(Nat.sub(d, 1n), off, 0.0, Array.get(F32, x, off)))

# rows i.. (left + 1 of them, d wide)
def rmsf_rows(left: Nat, +i: U32, +d: Nat, +df: F32, +eps: F32, s: R3) -> R3:
  match left:
    case 0n:
      rmsf_row(d, (i * U32.from_nat(d) : U32), df, eps, s)
    case 1n+p:
      rmsf_rows(p, (i + 1 : U32), d, df, eps, rmsf_row(d, (i * U32.from_nat(d) : U32), df, eps, s))

def rms_f2(-n: Nat, -d: Nat, s: R3) -> Fwd<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match s:
    case R3{x, g, y}:
      Fwd{TA.Mat{g}, TA.Mat{y}, TA.Mat{x}}

def rms_fwd(+n: Nat, +d: Nat, +eps: F32, p: TA.Mat<1n, d>, x: TA.Mat<n, d>) -> Fwd<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match p x:
    case TA.Mat{g} TA.Mat{xa}:
      rms_f2(n, d, rmsf_rows(Nat.sub(n, 1n), 0, d, F32.from_nat(d), eps, R3{xa, g, zeros_a(Nat.mul(n, d))}))

# with r = 1/rms, xh = x*r, dxh = dy*g: dx = r * (dxh - xh * mean(dxh * xh)); dg += dy * xh, row after row
type R5 is Type:
  R5{x: Array<F32>, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, dg: Array<F32>}

type R3S is Type:
  R3S{x: Array<F32>, dy: Array<F32>, g: Array<F32>, s: F32}

# acc + dxh[j]·xh[j] + ..., in order; rx, rd, rg hold x[off + jj], dy[off + jj], g[jj]
def rmsb_dot(k: Nat, +j: U32, +off: U32, +r: F32, acc: F32, rx: Array<F32> & F32, rd: Array<F32> & F32, rg: Array<F32> & F32) -> R3S:
  match k rx rd rg:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      R3S{x, dy, g, (acc + ((b * c : F32) * (a * r : F32) : F32) : F32)}
    case 1n+p Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      rmsb_dot(p, (j + 1 : U32), off, r, (acc + ((b * c : F32) * (a * r : F32) : F32) : F32), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, dy, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def dg_add(+j: U32, +v: F32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{dg, +old}:
      Array.set(F32, dg, j, (old + v : F32))

# dx[off + j] = ((dy[j]·g[j]) - (x[j]·r)·m)·r and dg[j] += dy[j]·(x[j]·r), j = jj..jj + k
def rmsb_w(k: Nat, +j: U32, +off: U32, +r: F32, +m: F32, dx: Array<F32>, dg: Array<F32>, rx: Array<F32> & F32, rd: Array<F32> & F32, rg: Array<F32> & F32) -> R5:
  match k rx rd rg:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = (a * r : F32)
      R5{x, dy, g, Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32)), dg_add(j, (b * xh : F32), Array.get(F32, dg, j))}
    case 1n+p Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = (a * r : F32)
      rmsb_w(p, (j + 1 : U32), off, r, m, Array.set(F32, dx, (off + j : U32), (((b * c : F32) - (xh * m : F32) : F32) * r : F32)), dg_add(j, (b * xh : F32), Array.get(F32, dg, j)), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, dy, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def rmsb_row3(+d: Nat, +off: U32, +df: F32, +r: F32, dx: Array<F32>, dg: Array<F32>, q: R3S) -> R5:
  match q:
    case R3S{x, dy, g, +s}:
      rmsb_w(Nat.sub(d, 1n), 0, off, r, (s / df : F32), dx, dg, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0))

def rmsb_row2(+d: Nat, +off: U32, +df: F32, +eps: F32, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, dg: Array<F32>, q: AS) -> R5:
  match q:
    case AS{x, +s}:
      +r = rms_inv(s, df, eps)
      rmsb_row3(d, off, df, r, dx, dg, rmsb_dot(Nat.sub(d, 1n), 0, off, r, 0.0, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0)))

def rmsb_row(+d: Nat, +off: U32, +df: F32, +eps: F32, s: R5) -> R5:
  match s:
    case R5{x, dy, g, dx, dg}:
      rmsb_row2(d, off, df, eps, dy, g, dx, dg, ssq(Nat.sub(d, 1n), off, 0.0, Array.get(F32, x, off)))

def rmsb_rows(left: Nat, +i: U32, +d: Nat, +df: F32, +eps: F32, s: R5) -> R5:
  match left:
    case 0n:
      rmsb_row(d, (i * U32.from_nat(d) : U32), df, eps, s)
    case 1n+p:
      rmsb_rows(p, (i + 1 : U32), d, df, eps, rmsb_row(d, (i * U32.from_nat(d) : U32), df, eps, s))

def gain_bwd_fin(-n: Nat, -d: Nat, -P: Type, p: P, dx: TA.Mat<n, d>, g: P) -> Bwd<P, TA.Mat<n, d>>:
  Bwd{p, dx, g}

def rms_b2(+d: Nat, -n: Nat, gacc: TA.Mat<1n, d>, s: R5) -> Bwd<TA.Mat<1n, d>, TA.Mat<n, d>>:
  match s:
    case R5{x, dy, g, dx, dg}:
      gain_bwd_fin(n, d, TA.Mat<1n, d>, TA.Mat{g}, TA.Mat{dx}, Mat.sum2(1n, d, gacc, TA.Mat{dg}))

def rms_bwd(+n: Nat, +d: Nat, +eps: F32, p: TA.Mat<1n, d>, x: TA.Mat<n, d>, dy: TA.Mat<n, d>, gacc: TA.Mat<1n, d>) -> Bwd<TA.Mat<1n, d>, TA.Mat<n, d>>:
  match p x dy:
    case TA.Mat{g} TA.Mat{xa} TA.Mat{da}:
      rms_b2(d, n, gacc, rmsb_rows(Nat.sub(n, 1n), 0, d, F32.from_nat(d), eps, R5{xa, da, g, zeros_a(Nat.mul(n, d)), zeros_a(d)}))

# RMSNorm over rows of d numbers, with a learned gain (Llama's norm); saves its input
def RMSNorm(+n: Nat, +d: Nat, +eps: F32) -> Layer<TA.Mat<1n, d>, TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  Layer{p => x => rms_fwd(n, d, eps, p, x), p => s => dy => g => rms_bwd(n, d, eps, p, s, dy, g), op => p => g => st => mat_step(1n, d, op, p, g, st), u => TA.Mat.zeros(1n, d), p => Mat.dump(1n, d, p), p => xs => off => Mat.restore(1n, d, p, xs, off), p => xs => off => Mat.store(1n, d, p, xs, off), u => U32.from_nat(d)}

# ---- LayerNorm: y = (x - mean) / sqrt(var + eps) * g + b, var the biased variance ----

type LN<-d: Nat> is Type:
  LN{g: TA.Mat<1n, d>, b: TA.Mat<1n, d>}

# acc + (a[j] - mu)·(a[j] - mu) for j = idx..idx + k, in order
def csq(k: Nat, +idx: U32, +mu: F32, acc: F32, r: Array<F32> & F32) -> AS:
  match k r:
    case 0n Tuple{a, +x}:
      AS{a, (acc + ((x - mu : F32) * (x - mu : F32) : F32) : F32)}
    case 1n+q Tuple{a, +x}:
      csq(q, (idx + 1 : U32), mu, (acc + ((x - mu : F32) * (x - mu : F32) : F32) : F32), Array.get(F32, a, (idx + 1 : U32)))

type ASM is Type:
  ASM{a: Array<F32>, mu: F32, r: F32}

def ln_mr2(+df: F32, +eps: F32, +mu: F32, q: AS) -> ASM:
  match q:
    case AS{a, +v}:
      ASM{a, mu, (1.0 / F32.sqrt(((v / df : F32) + eps : F32)) : F32)}

def ln_mr1(+d: Nat, +off: U32, +df: F32, +eps: F32, q: AS) -> ASM:
  match q:
    case AS{a, +s}:
      ln_mr2(df, eps, (s / df : F32), csq(Nat.sub(d, 1n), off, (s / df : F32), 0.0, Array.get(F32, a, off)))

# a row's mean and 1 / sqrt(var + eps), as ln_xh and ln_r computed them
def ln_mr(+d: Nat, +off: U32, +df: F32, +eps: F32, x: Array<F32>) -> ASM:
  ln_mr1(d, off, df, eps, asum(Nat.sub(d, 1n), off, 0.0, Array.get(F32, x, off)))

type R4 is Type:
  R4{x: Array<F32>, g: Array<F32>, b: Array<F32>, y: Array<F32>}

# y[off + j] = ((x - mu) * r) * g[j] + b[j], j = jj..jj + k
def lnf_w(k: Nat, +j: U32, +off: U32, +mu: F32, +r: F32, y: Array<F32>, rx: Array<F32> & F32, rg: Array<F32> & F32, rb: Array<F32> & F32) -> R4:
  match k rx rg rb:
    case 0n Tuple{x, +a} Tuple{g, +gv} Tuple{b, +bv}:
      R4{x, g, b, Array.set(F32, y, (off + j : U32), ((((a - mu : F32) * r : F32) * gv : F32) + bv : F32))}
    case 1n+q Tuple{x, +a} Tuple{g, +gv} Tuple{b, +bv}:
      lnf_w(q, (j + 1 : U32), off, mu, r, Array.set(F32, y, (off + j : U32), ((((a - mu : F32) * r : F32) * gv : F32) + bv : F32)), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)), Array.get(F32, b, (j + 1 : U32)))

def lnf_row2(+d: Nat, +off: U32, g: Array<F32>, b: Array<F32>, y: Array<F32>, m: ASM) -> R4:
  match m:
    case ASM{x, +mu, +r}:
      lnf_w(Nat.sub(d, 1n), 0, off, mu, r, y, Array.get(F32, x, off), Array.get(F32, g, 0), Array.get(F32, b, 0))

def lnf_row(+d: Nat, +off: U32, +df: F32, +eps: F32, s: R4) -> R4:
  match s:
    case R4{x, g, b, y}:
      lnf_row2(d, off, g, b, y, ln_mr(d, off, df, eps, x))

def lnf_rows(left: Nat, +i: U32, +d: Nat, +df: F32, +eps: F32, s: R4) -> R4:
  match left:
    case 0n:
      lnf_row(d, (i * U32.from_nat(d) : U32), df, eps, s)
    case 1n+q:
      lnf_rows(q, (i + 1 : U32), d, df, eps, lnf_row(d, (i * U32.from_nat(d) : U32), df, eps, s))

def ln_f2(-n: Nat, -d: Nat, s: R4) -> Fwd<LN<d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match s:
    case R4{x, g, b, y}:
      Fwd{LN{TA.Mat{g}, TA.Mat{b}}, TA.Mat{y}, TA.Mat{x}}

def ln_fwd(+n: Nat, +d: Nat, +eps: F32, p: LN<d>, x: TA.Mat<n, d>) -> Fwd<LN<d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  match p x:
    case LN{TA.Mat{g}, TA.Mat{b}} TA.Mat{xa}:
      ln_f2(n, d, lnf_rows(Nat.sub(n, 1n), 0, d, F32.from_nat(d), eps, R4{xa, g, b, zeros_a(Nat.mul(n, d))}))

# with xh = (x - mu) * r and dxh = dy * g: dx = r * ((dxh - mean(dxh)) - xh * mean(dxh * xh)); dg += dy * xh; db += dy
type R3SS is Type:
  R3SS{x: Array<F32>, dy: Array<F32>, g: Array<F32>, sa: F32, sb: F32}

# sa + dxh[j] and sb + dxh[j]·xh[j], j = jj..jj + k, each in order
def lnb_sums(k: Nat, +j: U32, +off: U32, +mu: F32, +r: F32, sa: F32, sb: F32, rx: Array<F32> & F32, rd: Array<F32> & F32, rg: Array<F32> & F32) -> R3SS:
  match k rx rd rg:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      R3SS{x, dy, g, (sa + (b * c : F32) : F32), (sb + ((b * c : F32) * ((a - mu : F32) * r : F32) : F32) : F32)}
    case 1n+q Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      lnb_sums(q, (j + 1 : U32), off, mu, r, (sa + (b * c : F32) : F32), (sb + ((b * c : F32) * ((a - mu : F32) * r : F32) : F32) : F32), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, dy, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def ln_dw(+j: U32, +v: F32, +w: F32, st: A2) -> A2:
  match st:
    case A2{dg, db}:
      A2{dg_add(j, v, Array.get(F32, dg, j)), dg_add(j, w, Array.get(F32, db, j))}

type R4B is Type:
  R4B{x: Array<F32>, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, w: A2}

# dx[off + j] = (((dxh - ma) - xh * mb) * r), dg[j] += dy * xh, db[j] += dy, j = jj..jj + k (ma, mb the two means)
def lnb_w(k: Nat, +j: U32, +off: U32, +mu: F32, +r: F32, +ma: F32, +mb: F32, dx: Array<F32>, w: A2, rx: Array<F32> & F32, rd: Array<F32> & F32, rg: Array<F32> & F32) -> R4B:
  match k rx rd rg:
    case 0n Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = ((a - mu : F32) * r : F32)
      R4B{x, dy, g, Array.set(F32, dx, (off + j : U32), ((((b * c : F32) - ma : F32) - (xh * mb : F32) : F32) * r : F32)), ln_dw(j, (b * xh : F32), b, w)}
    case 1n+q Tuple{x, +a} Tuple{dy, +b} Tuple{g, +c}:
      +xh = ((a - mu : F32) * r : F32)
      lnb_w(q, (j + 1 : U32), off, mu, r, ma, mb, Array.set(F32, dx, (off + j : U32), ((((b * c : F32) - ma : F32) - (xh * mb : F32) : F32) * r : F32)), ln_dw(j, (b * xh : F32), b, w), Array.get(F32, x, ((off + j : U32) + 1 : U32)), Array.get(F32, dy, ((off + j : U32) + 1 : U32)), Array.get(F32, g, (j + 1 : U32)))

def lnb_row3(+d: Nat, +off: U32, +df: F32, +mu: F32, +r: F32, dx: Array<F32>, w: A2, q: R3SS) -> R4B:
  match q:
    case R3SS{x, dy, g, +sa, +sb}:
      lnb_w(Nat.sub(d, 1n), 0, off, mu, r, (sa / df : F32), (sb / df : F32), dx, w, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0))

def lnb_row2(+d: Nat, +off: U32, +df: F32, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, w: A2, m: ASM) -> R4B:
  match m:
    case ASM{x, +mu, +r}:
      lnb_row3(d, off, df, mu, r, dx, w, lnb_sums(Nat.sub(d, 1n), 0, off, mu, r, 0.0, 0.0, Array.get(F32, x, off), Array.get(F32, dy, off), Array.get(F32, g, 0)))

def lnb_row(+d: Nat, +off: U32, +df: F32, +eps: F32, s: R4B) -> R4B:
  match s:
    case R4B{x, dy, g, dx, w}:
      lnb_row2(d, off, df, dy, g, dx, w, ln_mr(d, off, df, eps, x))

def lnb_rows(left: Nat, +i: U32, +d: Nat, +df: F32, +eps: F32, s: R4B) -> R4B:
  match left:
    case 0n:
      lnb_row(d, (i * U32.from_nat(d) : U32), df, eps, s)
    case 1n+q:
      lnb_rows(q, (i + 1 : U32), d, df, eps, lnb_row(d, (i * U32.from_nat(d) : U32), df, eps, s))

def ln_b2(+d: Nat, -n: Nat, b: TA.Mat<1n, d>, gg: LN<d>, s: R4B) -> Bwd<LN<d>, TA.Mat<n, d>>:
  match gg s:
    case LN{ga, gb} R4B{x, dy, g, dx, A2{dg, db}}:
      gain_bwd_fin(n, d, LN<d>, LN{TA.Mat{g}, b}, TA.Mat{dx}, LN{Mat.sum2(1n, d, ga, TA.Mat{dg}), Mat.sum2(1n, d, gb, TA.Mat{db})})

def ln_bwd(+n: Nat, +d: Nat, +eps: F32, p: LN<d>, x: TA.Mat<n, d>, dy: TA.Mat<n, d>, gg: LN<d>) -> Bwd<LN<d>, TA.Mat<n, d>>:
  match p x dy:
    case LN{TA.Mat{g}, b} TA.Mat{xa} TA.Mat{da}:
      ln_b2(d, n, b, gg, lnb_rows(Nat.sub(n, 1n), 0, d, F32.from_nat(d), eps, R4B{xa, da, g, zeros_a(Nat.mul(n, d)), A2{zeros_a(d), zeros_a(d)}}))

def ln_pair(-d: Nat, p: LN<d>) -> Two<TA.Mat<1n, d>, TA.Mat<1n, d>>:
  match p:
    case LN{w, b}: Two{w, b}

def ln_unpair(-d: Nat, p: Two<TA.Mat<1n, d>, TA.Mat<1n, d>>) -> LN<d>:
  match p:
    case Two{w, b}: LN{w, b}

def ln_s2(-d: Nat, a: MatStep<1n, d>, c: MatStep<1n, d>) -> Step<LN<d>>:
  match a c:
    case MatStep{w, gw, sw} MatStep{b, gb, sb}:
      Step{LN{w, b}, LN{gw, gb}, OptState.map(Two<TA.Mat<1n, d>, TA.Mat<1n, d>>, LN<d>, OptState.join(TA.Mat<1n, d>, TA.Mat<1n, d>, sw, sb), x => ln_unpair(d, x), x => ln_unpair(d, x))}

def ln_step2(+d: Nat, +op: Opt, p: LN<d>, g: LN<d>, st: Two<OptState<TA.Mat<1n, d>>, OptState<TA.Mat<1n, d>>>) -> Step<LN<d>>:
  match p g st:
    case LN{w, b} LN{gw, gb} Two{sw, sb}:
      ln_s2(d, Opt.mat(1n, d, op, w, gw, sw), Opt.mat(1n, d, op, b, gb, sb))

def ln_step(+d: Nat, +op: Opt, p: LN<d>, g: LN<d>, st: OptState<LN<d>>) -> Step<LN<d>>:
  ln_step2(d, op, p, g, OptState.split(TA.Mat<1n, d>, TA.Mat<1n, d>, OptState.map(LN<d>, Two<TA.Mat<1n, d>, TA.Mat<1n, d>>, st, x => ln_pair(d, x), x => ln_pair(d, x))))

def ln_dump2(-d: Nat, r: PL<Two<TA.Mat<1n, d>, TA.Mat<1n, d>>>) -> PL<LN<d>>:
  match r:
    case PL{Two{g, b}, xs}:
      PL{LN{g, b}, xs}

def ln_dump(+d: Nat, p: LN<d>) -> PL<LN<d>>:
  match p:
    case LN{g, b}:
      ln_dump2(d, pl_pair(TA.Mat<1n, d>, TA.Mat<1n, d>, Mat.dump(1n, d, g), Mat.dump(1n, d, b)))

def ln_load3(-d: Nat, r: PA<Two<TA.Mat<1n, d>, TA.Mat<1n, d>>>) -> PA<LN<d>>:
  match r:
    case PA{Two{g, b}, xs, off}: PA{LN{g, b}, xs, off}

def ln_load2(+d: Nat, b: TA.Mat<1n, d>, r: PA<TA.Mat<1n, d>>) -> PA<LN<d>>:
  match r:
    case PA{g, xs, off}:
      ln_load3(d, pa_second(TA.Mat<1n, d>, TA.Mat<1n, d>, g, Mat.restore(1n, d, b, xs, off)))

def ln_load(+d: Nat, p: LN<d>, xs: Array<F32>, off: U32) -> PA<LN<d>>:
  match p:
    case LN{g, b}: ln_load2(d, b, Mat.restore(1n, d, g, xs, off))

def ln_store2(+d: Nat, b: TA.Mat<1n, d>, r: PA<TA.Mat<1n, d>>) -> PA<LN<d>>:
  match r:
    case PA{g, xs, off}:
      ln_load3(d, pa_second(TA.Mat<1n, d>, TA.Mat<1n, d>, g, Mat.store(1n, d, b, xs, off)))

def ln_store(+d: Nat, p: LN<d>, xs: Array<F32>, off: U32) -> PA<LN<d>>:
  match p:
    case LN{g, b}: ln_store2(d, b, Mat.store(1n, d, g, xs, off))

def LayerNorm(+n: Nat, +d: Nat, +eps: F32) -> Layer<LN<d>, TA.Mat<n, d>, TA.Mat<n, d>, TA.Mat<n, d>>:
  Layer{p => x => ln_fwd(n, d, eps, p, x), p => s => dy => g => ln_bwd(n, d, eps, p, s, dy, g), op => p => g => st => ln_step(d, op, p, g, st), u => LN{TA.Mat.zeros(1n, d), TA.Mat.zeros(1n, d)}, p => ln_dump(d, p), p => xs => off => ln_load(d, p, xs, off), p => xs => off => ln_store(d, p, xs, off), u => U32.from_nat(Nat.mul(2n, d))}

# ---- GroupNorm: LayerNorm over one image's group of channels, with a gain and a bias per channel ----
# Images are channels last: x is (n·hw) x c with c = ng·cg, image s owns rows s·hw .. s·hw + hw - 1 and group k columns
# k·cg .. k·cg + cg - 1. Each (image, group) block of hw·cg numbers is normalized by its own mean and biased variance,
# then y = xh * g[ch] + b[ch], as nn.GroupNorm. Writing c as ng·cg in the type makes "ng divides c" hold by
# construction. A block is walked one row segment at a time with LayerNorm's loops: `off` is the row's first cell and
# `j` the group's first channel, so a segment's cells are off + j .. off + j + cg - 1 and its gains g[j .. j + cg - 1].

# acc + the segments of the block's remaining `left` rows, each one row (c cells) below the last, in order
def gn_sum(left: Nat, +off: U32, +c: U32, +cg: Nat, q: AS) -> AS:
  match left q:
    case 0n r:
      r
    case 1n+l AS{a, +s}:
      gn_sum(l, (off + c : U32), c, cg, asum(Nat.sub(cg, 1n), (off + c : U32), s, Array.get(F32, a, (off + c : U32))))

# acc + (x - mu)·(x - mu) over the same segments
def gn_csq(left: Nat, +off: U32, +c: U32, +cg: Nat, +mu: F32, q: AS) -> AS:
  match left q:
    case 0n r:
      r
    case 1n+l AS{a, +s}:
      gn_csq(l, (off + c : U32), c, cg, mu, csq(Nat.sub(cg, 1n), (off + c : U32), mu, s, Array.get(F32, a, (off + c : U32))))

def gn_mr1(+hw: Nat, +off: U32, +c: U32, +cg: Nat, +nf: F32, +eps: F32, q: AS) -> ASM:
  match q:
    case AS{a, +s}:
      ln_mr2(nf, eps, (s / nf : F32), gn_csq(Nat.sub(hw, 1n), off, c, cg, (s / nf : F32), csq(Nat.sub(cg, 1n), off, (s / nf : F32), 0.0, Array.get(F32, a, off))))

# a block's mean and 1 / sqrt(var + eps); off is its first cell, nf = hw·cg
def gn_mr(+hw: Nat, +off: U32, +c: U32, +cg: Nat, +nf: F32, +eps: F32, x: Array<F32>) -> ASM:
  gn_mr1(hw, off, c, cg, nf, eps, gn_sum(Nat.sub(hw, 1n), off, c, cg, asum(Nat.sub(cg, 1n), off, 0.0, Array.get(F32, x, off))))

# y over the block's rows, one segment each (LayerNorm's lnf_w with the group's gains)
def gnf_rows(left: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +mu: F32, +r: F32, s: R4) -> R4:
  match left s:
    case 0n R4{x, g, b, y}:
      lnf_w(Nat.sub(cg, 1n), j, off, mu, r, y, Array.get(F32, x, (off + j : U32)), Array.get(F32, g, j), Array.get(F32, b, j))
    case 1n+l R4{x, g, b, y}:
      gnf_rows(l, (off + c : U32), j, c, cg, mu, r, lnf_w(Nat.sub(cg, 1n), j, off, mu, r, y, Array.get(F32, x, (off + j : U32)), Array.get(F32, g, j), Array.get(F32, b, j)))

def gnf_grp2(+hw: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, g: Array<F32>, b: Array<F32>, y: Array<F32>, m: ASM) -> R4:
  match m:
    case ASM{x, +mu, +r}:
      gnf_rows(Nat.sub(hw, 1n), off, j, c, cg, mu, r, R4{x, g, b, y})

# one (image, group) block: off is the image's first row's first cell, j the group's first channel
def gnf_grp(+hw: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4) -> R4:
  match s:
    case R4{x, g, b, y}:
      gnf_grp2(hw, off, j, c, cg, g, b, y, gn_mr(hw, (off + j : U32), c, cg, nf, eps, x))

def gnf_groups(left: Nat, +off: U32, +j: U32, +hw: Nat, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4) -> R4:
  match left:
    case 0n:
      gnf_grp(hw, off, j, c, cg, nf, eps, s)
    case 1n+q:
      gnf_groups(q, off, (j + U32.from_nat(cg) : U32), hw, c, cg, nf, eps, gnf_grp(hw, off, j, c, cg, nf, eps, s))

def gnf_images(left: Nat, +off: U32, +ng: Nat, +hw: Nat, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4) -> R4:
  match left:
    case 0n:
      gnf_groups(Nat.sub(ng, 1n), off, 0, hw, c, cg, nf, eps, s)
    case 1n+q:
      gnf_images(q, (off + (U32.from_nat(hw) * c : U32) : U32), ng, hw, c, cg, nf, eps, gnf_groups(Nat.sub(ng, 1n), off, 0, hw, c, cg, nf, eps, s))

def gn_fwd(+n: Nat, +hw: Nat, +ng: Nat, +cg: Nat, +eps: F32, p: LN<Nat.mul(ng, cg)>, x: TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>) -> Fwd<LN<Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>>:
  match p x:
    case LN{TA.Mat{g}, TA.Mat{b}} TA.Mat{xa}:
      ln_f2(Nat.mul(n, hw), Nat.mul(ng, cg), gnf_images(Nat.sub(n, 1n), 0, ng, hw, U32.from_nat(Nat.mul(ng, cg)), cg, F32.from_nat(Nat.mul(hw, cg)), eps, R4{xa, g, b, zeros_a(Nat.mul(Nat.mul(n, hw), Nat.mul(ng, cg)))}))

# the backward pass is LayerNorm's with the block in place of the row: the two means (of dxh and of dxh·xh) are taken
# over the whole block, then each cell's dx, dg[ch] += dy·xh and db[ch] += dy, in order
def gnb_sums(left: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +mu: F32, +r: F32, q: R3SS) -> R3SS:
  match left q:
    case 0n t:
      t
    case 1n+l R3SS{x, dy, g, +sa, +sb}:
      gnb_sums(l, (off + c : U32), j, c, cg, mu, r, lnb_sums(Nat.sub(cg, 1n), j, (off + c : U32), mu, r, sa, sb, Array.get(F32, x, ((off + c : U32) + j : U32)), Array.get(F32, dy, ((off + c : U32) + j : U32)), Array.get(F32, g, j)))

def gnb_rows(left: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +mu: F32, +r: F32, +ma: F32, +mb: F32, s: R4B) -> R4B:
  match left s:
    case 0n R4B{x, dy, g, dx, w}:
      lnb_w(Nat.sub(cg, 1n), j, off, mu, r, ma, mb, dx, w, Array.get(F32, x, (off + j : U32)), Array.get(F32, dy, (off + j : U32)), Array.get(F32, g, j))
    case 1n+l R4B{x, dy, g, dx, w}:
      gnb_rows(l, (off + c : U32), j, c, cg, mu, r, ma, mb, lnb_w(Nat.sub(cg, 1n), j, off, mu, r, ma, mb, dx, w, Array.get(F32, x, (off + j : U32)), Array.get(F32, dy, (off + j : U32)), Array.get(F32, g, j)))

def gnb_grp3(+hw: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +nf: F32, +mu: F32, +r: F32, dx: Array<F32>, w: A2, q: R3SS) -> R4B:
  match q:
    case R3SS{x, dy, g, +sa, +sb}:
      gnb_rows(Nat.sub(hw, 1n), off, j, c, cg, mu, r, (sa / nf : F32), (sb / nf : F32), R4B{x, dy, g, dx, w})

def gnb_grp2(+hw: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +nf: F32, dy: Array<F32>, g: Array<F32>, dx: Array<F32>, w: A2, m: ASM) -> R4B:
  match m:
    case ASM{x, +mu, +r}:
      gnb_grp3(hw, off, j, c, cg, nf, mu, r, dx, w, gnb_sums(Nat.sub(hw, 1n), off, j, c, cg, mu, r, lnb_sums(Nat.sub(cg, 1n), j, off, mu, r, 0.0, 0.0, Array.get(F32, x, (off + j : U32)), Array.get(F32, dy, (off + j : U32)), Array.get(F32, g, j))))

def gnb_grp(+hw: Nat, +off: U32, +j: U32, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4B) -> R4B:
  match s:
    case R4B{x, dy, g, dx, w}:
      gnb_grp2(hw, off, j, c, cg, nf, dy, g, dx, w, gn_mr(hw, (off + j : U32), c, cg, nf, eps, x))

def gnb_groups(left: Nat, +off: U32, +j: U32, +hw: Nat, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4B) -> R4B:
  match left:
    case 0n:
      gnb_grp(hw, off, j, c, cg, nf, eps, s)
    case 1n+q:
      gnb_groups(q, off, (j + U32.from_nat(cg) : U32), hw, c, cg, nf, eps, gnb_grp(hw, off, j, c, cg, nf, eps, s))

def gnb_images(left: Nat, +off: U32, +ng: Nat, +hw: Nat, +c: U32, +cg: Nat, +nf: F32, +eps: F32, s: R4B) -> R4B:
  match left:
    case 0n:
      gnb_groups(Nat.sub(ng, 1n), off, 0, hw, c, cg, nf, eps, s)
    case 1n+q:
      gnb_images(q, (off + (U32.from_nat(hw) * c : U32) : U32), ng, hw, c, cg, nf, eps, gnb_groups(Nat.sub(ng, 1n), off, 0, hw, c, cg, nf, eps, s))

def gn_bwd(+n: Nat, +hw: Nat, +ng: Nat, +cg: Nat, +eps: F32, p: LN<Nat.mul(ng, cg)>, x: TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>, dy: TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>, gg: LN<Nat.mul(ng, cg)>) -> Bwd<LN<Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>>:
  match p x dy:
    case LN{TA.Mat{g}, b} TA.Mat{xa} TA.Mat{da}:
      ln_b2(Nat.mul(ng, cg), Nat.mul(n, hw), b, gg, gnb_images(Nat.sub(n, 1n), 0, ng, hw, U32.from_nat(Nat.mul(ng, cg)), cg, F32.from_nat(Nat.mul(hw, cg)), eps, R4B{xa, da, g, zeros_a(Nat.mul(Nat.mul(n, hw), Nat.mul(ng, cg))), A2{zeros_a(Nat.mul(ng, cg)), zeros_a(Nat.mul(ng, cg))}}))

# GroupNorm over n images of hw pixels and ng groups of cg channels, channels last; its parameters are LayerNorm's
def GroupNorm(+n: Nat, +hw: Nat, +ng: Nat, +cg: Nat, +eps: F32) -> Layer<LN<Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>, TA.Mat<Nat.mul(n, hw), Nat.mul(ng, cg)>>:
  Layer{p => x => gn_fwd(n, hw, ng, cg, eps, p, x), p => s => dy => g => gn_bwd(n, hw, ng, cg, eps, p, s, dy, g), op => p => g => st => ln_step(Nat.mul(ng, cg), op, p, g, st), u => LN{TA.Mat.zeros(1n, Nat.mul(ng, cg)), TA.Mat.zeros(1n, Nat.mul(ng, cg))}, p => ln_dump(Nat.mul(ng, cg), p), p => xs => off => ln_load(Nat.mul(ng, cg), p, xs, off), p => xs => off => ln_store(Nat.mul(ng, cg), p, xs, off), u => U32.from_nat(Nat.mul(2n, Nat.mul(ng, cg)))}

# =====================================================================================================
# Embedding: row i of the output is row ids[i] of the table (v x d); the gradient adds each row back
# =====================================================================================================

# row i of the output (d numbers) copied from row ids[i] of the table, for the first n ids
def emb_cp(left: Nat, ids: List<&2, Nat>, +i: U32, +d: Nat, st: A2) -> A2:
  match left ids st:
    case 1n+q Con{+id, t} A2{e, o}:
      emb_cp(q, t, (i + 1 : U32), d, copy_n(Nat.sub(d, 1n), (U32.from_nat(id) * U32.from_nat(d) : U32), (i * U32.from_nat(d) : U32), o, Array.get(F32, e, (U32.from_nat(id) * U32.from_nat(d) : U32))))
    case _ _ st:
      st

def emb_f2(-n: Nat, -v: Nat, -d: Nat, +ids: List<&2, Nat>, r: A2) -> Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>:
  match r:
    case A2{e, o}:
      Fwd{TA.Mat{e}, TA.Mat{o}, ids}

def emb_f(+n: Nat, +v: Nat, +d: Nat, +ids: List<&2, Nat>, e: TA.Mat<v, d>) -> Fwd<TA.Mat<v, d>, TA.Mat<n, d>, List<&2, Nat>>:
  match e:
    case TA.Mat{ea}:
      emb_f2(n, v, d, ids, emb_cp(n, ids, 0, d, A2{ea, zeros_a(Nat.mul(n, d))}))

# g[gi + j] = g[gi + j] + dy[di + j], j = 0..k
def emb_add(k: Nat, +gi: U32, +di: U32, rg: Array<F32> & F32, rd: Array<F32> & F32) -> A2:
  match k rg rd:
    case 0n Tuple{g, +a} Tuple{dy, +b}:
      A2{Array.set(F32, g, gi, (a + b : F32)), dy}
    case 1n+q Tuple{g, +a} Tuple{dy, +b}:
      emb_add(q, (gi + 1 : U32), (di + 1 : U32), Array.get(F32, Array.set(F32, g, gi, (a + b : F32)), (gi + 1 : U32)), Array.get(F32, dy, (di + 1 : U32)))

# each dY row added into the gradient's row ids[i], in order, by index
def emb_sc(left: Nat, ids: List<&2, Nat>, +i: U32, +d: Nat, st: A2) -> A2:
  match left ids st:
    case 1n+q Con{+id, t} A2{g, dy}:
      emb_sc(q, t, (i + 1 : U32), d, emb_add(Nat.sub(d, 1n), (U32.from_nat(id) * U32.from_nat(d) : U32), (i * U32.from_nat(d) : U32), Array.get(F32, g, (U32.from_nat(id) * U32.from_nat(d) : U32)), Array.get(F32, dy, (i * U32.from_nat(d) : U32))))
    case _ _ st:
      st

def emb_b2(-v: Nat, -d: Nat, e: TA.Mat<v, d>, r: A2) -> Bwd<TA.Mat<v, d>, List<&2, Nat>>:
  match r:
    case A2{g, dy}:
      Bwd{e, Nil{}, TA.Mat{g}}

def emb_bwd(+n: Nat, +v: Nat, +d: Nat, e: TA.Mat<v, d>, ids: List<&2, Nat>, dy: TA.Mat<n, d>, g: TA.Mat<v, d>) -> Bwd<TA.Mat<v, d>, List<&2, Nat>>:
  match dy g:
    case TA.Mat{da} TA.Mat{ga}:
      emb_b2(v, d, e, emb_sc(n, ids, 0, d, A2{ga, da}))

# a table of v rows of d numbers, looked up by n ids; an id has no gradient, so dX is the empty list
def Embedding(+n: Nat, +v: Nat, +d: Nat) -> Layer<TA.Mat<v, d>, List<&2, Nat>, TA.Mat<n, d>, List<&2, Nat>>:
  Layer{p => x => emb_f(n, v, d, x, p), p => s => dy => g => emb_bwd(n, v, d, p, s, dy, g), op => p => g => st => mat_step(v, d, op, p, g, st), u => TA.Mat.zeros(v, d), p => Mat.dump(v, d, p), p => xs => off => Mat.restore(v, d, p, xs, off), p => xs => off => Mat.store(v, d, p, xs, off), u => U32.from_nat(Nat.mul(v, d))}

# =====================================================================================================
# Linear without bias: y = x·Wᵀ (Llama's projections)
# =====================================================================================================

def lnb_f2(-n: Nat, -i: Nat, -o: Nat, r: TA.MMulNT<n, i, o>) -> Fwd<TA.Mat<o, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  match r:
    case TA.MMulNT{x, w, y}:
      Fwd{w, y, x}

def lnb_b3(-n: Nat, -i: Nat, -o: Nat, w: TA.Mat<o, i>, dx: TA.Mat<n, i>, r: TA.MMulTN<o, n, i>) -> Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMulTN{dy, x, g}:
      Bwd{w, dx, g}

def lnb_b2(+n: Nat, +i: Nat, +o: Nat, +par: Nat, x: TA.Mat<n, i>, g: TA.Mat<o, i>, r: TA.MMul<n, o, i>) -> Bwd<TA.Mat<o, i>, TA.Mat<n, i>>:
  match r:
    case TA.MMul{dy, w, dx}:
      lnb_b3(n, i, o, w, dx, TA.Mat.matmul_tn_acc(o, n, i, par, dy, x, g))

def LinearNB(+n: Nat, +i: Nat, +o: Nat, +par: Nat) -> Layer<TA.Mat<o, i>, TA.Mat<n, i>, TA.Mat<n, o>, TA.Mat<n, i>>:
  Layer{w => x => lnb_f2(n, i, o, TA.Mat.matmul_nt(n, i, o, par, x, w)), w => s => dy => g => lnb_b2(n, i, o, par, s, g, TA.Mat.matmul_precise(n, o, i, par, dy, w)), op => p => g => st => mat_step(o, i, op, p, g, st), u => TA.Mat.zeros(o, i), p => Mat.dump(o, i, p), p => xs => off => Mat.restore(o, i, p, xs, off), p => xs => off => Mat.store(o, i, p, xs, off), u => U32.from_nat(Nat.mul(o, i))}

# =====================================================================================================
# Combinators: Id, Fork (one input to two layers), Both (two inputs, two layers), Add2, Mul2, Residual
# =====================================================================================================

def Id(~X: Type) -> Layer<Unit, X, X, Unit>:
  Layer{p => x => Fwd{p, x, Unit{}}, p => s => dy => g => Bwd{p, dy, g}, op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# one optimizer step on two layers' parameters, of any types
def pair_step2(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: Two<OptState<P1>, OptState<P2>>) -> Step<Two<P1, P2>>:
  match p g st:
    case Two{p1, p2} Two{g1, g2} Two{s1, s2}:
      pair_s3(P1, P2, Layer.step(P1, X1, Y1, S1, la, op, p1, g1, s1), Layer.step(P2, X2, Y2, S2, lb, op, p2, g2, s2))

def pair_step(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, +op: Opt, p: Two<P1, P2>, g: Two<P1, P2>, st: OptState<Two<P1, P2>>) -> Step<Two<P1, P2>>:
  pair_step2(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb, op, p, g, OptState.split(P1, P2, st))

def pair_f4(-P1: Type, -P2: Type, -Y1: Type, -Y2: Type, -S1: Type, -S2: Type, p1: P1, y1: Y1, s1: S1, r: Fwd<P2, Y2, S2>) -> Fwd<Two<P1, P2>, Two<Y1, Y2>, Two<S1, S2>>:
  match r:
    case Fwd{p2, y2, s2}:
      Fwd{Two{p1, p2}, Two{y1, y2}, Two{s1, s2}}

# runs lb on x2 once la's forward pass is done
def pair_f3(~P2: Type, ~X2: Type, ~Y2: Type, ~S2: Type, ~lb: Layer<P2, X2, Y2, S2>, -P1: Type, -Y1: Type, -S1: Type, p2: P2, x2: X2, r: Fwd<P1, Y1, S1>) -> Fwd<Two<P1, P2>, Two<Y1, Y2>, Two<S1, S2>>:
  match r:
    case Fwd{p1, y1, s1}:
      pair_f4(P1, P2, Y1, Y2, S1, S2, p1, y1, s1, Layer.fwd(P2, X2, Y2, S2, lb, p2, x2))

# ---- Fork: x goes to both layers (copied by `dup`); the two input gradients are summed by `add` ----

def fork_f2(~P1: Type, ~P2: Type, ~X: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X, Y1, S1>, ~lb: Layer<P2, X, Y2, S2>, p1: P1, p2: P2, d: Two<X, X>) -> Fwd<Two<P1, P2>, Two<Y1, Y2>, Two<S1, S2>>:
  match d:
    case Two{xa, xb}:
      pair_f3(~P2, ~X, ~Y2, ~S2, ~lb, P1, Y1, S1, p2, xb, Layer.fwd(P1, X, Y1, S1, la, p1, xa))

def fork_b3(~X: Type, ~add: X -> X -> X, -P1: Type, -P2: Type, p2: P2, g2: P2, dx2: X, r: Bwd<P1, X>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p1, dx1, g1}:
      Bwd{Two{p1, p2}, add(dx1, dx2), Two{g1, g2}}

def fork_b2(~P1: Type, ~X: Type, ~Y1: Type, ~S1: Type, ~add: X -> X -> X, ~la: Layer<P1, X, Y1, S1>, -P2: Type, p1: P1, s1: S1, d1: Y1, g1: P1, r: Bwd<P2, X>) -> Bwd<Two<P1, P2>, X>:
  match r:
    case Bwd{p2, dx2, g2}:
      fork_b3(~X, ~add, P1, P2, p2, g2, dx2, Layer.bwd(P1, X, Y1, S1, la, p1, s1, d1, g1))

def fork_bwd(~P1: Type, ~P2: Type, ~X: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~add: X -> X -> X, ~la: Layer<P1, X, Y1, S1>, ~lb: Layer<P2, X, Y2, S2>, p: Two<P1, P2>, s: Two<S1, S2>, dy: Two<Y1, Y2>, g: Two<P1, P2>) -> Bwd<Two<P1, P2>, X>:
  match p s dy g:
    case Two{p1, p2} Two{s1, s2} Two{d1, d2} Two{g1, g2}:
      fork_b2(~P1, ~X, ~Y1, ~S1, ~add, ~la, P2, p1, s1, d1, g1, Layer.bwd(P2, X, Y2, S2, lb, p2, s2, d2, g2))

def fork_fwd(~P1: Type, ~P2: Type, ~X: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~dup: X -> Two<X, X>, ~la: Layer<P1, X, Y1, S1>, ~lb: Layer<P2, X, Y2, S2>, p: Two<P1, P2>, x: X) -> Fwd<Two<P1, P2>, Two<Y1, Y2>, Two<S1, S2>>:
  match p:
    case Two{p1, p2}:
      fork_f2(~P1, ~P2, ~X, ~Y1, ~Y2, ~S1, ~S2, ~la, ~lb, p1, p2, dup(x))

def Fork(~P1: Type, ~P2: Type, ~X: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~dup: X -> Two<X, X>, ~add: X -> X -> X, ~la: Layer<P1, X, Y1, S1>, ~lb: Layer<P2, X, Y2, S2>) -> Layer<Two<P1, P2>, X, Two<Y1, Y2>, Two<S1, S2>>:
  # bend-lint: disable-next bend-lint-rules/performance/gpu_fit -- zeros are made once per run (Train.new, OptState.new), and pair_zeros already forks not-expected-because: the combinators nest, so a bang here would run inside a bang
  Layer{p => x => fork_fwd(~P1, ~P2, ~X, ~Y1, ~Y2, ~S1, ~S2, ~dup, ~la, ~lb, p, x), p => s => dy => g => fork_bwd(~P1, ~P2, ~X, ~Y1, ~Y2, ~S1, ~S2, ~add, ~la, ~lb, p, s, dy, g), op => p => g => st => pair_step(~P1, ~X, ~Y1, ~S1, ~P2, ~X, ~Y2, ~S2, ~la, ~lb, op, p, g, st), u => pair_zeros(~P1, ~X, ~Y1, ~S1, ~P2, ~X, ~Y2, ~S2, ~la, ~lb), p => pair_dump(~P1, ~X, ~Y1, ~S1, ~P2, ~X, ~Y2, ~S2, ~la, ~lb, p), p => xs => off => pair_load(~P1, ~X, ~Y1, ~S1, ~P2, ~X, ~Y2, ~S2, ~la, ~lb, p, xs, off), p => xs => off => pair_store(~P1, ~X, ~Y1, ~S1, ~P2, ~X, ~Y2, ~S2, ~la, ~lb, p, xs, off), u => (Layer.count(P1, X, Y1, S1, la) + Layer.count(P2, X, Y2, S2, lb) : U32)}

# ---- Both: la on the first of a pair, lb on the second ----

def both_b3(-P1: Type, -P2: Type, -X1: Type, -X2: Type, p2: P2, g2: P2, dx2: X2, r: Bwd<P1, X1>) -> Bwd<Two<P1, P2>, Two<X1, X2>>:
  match r:
    case Bwd{p1, dx1, g1}:
      Bwd{Two{p1, p2}, Two{dx1, dx2}, Two{g1, g2}}

def both_b2(~P1: Type, ~X1: Type, ~Y1: Type, ~S1: Type, ~la: Layer<P1, X1, Y1, S1>, -P2: Type, -X2: Type, p1: P1, s1: S1, d1: Y1, g1: P1, r: Bwd<P2, X2>) -> Bwd<Two<P1, P2>, Two<X1, X2>>:
  match r:
    case Bwd{p2, dx2, g2}:
      both_b3(P1, P2, X1, X2, p2, g2, dx2, Layer.bwd(P1, X1, Y1, S1, la, p1, s1, d1, g1))

def both_bwd(~P1: Type, ~P2: Type, ~X1: Type, ~X2: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, p: Two<P1, P2>, s: Two<S1, S2>, dy: Two<Y1, Y2>, g: Two<P1, P2>) -> Bwd<Two<P1, P2>, Two<X1, X2>>:
  match p s dy g:
    case Two{p1, p2} Two{s1, s2} Two{d1, d2} Two{g1, g2}:
      both_b2(~P1, ~X1, ~Y1, ~S1, ~la, P2, X2, p1, s1, d1, g1, Layer.bwd(P2, X2, Y2, S2, lb, p2, s2, d2, g2))

def both_fwd(~P1: Type, ~P2: Type, ~X1: Type, ~X2: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>, p: Two<P1, P2>, x: Two<X1, X2>) -> Fwd<Two<P1, P2>, Two<Y1, Y2>, Two<S1, S2>>:
  match p x:
    case Two{p1, p2} Two{x1, x2}:
      pair_f3(~P2, ~X2, ~Y2, ~S2, ~lb, P1, Y1, S1, p2, x2, Layer.fwd(P1, X1, Y1, S1, la, p1, x1))

def Both(~P1: Type, ~P2: Type, ~X1: Type, ~X2: Type, ~Y1: Type, ~Y2: Type, ~S1: Type, ~S2: Type, ~la: Layer<P1, X1, Y1, S1>, ~lb: Layer<P2, X2, Y2, S2>) -> Layer<Two<P1, P2>, Two<X1, X2>, Two<Y1, Y2>, Two<S1, S2>>:
  # bend-lint: disable-next bend-lint-rules/performance/gpu_fit -- zeros are made once per run (Train.new, OptState.new), and pair_zeros already forks not-expected-because: the combinators nest, so a bang here would run inside a bang
  Layer{p => x => both_fwd(~P1, ~P2, ~X1, ~X2, ~Y1, ~Y2, ~S1, ~S2, ~la, ~lb, p, x), p => s => dy => g => both_bwd(~P1, ~P2, ~X1, ~X2, ~Y1, ~Y2, ~S1, ~S2, ~la, ~lb, p, s, dy, g), op => p => g => st => pair_step(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb, op, p, g, st), u => pair_zeros(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb), p => pair_dump(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb, p), p => xs => off => pair_load(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb, p, xs, off), p => xs => off => pair_store(~P1, ~X1, ~Y1, ~S1, ~P2, ~X2, ~Y2, ~S2, ~la, ~lb, p, xs, off), u => (Layer.count(P1, X1, Y1, S1, la) + Layer.count(P2, X2, Y2, S2, lb) : U32)}

# ---- Tie: first, mid, last, where first and last share one parameter (tied embeddings: the table is also the output
# layer's weight). Forward runs the three in order with the same P for first and last; backward runs last, mid, first,
# both adding into the same gradient, as PyTorch does for a tensor used twice. ----

type Three<-A: Type, -B: Type, -C: Type> is Type:
  Three{a: A, b: B, c: C}

def tie_f4(-P: Type, -Q: Type, -W: Type, -S1: Type, -S2: Type, -S3: Type, q: Q, s1: S1, s2: S2, r: Fwd<P, W, S3>) -> Fwd<Two<P, Q>, W, Three<S1, S2, S3>>:
  match r:
    case Fwd{p, w, s3}:
      Fwd{Two{p, q}, w, Three{s1, s2, s3}}

def tie_f3(~P: Type, ~Z: Type, ~W: Type, ~S3: Type, ~lc: Layer<P, Z, W, S3>, -Q: Type, -S1: Type, -S2: Type, p: P, s1: S1, r: Fwd<Q, Z, S2>) -> Fwd<Two<P, Q>, W, Three<S1, S2, S3>>:
  match r:
    case Fwd{q, z, s2}:
      tie_f4(P, Q, W, S1, S2, S3, q, s1, s2, Layer.fwd(P, Z, W, S3, lc, p, z))

def tie_f2(~P: Type, ~Q: Type, ~Y: Type, ~Z: Type, ~W: Type, ~S2: Type, ~S3: Type, ~lb: Layer<Q, Y, Z, S2>, ~lc: Layer<P, Z, W, S3>, -S1: Type, q: Q, r: Fwd<P, Y, S1>) -> Fwd<Two<P, Q>, W, Three<S1, S2, S3>>:
  match r:
    case Fwd{p, y, s1}:
      tie_f3(~P, ~Z, ~W, ~S3, ~lc, Q, S1, S2, p, s1, Layer.fwd(Q, Y, Z, S2, lb, q, y))

def tie_fwd(~P: Type, ~Q: Type, ~X: Type, ~Y: Type, ~Z: Type, ~W: Type, ~S1: Type, ~S2: Type, ~S3: Type, ~la: Layer<P, X, Y, S1>, ~lb: Layer<Q, Y, Z, S2>, ~lc: Layer<P, Z, W, S3>, pq: Two<P, Q>, x: X) -> Fwd<Two<P, Q>, W, Three<S1, S2, S3>>:
  match pq:
    case Two{p, q}:
      tie_f2(~P, ~Q, ~Y, ~Z, ~W, ~S2, ~S3, ~lb, ~lc, S1, q, Layer.fwd(P, X, Y, S1, la, p, x))

def tie_b4(-P: Type, -Q: Type, -X: Type, q: Q, gq: Q, r: Bwd<P, X>) -> Bwd<Two<P, Q>, X>:
  match r:
    case Bwd{p, dx, gp}:
      Bwd{Two{p, q}, dx, Two{gp, gq}}

def tie_b3(~P: Type, ~X: Type, ~Y: Type, ~S1: Type, ~la: Layer<P, X, Y, S1>, -Q: Type, p: P, s1: S1, gp: P, r: Bwd<Q, Y>) -> Bwd<Two<P, Q>, X>:
  match r:
    case Bwd{q, dy, gq}:
      tie_b4(P, Q, X, q, gq, Layer.bwd(P, X, Y, S1, la, p, s1, dy, gp))

def tie_b2(~P: Type, ~Q: Type, ~X: Type, ~Y: Type, ~Z: Type, ~S1: Type, ~S2: Type, ~la: Layer<P, X, Y, S1>, ~lb: Layer<Q, Y, Z, S2>, q: Q, s1: S1, s2: S2, gq: Q, r: Bwd<P, Z>) -> Bwd<Two<P, Q>, X>:
  match r:
    case Bwd{p, dz, gp}:
      tie_b3(~P, ~X, ~Y, ~S1, ~la, Q, p, s1, gp, Layer.bwd(Q, Y, Z, S2, lb, q, s2, dz, gq))

def tie_bwd(~P: Type, ~Q: Type, ~X: Type, ~Y: Type, ~Z: Type, ~W: Type, ~S1: Type, ~S2: Type, ~S3: Type, ~la: Layer<P, X, Y, S1>, ~lb: Layer<Q, Y, Z, S2>, ~lc: Layer<P, Z, W, S3>, pq: Two<P, Q>, s: Three<S1, S2, S3>, dw: W, g: Two<P, Q>) -> Bwd<Two<P, Q>, X>:
  match pq s g:
    case Two{p, q} Three{s1, s2, s3} Two{gp, gq}:
      tie_b2(~P, ~Q, ~X, ~Y, ~Z, ~S1, ~S2, ~la, ~lb, q, s1, s2, gq, Layer.bwd(P, Z, W, S3, lc, p, s3, dw, gp))

# The shared parameter steps once (through first's step; last's must be the same kind of layer for P, e.g. Embedding
# and LinearNB over the same v x d matrix), the middle with its own.
def Tie(~P: Type, ~Q: Type, ~X: Type, ~Y: Type, ~Z: Type, ~W: Type, ~S1: Type, ~S2: Type, ~S3: Type, ~la: Layer<P, X, Y, S1>, ~lb: Layer<Q, Y, Z, S2>, ~lc: Layer<P, Z, W, S3>) -> Layer<Two<P, Q>, X, W, Three<S1, S2, S3>>:
  # bend-lint: disable-next bend-lint-rules/performance/gpu_fit -- zeros are made once per run (Train.new, OptState.new), and pair_zeros already forks not-expected-because: the combinators nest, so a bang here would run inside a bang
  Layer{pq => x => tie_fwd(~P, ~Q, ~X, ~Y, ~Z, ~W, ~S1, ~S2, ~S3, ~la, ~lb, ~lc, pq, x), pq => s => dw => g => tie_bwd(~P, ~Q, ~X, ~Y, ~Z, ~W, ~S1, ~S2, ~S3, ~la, ~lb, ~lc, pq, s, dw, g), op => p => g => st => pair_step(~P, ~X, ~Y, ~S1, ~Q, ~Y, ~Z, ~S2, ~la, ~lb, op, p, g, st), u => pair_zeros(~P, ~X, ~Y, ~S1, ~Q, ~Y, ~Z, ~S2, ~la, ~lb), p => pair_dump(~P, ~X, ~Y, ~S1, ~Q, ~Y, ~Z, ~S2, ~la, ~lb, p), p => xs => off => pair_load(~P, ~X, ~Y, ~S1, ~Q, ~Y, ~Z, ~S2, ~la, ~lb, p, xs, off), p => xs => off => pair_store(~P, ~X, ~Y, ~S1, ~Q, ~Y, ~Z, ~S2, ~la, ~lb, p, xs, off), u => (Layer.count(P, X, Y, S1, la) + Layer.count(Q, Y, Z, S2, lb) : U32)}

# ---- Add2 and Mul2: element-wise sum and product of a pair of matrices ----

def add2_f(+n: Nat, +c: Nat, x: Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> Fwd<Unit, TA.Mat<n, c>, Unit>:
  match x:
    case Two{a, b}:
      Fwd{Unit{}, Mat.sum2(n, c, a, b), Unit{}}

def Add2(+n: Nat, +c: Nat) -> Layer<Unit, Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, Unit>:
  Layer{p => x => add2_f(n, c, x), p => s => dy => g => Bwd{p, Mat.dup(n, c, dy), g}, op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

def mul2_f3(-n: Nat, -c: Nat, a2: TA.Mat<n, c>, b2: TA.Mat<n, c>, r: TA.MPair<n, c>) -> Fwd<Unit, TA.Mat<n, c>, Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match r:
    case TA.MPair{y, b}:
      Fwd{Unit{}, y, Two{a2, b2}}

def mul2_f2(+n: Nat, +c: Nat, da: Two<TA.Mat<n, c>, TA.Mat<n, c>>, db: Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> Fwd<Unit, TA.Mat<n, c>, Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match da db:
    case Two{a1, a2} Two{b1, b2}:
      mul2_f3(n, c, a2, b2, Ew.zip(~f32_mul, n, c, a1, b1))

def mul2_f(+n: Nat, +c: Nat, x: Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> Fwd<Unit, TA.Mat<n, c>, Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match x:
    case Two{a, b}:
      mul2_f2(n, c, Mat.dup(n, c, a), Mat.dup(n, c, b))

def mpair_a(-n: Nat, -c: Nat, r: TA.MPair<n, c>) -> TA.Mat<n, c>:
  match r:
    case TA.MPair{a, b}:
      a

# d(a*b) = (dy*b, dy*a)
def mul2_b2(+n: Nat, +c: Nat, s: Two<TA.Mat<n, c>, TA.Mat<n, c>>, d: Two<TA.Mat<n, c>, TA.Mat<n, c>>) -> Bwd<Unit, Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  match s d:
    case Two{a, b} Two{d1, d2}:
      Bwd{Unit{}, Two{mpair_a(n, c, Ew.zip(~f32_mul, n, c, d1, b)), mpair_a(n, c, Ew.zip(~f32_mul, n, c, d2, a))}, Unit{}}

def Mul2(+n: Nat, +c: Nat) -> Layer<Unit, Two<TA.Mat<n, c>, TA.Mat<n, c>>, TA.Mat<n, c>, Two<TA.Mat<n, c>, TA.Mat<n, c>>>:
  Layer{p => x => mul2_f(n, c, x), p => s => dy => g => mul2_b2(n, c, s, Mat.dup(n, c, dy)), op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# ---- Residual: y = x + f(x), for any layer f from n x c to n x c ----

# over any X with a copy (dup), an addition (add) and a layer that adds a pair (join): Seq(Fork(Id, f), join). Residual
# is this over matrices; the law residual runs it over the exact model of torch/spec.bend
def ResidualOf(~X: Type, ~P: Type, ~S: Type, ~dup: X -> Two<X, X>, ~add: X -> X -> X, ~join: Layer<Unit, Two<X, X>, X, Unit>, ~f: Layer<P, X, X, S>) -> Layer<Two<Two<Unit, P>, Unit>, X, X, Two<Two<Unit, S>, Unit>>:
  Seq(~Two<Unit, P>, ~Unit, ~X, ~Two<X, X>, ~X, ~Two<Unit, S>, ~Unit, ~Fork(~Unit, ~P, ~X, ~X, ~X, ~Unit, ~S, ~dup, ~add, ~Id(~X), ~f), ~join)

def Residual(~n: Nat, ~c: Nat, ~P: Type, ~S: Type, ~f: Layer<P, TA.Mat<n, c>, TA.Mat<n, c>, S>) -> Layer<Two<Two<Unit, P>, Unit>, TA.Mat<n, c>, TA.Mat<n, c>, Two<Two<Unit, S>, Unit>>:
  ResidualOf(~TA.Mat<n, c>, ~P, ~S, ~(m => Mat.dup(n, c, m)), ~(a => b => Mat.sum2(n, c, a, b)), ~Add2(n, c), ~f)

# =====================================================================================================
# RoPE (the rotate-half form of Llama and `transformers`): rows are positions (row r is position r mod t), each row
# holds h heads of 2*half numbers, and the pair (x[c], x[c + half]) of a head is rotated by pos * inv[c], with
# inv[c] = 1 / base^(2c / (2*half)). No parameters, nothing saved: the backward pass rotates by the opposite angle.
# =====================================================================================================

# Over the arrays: a table of cos and sin for each position mod t and each c (t x half), computed once a call with the
# operations the list version used, then every pair rotated in place, so the bits are the list version's.

type CS is Type:
  CS{c: Array<F32>, s: Array<F32>}

# the angle of table entry e: inv[c] * position, position e / half, c = e mod half, inv[c] = 1 / base^(2c / (2*half))
def rope_ang(+e: U32, +half: U32, +base: F32) -> F32:
  ((1.0 / F32.pow(base, (U32.to_f32((U32.mod(e, half) + U32.mod(e, half) : U32)) / U32.to_f32((half + half : U32)) : F32)) : F32) * U32.to_f32(U32.div(e, half)) : F32)

# entries e.. of the tables, left more after e
def rope_tab(left: Nat, +e: U32, +half: U32, +base: F32, tb: CS) -> CS:
  match left tb:
    case 0n CS{cs, sn}:
      +ang = rope_ang(e, half, base)
      CS{Array.set(F32, cs, e, F32.cos(ang)), Array.set(F32, sn, e, F32.sin(ang))}
    case 1n+q CS{cs, sn}:
      +ang = rope_ang(e, half, base)
      rope_tab(q, (e + 1 : U32), half, base, CS{Array.set(F32, cs, e, F32.cos(ang)), Array.set(F32, sn, e, F32.sin(ang))})

type RS3 is Type:
  RS3{x: Array<F32>, c: Array<F32>, s: Array<F32>}

# one pair (a = x[i1], b = x[i2]) rotated in place; fwd picks the direction: (a cos - b sin, b cos + a sin) forward,
# (a cos + b sin, b cos - a sin) backward, in PyTorch's order of operations for each
def rope_p3(fwd: Bool, +i1: U32, +i2: U32, +a: F32, r2: Array<F32> & F32, rc: Array<F32> & F32, rs: Array<F32> & F32) -> RS3:
  match fwd r2 rc rs:
    case True{} Tuple{x, +b} Tuple{cs, +c} Tuple{sn, +z}:
      RS3{Array.set(F32, Array.set(F32, x, i1, ((a * c : F32) + (F32.neg(b) * z : F32) : F32)), i2, ((b * c : F32) + (a * z : F32) : F32)), cs, sn}
    case False{} Tuple{x, +b} Tuple{cs, +c} Tuple{sn, +z}:
      RS3{Array.set(F32, Array.set(F32, x, i1, ((a * c : F32) + (b * z : F32) : F32)), i2, ((b * c : F32) + F32.neg((a * z : F32)) : F32)), cs, sn}

def rope_p2(+fwd: Bool, +i1: U32, +i2: U32, +k: U32, cs: Array<F32>, sn: Array<F32>, r1: Array<F32> & F32) -> RS3:
  match r1:
    case Tuple{x, +a}:
      rope_p3(fwd, i1, i2, a, Array.get(F32, x, i2), Array.get(F32, cs, k), Array.get(F32, sn, k))

# pair e of the n x w matrix: row e / hh (hh = h * half pairs a row), head (e mod hh) / half, column c of it
def rope_one(+e: U32, +fwd: Bool, +hh: U32, +half: U32, +w: U32, +t: U32, st: RS3) -> RS3:
  match st:
    case RS3{x, cs, sn}:
      +r = U32.div(e, hh)
      +c = U32.mod(U32.mod(e, hh), half)
      +i1 = (((r * w : U32) + (U32.div(U32.mod(e, hh), half) * (half + half : U32) : U32) : U32) + c : U32)
      rope_p2(fwd, i1, (i1 + half : U32), ((U32.mod(r, t) * half : U32) + c : U32), cs, sn, Array.get(F32, x, i1))

# pairs e.., left more after e
def rope_e(left: Nat, +e: U32, +fwd: Bool, +hh: U32, +half: U32, +w: U32, +t: U32, st: RS3) -> RS3:
  match left:
    case 0n:
      rope_one(e, fwd, hh, half, w, t, st)
    case 1n+q:
      rope_e(q, (e + 1 : U32), fwd, hh, half, w, t, rope_one(e, fwd, hh, half, w, t, st))

def rope_fin(-n: Nat, -w: Nat, st: RS3) -> TA.Mat<n, w>:
  match st:
    case RS3{x, cs, sn}:
      TA.Mat{x}

def rope_go2(+n: Nat, +w: Nat, +fwd: Bool, +t: Nat, +h: Nat, +half: Nat, x: Array<F32>, tb: CS) -> TA.Mat<n, w>:
  match tb:
    case CS{cs, sn}:
      rope_fin(n, w, rope_e(Nat.sub(Nat.mul(n, Nat.mul(h, half)), 1n), 0, fwd, U32.from_nat(Nat.mul(h, half)), U32.from_nat(half), U32.from_nat(w), U32.from_nat(t), RS3{x, cs, sn}))

def rope_go(+n: Nat, +w: Nat, +fwd: Bool, +t: Nat, +h: Nat, +half: Nat, +base: F32, m: TA.Mat<n, w>) -> TA.Mat<n, w>:
  match m:
    case TA.Mat{x}:
      rope_go2(n, w, fwd, t, h, half, x, rope_tab(Nat.sub(Nat.mul(t, half), 1n), 0, U32.from_nat(half), base, CS{zeros_a(Nat.mul(t, half)), zeros_a(Nat.mul(t, half))}))

# n rows (b sequences of t positions), w = h heads of 2*half numbers each
def RoPE(+n: Nat, +w: Nat, +t: Nat, +h: Nat, +half: Nat, +base: F32, -e: {w == Nat.mul(h, Nat.add(half, half)) : Nat}) -> Layer<Unit, TA.Mat<n, w>, TA.Mat<n, w>, Unit>:
  Layer{p => x => Fwd{p, rope_go(n, w, True{}, t, h, half, base, x), Unit{}}, p => s => dy => g => Bwd{p, rope_go(n, w, False{}, t, h, half, base, dy), g}, op => p => g => st => no_step(Unit, op, p, g, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# =====================================================================================================
# Causal attention with grouped-query heads: queries n x (h*hd), keys and values n x (kv*hd), h = kv*g; query head
# i reads key/value head i / g. Rows are b sequences of t positions; position p sees positions 0..p of its sequence.
#   S = (Q·Kᵀ) * scale, P = softmax(S) over the visible positions, O = P·V
# The forward pass keeps Q, K, V; the backward pass computes P again from them (recomputation, not storage).
# =====================================================================================================

# Q, K and V stay in their arrays. Each sequence's t rows are copied out once into arrays of their own, and every
# head's probabilities P (t x t) are an array too. Each sum is added in the order the earlier list version added it
# (left to right from +0.0), and each row of P is softmaxed in place over the row's visible positions (prow), so the
# bits are the list version's.

type AttnS is Type:
  AttnS{q: Array<F32>, k: Array<F32>, v: Array<F32>}

# the key/value head that query head hi reads, with g query heads per key/value head; the forward and the backward pass
# both pick it here (the law kv_of_group says which heads share one)
def kv_of(+hi: Nat, +g: Nat) -> Nat:
  Nat.div(hi, g)

# xs reversed onto acc
def rcat(xs: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{+x, t}:
      rcat(t, x <> acc)

type DR is Type:
  DR{a: Array<F32>, b: Array<F32>, row: List<&2, F32>}

# the dots of one row of a (hd numbers from ia) with consecutive rows of b (hd numbers from ib, each bs further): the
# current one r and m more, in order
def drow(m: Nat, +hd: Nat, +ia: U32, +ib: U32, +bs: U32, acc: List<&2, F32>, r: D2) -> DR:
  match m r:
    case 0n D2{a, b, +s}:
      DR{a, b, rcat(s <> acc, Nil{})}
    case 1n+p D2{a, b, +s}:
      drow(p, hd, ia, (ib + bs : U32), bs, s <> acc, dot0(hd, ia, (ib + bs : U32), 1, 1, a, b))

def dr_row(r: DR) -> List<&2, F32>:
  match r:
    case DR{a, b, row}:
      row

# the scores of the query at position i (row i of q, qs apart, the head's hd columns from qc) against keys 0..i (rows
# of k, ks apart, from kc): the one place attention reads keys, i + 1 of them (the law causal_keys)
def qrow(+i: Nat, +hd: Nat, +qs: U32, +qc: U32, +ks: U32, +kc: U32, q: Array<F32>, k: Array<F32>) -> DR:
  drow(i, hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, ks, Nil{}, dot0(hd, ((U32.from_nat(i) * qs : U32) + qc : U32), kc, 1, 1, q, k))

type QK is Type:
  QK{q: Array<F32>, k: Array<F32>, p: Array<F32>}

type PW is Type:
  PW{p: Array<F32>, n: Nat}

# xs scaled by sc, written into p from idx on; n counts the numbers written (from the n given)
def pw(xs: List<&2, F32>, +idx: U32, +sc: F32, n: Nat, p: Array<F32>) -> PW:
  match xs:
    case Nil{}:
      PW{p, n}
    case Con{x, t}:
      pw(t, (idx + 1 : U32), sc, 1n+n, Array.set(F32, p, idx, (x * sc : F32)))

type PR is Type:
  PR{q: Array<F32>, k: Array<F32>, p: Array<F32>, n: Nat}

def pr_n(r: PR) -> Nat:
  match r:
    case PR{q, k, p, n}:
      n

def pr_of(q: Array<F32>, k: Array<F32>, r: PC) -> PR:
  match r:
    case PC{p, c}:
      PR{q, k, p, c}

# the softmax over the n numbers pw wrote; the row's count is the one the softmax reports
def prow3(+off: U32, q: Array<F32>, k: Array<F32>, w: PW) -> PR:
  match w:
    case PW{p, +n}:
      pr_of(q, k, psmc(n, off, p))

def prow2(+sc: F32, +off: U32, p: Array<F32>, r: DR) -> PR:
  match r:
    case DR{q, k, row}:
      prow3(off, q, k, pw(row, off, sc, 0n, p))

# row i of a head's P (t x t, row-major, created zeros): the scaled scores of query i against keys 0..i (qrow) written
# from the row's start, then the softmax over as many positions as were written: i + 1 (the law causal). The positions
# after i are never written and keep their zeros.
def prow(+i: Nat, +t: Nat, +hd: Nat, +sc: F32, +qs: U32, +qc: U32, +ks: U32, +kc: U32, q: Array<F32>, k: Array<F32>, p: Array<F32>) -> PR:
  prow2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, qrow(i, hd, qs, qc, ks, kc, q, k))

# rows i.. of a head's P: r holds rows up to i; left rows more after it. Queries are rows of q (qs apart, the head's
# columns from qc), keys rows of k (ks apart, from kc).
def prows(left: Nat, +i: Nat, +t: Nat, +hd: Nat, +sc: F32, +qs: U32, +qc: U32, +ks: U32, +kc: U32, r: PR) -> QK:
  match left r:
    case 0n PR{q, k, p, n}:
      QK{q, k, p}
    case 1n+m PR{q, k, p, n}:
      prows(m, 1n+i, t, hd, sc, qs, qc, ks, kc, prow(1n+i, t, hd, sc, qs, qc, ks, kc, q, k, p))

# P of one head: queries in q (row stride qs, columns from qc), keys in k (row stride ks, columns from kc)
def probs(+t: Nat, +hd: Nat, +sc: F32, +qs: U32, +qc: U32, +ks: U32, +kc: U32, q: Array<F32>, k: Array<F32>) -> QK:
  prows(Nat.sub(t, 1n), 0n, t, hd, sc, qs, qc, ks, kc, prow(0n, t, hd, sc, qs, qc, ks, kc, q, k, zeros_a(Nat.mul(t, t))))

type PO is Type:
  PO{a: Array<F32>, b: Array<F32>, o: Array<F32>}

# o[i·ow + oc + d] = a[i·t + 0]·b[0·bs + bc + d] + ... + a[i·t + t-1]·b[(t-1)·bs + bc + d], for idx = i·c + d from idx
# on, left more after r (P·V, and dQ = G·K): each sum left to right from +0.0
def pmul(left: Nat, +idx: U32, +c: U32, +t: Nat, +bs: U32, +bc: U32, +ow: U32, +oc: U32, o: Array<F32>, r: D2) -> PO:
  match left r:
    case 0n D2{a, b, +s}:
      PO{a, b, Array.set(F32, o, ((U32.div(idx, c) * ow : U32) + (oc + U32.mod(idx, c) : U32) : U32), s)}
    case 1n+n D2{a, b, +s}:
      pmul(n, (idx + 1 : U32), c, t, bs, bc, ow, oc, Array.set(F32, o, ((U32.div(idx, c) * ow : U32) + (oc + U32.mod(idx, c) : U32) : U32), s), dot0(t, (U32.div((idx + 1 : U32), c) * U32.from_nat(t) : U32), (bc + U32.mod((idx + 1 : U32), c) : U32), 1, bs, a, b))

def acc_set(+idx: U32, +s: F32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{o, +old}:
      Array.set(F32, o, idx, (old + s : F32))

# o[j·c + d] += a[0·t + j]·b[0·bs + bc + d] + ... + a[(t-1)·t + j]·b[(t-1)·bs + bc + d], for idx = j·c + d from idx on
# (dK += Gᵀ·Q and dV += Pᵀ·dO): each sum left to right from +0.0, then added into the group's running sum
def pacc(left: Nat, +idx: U32, +c: U32, +t: Nat, +bs: U32, +bc: U32, o: Array<F32>, r: D2) -> PO:
  match left r:
    case 0n D2{a, b, +s}:
      PO{a, b, acc_set(idx, s, Array.get(F32, o, idx))}
    case 1n+n D2{a, b, +s}:
      pacc(n, (idx + 1 : U32), c, t, bs, bc, acc_set(idx, s, Array.get(F32, o, idx)), dot0(t, U32.div((idx + 1 : U32), c), (bc + U32.mod((idx + 1 : U32), c) : U32), U32.from_nat(t), bs, a, b))

# rows r0.. (t of them, c wide) of src into a new array, src handed back
def rows_out(+t: Nat, +c: Nat, +r0: Nat, src: Array<F32>) -> A2:
  copy_n(Nat.sub(Nat.mul(t, c), 1n), U32.from_nat(Nat.mul(r0, c)), 0, zeros_a(Nat.mul(t, c)), Array.get(F32, src, U32.from_nat(Nat.mul(r0, c))))

def a2_b(r: A2) -> Array<F32>:
  match r:
    case A2{a, b}:
      b

# a block of t rows, c wide, written into dst from row r0; the block is dropped
def rows_in(+t: Nat, +c: Nat, +r0: Nat, blk: Array<F32>, dst: Array<F32>) -> Array<F32>:
  a2_b(copy_n(Nat.sub(Nat.mul(t, c), 1n), 0, U32.from_nat(Nat.mul(r0, c)), dst, Array.get(F32, blk, 0)))

def pmul_copy(left: Nat, +idx: U32, +c: U32, +ow: U32, +oc: U32, o: Array<F32>, r: Array<F32> & F32) -> A2:
  match left r:
    case 0n Tuple{s, +x}:
      A2{s, Array.set(F32, o, ((U32.div(idx, c) * ow : U32) + (oc + U32.mod(idx, c) : U32) : U32), x)}
    case 1n+n Tuple{s, +x}:
      pmul_copy(n, (idx + 1 : U32), c, ow, oc, Array.set(F32, o, ((U32.div(idx, c) * ow : U32) + (oc + U32.mod(idx, c) : U32) : U32), x), Array.get(F32, s, (idx + 1 : U32)))

# dst's columns oc.. (c of them, ow wide) of rows 0..t-1 from src (t x c): src dropped
def cols_in(+t: Nat, +c: Nat, +ow: U32, +oc: U32, src: Array<F32>, dst: Array<F32>) -> Array<F32>:
  a2_b(pmul_copy(Nat.sub(Nat.mul(t, c), 1n), 0, U32.from_nat(c), ow, oc, dst, Array.get(F32, src, 0)))

# t x c numbers of src read from rows r0.. and columns c0.. (src ow wide), written in order into d; idx counts d's cells
def pick_copy(left: Nat, +idx: U32, +c: U32, +ow: U32, +oc: U32, d: Array<F32>, r: Array<F32> & F32) -> A2:
  match left r:
    case 0n Tuple{s, +x}:
      A2{s, Array.set(F32, d, idx, x)}
    case 1n+n Tuple{s, +x}:
      pick_copy(n, (idx + 1 : U32), c, ow, oc, Array.set(F32, d, idx, x), Array.get(F32, s, ((U32.div((idx + 1 : U32), c) * ow : U32) + (oc + U32.mod((idx + 1 : U32), c) : U32) : U32)))

# the block of rows r0.. (t of them) and columns c0.. (c of them) of src (ow wide) into a new t x c array, src handed back
def block_out(+t: Nat, +c: Nat, +ow: Nat, +r0: Nat, +c0: Nat, src: Array<F32>) -> A2:
  pick_copy(Nat.sub(Nat.mul(t, c), 1n), 0, U32.from_nat(c), U32.from_nat(ow), U32.from_nat(Nat.add(Nat.mul(r0, ow), c0)), zeros_a(Nat.mul(t, c)), Array.get(F32, src, U32.from_nat(Nat.add(Nat.mul(r0, ow), c0))))

# a t x c block into dst's rows r0.. and columns c0.. (dst ow wide); the block is dropped
def block_in(+t: Nat, +c: Nat, +ow: Nat, +r0: Nat, +c0: Nat, blk: Array<F32>, dst: Array<F32>) -> Array<F32>:
  cols_in(t, c, U32.from_nat(ow), U32.from_nat(Nat.add(Nat.mul(r0, ow), c0)), blk, dst)

# ---- the forward pass of one sequence: every head's O = P·V into its columns of o ----

type FS is Type:
  FS{q: Array<F32>, k: Array<F32>, v: Array<F32>, o: Array<F32>}

type Dm is Data:
  Dm{t: Nat, g: Nat, hd: Nat, w: Nat, kvw: Nat, sc: F32}

# the columns query head hi reads: its own hd columns of q (hc) and those of its key/value head kv_of(hi, g) in k and v
# (kc); both passes take their columns from here (the law kv_cols)
type HC is Data:
  HC{hc: U32, kc: U32}

def head_cols(+dm: Dm, +hi: Nat) -> HC:
  match dm:
    case Dm{+t, +g, +hd, +w, +kvw, +sc}:
      HC{U32.from_nat(Nat.mul(hi, hd)), U32.from_nat(Nat.mul(kv_of(hi, g), hd))}

def hc_kc(+c: HC) -> U32:
  match c:
    case HC{+hc, +kc}:
      kc

def fs_fin(q: Array<F32>, k: Array<F32>, r: PO) -> FS:
  match r:
    case PO{p, v, o}:
      FS{q, k, v, o}

def head_f2(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, v: Array<F32>, o: Array<F32>, r: QK) -> FS:
  match r:
    case QK{q, k, p}:
      fs_fin(q, k, pmul(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(kvw), kc, U32.from_nat(w), hc, o, dot0(t, 0, kc, 1, U32.from_nat(kvw), p, v)))

# one head's forward pass at the columns hc (its queries) and kc (its key/value head) that head_cols gives; it has no
# head index, so it reads no other columns
def head_fwd(+dm: Dm, +cols: HC, s: FS) -> FS:
  match dm cols s:
    case Dm{+t, +g, +hd, +w, +kvw, +sc} HC{+hc, +kc} FS{q, k, v, o}:
      head_f2(t, hd, w, kvw, hc, kc, v, o, probs(t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, q, k))

def fheads(left: Nat, +hi: Nat, +dm: Dm, s: FS) -> FS:
  match left:
    case 0n:
      s
    case 1n+n:
      fheads(n, 1n+hi, dm, head_fwd(dm, head_cols(dm, hi), s))

def fs_o(s: FS) -> Array<F32>:
  match s:
    case FS{q, k, v, o}:
      o

# ---- the backward pass of one sequence: per key/value head, its g query heads' dQ into their columns of dq, their dK
# and dV summed in head order (from +0.0), then copied into the head's columns ----

type BS is Type:
  BS{q: Array<F32>, k: Array<F32>, v: Array<F32>, d: Array<F32>, dq: Array<F32>, dka: Array<F32>, dva: Array<F32>}

# g[j] = (p[j] * (g[j] - s)) * sc for j = idx..idx + k; rg and rp hold g[idx] and p[idx]
def gfix(k: Nat, +idx: U32, +s: F32, +sc: F32, rg: Array<F32> & F32, rp: Array<F32> & F32) -> A2:
  match k rg rp:
    case 0n Tuple{g, +d} Tuple{p, +y}:
      A2{p, Array.set(F32, g, idx, ((y * (d - s : F32) : F32) * sc : F32))}
    case 1n+q Tuple{g, +d} Tuple{p, +y}:
      gfix(q, (idx + 1 : U32), s, sc, Array.get(F32, Array.set(F32, g, idx, ((y * (d - s : F32) : F32) * sc : F32)), (idx + 1 : U32)), Array.get(F32, p, (idx + 1 : U32)))

def g_set2(+t: Nat, +sc: F32, +off: U32, r: D2) -> A2:
  match r:
    case D2{g, p, +s}:
      gfix(Nat.sub(t, 1n), off, s, sc, Array.get(F32, g, off), Array.get(F32, p, off))

type GW is Type:
  GW{d: Array<F32>, v: Array<F32>, p: Array<F32>, g: Array<F32>}

type GN is Type:
  GN{g: Array<F32>, n: Nat}

# xs written into g from idx, counted onto n
def gw(xs: List<&2, F32>, +idx: U32, n: Nat, g: Array<F32>) -> GN:
  match xs:
    case Nil{}:
      GN{g, n}
    case Con{x, t}:
      gw(t, (idx + 1 : U32), 1n+n, Array.set(F32, g, idx, x))

type GR is Type:
  GR{d: Array<F32>, v: Array<F32>, pg: A2, n: Nat}

def gr_n(r: GR) -> Nat:
  match r:
    case GR{d, v, pg, n}:
      n

# the n positions of the row written: s = sum(dP * P) over them (from +0.0, in order), then G = (P * (dP - s)) * scale
def gr3(+sc: F32, +off: U32, d: Array<F32>, v: Array<F32>, p: Array<F32>, w: GN) -> GR:
  match w:
    case GN{g, +n}:
      GR{d, v, g_set2(n, sc, off, dot0(n, off, off, 1, 1, g, p)), n}

def gr2(+sc: F32, +off: U32, p: Array<F32>, g: Array<F32>, r: DR) -> GR:
  match r:
    case DR{d, v, row}:
      gr3(sc, off, d, v, p, gw(row, off, 0n, g))

# row i of G (from i·t, t wide; softmax's backward pass, then the scale): dP's row against values 0..i only (drow, as
# qrow reads keys 0..i), written from the row's start, and s and G over as many positions as were written: i + 1 (the
# law causal_grad). G starts as zeros and nothing else writes the row, so its positions after i stay 0: no gradient
# reaches a key or value after the query's position. (P is 0 there too, so the full row gave the same G.)
def grow(+i: Nat, +t: Nat, +hd: Nat, +sc: F32, +ds: U32, +dc: U32, +vs: U32, +vc: U32, d: Array<F32>, v: Array<F32>, p: Array<F32>, g: Array<F32>) -> GR:
  gr2(sc, (U32.from_nat(i) * U32.from_nat(t) : U32), p, g, drow(i, hd, ((U32.from_nat(i) * ds : U32) + dc : U32), vc, vs, Nil{}, dot0(hd, ((U32.from_nat(i) * ds : U32) + dc : U32), vc, 1, 1, d, v)))

# rows i.. of G; r is row i
def grows(left: Nat, +i: Nat, +t: Nat, +hd: Nat, +sc: F32, +ds: U32, +dc: U32, +vs: U32, +vc: U32, r: GR) -> GW:
  match left r:
    case 0n GR{d, v, A2{p, g}, n}:
      GW{d, v, p, g}
    case 1n+m GR{d, v, A2{p, g}, n}:
      grows(m, 1n+i, t, hd, sc, ds, dc, vs, vc, grow(1n+i, t, hd, sc, ds, dc, vs, vc, d, v, p, g))

def hb5(q: Array<F32>, k: Array<F32>, v: Array<F32>, dq: Array<F32>, dka: Array<F32>, r: PO) -> BS:
  match r:
    case PO{p, d, dva}:
      BS{q, k, v, d, dq, dka, dva}

# dK += Gᵀ·Q, then dV += Pᵀ·dO
def hb4(+t: Nat, +hd: Nat, +w: Nat, +hc: U32, k: Array<F32>, d: Array<F32>, v: Array<F32>, p: Array<F32>, dq: Array<F32>, dva: Array<F32>, r: PO) -> BS:
  match r:
    case PO{g, q, dka}:
      hb5(q, k, v, dq, dka, pacc(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(w), hc, dva, dot0(t, 0, hc, U32.from_nat(t), U32.from_nat(w), p, d)))

# dQ = G·K into the head's columns, then dK
def hb3(+t: Nat, +hd: Nat, +w: Nat, +hc: U32, q: Array<F32>, d: Array<F32>, v: Array<F32>, p: Array<F32>, dka: Array<F32>, dva: Array<F32>, r: PO) -> BS:
  match r:
    case PO{g, k, dq}:
      hb4(t, hd, w, hc, k, d, v, p, dq, dva, pacc(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(w), hc, dka, dot0(t, 0, hc, U32.from_nat(t), U32.from_nat(w), g, q)))

def hb2(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, q: Array<F32>, k: Array<F32>, dq: Array<F32>, dka: Array<F32>, dva: Array<F32>, r: GW) -> BS:
  match r:
    case GW{d, v, p, g}:
      hb3(t, hd, w, hc, q, d, v, p, dka, dva, pmul(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(kvw), kc, U32.from_nat(w), hc, dq, dot0(t, 0, kc, 1, U32.from_nat(kvw), g, k)))

# P again, then G
def hb1(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +sc: F32, +hc: U32, +kc: U32, v: Array<F32>, d: Array<F32>, dq: Array<F32>, dka: Array<F32>, dva: Array<F32>, r: QK) -> BS:
  match r:
    case QK{q, k, p}:
      hb2(t, hd, w, kvw, hc, kc, q, k, dq, dka, dva, grows(Nat.sub(t, 1n), 0n, t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, grow(0n, t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, d, v, p, zeros_a(Nat.mul(t, t)))))

# one head's backward pass at head_cols' columns (no head index, as head_fwd)
def head_bwd(+dm: Dm, +cols: HC, s: BS) -> BS:
  match dm cols s:
    case Dm{+t, +g, +hd, +w, +kvw, +sc} HC{+hc, +kc} BS{q, k, v, d, dq, dka, dva}:
      hb1(t, hd, w, kvw, sc, hc, kc, v, d, dq, dka, dva, probs(t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, q, k))

def bheads(left: Nat, +hi: Nat, +dm: Dm, s: BS) -> BS:
  match left:
    case 0n:
      s
    case 1n+n:
      bheads(n, 1n+hi, dm, head_bwd(dm, head_cols(dm, hi), s))

type BG is Type:
  BG{q: Array<F32>, k: Array<F32>, v: Array<F32>, d: Array<F32>, dq: Array<F32>, dk: Array<F32>, dv: Array<F32>}

# the group's summed dK and dV into key/value head j's columns of dk and dv
def bg_fin(+t: Nat, +hd: Nat, +kvw: Nat, +j: Nat, dk: Array<F32>, dv: Array<F32>, s: BS) -> BG:
  match s:
    case BS{q, k, v, d, dq, dka, dva}:
      BG{q, k, v, d, dq, cols_in(t, hd, U32.from_nat(kvw), U32.from_nat(Nat.mul(j, hd)), dka, dk), cols_in(t, hd, U32.from_nat(kvw), U32.from_nat(Nat.mul(j, hd)), dva, dv)}

def bgroup(+dm: Dm, +j: Nat, s: BG) -> BG:
  match dm s:
    case Dm{+t, +g, +hd, +w, +kvw, +sc} BG{q, k, v, d, dq, dk, dv}:
      bg_fin(t, hd, kvw, j, dk, dv, bheads(g, Nat.mul(j, g), dm, BS{q, k, v, d, dq, zeros_a(Nat.mul(t, hd)), zeros_a(Nat.mul(t, hd))}))

def bgroups(left: Nat, +j: Nat, +dm: Dm, s: BG) -> BG:
  match left:
    case 0n:
      s
    case 1n+n:
      bgroups(n, 1n+j, dm, bgroup(dm, j, s))

# ---- the sequences: each one's rows copied out into arrays of its own, the sequences computed in a tree of parallel
# lets (each leaf holds its own arrays, nothing shared), the results written back in order ----

type Halves<-A: Type> is Type:
  Halves{l: List<&1, A>, r: List<&1, A>}

# the two halves back as one list, in order (what the laws attn_par_fwd and attn_par_bwd compare the trees with)
def hcat(-A: Type, h: Halves<A>) -> List<&1, A>:
  match h:
    case Halves{l, r}:
      List.append(&1, A, l, r)

# xs cut after its first n (a tail loop: a helper in the fork tree must not hold the stack)
def cut(-A: Type, n: Nat, xs: List<&1, A>, acc: List<&1, A>) -> Halves<A>:
  match n xs:
    case 1n+m Con{x, xt}:
      cut(A, m, xt, x <> acc)
    case _ _:
      Halves{List.reverse(&1, A, acc), xs}

type FL is Type:
  FL{q: Array<F32>, k: Array<F32>, v: Array<F32>, xs: List<&1, FS>}

# a unit of the parallel work is one sequence's key/value group j: its g query heads' columns of Q (t x g·hd), head j's
# columns of K and V (t x hd), and an output of its own (t x g·hd). Inside a unit the heads keep their order and their
# arithmetic (head_cols with j = 0: the law kv_cols), so the units give the bits of the whole sequence computed at once.
def fu3(+t: Nat, +gw: Nat, qq: Array<F32>, kk: Array<F32>, xs: List<&1, FS>, q: Array<F32>, k: Array<F32>, r: A2) -> FL:
  match r:
    case A2{vv, v}:
      FL{qq, kk, vv, FS{q, k, v, zeros_a(Nat.mul(t, gw))} <> xs}

def fu2(+t: Nat, +hd: Nat, +kvw: Nat, +r0: Nat, +kc: Nat, +gw: Nat, qq: Array<F32>, vv: Array<F32>, xs: List<&1, FS>, q: Array<F32>, r: A2) -> FL:
  match r:
    case A2{kk, k}:
      fu3(t, gw, qq, kk, xs, q, k, block_out(t, hd, kvw, r0, kc, vv))

def fu1(+t: Nat, +hd: Nat, +kvw: Nat, +r0: Nat, +kc: Nat, +gw: Nat, kk: Array<F32>, vv: Array<F32>, xs: List<&1, FS>, r: A2) -> FL:
  match r:
    case A2{qq, q}:
      fu2(t, hd, kvw, r0, kc, gw, qq, vv, xs, q, block_out(t, hd, kvw, r0, kc, kk))

def funit(+t: Nat, +w: Nat, +kvw: Nat, +g: Nat, +hd: Nat, +bi: Nat, +j: Nat, s: FL) -> FL:
  match s:
    case FL{qq, kk, vv, xs}:
      fu1(t, hd, kvw, Nat.mul(bi, t), Nat.mul(j, hd), Nat.mul(g, hd), kk, vv, xs, block_out(t, Nat.mul(g, hd), w, Nat.mul(bi, t), Nat.mul(Nat.mul(j, g), hd), qq))

def fgroups(left: Nat, +j: Nat, +bi: Nat, +t: Nat, +w: Nat, +kvw: Nat, +g: Nat, +hd: Nat, s: FL) -> FL:
  match left:
    case 0n:
      s
    case 1n+n:
      fgroups(n, 1n+j, bi, t, w, kvw, g, hd, funit(t, w, kvw, g, hd, bi, j, s))

# every sequence's units, sequence by sequence and group by group, newest first
def fslices(left: Nat, +bi: Nat, +t: Nat, +w: Nat, +kvw: Nat, +kv: Nat, +g: Nat, +hd: Nat, s: FL) -> FL:
  match left:
    case 0n:
      s
    case 1n+n:
      fslices(n, 1n+bi, t, w, kvw, kv, g, hd, fgroups(kv, 0n, bi, t, w, kvw, g, hd, s))

def fleaf(xs: List<&1, FS>, +h: Nat, +dm: Dm, acc: List<&1, Array<F32>>) -> List<&1, Array<F32>>:
  match xs:
    case Nil{}:
      List.reverse(&1, Array<F32>, acc)
    case Con{x, rest}:
      fleaf(rest, h, dm, fs_o(fheads(h, 0n, dm, x)) <> acc)

# the n sequences of gs (cut in two halves) in a tree of parallel lets of depth d; the outputs come back in order
def fpar(d: Nat, gs: Halves<FS>, +n: Nat, +h: Nat, +dm: Dm) -> List<&1, Array<F32>>:
  match d gs:
    case 0n Halves{l, r}:
      a b = fleaf(l, h, dm, Nil{}) fleaf(r, h, dm, Nil{})
      List.append(&1, Array<F32>, a, b)
    case 1n++e Halves{l, r}:
      a b = fpar(e, cut(FS, TA.half(TA.half(n)), l, Nil{}), TA.half(n), h, dm) fpar(e, cut(FS, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), h, dm)
      List.append(&1, Array<F32>, a, b)

type UO is Type:
  UO{xs: List<&1, Array<F32>>, o: Array<F32>}

# one sequence's units (t x gw each) into o's rows r0.. at columns 0, gw, 2·gw, ...
def ub_row(left: Nat, +j: Nat, +t: Nat, +gw: Nat, +w: Nat, +r0: Nat, r: UO) -> UO:
  match left r:
    case 1n+n UO{Con{x, rest}, o}:
      ub_row(n, 1n+j, t, gw, w, r0, UO{rest, block_in(t, gw, w, r0, Nat.mul(j, gw), x, o)})
    case _ r:
      r

# the units' outputs into o (n x w), in the order fslices made them
def blocks_in(left: Nat, +bi: Nat, +t: Nat, +gw: Nat, +w: Nat, +kv: Nat, r: UO) -> Array<F32>:
  match left r:
    case 0n UO{xs, o}:
      o
    case 1n+n r:
      blocks_in(n, 1n+bi, t, gw, w, kv, ub_row(kv, 0n, t, gw, w, Nat.mul(bi, t), r))

type BL is Type:
  BL{q: Array<F32>, k: Array<F32>, v: Array<F32>, d: Array<F32>, xs: List<&1, BG>}

# the backward pass's unit: one sequence's group j, as fslices' (Q, K, V, and dO's columns of the g heads), with
# dQ, dK and dV of its own
def bu4(+t: Nat, +gw: Nat, +hd: Nat, qq: Array<F32>, kk: Array<F32>, vv: Array<F32>, xs: List<&1, BG>, q: Array<F32>, k: Array<F32>, v: Array<F32>, r: A2) -> BL:
  match r:
    case A2{dd, d}:
      BL{qq, kk, vv, dd, BG{q, k, v, d, zeros_a(Nat.mul(t, gw)), zeros_a(Nat.mul(t, hd)), zeros_a(Nat.mul(t, hd))} <> xs}

def bu3(+t: Nat, +w: Nat, +hd: Nat, +r0: Nat, +qc: Nat, +gw: Nat, qq: Array<F32>, kk: Array<F32>, dd: Array<F32>, xs: List<&1, BG>, q: Array<F32>, k: Array<F32>, r: A2) -> BL:
  match r:
    case A2{vv, v}:
      bu4(t, gw, hd, qq, kk, vv, xs, q, k, v, block_out(t, gw, w, r0, qc, dd))

def bu2(+t: Nat, +w: Nat, +hd: Nat, +kvw: Nat, +r0: Nat, +qc: Nat, +kc: Nat, +gw: Nat, qq: Array<F32>, vv: Array<F32>, dd: Array<F32>, xs: List<&1, BG>, q: Array<F32>, r: A2) -> BL:
  match r:
    case A2{kk, k}:
      bu3(t, w, hd, r0, qc, gw, qq, kk, dd, xs, q, k, block_out(t, hd, kvw, r0, kc, vv))

def bu1(+t: Nat, +w: Nat, +hd: Nat, +kvw: Nat, +r0: Nat, +qc: Nat, +kc: Nat, +gw: Nat, kk: Array<F32>, vv: Array<F32>, dd: Array<F32>, xs: List<&1, BG>, r: A2) -> BL:
  match r:
    case A2{qq, q}:
      bu2(t, w, hd, kvw, r0, qc, kc, gw, qq, vv, dd, xs, q, block_out(t, hd, kvw, r0, kc, kk))

def bunit(+t: Nat, +w: Nat, +kvw: Nat, +g: Nat, +hd: Nat, +bi: Nat, +j: Nat, s: BL) -> BL:
  match s:
    case BL{qq, kk, vv, dd, xs}:
      bu1(t, w, hd, kvw, Nat.mul(bi, t), Nat.mul(Nat.mul(j, g), hd), Nat.mul(j, hd), Nat.mul(g, hd), kk, vv, dd, xs, block_out(t, Nat.mul(g, hd), w, Nat.mul(bi, t), Nat.mul(Nat.mul(j, g), hd), qq))

def bgroups_u(left: Nat, +j: Nat, +bi: Nat, +t: Nat, +w: Nat, +kvw: Nat, +g: Nat, +hd: Nat, s: BL) -> BL:
  match left:
    case 0n:
      s
    case 1n+n:
      bgroups_u(n, 1n+j, bi, t, w, kvw, g, hd, bunit(t, w, kvw, g, hd, bi, j, s))

def bslices(left: Nat, +bi: Nat, +t: Nat, +w: Nat, +kvw: Nat, +kv: Nat, +g: Nat, +hd: Nat, s: BL) -> BL:
  match left:
    case 0n:
      s
    case 1n+n:
      bslices(n, 1n+bi, t, w, kvw, kv, g, hd, bgroups_u(kv, 0n, bi, t, w, kvw, g, hd, s))

type DB is Type:
  DB{dq: Array<F32>, dk: Array<F32>, dv: Array<F32>}

def db_of(s: BG) -> DB:
  match s:
    case BG{q, k, v, d, dq, dk, dv}:
      DB{dq, dk, dv}

def bleaf(xs: List<&1, BG>, +kv: Nat, +dm: Dm, acc: List<&1, DB>) -> List<&1, DB>:
  match xs:
    case Nil{}:
      List.reverse(&1, DB, acc)
    case Con{x, rest}:
      bleaf(rest, kv, dm, db_of(bgroups(kv, 0n, dm, x)) <> acc)

def bpar(d: Nat, gs: Halves<BG>, +n: Nat, +kv: Nat, +dm: Dm) -> List<&1, DB>:
  match d gs:
    case 0n Halves{l, r}:
      a b = bleaf(l, kv, dm, Nil{}) bleaf(r, kv, dm, Nil{})
      List.append(&1, DB, a, b)
    case 1n++e Halves{l, r}:
      a b = bpar(e, cut(BG, TA.half(TA.half(n)), l, Nil{}), TA.half(n), kv, dm) bpar(e, cut(BG, TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), kv, dm)
      List.append(&1, DB, a, b)

type D3 is Type:
  D3{dq: Array<F32>, dk: Array<F32>, dv: Array<F32>}

type UD is Type:
  UD{xs: List<&1, DB>, o: D3}

# one sequence's units into rows r0..: dQ at columns j·gw (w wide), dK and dV at j·hd (kvw wide)
def ud_row(left: Nat, +j: Nat, +t: Nat, +gw: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +r0: Nat, r: UD) -> UD:
  match left r:
    case 1n+n UD{Con{DB{bq, bk, bv}, rest}, D3{dq, dk, dv}}:
      ud_row(n, 1n+j, t, gw, hd, w, kvw, r0, UD{rest, D3{block_in(t, gw, w, r0, Nat.mul(j, gw), bq, dq), block_in(t, hd, kvw, r0, Nat.mul(j, hd), bk, dk), block_in(t, hd, kvw, r0, Nat.mul(j, hd), bv, dv)}})
    case _ r:
      r

def dblocks_in(left: Nat, +bi: Nat, +t: Nat, +gw: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +kv: Nat, r: UD) -> D3:
  match left r:
    case 0n UD{xs, o}:
      o
    case 1n+n r:
      dblocks_in(n, 1n+bi, t, gw, hd, w, kvw, kv, ud_row(kv, 0n, t, gw, hd, w, kvw, Nat.mul(bi, t), r))

# ---- the layer ----

type AttnDims is Data:
  AttnDims{b: Nat, t: Nat, kv: Nat, g: Nat, hd: Nat, sc: F32}

# a unit's dimensions: g query heads (g·hd columns) over one key/value head (hd columns)
def dm_of(+dm: AttnDims) -> Dm:
  match dm:
    case AttnDims{+b, +t, +kv, +g, +hd, +sc}:
      Dm{t, g, hd, Nat.mul(g, hd), hd, sc}

# whether the units (a sequence's key/value group each, fslices) run in the tree: from 8 units on, or from 2 when each is
# big. With the tiny Llama's sequences (t = 16, w = 32) the tree cost more than it gave at 2 sequences (0.48 against
# 0.37 ms on 16 threads) and gained from 8 on (1.32 against 1.52 ms; 1.5x at 32, 1.7x at 128); with SmolLM2's (t = 128,
# w = 576, about half a second of a step each) the two sequences one after another kept a single thread busy for a
# third of the step (NOTES, 2026-10-21), and its 2 sequences x 3 groups now give 6 units. t w counts in U32 (a Nat
# comparison walks its numbers one unit at a time). The answer never changes the result: the laws attn_par_fwd and
# attn_par_bwd show the tree gives the units in order, as one after another.
def attn_tree(+b: Nat, +t: Nat, +w: Nat) -> Bool:
  Bool.or(Nat.is_le(8n, b), Bool.and(Nat.is_le(2n, b), U32.is_le(4096, (U32.from_nat(t) * U32.from_nat(w) : U32))))

def fgo(big: Bool, +b: Nat, +h: Nat, +dm: Dm, xs: List<&1, FS>) -> List<&1, Array<F32>>:
  match big:
    case True{}:
      fpar(Nat.sub(depth(b), 1n), cut(FS, TA.half(b), xs, Nil{}), b, h, dm)
    case False{}:
      fleaf(xs, h, dm, Nil{})

def bgo(big: Bool, +b: Nat, +kv: Nat, +dm: Dm, xs: List<&1, BG>) -> List<&1, DB>:
  match big:
    case True{}:
      bpar(Nat.sub(depth(b), 1n), cut(BG, TA.half(b), xs, Nil{}), b, kv, dm)
    case False{}:
      bleaf(xs, kv, dm, Nil{})

def af2(-n: Nat, +w: Nat, +b: Nat, +kv: Nat, +dm: Dm, s: FL) -> Fwd<Unit, TA.Mat<n, w>, AttnS>:
  match dm s:
    case Dm{+t, +g, +hd, +gw, +kw, +sc} FL{q, k, v, xs}:
      Fwd{Unit{}, TA.Mat{blocks_in(b, 0n, t, gw, w, kv, UO{fgo(attn_tree(Nat.mul(b, kv), t, gw), Nat.mul(b, kv), g, dm, List.reverse(&1, FS, xs)), zeros_a(Nat.mul(Nat.mul(b, t), w))})}, AttnS{q, k, v}}

def attn_fwd(+n: Nat, +w: Nat, +kvw: Nat, +dm: AttnDims, x: Two<TA.Mat<n, w>, Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>) -> Fwd<Unit, TA.Mat<n, w>, AttnS>:
  match dm x:
    case AttnDims{+b, +t, +kv, +g, +hd, +sc} Two{TA.Mat{q}, Two{TA.Mat{k}, TA.Mat{v}}}:
      af2(n, w, b, kv, dm_of(dm), fslices(b, 0n, t, w, kvw, kv, g, hd, FL{q, k, v, Nil{}}))

def ab3(-n: Nat, -w: Nat, -kvw: Nat, o: D3) -> Bwd<Unit, Two<TA.Mat<n, w>, Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>>:
  match o:
    case D3{dq, dk, dv}:
      Bwd{Unit{}, Two{TA.Mat{dq}, Two{TA.Mat{dk}, TA.Mat{dv}}}, Unit{}}

def ab2(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +kv: Nat, +dm: Dm, s: BL) -> Bwd<Unit, Two<TA.Mat<n, w>, Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>>:
  match dm s:
    case Dm{+t, +g, +hd, +gw, +kw, +sc} BL{q, k, v, d, xs}:
      ab3(n, w, kvw, dblocks_in(b, 0n, t, gw, hd, w, kvw, kv, UD{bgo(attn_tree(Nat.mul(b, kv), t, gw), Nat.mul(b, kv), 1n, dm, List.reverse(&1, BG, xs)), D3{zeros_a(Nat.mul(n, w)), zeros_a(Nat.mul(n, kvw)), zeros_a(Nat.mul(n, kvw))}}))

def attn_bwd(+n: Nat, +w: Nat, +kvw: Nat, +dm: AttnDims, s: AttnS, dy: TA.Mat<n, w>) -> Bwd<Unit, Two<TA.Mat<n, w>, Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>>:
  match dm s dy:
    case AttnDims{+b, +t, +kv, +g, +hd, +sc} AttnS{q, k, v} TA.Mat{d}:
      ab2(n, w, kvw, b, kv, dm_of(dm), bslices(b, 0n, t, w, kvw, kv, g, hd, BL{q, k, v, d, Nil{}}))

# the core of attention, after the projections and RoPE; the erased arguments are the shape facts it relies on
def Attention(+n: Nat, +w: Nat, +kvw: Nat, +b: Nat, +t: Nat, +kv: Nat, +g: Nat, +hd: Nat, +sc: F32, -e1: {n == Nat.mul(b, t) : Nat}, -e2: {w == Nat.mul(Nat.mul(kv, g), hd) : Nat}, -e3: {kvw == Nat.mul(kv, hd) : Nat}) -> Layer<Unit, Two<TA.Mat<n, w>, Two<TA.Mat<n, kvw>, TA.Mat<n, kvw>>>, TA.Mat<n, w>, AttnS>:
  Layer{p => x => attn_fwd(n, w, kvw, AttnDims{b, t, kv, g, hd, sc}, x), p => s => dy => g2 => attn_bwd(n, w, kvw, AttnDims{b, t, kv, g, hd, sc}, s, dy), op => p => g2 => st => no_step(Unit, op, p, g2, st), u => Unit{}, p => PL{p, Nil{}}, p => xs => off => PA{p, xs, off}, p => xs => off => PA{p, xs, off}, u => 0}

# =====================================================================================================
# Losses over arrays (cross-entropy, MSE, NLL)
# =====================================================================================================

# ---- cross-entropy of the logits z (n x c) against one label per row, mean over the rows (PyTorch's F.cross_entropy,
# reduction="mean"); its gradient is (softmax(z) - onehot) / n. Over z's own array, a row at a time, with
# bend-ml-tensor-array's Mat.softmax_ce's operations in its order, so its bits. ----

def hit(b: Bool) -> F32:
  match b:
    case True{}:
      1.0
    case False{}:
      0.0

# the sum of a[0] .. a[m-1] as a balanced tree of additions (pairwise summation), in place: each pass adds neighbours
# (a[i] = a[2i] + a[2i+1], an odd last one carried), until one number is left. Its rounding error grows with log2(m),
# not m, as in PyTorch's cascade sum: a running sum of 295k squares was 1e-4 off (NOTES, 2026-10-17)
def ts_add(+i: U32, +x: F32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{a, +y}:
      Array.set(F32, a, i, (x + y : F32))

def ts_pair(+i: U32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{a, +x}:
      ts_add(i, x, Array.get(F32, a, ((i + i : U32) + 1 : U32)))

def ts_pairs(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n:
      a
    case 1n+q:
      ts_pairs(q, (i + 1 : U32), ts_pair(i, Array.get(F32, a, (i + i : U32))))

def ts_put(+i: U32, r: Array<F32> & F32) -> Array<F32>:
  match r:
    case Tuple{a, +x}:
      Array.set(F32, a, i, x)

def ts_carry(odd: Bool, +m: U32, a: Array<F32>) -> Array<F32>:
  match odd:
    case True{}:
      ts_put(U32.div(m, 2), Array.get(F32, a, (m - 1 : U32)))
    case False{}:
      a

def ts_pass(+m: U32, a: Array<F32>) -> Array<F32>:
  ts_carry(U32.is_eq(U32.mod(m, 2), 1), m, ts_pairs(U32.to_nat(U32.div(m, 2)), 0, a))

def ts_first(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{a, x}:
      x

def ts_go(f: Nat, done: Bool, +m: U32, a: Array<F32>) -> F32:
  match f done:
    case _ True{}:
      ts_first(Array.get(F32, a, 0))
    case 0n False{}:
      ts_first(Array.get(F32, a, 0))
    case 1n+g False{}:
      ts_go(g, U32.is_le(U32.div((m + 1 : U32), 2), 1), U32.div((m + 1 : U32), 2), ts_pass(m, a))

def tree_sum(+m: U32, a: Array<F32>) -> F32:
  ts_go(32n, U32.is_le(m, 1), m, a)

# Cross-entropy's softmax over a row of c classes: as psm, but the row's exponentials are summed as a balanced tree
# (tree_sum, on a copy), since a running F32 sum of SmolLM2's 49152 classes was 3.4 tolerances off PyTorch's loss
# (NOTES, M10). psm itself keeps its order: attention's rows are short and the law causal reads its count.
# a row's probabilities, its max m and log of its sum of exp(x - m) (log_softmax's pieces)
type CSM is Type:
  CSM{a: Array<F32>, m: F32, ls: F32}

def cesm4(+k: Nat, +off: U32, +m: F32, +s: F32, a: Array<F32>) -> CSM:
  CSM{pc_a(pdiv(k, off, s, 0n, Array.get(F32, a, off))), m, F32.log(s)}

def cesm3(+k: Nat, +off: U32, +m: F32, r: A2) -> CSM:
  match r:
    case A2{a, sc}:
      cesm4(k, off, m, tree_sum(U32.from_nat(1n+k), sc), a)

def cesm_e(+k: Nat, +off: U32, +m: F32, r: AS) -> CSM:
  match r:
    case AS{a, s}:
      cesm3(k, off, m, copy_n(k, off, 0, Array.new(F32, depth(1n+k), 0.0), Array.get(F32, a, off)))

def cesm2(+k: Nat, +off: U32, r: AS) -> CSM:
  match r:
    case AS{a, +m}:
      cesm_e(k, off, m, pexp(k, off, m, 0.0, Array.get(F32, a, off)))

def ce_sm(+c: Nat, +off: U32, z: Array<F32>) -> CSM:
  cesm2(Nat.sub(c, 1n), off, pmax0(Nat.sub(c, 1n), off, Array.get(F32, z, off)))

# z[idx] = (p - [j == y]) / nf for j = jj..jj + k, p the row's probabilities
def ce_g(k: Nat, +idx: U32, +j: U32, +y: U32, +nf: F32, r: Array<F32> & F32) -> Array<F32>:
  match k r:
    case 0n Tuple{z, +x}:
      Array.set(F32, z, idx, ((x - hit(U32.is_eq(j, y)) : F32) / nf : F32))
    case 1n+q Tuple{z, +x}:
      ce_g(q, (idx + 1 : U32), (j + 1 : U32), y, nf, Array.get(F32, Array.set(F32, z, idx, ((x - hit(U32.is_eq(j, y)) : F32) / nf : F32)), (idx + 1 : U32)))

# loss - log_softmax(x)[y], log_softmax as PyTorch computes it: (x[y] - m) - log(sum exp(x - m)) (since 2026-10-18; it
# was loss - log(max(p[y], 1e-12)), which capped a row's loss at 27.63 where PyTorch's grows, NOTES)
def ce_pick2(+xy: F32, loss: F32, r: CSM) -> AS:
  match r:
    case CSM{z, +m, +ls}:
      AS{z, (loss - ((xy - m : F32) - ls : F32) : F32)}

def ce_pick3(+c: Nat, +off: U32, loss: F32, r: Array<F32> & F32) -> AS:
  match r:
    case Tuple{z, +xy}:
      ce_pick2(xy, loss, ce_sm(c, off, z))

def ce_bad(loss: F32, r: CSM) -> AS:
  match r:
    case CSM{z, m, ls}:
      AS{z, (loss - F32.log(F32.max(0.0, 0.000000000001)) : F32)}

# a label past the last class picks probability 0.0, clamped at 1e-12, as tensor-array's pick
def ce_pick(ok: Bool, +c: Nat, +off: U32, +idx: U32, loss: F32, z: Array<F32>) -> AS:
  match ok:
    case True{}:
      ce_pick3(c, off, loss, Array.get(F32, z, idx))
    case False{}:
      ce_bad(loss, ce_sm(c, off, z))

def ce_row2(+c: Nat, +off: U32, +y: Nat, +nf: F32, r: AS) -> AS:
  match r:
    case AS{z, +loss}:
      AS{ce_g(Nat.sub(c, 1n), off, 0, U32.from_nat(y), nf, Array.get(F32, z, off)), loss}

def ce_row(+c: Nat, +off: U32, +y: Nat, +nf: F32, s: AS) -> AS:
  match s:
    case AS{z, loss}:
      ce_row2(c, off, y, nf, ce_pick(Nat.is_le(1n+y, c), c, off, (off + U32.from_nat(y) : U32), loss, z))

def ce_rows(labels: List<&2, Nat>, +c: Nat, +i: U32, +nf: F32, s: AS) -> AS:
  match labels:
    case Nil{}:
      s
    case Con{+y, t}:
      ce_rows(t, c, (i + 1 : U32), nf, ce_row(c, (i * U32.from_nat(c) : U32), y, nf, s))

def ce_fin(-n: Nat, -c: Nat, +nf: F32, r: AS) -> LossOut<TA.Mat<n, c>>:
  match r:
    case AS{d, +l}:
      LossOut{(l / nf : F32), TA.Mat{d}}

def CrossEntropy(+n: Nat, +c: Nat, z: TA.Mat<n, c>, labels: List<&2, Nat>) -> LossOut<TA.Mat<n, c>>:
  match z:
    case TA.Mat{d}:
      ce_fin(n, c, F32.from_nat(n), ce_rows(labels, c, 0, F32.from_nat(n), AS{d, 0.0}))


type D3S is Type:
  D3S{y: Array<F32>, t: Array<F32>, q: Array<F32>}

# over every number from idx on (k more): d = y - t, q[idx] = d·d, y[idx] = d * kk
def mse_l(k: Nat, +idx: U32, +kk: F32, q: Array<F32>, ry: Array<F32> & F32, rt: Array<F32> & F32) -> D3S:
  match k ry rt:
    case 0n Tuple{y, +a} Tuple{t, +b}:
      D3S{Array.set(F32, y, idx, ((a - b : F32) * kk : F32)), t, Array.set(F32, q, idx, ((a - b : F32) * (a - b : F32) : F32))}
    case 1n+n Tuple{y, +a} Tuple{t, +b}:
      mse_l(n, (idx + 1 : U32), kk, Array.set(F32, q, idx, ((a - b : F32) * (a - b : F32) : F32)), Array.get(F32, Array.set(F32, y, idx, ((a - b : F32) * kk : F32)), (idx + 1 : U32)), Array.get(F32, t, (idx + 1 : U32)))

def mse_fin(-n: Nat, -c: Nat, +m: U32, +nc: F32, r: D3S) -> LossOut<TA.Mat<n, c>>:
  match r:
    case D3S{y, t, q}:
      LossOut{(tree_sum(m, q) / nc : F32), TA.Mat{y}}

# mean squared error over every number (PyTorch's F.mse_loss); its gradient is 2 (y - t) / (n*c); the squares summed
# as a tree
def MSE(+n: Nat, +c: Nat, y: TA.Mat<n, c>, t: TA.Mat<n, c>) -> LossOut<TA.Mat<n, c>>:
  match y t:
    case TA.Mat{ya} TA.Mat{ta}:
      mse_fin(n, c, U32.from_nat(Nat.mul(n, c)), F32.from_nat(Nat.mul(n, c)), mse_l(Nat.sub(Nat.mul(n, c), 1n), 0, (2.0 / F32.from_nat(Nat.mul(n, c)) : F32), zeros_a(Nat.mul(n, c)), Array.get(F32, ya, 0), Array.get(F32, ta, 0)))

type NS is Type:
  NS{lp: Array<F32>, g: Array<F32>, s: F32}

def nll_hit(+idx: U32, +v: F32, g: Array<F32>, acc: F32, r: Array<F32> & F32) -> NS:
  match r:
    case Tuple{lp, +x}:
      NS{lp, Array.set(F32, g, idx, v), (acc + x : F32)}

# a label past the last class picks 0.0 and leaves its gradient row zero, as l_at and onehot do
def nll_one(ok: Bool, +idx: U32, +v: F32, acc: F32, lp: Array<F32>, g: Array<F32>) -> NS:
  match ok:
    case True{}:
      nll_hit(idx, v, g, acc, Array.get(F32, lp, idx))
    case False{}:
      NS{lp, g, (acc + 0.0 : F32)}

# the sum of the picked log-probabilities (in row order, from +0.0), and the gradient (-1/n at each label)
def nll_rows(left: Nat, labels: List<&2, Nat>, +i: U32, +c: Nat, +v: F32, st: NS) -> NS:
  match left labels st:
    case 1n+q Con{+y, t} NS{lp, g, acc}:
      nll_rows(q, t, (i + 1 : U32), c, v, nll_one(Nat.is_le(1n+y, c), ((i * U32.from_nat(c) : U32) + U32.from_nat(y) : U32), v, acc, lp, g))
    case _ _ st:
      st

def nll_fin(-c: Nat, +n: Nat, r: NS) -> LossOut<TA.Mat<n, c>>:
  match r:
    case NS{lp, g, +s}:
      LossOut{F32.neg((s / F32.from_nat(n) : F32)), TA.Mat{g}}

# negative log-likelihood of log-probabilities (PyTorch's F.nll_loss, mean over the rows)
def NLL(+n: Nat, +c: Nat, logp: TA.Mat<n, c>, labels: List<&2, Nat>) -> LossOut<TA.Mat<n, c>>:
  match logp:
    case TA.Mat{lp}:
      nll_fin(c, n, nll_rows(n, labels, 0, c, F32.neg((1.0 / F32.from_nat(n) : F32)), NS{lp, zeros_a(Nat.mul(n, c)), 0.0}))

# =====================================================================================================
# Checkpoints: parameters, only the buffers in use, and Adam powers, as little-endian F32 (the gradient is zero
# between steps and is not saved). A run stopped after step k and resumed from its checkpoint computes the same bits
# as a run that never stopped.
# =====================================================================================================

def byte_of(+u: U32, +k: Nat) -> U32:
  U32.and(U32.shrn(u, k), 255)

def read_all(+path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 2000000000)
    rd_done(m)

# ---- packed BF16 moments in a checkpoint (CAdamWBF16) ----
# A PTree is written as F32 cells holding U32 bits, before 1 - b1^t and 1 - b2^t, in the order the step walks it:
# PNew: 0, v16 (1 or 0); PNode: 1, its left part, its right part; PLeaf: 2, n, m's ceil(n / 2) words, v's; PLeafV: 3, n,
# m's words, v's n F32 numbers. The tree carries its own shape and sizes, so a fresh state (all PNew) reads it back.

def ph_hw(+n: U32) -> U32:
  U32.shrn((n + 1 : U32), 1n)

def ph_bool(b: Bool) -> U32:
  match b:
    case True{}: 1
    case False{}: 0

type PTU is Type:
  PTU{t: PTree, n: U32}

def ptu_node(ra: PTU, rb: PTU) -> PTU:
  match ra rb:
    case PTU{a, na} PTU{b, nb}: PTU{PNode{a, b}, ((1 + na : U32) + nb : U32)}

# the number of cells a tree takes (the tree comes back)
def pt_cells(t: PTree) -> PTU:
  match t:
    case PNew{v16}: PTU{PNew{v16}, 2}
    case PLeaf{+n, m, v}: PTU{PLeaf{n, m, v}, (2 + (2 * ph_hw(n) : U32) : U32)}
    case PLeafV{+n, m, v}: PTU{PLeafV{n, m, v}, ((2 + ph_hw(n) : U32) + n : U32)}
    # bend-lint: disable-next bend-lint-rules/performance/missed_fork -- a tree of one node per layer, walked once per checkpoint not-expected-because: a fork costs more than the walk
    case PNode{a, b}: ptu_node(pt_cells(a), pt_cells(b))

type PTA is Type:
  PTA{t: PTree, xs: Array<F32>, off: U32}

type UA is Type:
  UA{a: Array<U32>, xs: Array<F32>}

# words i.. of a packed buffer into xs from off, as F32 cells with their bits (both arrays come back)
def words_out(left: Nat, +i: U32, +off: U32, ra: Array<U32> & U32, xs: Array<F32>) -> UA:
  match left ra:
    case 0n Tuple{a, w}: UA{a, xs}
    case 1n+q Tuple{a, +w}:
      words_out(q, (i + 1 : U32), (off + 1 : U32), Array.get(U32, a, (i + 1 : U32)), Array.set(F32, xs, off, ph_u2f(w)))

def words_in(left: Nat, +i: U32, +off: U32, rx: Array<F32> & F32, a: Array<U32>) -> UA:
  match left rx:
    case 0n Tuple{xs, x}: UA{a, xs}
    case 1n+q Tuple{xs, x}:
      words_in(q, (i + 1 : U32), (off + 1 : U32), Array.get(F32, xs, (off + 1 : U32)), Array.set(U32, a, i, ph_f2u(x)))

def pta_then(r: PTA, k: PTree -> Array<F32> -> U32 -> PTA) -> PTA:
  match r:
    case PTA{t, xs, off}: k(t, xs, off)

def pta_node(a: PTree, r: PTA) -> PTA:
  match r:
    case PTA{b, xs, off}: PTA{PNode{a, b}, xs, off}

def pts_v(+n: U32, m: Array<U32>, +off: U32, r: UA) -> PTA:
  match r:
    case UA{v, xs}: PTA{PLeaf{n, m, v}, xs, (off + ph_hw(n) : U32)}

def pts_m(+n: U32, v: Array<U32>, +off: U32, r: UA) -> PTA:
  match r:
    case UA{m, xs}: pts_v(n, m, (off + ph_hw(n) : U32), words_out(U32.to_nat(ph_hw(n)), 0, (off + ph_hw(n) : U32), Array.get(U32, v, 0), xs))

def ptsv_v(+n: U32, m: Array<U32>, +off: U32, r: TA.P2) -> PTA:
  match r:
    case TA.P2{v, xs}: PTA{PLeafV{n, m, v}, xs, (off + n : U32)}

def ptsv_m(+n: U32, v: Array<F32>, +off: U32, r: UA) -> PTA:
  match r:
    case UA{m, xs}: ptsv_v(n, m, (off + ph_hw(n) : U32), store_cells(U32.to_nat(n), 0, (off + ph_hw(n) : U32), Array.get(F32, v, 0), xs))

# the tree's cells into xs from off
def pt_store(t: PTree, xs: Array<F32>, +off: U32) -> PTA:
  match t:
    case PNew{+v16}:
      PTA{PNew{v16}, Array.set(F32, Array.set(F32, xs, off, ph_u2f(0)), (off + 1 : U32), ph_u2f(ph_bool(v16))), (off + 2 : U32)}
    case PLeaf{+n, m, v}:
      pts_m(n, v, (off + 2 : U32), words_out(U32.to_nat(ph_hw(n)), 0, (off + 2 : U32), Array.get(U32, m, 0), Array.set(F32, Array.set(F32, xs, off, ph_u2f(2)), (off + 1 : U32), ph_u2f(n))))
    case PLeafV{+n, m, v}:
      ptsv_m(n, v, (off + 2 : U32), words_out(U32.to_nat(ph_hw(n)), 0, (off + 2 : U32), Array.get(U32, m, 0), Array.set(F32, Array.set(F32, xs, off, ph_u2f(3)), (off + 1 : U32), ph_u2f(n))))
    case PNode{a, b}:
      pta_then(pt_store(a, Array.set(F32, xs, off, ph_u2f(1)), (off + 1 : U32)), ta => xs2 => off2 => pta_node(ta, pt_store(b, xs2, off2)))

# reading a leaf back: fresh buffers filled from xs
def ptl_v(+n: U32, m: Array<U32>, +off: U32, r: UA) -> PTA:
  match r:
    case UA{v, xs}: PTA{PLeaf{n, m, v}, xs, (off + ph_hw(n) : U32)}

def ptl_m(+n: U32, +off: U32, r: UA) -> PTA:
  match r:
    case UA{m, xs}: ptl_v(n, m, (off + ph_hw(n) : U32), words_in(U32.to_nat(ph_hw(n)), 0, (off + ph_hw(n) : U32), Array.get(F32, xs, (off + ph_hw(n) : U32)), ph_words(U32.to_nat(n))))

def ptlv_v(+n: U32, m: Array<U32>, +off: U32, r: TA.P2) -> PTA:
  match r:
    case TA.P2{xs, v}: PTA{PLeafV{n, m, v}, xs, (off + n : U32)}

def ptlv_m(+n: U32, +off: U32, r: UA) -> PTA:
  match r:
    case UA{m, xs}: ptlv_v(n, m, (off + ph_hw(n) : U32), load_cells(U32.to_nat(n), 0, (off + ph_hw(n) : U32), Array.get(F32, xs, (off + ph_hw(n) : U32)), Array.new(F32, depth(U32.to_nat(n)), 0.0)))

# what a tag says
type PK is Data:
  KNew{}
  KNode{}
  KLeaf{}
  KLeafV{}

def pt_kind2(z: Bool, o: Bool, l: Bool) -> PK:
  match z o l:
    case True{} _ _: KNew{}
    case False{} True{} _: KNode{}
    case False{} False{} True{}: KLeaf{}
    case False{} False{} False{}: KLeafV{}

def pt_kind(+tag: U32) -> PK:
  pt_kind2(U32.is_eq(tag, 0), U32.is_eq(tag, 1), U32.is_eq(tag, 2))

# the tag at off and the cell after it, to a continuation
def pt_next(+off: U32, r: Array<F32> & F32, k: PK -> (Array<F32> & F32) -> PTA) -> PTA:
  match r:
    case Tuple{xs, x}: k(pt_kind(ph_f2u(x)), Array.get(F32, xs, (off + 1 : U32)))

# the part at off, of kind k, with its second cell rn already read; fuel bounds the depth (a model's pairs nest far
# less than 1000 deep)
def pt_read(fuel: Nat, k: PK, +off: U32, rn: Array<F32> & F32) -> PTA:
  match fuel k rn:
    case 0n _ Tuple{xs, x}:
      PTA{PNew{False{}}, xs, off}
    case 1n+f KNew{} Tuple{xs, x}:
      PTA{PNew{U32.is_eq(ph_f2u(x), 1)}, xs, (off + 2 : U32)}
    case 1n++f KNode{} Tuple{xs, x}:
      pta_then(pt_read(f, pt_kind(ph_f2u(x)), (off + 1 : U32), Array.get(F32, xs, (off + 2 : U32))), ta => xs2 => +off2 => pta_node(ta, pt_next(off2, Array.get(F32, xs2, off2), kb => rb => pt_read(f, kb, off2, rb))))
    case 1n+f KLeaf{} Tuple{xs, x}:
      +n = ph_f2u(x)
      ptl_m(n, (off + 2 : U32), words_in(U32.to_nat(ph_hw(n)), 0, (off + 2 : U32), Array.get(F32, xs, (off + 2 : U32)), ph_words(U32.to_nat(n))))
    case 1n+f KLeafV{} Tuple{xs, x}:
      +n = ph_f2u(x)
      ptlv_m(n, (off + 2 : U32), words_in(U32.to_nat(ph_hw(n)), 0, (off + 2 : U32), Array.get(F32, xs, (off + 2 : U32)), ph_words(U32.to_nat(n))))

# a tree from xs at off
def pt_load(xs: Array<F32>, +off: U32) -> PTA:
  pt_next(off, Array.get(F32, xs, off), k => rn => pt_read(1000n, k, off, rn))

type TrainPL<-P: Type> is Type:
  TrainPL{tr: Train<P>, xs: List<&2, F32>}

def td0(-P: Type, g: P, +p1: F32, +p2: F32, +l: F32, rp: PL<P>) -> TrainPL<P>:
  match rp:
    case PL{p, xp}: TrainPL{Train{p, g, NoMom{}, p1, p2, l}, xp}

def td1(-P: Type, g: P, +p1: F32, +p2: F32, +l: F32, rp: PL<P>, rm: PL<P>) -> TrainPL<P>:
  match rp rm:
    case PL{p, xp} PL{mm, xm}: TrainPL{Train{p, g, OneMom{mm}, p1, p2, l}, List.append(&2, F32, xp, xm)}

def td3(-P: Type, g: P, +p1: F32, +p2: F32, +l: F32, rp: PL<P>, rm: PL<P>, rv: PL<P>) -> TrainPL<P>:
  match rp rm rv:
    case PL{p, xp} PL{m, xm} PL{v, xv}:
      TrainPL{Train{p, g, TwoMom{m, v}, p1, p2, l}, List.append(&2, F32, xp, List.append(&2, F32, xm, List.append(&2, F32, xv, [p1, p2])))}

# the first k cells of an array, in order
def cells_l(k: Nat, +i: U32, acc: List<&2, F32>, ra: Array<F32> & F32) -> List<&2, F32>:
  match k ra:
    case 0n Tuple{a, v}:
      List.reverse(&2, F32, acc)
    case 1n+q Tuple{a, +v}:
      cells_l(q, (i + 1 : U32), v <> acc, Array.get(F32, a, (i + 1 : U32)))

def tdp2(-P: Type, p: P, g: P, +p1: F32, +p2: F32, +l: F32, xp: List<&2, F32>, r: PTA) -> TrainPL<P>:
  match r:
    case PTA{t, xs, +off}:
      TrainPL{Train{p, g, Packed{t}, p1, p2, l}, List.append(&2, F32, xp, List.append(&2, F32, cells_l(U32.to_nat(off), 0, [], Array.get(F32, xs, 0)), [p1, p2]))}

def tdp(-P: Type, g: P, +p1: F32, +p2: F32, +l: F32, rp: PL<P>, rt: PTU) -> TrainPL<P>:
  match rp rt:
    case PL{p, xp} PTU{t, +n}:
      tdp2(P, p, g, p1, p2, l, xp, pt_store(t, Array.new(F32, depth(U32.to_nat(n)), 0.0), 0))

# every number of a training state: the parameters, then the first moments, the second moments, b1^t and b2^t (packed
# moments: their tree's cells, pt_store's layout, in place of the two moments)
def Train.dump(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, tr: Train<P>) -> TrainPL<P>:
  match tr:
    case Train{p, g, NoMom{}, +p1, +p2, +l}:
      td0(P, g, p1, p2, l, Layer.dump(P, X, Y, S, m, p))
    case Train{p, g, OneMom{mm}, +p1, +p2, +l}:
      td1(P, g, p1, p2, l, Layer.dump(P, X, Y, S, m, p), Layer.dump(P, X, Y, S, m, mm))
    case Train{p, g, TwoMom{mm, vv}, +p1, +p2, +l}:
      td3(P, g, p1, p2, l, Layer.dump(P, X, Y, S, m, p), Layer.dump(P, X, Y, S, m, mm), Layer.dump(P, X, Y, S, m, vv))
    case Train{p, g, Packed{t}, +p1, +p2, +l}:
      tdp(P, g, p1, p2, l, Layer.dump(P, X, Y, S, m, p), pt_cells(t))

def f32_at(xs: List<&2, F32>, n: Nat) -> F32:
  match xs n:
    case Con{x, t} 0n:
      x
    case Con{x, t} 1n+q:
      f32_at(t, q)
    case Nil{} _:
      0.0

def tr0(-P: Type, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{p, xs, off}: Train{p, g, NoMom{}, 0.0, 0.0, 0.0}

def tr_m2(-P: Type, p: P, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{mm, xs, off}: Train{p, g, OneMom{mm}, 0.0, 0.0, 0.0}

def tr_m1(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, mm: P, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{p, xs, off}: tr_m2(P, p, g, Layer.load(P, X, Y, S, m, mm, xs, off))

def tr_adam2(-P: Type, p: P, mm: P, vv: P, g: P, p1: F32, r: Array<F32> & F32) -> Train<P>:
  match r:
    case Tuple{xs, p2}: Train{p, g, TwoMom{mm, vv}, p1, p2, 0.0}

def tr_adam(-P: Type, p: P, mm: P, vv: P, g: P, r: Array<F32> & F32, +off: U32) -> Train<P>:
  match r:
    case Tuple{xs, p1}: tr_adam2(P, p, mm, vv, g, p1, Array.get(F32, xs, off))

def tr3(-P: Type, p: P, mm: P, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{vv, xs, +off}: tr_adam(P, p, mm, vv, g, Array.get(F32, xs, off), (off + 1 : U32))

def tr2(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, p: P, vv: P, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{mm, xs, off}:
      tr3(P, p, mm, g, Layer.load(P, X, Y, S, m, vv, xs, off))

def tr1(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, mm: P, vv: P, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{p, xs, off}:
      tr2(~P, ~X, ~Y, ~S, ~m, p, vv, g, Layer.load(P, X, Y, S, m, mm, xs, off))

def trp4(-P: Type, p: P, g: P, t: PTree, +p1: F32, r: Array<F32> & F32) -> Train<P>:
  match r:
    case Tuple{xs, p2}: Train{p, g, Packed{t}, p1, p2, 0.0}

def trp3(-P: Type, p: P, g: P, t: PTree, r: Array<F32> & F32, +off: U32) -> Train<P>:
  match r:
    case Tuple{xs, p1}: trp4(P, p, g, t, p1, Array.get(F32, xs, off))

def trp2(-P: Type, p: P, g: P, r: PTA) -> Train<P>:
  match r:
    case PTA{t, xs, +off}: trp3(P, p, g, t, Array.get(F32, xs, off), (off + 1 : U32))

def trp(-P: Type, g: P, r: PA<P>) -> Train<P>:
  match r:
    case PA{p, xs, +off}: trp2(P, p, g, pt_load(xs, off))

# a training state filled from the numbers Train.dump wrote (its buffers are overwritten in place; packed moments are
# built again from the tree's cells)
def Train.restore(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, tr: Train<P>, xs: Array<F32>) -> Train<P>:
  match tr:
    case Train{p, g, NoMom{}, p1, p2, l}:
      tr0(P, g, Layer.load(P, X, Y, S, m, p, xs, 0))
    case Train{p, g, OneMom{mm}, p1, p2, l}:
      tr_m1(~P, ~X, ~Y, ~S, ~m, mm, g, Layer.load(P, X, Y, S, m, p, xs, 0))
    case Train{p, g, TwoMom{mm, vv}, p1, p2, l}:
      tr1(~P, ~X, ~Y, ~S, ~m, mm, vv, g, Layer.load(P, X, Y, S, m, p, xs, 0))
    case Train{p, g, Packed{t}, p1, p2, l}:
      trp(P, g, Layer.load(P, X, Y, S, m, p, xs, 0))

# ---- writing numbers straight from an array, a chunk at a time ----
# File.write_bytes takes a list with one cell per byte; one list for a whole model would take several GB at SmolLM2's
# sizes, so each chunk of WCHUNK numbers gets its own list, written and dropped before the next.

def WCHUNK() -> U32:
  65536

# the bytes of xs[i - k + 1 .. i] onto acc, built from the last number back (no reverse)
def chunk_bytes(k: Nat, +i: U32, rx: Array<F32> & F32, acc: List<&2, U32>) -> Array<F32> & List<&2, U32>:
  match k rx:
    case 0n Tuple{a, v}: Tuple{a, acc}
    case 1n+q Tuple{a, v}:
      +u = bits(v)
      chunk_bytes(q, U32.sub(i, 1), Array.get(F32, a, U32.sub(i, 1)), byte_of(u, 0n) <> byte_of(u, 8n) <> byte_of(u, 16n) <> byte_of(u, 24n) <> acc)

def wc_ok(-R: Type, rr: File & Result<&1, &1, U32 & String, Unit>, a: Array<F32>, k: File -> Array<F32> -> IO(R)) -> IO(R):
  match rr:
    case Tuple{f, res}:
      do IO<R>:
        u : Unit <- IO.pass(Unit, res)
        k(f, a)

def wc_bytes(-R: Type, f: File, r: Array<F32> & List<&2, U32>, k: File -> Array<F32> -> IO(R)) -> IO(R):
  match r:
    case Tuple{a, bs}:
      do IO<R>:
        rr : File & Result<&1, &1, U32 & String, Unit> <- File.write_bytes(f, bs)
        wc_ok(R, rr, a, k)

# k chunks of a's numbers lo .. n - 1
def write_chunks(k: Nat, f: File, a: Array<F32>, +lo: U32, +n: U32) -> IO(File & Array<F32>):
  match k:
    case 0n:
      IO.pure(File & Array<F32>, Tuple{f, a})
    case 1n+q:
      +hi = U32.min((lo + WCHUNK() : U32), n)
      wc_bytes(File & Array<F32>, f, chunk_bytes(U32.to_nat(U32.sub(hi, lo)), U32.sub(hi, 1), Array.get(F32, a, U32.sub(hi, 1)), []), f2 => a2 => write_chunks(q, f2, a2, hi, n))

def wf_close(r: File & Array<F32>) -> IO(Array<F32>):
  match r:
    case Tuple{f, a}:
      do IO<Array<F32>>:
        File.close(f)
        return a

# cells 0 .. n - 1 of a, as little-endian F32, into a new file at path; a comes back
def write_f32(+path: String, a: Array<F32>, +n: U32) -> IO(Array<F32>):
  do IO<Array<F32>>:
    f : File <- IO.try(File, File.open(path, "w"))
    r : File & Array<F32> <- write_chunks(U32.to_nat(U32.div((n + U32.sub(WCHUNK(), 1) : U32), WCHUNK())), f, a, 0, n)
    wf_close(r)

def ps_fin(-P: Type, +path: String, +n: U32, r: PA<P>) -> IO(P):
  match r:
    case PA{p, xs, off}:
      do IO<P>:
        a : Array<F32> <- write_f32(path, xs, n)
        return p

# every parameter of p, in dump order, into a file at path (Layer.count numbers), through one array
def Params.save(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, p: P) -> IO(P):
  +n = Layer.count(P, X, Y, S, m)
  ps_fin(P, path, n, Layer.store(P, X, Y, S, m, p, Array.new(F32, depth(U32.to_nat(n)), 0.0), 0))

# ---- a training state into a file: the parameters, then the moments in use, then Adam's b1^t and b2^t ----

def tsa_fin(-P: Type, +path: String, g: P, st: OptState<P>, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{p, xs, +off}:
      do IO<Train<P>>:
        a : Array<F32> <- write_f32(path, xs, off)
        return Train{p, g, st, p1, p2, l}

def tsa_m(-P: Type, +path: String, p: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{mm, xs, off}: tsa_fin(P, path, g, OneMom{mm}, p1, p2, l, PA{p, xs, off})

def tsa_pw(-P: Type, +path: String, p: P, mm: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{vv, xs, +off}:
      tsa_fin(P, path, g, TwoMom{mm, vv}, p1, p2, l, PA{p, Array.set(F32, Array.set(F32, xs, off, p1), (off + 1 : U32), p2), (off + 2 : U32)})

def tsa_v(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, p: P, vv: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{mm, xs, off}: tsa_pw(P, path, p, mm, g, p1, p2, l, Layer.store(P, X, Y, S, m, vv, xs, off))

def tsa_mm(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, mm: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{p, xs, off}: tsa_m(P, path, p, g, p1, p2, l, Layer.store(P, X, Y, S, m, mm, xs, off))

def tsa_mv(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, mm: P, vv: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{p, xs, off}: tsa_v(~P, ~X, ~Y, ~S, ~m, path, p, vv, g, p1, p2, l, Layer.store(P, X, Y, S, m, mm, xs, off))

def tsp3(-P: Type, +path: String, p: P, g: P, +p1: F32, +p2: F32, +l: F32, r: PTA) -> IO(Train<P>):
  match r:
    case PTA{t, xs, +off}:
      tsa_fin(P, path, g, Packed{t}, p1, p2, l, PA{p, Array.set(F32, Array.set(F32, xs, off, p1), (off + 1 : U32), p2), (off + 2 : U32)})

def tsp2(-P: Type, +path: String, g: P, +p1: F32, +p2: F32, +l: F32, t: PTree, r: PA<P>) -> IO(Train<P>):
  match r:
    case PA{p, xs, +off}: tsp3(P, path, p, g, p1, p2, l, pt_store(t, xs, off))

def tsp(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, p: P, g: P, +p1: F32, +p2: F32, +l: F32, rt: PTU) -> IO(Train<P>):
  match rt:
    case PTU{t, +n}:
      tsp2(P, path, g, p1, p2, l, t, Layer.store(P, X, Y, S, m, p, Array.new(F32, depth(U32.to_nat(((Layer.count(P, X, Y, S, m) + n : U32) + 2 : U32))), 0.0), 0))

# the layout Train.dump gives, written from one array instead of a list (the same bytes)
def Train.save(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: Layer<P, X, Y, S>, +path: String, tr: Train<P>) -> IO(Train<P>):
  match tr:
    case Train{p, g, NoMom{}, +p1, +p2, +l}:
      tsa_fin(P, path, g, NoMom{}, p1, p2, l, Layer.store(P, X, Y, S, m, p, Array.new(F32, depth(U32.to_nat(Layer.count(P, X, Y, S, m))), 0.0), 0))
    case Train{p, g, OneMom{mm}, +p1, +p2, +l}:
      tsa_mm(~P, ~X, ~Y, ~S, ~m, path, mm, g, p1, p2, l, Layer.store(P, X, Y, S, m, p, Array.new(F32, depth(U32.to_nat((2 * Layer.count(P, X, Y, S, m) : U32))), 0.0), 0))
    case Train{p, g, TwoMom{mm, vv}, +p1, +p2, +l}:
      tsa_mv(~P, ~X, ~Y, ~S, ~m, path, mm, vv, g, p1, p2, l, Layer.store(P, X, Y, S, m, p, Array.new(F32, depth(U32.to_nat(((3 * Layer.count(P, X, Y, S, m) : U32) + 2 : U32))), 0.0), 0))
    case Train{p, g, Packed{t}, +p1, +p2, +l}:
      tsp(~P, ~X, ~Y, ~S, ~m, path, p, g, p1, p2, l, pt_cells(t))
