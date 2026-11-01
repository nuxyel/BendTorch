# BendTorch tensors: Tensor<s>, an n-dimensional F32 array whose shape s (a list of sizes) is in its type.
#
#   import ../torch/tensor.bend as TT
#
# A Tensor<s> holds product(s) numbers in one flat Array<F32>, row-major (the last axis varies fastest), as bend-ml's
# Mat<r, c> does for two axes: Tensor<[r, c]> and Mat<r, c> are the same array under two types, and `of_mat`/`to_mat`
# convert without a copy. Every operation hands back what it reads, as the Mat operations do.
#
# Each operation is one of two kernels over an index map (for each position of one tensor, the position of another):
#   gather       out[j] = f(x[m(j)])           (and its two-input form, for broadcasting)
#   scatter-add  g[m(j)] = g[m(j)] + x[j]      (reductions, and the gradient of a broadcast)
# An index map is a list of axes, each with its size and the stride of the tensor being read (0 on a broadcast axis),
# or an axis map (the position with one axis taken out). The shape conditions of an operation are Bool functions of the
# shapes (`bc_ok` for broadcasting); a caller passes the proof that the condition is True, which the checker computes for
# literal shapes and a program checks once where shapes read at run time enter (`Bool.check`).

import Base
import bend-ml-tensor-array@0.1.18.0/main.bend as TA
import bend-ml-nat-lemmas@0.1.2.0/main.bend as NL

# =====================================================================================================
# Shapes
# =====================================================================================================

type Tensor<-s: List<&2, Nat>> is Type:
  Tensor{data: Array<F32>}

# the number of elements of a tensor of shape s
def numel(+s: List<&2, Nat>) -> Nat:
  NL.product(s)

# size of axis k (1 past the last axis, so a missing axis counts as a size-1 one)
def at1(s: List<&2, Nat>, k: Nat) -> Nat:
  match s k:
    case Nil{} _:
      1n
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      at1(t, p)

# s without axis k (s itself past the last axis)
def remove(s: List<&2, Nat>, k: Nat) -> List<&2, Nat>:
  match s k:
    case Nil{} _:
      Nil{}
    case Con{h, t} 0n:
      t
    case Con{h, t} 1n+p:
      h <> remove(t, p)

# n copies of 1, then s: s seen with n more leading axes
def ones_then(n: Nat, s: List<&2, Nat>) -> List<&2, Nat>:
  match n:
    case 0n:
      s
    case 1n+p:
      1n <> ones_then(p, s)

# s aligned to rank r by leading size-1 axes, as broadcasting aligns shapes from the right
def pad(+s: List<&2, Nat>, +r: Nat) -> List<&2, Nat>:
  ones_then(Nat.sub(r, List.length(&2, Nat, s)), s)

# ---- broadcasting (PyTorch's rule): shapes aligned from the right; on each axis the sizes are equal or one is 1 ----

# one axis: z is x, and y is x or 1
def bc_pair(+x: Nat, +y: Nat, +z: Nat) -> Bool:
  Bool.and(Nat.is_eq(z, x), Bool.or(Nat.is_eq(y, x), Nat.is_eq(y, 1n)))

# one axis, either way round
def bc_dim(+x: Nat, +y: Nat, +z: Nat) -> Bool:
  Bool.or(bc_pair(x, y, z), bc_pair(y, x, z))

# three shapes of one rank, axis by axis
def bc_eq(a: List<&2, Nat>, b: List<&2, Nat>, o: List<&2, Nat>) -> Bool:
  match a b o:
    case Nil{} Nil{} Nil{}:
      True{}
    case Con{+x, ra} Con{+y, rb} Con{+z, ro}:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
      Bool.and(bc_dim(x, y, z), bc_eq(ra, rb, ro))
    case _ _ _:
      False{}

# the ranks: o's is the larger of a's and b's
def bc_rank(+la: Nat, +lb: Nat, +lo: Nat) -> Bool:
  Bool.and(Bool.and(Nat.is_le(la, lo), Nat.is_le(lb, lo)), Bool.or(Nat.is_eq(la, lo), Nat.is_eq(lb, lo)))

# a and b broadcast to o
def bc_ok(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>) -> Bool:
  # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
  Bool.and(bc_rank(List.length(&2, Nat, a), List.length(&2, Nat, b), List.length(&2, Nat, o)), bc_eq(pad(a, List.length(&2, Nat, o)), pad(b, List.length(&2, Nat, o)), o))

# a proof that b is True, when it is: where shapes read at run time enter, a program checks a condition once
def Bool.check(b: Bool) -> Maybe<&1, {True{} == b : Bool}>:
  match b:
    case True{}:
      Some{{==}}
    case False{}:
      None{}

def nat_eq_succ(-a: Nat, -b: Nat, m: Maybe<&1, {a == b : Nat}>) -> Maybe<&1, {1n+a == 1n+b : Nat}>:
  match m:
    case Some{e}:
      Some{Equal.cong(Nat, Nat, x => 1n+x, a, b, e)}
    case None{}:
      None{}

# a proof that a == b, when it holds (reshape's count, read at run time)
def Nat.check_eq(a: Nat, b: Nat) -> Maybe<&1, {a == b : Nat}>:
  match a b:
    case 0n 0n:
      Some{{==}}
    case 1n+p 1n+q:
      nat_eq_succ(p, q, Nat.check_eq(p, q))
    case _ _:
      None{}

# =====================================================================================================
# Index maps
# =====================================================================================================

# one axis of an index map: its size, and the stride of the tensor read along it (0 where it is broadcast)
type Ax is Data:
  Ax{n: U32, st: U32}

type IMap is Data:
  # base plus, for every axis (the last one first), its coordinate times its stride
  Strided{base: U32, axes: List<&2, Ax>}
  # the position with one axis taken out: inner is the product of the sizes after it, size its own size
  Drop{inner: U32, size: U32}

def stride_of(+d: Nat, +st: U32) -> U32:
  Bool.pick(U32, Nat.is_eq(d, 1n), 0, st)

# the axes of reading a tensor of shape a (aligned to o, same rank) at every position of o; the last axis first
def axes(ra: List<&2, Nat>, ro: List<&2, Nat>, +st: U32, acc: List<&2, Ax>) -> List<&2, Ax>:
  match ra ro:
    case Con{+d, ta} Con{+n, to}:
      axes(ta, to, (st * U32.from_nat(d) : U32), Ax{U32.from_nat(n), stride_of(d, st)} <> acc)
    case _ _:
      acc

# reading a (any rank up to o's) along o
def bmap(+a: List<&2, Nat>, +o: List<&2, Nat>) -> IMap:
  Strided{0, List.reverse(&2, Ax, axes(List.reverse(&2, Nat, pad(a, List.length(&2, Nat, o))), List.reverse(&2, Nat, o), 1, Nil{}))}

def strided(axs: List<&2, Ax>, +j: U32, acc: U32) -> U32:
  match axs:
    case Nil{}:
      acc
    case Con{Ax{+n, +st}, r}:
      strided(r, U32.div(j, n), (acc + (U32.mod(j, n) * st : U32) : U32))

# the position m(j)
def imap(+m: IMap, +j: U32) -> U32:
  match m:
    case Strided{+base, axs}:
      strided(axs, j, base)
    case Drop{+inner, +size}:
      ((U32.div(j, (inner * size : U32)) * inner : U32) + U32.mod(j, inner) : U32)

# =====================================================================================================
# Kernels
# =====================================================================================================

type P2 is Type:
  P2{a: Array<F32>, b: Array<F32>}

type P3 is Type:
  P3{a: Array<F32>, b: Array<F32>, c: Array<F32>}

# o[j] = f(x[m(j)], y[n(j)]) for k positions from j; rx and ry hold x[m(j)] and y[n(j)], already read
def gather2_l(~f: @+a: F32 -> @+b: F32 -> F32, k: Nat, +j: U32, +m: IMap, +n: IMap, rx: Array<F32> & F32, ry: Array<F32> & F32, o: Array<F32>) -> P3:
  match k rx ry:
    case 0n Tuple{x, u} Tuple{y, v}:
      P3{x, y, o}
    case 1n+p Tuple{x, +u} Tuple{y, +v}:
      gather2_l(~f, p, (j + 1 : U32), m, n, Array.get(F32, x, imap(m, (j + 1 : U32))), Array.get(F32, y, imap(n, (j + 1 : U32))), Array.set(F32, o, j, f(u, v)))

# o[j] = x[m(j)] for k positions from j
def gather_l(k: Nat, +j: U32, +m: IMap, rx: Array<F32> & F32, o: Array<F32>) -> P2:
  match k rx:
    case 0n Tuple{x, u}:
      P2{x, o}
    case 1n+p Tuple{x, +u}:
      gather_l(p, (j + 1 : U32), m, Array.get(F32, x, imap(m, (j + 1 : U32))), Array.set(F32, o, j, u))

# g[m(j)] = g[m(j)] + x[j] for k positions from j; rx holds x[j], rg holds g[m(j)]
def scatter_l(k: Nat, +j: U32, +m: IMap, rx: Array<F32> & F32, rg: Array<F32> & F32) -> P2:
  match k rx rg:
    case 0n Tuple{x, u} Tuple{g, w}:
      P2{x, g}
    case 1n+p Tuple{x, +u} Tuple{g, +w}:
      scatter_l(p, (j + 1 : U32), m, Array.get(F32, x, (j + 1 : U32)), Array.get(F32, Array.set(F32, g, imap(m, j), (w + u : F32)), imap(m, (j + 1 : U32))))

def new(+n: Nat) -> Array<F32>:
  Array.new(F32, TA.cap_depth(n), 0.0)

# =====================================================================================================
# Operations
# =====================================================================================================

def zeros(+s: List<&2, Nat>) -> Tensor<s>:
  Tensor{new(numel(s))}

def of_mat(-r: Nat, -c: Nat, m: TA.Mat<r, c>) -> Tensor<[r, c]>:
  match m:
    case TA.Mat{d}:
      Tensor{d}

def to_mat(-r: Nat, -c: Nat, t: Tensor<[r, c]>) -> TA.Mat<r, c>:
  match t:
    case Tensor{d}:
      TA.Mat{d}

# a binary operation, both operands handed back with the result
type T3<-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>> is Type:
  T3{x: Tensor<a>, y: Tensor<b>, z: Tensor<o>}

def t3_of(-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>, p: P3) -> T3<a, b, o>:
  match p:
    case P3{x, y, z}:
      T3{Tensor{x}, Tensor{y}, Tensor{z}}

def f32_add(+a: F32, +b: F32) -> F32:
  (a + b : F32)

def f32_sub(+a: F32, +b: F32) -> F32:
  (a - b : F32)

def f32_mul(+a: F32, +b: F32) -> F32:
  (a * b : F32)

def f32_div(+a: F32, +b: F32) -> F32:
  (a / b : F32)

# z = f(x, y) element by element, x and y broadcast to o
def map2(~f: @+a: F32 -> @+b: F32 -> F32, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == bc_ok(a, b, o) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, o>:
  match x y:
    case Tensor{dx} Tensor{dy}:
      t3_of(a, b, o, gather2_l(~f, numel(o), 0, bmap(a, o), bmap(b, o), Array.get(F32, dx, imap(bmap(a, o), 0)), Array.get(F32, dy, imap(bmap(b, o), 0)), new(numel(o))))

def add(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == bc_ok(a, b, o) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, o>:
  map2(~f32_add, a, b, o, pf, x, y)

# a tensor and the gradient that comes back with it
type TP<-a: List<&2, Nat>, -b: List<&2, Nat>> is Type:
  TP{x: Tensor<a>, y: Tensor<b>}

def tp_of(-a: List<&2, Nat>, -b: List<&2, Nat>, p: P2) -> TP<a, b>:
  match p:
    case P2{x, y}:
      TP{Tensor{x}, Tensor{y}}

# g + (dz summed down to a's shape): the gradient of an operand that was broadcast from a to o; dz comes back
def sum_to(+a: List<&2, Nat>, +o: List<&2, Nat>, dz: Tensor<o>, g: Tensor<a>) -> TP<o, a>:
  match dz g:
    case Tensor{d} Tensor{e}:
      tp_of(o, a, scatter_l(numel(o), 0, bmap(a, o), Array.get(F32, d, 0), Array.get(F32, e, imap(bmap(a, o), 0))))

# ---- sum over one axis ----

# the product of the sizes after axis k
def inner(s: List<&2, Nat>, k: Nat) -> Nat:
  match s k:
    case Nil{} _:
      1n
    case Con{h, t} 0n:
      NL.product(t)
    case Con{h, t} 1n+p:
      inner(t, p)

def drop_map(+s: List<&2, Nat>, +k: Nat) -> IMap:
  Drop{U32.from_nat(inner(s, k)), U32.from_nat(at1(s, k))}

# x summed over axis k: shape remove(s, k); x comes back
def sum_axis(+s: List<&2, Nat>, +k: Nat, x: Tensor<s>) -> TP<s, remove(s, k)>:
  match x:
    case Tensor{d}:
      tp_of(s, remove(s, k), scatter_l(numel(s), 0, drop_map(s, k), Array.get(F32, d, 0), Array.get(F32, new(numel(remove(s, k))), 0)))

# the gradient of sum_axis: dy (shape remove(s, k)) copied along axis k; dy comes back
def sum_axis_bwd(+s: List<&2, Nat>, +k: Nat, dy: Tensor<remove(s, k)>) -> TP<remove(s, k), s>:
  match dy:
    case Tensor{d}:
      tp_of(remove(s, k), s, gather_l(numel(s), 0, drop_map(s, k), Array.get(F32, d, imap(drop_map(s, k), 0)), new(numel(s))))

# =====================================================================================================
# More shapes
# =====================================================================================================

# s with the size of axis k replaced by v (s itself past the last axis)
def setd(s: List<&2, Nat>, k: Nat, +v: Nat) -> List<&2, Nat>:
  match s k:
    case Nil{} _:
      Nil{}
    case Con{h, t} 0n:
      v <> t
    case Con{h, t} 1n+p:
      h <> setd(t, p, v)

def rank(+s: List<&2, Nat>) -> Nat:
  List.length(&2, Nat, s)

# the index map of reading a tensor of strides st (given by its shape st) at every position of a tensor of shape o, from
# a base: axis e has o's size and st's stride (inner(st, e)); the last axis first
def win_axes(n: Nat, +o: List<&2, Nat>, +st: List<&2, Nat>, +e: Nat, acc: List<&2, Ax>) -> List<&2, Ax>:
  match n:
    case 0n:
      acc
    case 1n+p:
      win_axes(p, o, st, 1n+e, Ax{U32.from_nat(at1(o, e)), U32.from_nat(inner(st, e))} <> acc)

def window(+o: List<&2, Nat>, +st: List<&2, Nat>, +base: U32) -> IMap:
  Strided{base, win_axes(rank(o), o, st, 0n, Nil{})}

# g[m(j)] = x[j] for k positions from j (a copy into a window of g; no addition, so -0 stays -0)
def place_l(k: Nat, +j: U32, +m: IMap, rx: Array<F32> & F32, g: Array<F32>) -> P2:
  match k rx:
    case 0n Tuple{x, u}:
      P2{x, g}
    case 1n+p Tuple{x, +u}:
      place_l(p, (j + 1 : U32), m, Array.get(F32, x, (j + 1 : U32)), Array.set(F32, g, imap(m, j), u))

# a[j] = f(a[j]) for k positions from j, with a constant c
def ewc_l(~f: @+a: F32 -> @+c: F32 -> F32, k: Nat, +j: U32, +c: F32, ra: Array<F32> & F32) -> Array<F32>:
  match k ra:
    case 0n Tuple{a, v}:
      a
    case 1n+p Tuple{a, +v}:
      ewc_l(~f, p, (j + 1 : U32), c, Array.get(F32, Array.set(F32, a, j, f(v, c)), (j + 1 : U32)))

def ew_div(+s: List<&2, Nat>, +c: F32, t: Tensor<s>) -> Tensor<s>:
  match t:
    case Tensor{d}:
      Tensor{ewc_l(~f32_div, numel(s), 0, c, Array.get(F32, d, 0))}

def n_f32(+n: Nat) -> F32:
  U32.to_f32(U32.from_nat(n))

# =====================================================================================================
# Binary operations with broadcasting: gradients
# =====================================================================================================

# g[mg(j)] = g[mg(j)] + f(dz[j], x[ma(j)], y[mb(j)]) for k positions from j: the gradient of one operand of map2,
# with both operands at hand (mul and div need them)
def grad2_l(~f: @+d: F32 -> @+p: F32 -> @+q: F32 -> F32, k: Nat, +j: U32, +mg: IMap, +ma: IMap, +mb: IMap, rd: Array<F32> & F32, rx: Array<F32> & F32, ry: Array<F32> & F32, rg: Array<F32> & F32) -> P4:
  match k rd rx ry rg:
    case 0n Tuple{d, u} Tuple{x, p} Tuple{y, q} Tuple{g, w}:
      P4{d, x, y, g}
    case 1n+n Tuple{d, +u} Tuple{x, +p} Tuple{y, +q} Tuple{g, +w}:
      grad2_l(~f, n, (j + 1 : U32), mg, ma, mb, Array.get(F32, d, (j + 1 : U32)), Array.get(F32, x, imap(ma, (j + 1 : U32))), Array.get(F32, y, imap(mb, (j + 1 : U32))),
        Array.get(F32, Array.set(F32, g, imap(mg, j), (w + f(u, p, q) : F32)), imap(mg, (j + 1 : U32))))

type P4 is Type:
  P4{a: Array<F32>, b: Array<F32>, c: Array<F32>, d: Array<F32>}

# dz, x, y and the gradient, handed back
type G2<-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>, -g: List<&2, Nat>> is Type:
  G2{dz: Tensor<o>, x: Tensor<a>, y: Tensor<b>, g: Tensor<g>}

def g2_of(-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>, -g: List<&2, Nat>, p: P4) -> G2<a, b, o, g>:
  match p:
    case P4{d, x, y, w}:
      G2{Tensor{d}, Tensor{x}, Tensor{y}, Tensor{w}}

# g + the gradient, through f, of the operand of shape gs (a or b) of z = op(x, y): f(dz, x, y) at each position of o
def grad2(~f: @+d: F32 -> @+p: F32 -> @+q: F32 -> F32, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, +gs: List<&2, Nat>, dz: Tensor<o>, x: Tensor<a>, y: Tensor<b>, g: Tensor<gs>) -> G2<a, b, o, gs>:
  match dz x y g:
    case Tensor{dd} Tensor{dx} Tensor{dy} Tensor{dg}:
      g2_of(a, b, o, gs, grad2_l(~f, numel(o), 0, bmap(gs, o), bmap(a, o), bmap(b, o), Array.get(F32, dd, 0), Array.get(F32, dx, imap(bmap(a, o), 0)), Array.get(F32, dy, imap(bmap(b, o), 0)), Array.get(F32, dg, imap(bmap(gs, o), 0))))

# the partial derivatives, dz times d op / d operand (PyTorch's formulas and order of operations)
def d_id(+d: F32, +p: F32, +q: F32) -> F32:
  d

def d_neg(+d: F32, +p: F32, +q: F32) -> F32:
  (0.0 - d : F32)

def d_mul_a(+d: F32, +p: F32, +q: F32) -> F32:
  (d * q : F32)

def d_mul_b(+d: F32, +p: F32, +q: F32) -> F32:
  (d * p : F32)

def d_div_a(+d: F32, +p: F32, +q: F32) -> F32:
  (d / q : F32)

# -dz * ((x / y) / y), as PyTorch's div backward for the divisor
def d_div_b(+d: F32, +p: F32, +q: F32) -> F32:
  (0.0 - (d * ((p / q : F32) / q : F32) : F32) : F32)

def sub(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == bc_ok(a, b, o) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, o>:
  map2(~f32_sub, a, b, o, pf, x, y)

def mul(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == bc_ok(a, b, o) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, o>:
  map2(~f32_mul, a, b, o, pf, x, y)

def div(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == bc_ok(a, b, o) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, o>:
  map2(~f32_div, a, b, o, pf, x, y)

# =====================================================================================================
# Reductions
# =====================================================================================================

# the sum over axis k with the axis kept (size 1): the same numbers as sum_axis, in the same layout
def sum_axis_keep(+s: List<&2, Nat>, +k: Nat, x: Tensor<s>) -> TP<s, setd(s, k, 1n)>:
  match x:
    case Tensor{d}:
      tp_of(s, setd(s, k, 1n), scatter_l(numel(s), 0, drop_map(s, k), Array.get(F32, d, 0), Array.get(F32, new(numel(remove(s, k))), 0)))

def tp_div(-s: List<&2, Nat>, +u: List<&2, Nat>, +c: F32, r: TP<s, u>) -> TP<s, u>:
  match r:
    case TP{x, y}:
      TP{x, ew_div(u, c, y)}

# the mean over axis k: the sum divided by the axis' size
def mean_axis(+s: List<&2, Nat>, +k: Nat, x: Tensor<s>) -> TP<s, remove(s, k)>:
  tp_div(s, remove(s, k), n_f32(at1(s, k)), sum_axis(s, k, x))

# the gradient of mean_axis: dy divided by the axis' size, copied along it
def mean_axis_bwd(+s: List<&2, Nat>, +k: Nat, dy: Tensor<remove(s, k)>) -> TP<remove(s, k), s>:
  tp_div(remove(s, k), s, n_f32(at1(s, k)), sum_axis_bwd(s, k, dy))

# every position to position 0: the whole-tensor reductions
def all_map(+s: List<&2, Nat>) -> IMap:
  Drop{1, U32.from_nat(numel(s))}

# the sum of every element: a tensor of shape [] (one number)
def sum_all(+s: List<&2, Nat>, x: Tensor<s>) -> TP<s, []>:
  match x:
    case Tensor{d}:
      tp_of(s, [], scatter_l(numel(s), 0, all_map(s), Array.get(F32, d, 0), Array.get(F32, new(1n), 0)))

def sum_all_bwd(+s: List<&2, Nat>, dy: Tensor<[]>) -> TP<[], s>:
  match dy:
    case Tensor{d}:
      tp_of([], s, gather_l(numel(s), 0, all_map(s), Array.get(F32, d, 0), new(numel(s))))

def mean_all(+s: List<&2, Nat>, x: Tensor<s>) -> TP<s, []>:
  tp_div(s, [], n_f32(numel(s)), sum_all(s, x))

def mean_all_bwd(+s: List<&2, Nat>, dy: Tensor<[]>) -> TP<[], s>:
  tp_div([], s, n_f32(numel(s)), sum_all_bwd(s, dy))

# =====================================================================================================
# Reshaping: the same numbers under another shape, with a proof that the count does not change
# =====================================================================================================

def reshape(+s: List<&2, Nat>, +t: List<&2, Nat>, -e: {numel(s) == numel(t) : Nat}, x: Tensor<s>) -> Tensor<t>:
  match x:
    case Tensor{d}:
      Tensor{d}

# n * 1 == n, so a shape and its flattening hold the same count
def flat_numel(+s: List<&2, Nat>) -> {numel(s) == numel([numel(s)]) : Nat}:
  Equal.sym(Nat, Nat.mul(numel(s), 1n), numel(s), NL.mul_one_r(numel(s)))

def flatten(+s: List<&2, Nat>, x: Tensor<s>) -> Tensor<[numel(s)]>:
  reshape(s, [numel(s)], flat_numel(s), x)

# =====================================================================================================
# Permutations: a copy with the axes reordered
# =====================================================================================================

# the shape with axes reordered by p: axis d of the result is axis p[d] of s
def permuted(+s: List<&2, Nat>, p: List<&2, Nat>) -> List<&2, Nat>:
  match p:
    case Nil{}:
      Nil{}
    case Con{k, r}:
      at1(s, k) <> permuted(s, r)

def count(+k: Nat, p: List<&2, Nat>) -> Nat:
  match p:
    case Nil{}:
      0n
    case Con{+h, r}:
      Nat.add(Bool.pick(Nat, Nat.is_eq(h, k), 1n, 0n), count(k, r))

# every axis 0..n-1 exactly once, from axis i
def all_once(left: Nat, +p: List<&2, Nat>, +i: Nat) -> Bool:
  match left:
    case 0n:
      True{}
    case 1n+r:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
      Bool.and(Nat.is_eq(count(i, p), 1n), all_once(r, p, 1n+i))

# p is a permutation of the axes of a rank-n tensor
def is_perm(+p: List<&2, Nat>, +n: Nat) -> Bool:
  # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
  Bool.and(Nat.is_eq(List.length(&2, Nat, p), n), all_once(n, p, 0n))

# reading s at every position of the permuted shape: axis d has s's size and s's stride of axis p[d]; the last first
def perm_axes(p: List<&2, Nat>, +s: List<&2, Nat>, acc: List<&2, Ax>) -> List<&2, Ax>:
  match p:
    case Nil{}:
      acc
    case Con{+k, r}:
      perm_axes(r, s, Ax{U32.from_nat(at1(s, k)), U32.from_nat(inner(s, k))} <> acc)

def perm_map(+s: List<&2, Nat>, +p: List<&2, Nat>) -> IMap:
  Strided{0, perm_axes(p, s, Nil{})}

def permute(+s: List<&2, Nat>, +p: List<&2, Nat>, -pf: {True{} == is_perm(p, rank(s)) : Bool}, x: Tensor<s>) -> TP<s, permuted(s, p)>:
  match x:
    case Tensor{d}:
      tp_of(s, permuted(s, p), gather_l(numel(s), 0, perm_map(s, p), Array.get(F32, d, imap(perm_map(s, p), 0)), new(numel(s))))

# the gradient of permute: dy put back where each number came from
def permute_bwd(+s: List<&2, Nat>, +p: List<&2, Nat>, dy: Tensor<permuted(s, p)>) -> TP<permuted(s, p), s>:
  match dy:
    case Tensor{d}:
      tp_of(permuted(s, p), s, place_l(numel(s), 0, perm_map(s, p), Array.get(F32, d, 0), new(numel(s))))

# 0..n-1 with i and j swapped
def swap_perm(n: Nat, +i: Nat, +a: Nat, +b: Nat) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+r:
      Bool.pick(Nat, Nat.is_eq(i, a), b, Bool.pick(Nat, Nat.is_eq(i, b), a, i)) <> swap_perm(r, 1n+i, a, b)

def transpose_p(+s: List<&2, Nat>, +i: Nat, +j: Nat) -> List<&2, Nat>:
  swap_perm(rank(s), 0n, i, j)

# =====================================================================================================
# Windows: narrow (a range of one axis) and cat (two tensors along one axis)
# =====================================================================================================

# axis d exists and start + len fits in it
def narrow_ok(+s: List<&2, Nat>, +d: Nat, +st: Nat, +l: Nat) -> Bool:
  Bool.and(Nat.is_lt(d, rank(s)), Nat.is_le(Nat.add(st, l), at1(s, d)))

def narrow_map(+s: List<&2, Nat>, +d: Nat, +st: Nat, +l: Nat) -> IMap:
  window(setd(s, d, l), s, U32.from_nat(Nat.mul(st, inner(s, d))))

# positions st .. st + l - 1 of axis d
def narrow(+s: List<&2, Nat>, +d: Nat, +st: Nat, +l: Nat, -pf: {True{} == narrow_ok(s, d, st, l) : Bool}, x: Tensor<s>) -> TP<s, setd(s, d, l)>:
  match x:
    case Tensor{dx}:
      tp_of(s, setd(s, d, l), gather_l(numel(setd(s, d, l)), 0, narrow_map(s, d, st, l), Array.get(F32, dx, imap(narrow_map(s, d, st, l), 0)), new(numel(setd(s, d, l)))))

# the gradient of narrow: dy in its window, zeros elsewhere
def narrow_bwd(+s: List<&2, Nat>, +d: Nat, +st: Nat, +l: Nat, dy: Tensor<setd(s, d, l)>) -> TP<setd(s, d, l), s>:
  match dy:
    case Tensor{dd}:
      tp_of(setd(s, d, l), s, place_l(numel(setd(s, d, l)), 0, narrow_map(s, d, st, l), Array.get(F32, dd, 0), new(numel(s))))

# the same sizes on every axis
def same(a: List<&2, Nat>, b: List<&2, Nat>) -> Bool:
  match a b:
    case Nil{} Nil{}:
      True{}
    case Con{+x, ra} Con{+y, rb}:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
      Bool.and(Nat.is_eq(x, y), same(ra, rb))
    case _ _:
      False{}

# a and b have the same rank, and the same sizes on every axis but d
def cat_eq(a: List<&2, Nat>, b: List<&2, Nat>, d: Nat) -> Bool:
  match a b d:
    case Nil{} Nil{} _:
      True{}
    case Con{x, ra} Con{y, rb} 0n:
      same(ra, rb)
    case Con{+x, ra} Con{+y, rb} 1n+p:
      # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
      Bool.and(Nat.is_eq(x, y), cat_eq(ra, rb, p))
    case _ _ _:
      False{}

def cat_ok(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat) -> Bool:
  # bend-lint: disable-next bend-lint-rules/performance/eager_branch -- a shape predicate over a list of rank <= 4, once per operation; the types and LAWS.bend read its Bool form not-expected-because: a match would change the definition the proofs reduce, for no measurable gain
  Bool.and(Nat.is_lt(d, rank(a)), cat_eq(a, b, d))

def cat_shape(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat) -> List<&2, Nat>:
  setd(a, d, Nat.add(at1(a, d), at1(b, d)))

# where a's and b's positions go in the result
def cat_map_a(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat) -> IMap:
  window(a, cat_shape(a, b, d), 0)

def cat_map_b(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat) -> IMap:
  window(b, cat_shape(a, b, d), U32.from_nat(Nat.mul(at1(a, d), inner(cat_shape(a, b, d), d))))

def t3_cat(-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>, ax: Array<F32>, r: P2) -> T3<a, b, o>:
  match r:
    case P2{bx, o2}:
      T3{Tensor{ax}, Tensor{bx}, Tensor{o2}}

def cat_b(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, rb: Array<F32>, pa: P2) -> T3<a, b, cat_shape(a, b, d)>:
  match pa:
    case P2{ax, o}:
      t3_cat(a, b, cat_shape(a, b, d), ax, place_l(numel(b), 0, cat_map_b(a, b, d), Array.get(F32, rb, 0), o))

# a then b along axis d
def cat(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, -pf: {True{} == cat_ok(a, b, d) : Bool}, x: Tensor<a>, y: Tensor<b>) -> T3<a, b, cat_shape(a, b, d)>:
  match x y:
    case Tensor{dx} Tensor{dy}:
      cat_b(a, b, d, dy, place_l(numel(a), 0, cat_map_a(a, b, d), Array.get(F32, dx, 0), new(numel(cat_shape(a, b, d)))))

# the gradients of cat: dz's window of a, and dz's window of b (dz comes back); returned as TP<a, b> after dz
type CatG<-a: List<&2, Nat>, -b: List<&2, Nat>, -o: List<&2, Nat>> is Type:
  CatG{dz: Tensor<o>, ga: Tensor<a>, gb: Tensor<b>}

def catg_c(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, ga: Array<F32>, r: P2) -> CatG<a, b, cat_shape(a, b, d)>:
  match r:
    case P2{dz, gb}:
      CatG{Tensor{dz}, Tensor{ga}, Tensor{gb}}

def catg_b(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, r: P2) -> CatG<a, b, cat_shape(a, b, d)>:
  match r:
    case P2{dz, ga}:
      catg_c(a, b, d, ga, gather_l(numel(b), 0, cat_map_b(a, b, d), Array.get(F32, dz, imap(cat_map_b(a, b, d), 0)), new(numel(b))))

def cat_bwd(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, dz: Tensor<cat_shape(a, b, d)>) -> CatG<a, b, cat_shape(a, b, d)>:
  match dz:
    case Tensor{dd}:
      catg_b(a, b, d, gather_l(numel(a), 0, cat_map_a(a, b, d), Array.get(F32, dd, imap(cat_map_a(a, b, d), 0)), new(numel(a))))

# =====================================================================================================
# Batched matrix product: [bs, n, k] · [bs, k, m] = [bs, n, m]
# =====================================================================================================

# where the operands of out[b, i, j] = sum over t of X[b·xb + i·xi + t·xt] · Y[b·yb + t·yt + j·yj] are
type Geo is Data:
  Geo{n: U32, m: U32, k: Nat, xb: U32, xi: U32, xt: U32, yb: U32, yt: U32, yj: U32}

type Dot is Type:
  Dot{x: Array<F32>, y: Array<F32>, v: F32}

def Dot.then(-R: Type, d: Dot, k: Array<F32> -> Array<F32> -> F32 -> R) -> R:
  match d:
    case Dot{x, y, v}:
      k(x, y, v)

# acc + the k products from x[xo] and y[yo], stepping xt and yt; rx and ry hold the first pair, already read
def dot_l(k: Nat, +xo: U32, +xt: U32, +yo: U32, +yt: U32, rx: Array<F32> & F32, ry: Array<F32> & F32, acc: F32) -> Dot:
  match k rx ry:
    case 0n Tuple{x, u} Tuple{y, v}:
      Dot{x, y, acc}
    case 1n+p Tuple{x, +u} Tuple{y, +v}:
      dot_l(p, (xo + xt : U32), xt, (yo + yt : U32), yt, Array.get(F32, x, (xo + xt : U32)), Array.get(F32, y, (yo + yt : U32)), (acc + (u * v : F32) : F32))

def x_off(+g: Geo, +j: U32) -> U32:
  match g:
    case Geo{+n, +m, k, +xb, +xi, xt, yb, yt, yj}:
      ((U32.div(j, (n * m : U32)) * xb : U32) + (U32.mod(U32.div(j, m), n) * xi : U32) : U32)

def y_off(+g: Geo, +j: U32) -> U32:
  match g:
    case Geo{+n, +m, k, xb, xi, xt, +yb, yt, +yj}:
      ((U32.div(j, (n * m : U32)) * yb : U32) + (U32.mod(j, m) * yj : U32) : U32)

def geo_k(+g: Geo) -> Nat:
  match g:
    case Geo{n, m, +k, xb, xi, xt, yb, yt, yj}:
      k

def geo_xt(+g: Geo) -> U32:
  match g:
    case Geo{n, m, k, xb, xi, +xt, yb, yt, yj}:
      xt

def geo_yt(+g: Geo) -> U32:
  match g:
    case Geo{n, m, k, xb, xi, xt, yb, +yt, yj}:
      yt

# out[j] for cnt positions from j
def bmm_l(cnt: Nat, +j: U32, +g: Geo, x: Array<F32>, y: Array<F32>, o: Array<F32>) -> P3:
  match cnt:
    case 0n:
      P3{x, y, o}
    case 1n+p:
      Dot.then(P3, dot_l(geo_k(g), x_off(g, j), geo_xt(g), y_off(g, j), geo_yt(g), Array.get(F32, x, x_off(g, j)), Array.get(F32, y, y_off(g, j)), 0.0),
        x2 => y2 => v => bmm_l(p, (j + 1 : U32), g, x2, y2, Array.set(F32, o, j, v)))

def u(+n: Nat) -> U32:
  U32.from_nat(n)

def bmm(+bs: Nat, +n: Nat, +k: Nat, +m: Nat, x: Tensor<[bs, n, k]>, y: Tensor<[bs, k, m]>) -> T3<[bs, n, k], [bs, k, m], [bs, n, m]>:
  match x y:
    case Tensor{dx} Tensor{dy}:
      t3_of([bs, n, k], [bs, k, m], [bs, n, m], bmm_l(Nat.mul(bs, Nat.mul(n, m)), 0, Geo{u(n), u(m), k, u(Nat.mul(n, k)), u(k), 1, u(Nat.mul(k, m)), u(m), 1}, dx, dy, new(Nat.mul(bs, Nat.mul(n, m)))))

# dX = dZ · Yᵀ per batch: [bs, n, m] · [bs, k, m]ᵀ = [bs, n, k]; dZ and Y come back
def bmm_dx(+bs: Nat, +n: Nat, +k: Nat, +m: Nat, dz: Tensor<[bs, n, m]>, y: Tensor<[bs, k, m]>) -> T3<[bs, n, m], [bs, k, m], [bs, n, k]>:
  match dz y:
    case Tensor{dd} Tensor{dy}:
      t3_of([bs, n, m], [bs, k, m], [bs, n, k], bmm_l(Nat.mul(bs, Nat.mul(n, k)), 0, Geo{u(n), u(k), m, u(Nat.mul(n, m)), u(m), 1, u(Nat.mul(k, m)), 1, u(m)}, dd, dy, new(Nat.mul(bs, Nat.mul(n, k)))))

# dY = Xᵀ · dZ per batch: [bs, n, k]ᵀ · [bs, n, m] = [bs, k, m]; X and dZ come back
def bmm_dy(+bs: Nat, +n: Nat, +k: Nat, +m: Nat, x: Tensor<[bs, n, k]>, dz: Tensor<[bs, n, m]>) -> T3<[bs, n, k], [bs, n, m], [bs, k, m]>:
  match x dz:
    case Tensor{dx} Tensor{dd}:
      t3_of([bs, n, k], [bs, n, m], [bs, k, m], bmm_l(Nat.mul(bs, Nat.mul(k, m)), 0, Geo{u(k), u(m), n, u(Nat.mul(n, k)), 1, u(k), u(Nat.mul(n, m)), u(m), 1}, dx, dd, new(Nat.mul(bs, Nat.mul(k, m)))))
