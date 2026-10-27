# The mathematical model the laws speak about: vectors and matrices as lists of Nat, a matrix as its list of rows.
# Nothing here runs in a model; it is the specification that LAWS.bend states facts about. Nat makes the algebra
# exact (no rounding), so a law proved here says the formulas are right; whether the F32 code computes them to F32's
# precision is tested against PyTorch, never proved (docs/design.md, "Proved, tested, trusted").

import Base
import ./main.bend as T

# the dot product of the common prefix
def dotn(xs: List<&2, Nat>, ys: List<&2, Nat>) -> Nat:
  match xs ys:
    case Con{x, xt} Con{y, yt}:
      Nat.add(Nat.mul(x, y), dotn(xt, yt))
    case _ _:
      0n

# vector sum; the shorter vector counts as padded with zeros
def vadd(xs: List<&2, Nat>, ys: List<&2, Nat>) -> List<&2, Nat>:
  match xs ys:
    case Nil{} _:
      ys
    case Con{x, xt} Nil{}:
      Con{x, xt}
    case Con{x, xt} Con{y, yt}:
      Nat.add(x, y) <> vadd(xt, yt)

def vscale(+k: Nat, xs: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{x, t}:
      Nat.mul(k, x) <> vscale(k, t)

# M·v: one dot product per row (a linear layer's forward pass, y = W·x)
def mv(rows: List<&2, List<&2, Nat>>, +v: List<&2, Nat>) -> List<&2, Nat>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{r, t}:
      dotn(r, v) <> mv(t, v)

# Mᵀ·g: the rows weighted by g and summed (a linear layer's backward pass to its input, dx = Wᵀ·dy)
def mtv(rows: List<&2, List<&2, Nat>>, g: List<&2, Nat>) -> List<&2, Nat>:
  match rows g:
    case Con{r, rt} Con{gi, gt}:
      vadd(vscale(gi, r), mtv(rt, gt))
    case _ _:
      Nil{}

# g ⊗ v: row i is g[i]·v (a linear layer's weight gradient, dW = dy ⊗ x)
def outer(g: List<&2, Nat>, +v: List<&2, Nat>) -> List<&2, List<&2, Nat>>:
  match g:
    case Nil{}:
      Nil{}
    case Con{gi, gt}:
      vscale(gi, v) <> outer(gt, v)

# the sum of the products of matching entries of two matrices (Frobenius inner product)
def frob(a: List<&2, List<&2, Nat>>, b: List<&2, List<&2, Nat>>) -> Nat:
  match a b:
    case Con{ar, at} Con{br, bt}:
      Nat.add(dotn(ar, br), frob(at, bt))
    case _ _:
      0n

# a <= b, by recursion (simpler to reason about than Nat.is_le, which goes through Nat.cmp)
def le(a: Nat, b: Nat) -> Bool:
  match a b:
    case 0n _:
      True{}
    case 1n+p 0n:
      False{}
    case 1n+p 1n+q:
      le(p, q)

# ---- linear layers of the model as BendTorch layers, so the laws can run T.Seq and T.Fork's own code on them ----

# row-wise sum of two matrices (the shorter counts as padded)
def madd(a: List<&2, List<&2, Nat>>, b: List<&2, List<&2, Nat>>) -> List<&2, List<&2, Nat>>:
  match a b:
    case Nil{} _:
      b
    case Con{r, rt} Nil{}:
      Con{r, rt}
    case Con{r, rt} Con{s, st}:
      vadd(r, s) <> madd(rt, st)

# y = W·x, saving x
def lin_fwd(+w: List<&2, List<&2, Nat>>, +x: List<&2, Nat>) -> T.Fwd<List<&2, List<&2, Nat>>, List<&2, Nat>, List<&2, Nat>>:
  T.Fwd{w, mv(w, x), x}

# dx = Wᵀ·dy, and dW += dy ⊗ x
def lin_bwd(+w: List<&2, List<&2, Nat>>, +x: List<&2, Nat>, +dy: List<&2, Nat>, g: List<&2, List<&2, Nat>>) -> T.Bwd<List<&2, List<&2, Nat>>, List<&2, Nat>>:
  T.Bwd{w, mtv(w, dy), madd(g, outer(dy, x))}

def LinL() -> T.Layer<List<&2, List<&2, Nat>>, List<&2, Nat>, List<&2, Nat>, List<&2, Nat>>:
  T.Layer{p => x => lin_fwd(p, x), p => s => dy => g => lin_bwd(p, s, dy, g), op => p => g => st => T.no_step(List<&2, List<&2, Nat>>, op, p, g, st), u => Nil{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# two linear layers in sequence, and side by side on one input, through BendTorch's combinators
def SeqL() -> T.Layer<T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>>:
  T.Seq(~List<&2, List<&2, Nat>>, ~List<&2, List<&2, Nat>>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~LinL(), ~LinL())

def dupv(+x: List<&2, Nat>) -> T.Two<List<&2, Nat>, List<&2, Nat>>:
  T.Two{x, x}

def ForkL() -> T.Layer<T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, List<&2, Nat>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>>:
  T.Fork(~List<&2, List<&2, Nat>>, ~List<&2, List<&2, Nat>>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~(x => dupv(x)), ~(a => b => vadd(a, b)), ~LinL(), ~LinL())

# two linear layers on the two vectors of a pair, through T.Both
def BothL() -> T.Layer<T.Two<List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>, T.Two<List<&2, Nat>, List<&2, Nat>>>:
  T.Both(~List<&2, List<&2, Nat>>, ~List<&2, List<&2, Nat>>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~List<&2, Nat>, ~LinL(), ~LinL())

# a + b for a pair of vectors; the backward pass hands dy to both (T.Add2 over vectors)
def add_fwd(+p: Unit, x: T.Two<List<&2, Nat>, List<&2, Nat>>) -> T.Fwd<Unit, List<&2, Nat>, Unit>:
  match x:
    case T.Two{a, b}:
      T.Fwd{p, vadd(a, b), Unit{}}

def AddL() -> T.Layer<Unit, T.Two<List<&2, Nat>, List<&2, Nat>>, List<&2, Nat>, Unit>:
  T.Layer{p => x => add_fwd(p, x), p => s => dy => g => T.Bwd{p, dupv(dy), g}, op => p => g => st => T.no_step(Unit, op, p, g, st), u => Unit{}, p => T.PL{p, Nil{}}, p => xs => off => T.PA{p, xs, off}, p => xs => off => T.PA{p, xs, off}, u => 0}

# x + W·x through T.ResidualOf, the code of T.Residual: Seq(Fork(Id, linear), AddL)
def ResL() -> T.Layer<T.Two<T.Two<Unit, List<&2, List<&2, Nat>>>, Unit>, List<&2, Nat>, List<&2, Nat>, T.Two<T.Two<Unit, List<&2, Nat>>, Unit>>:
  T.ResidualOf(~List<&2, Nat>, ~List<&2, List<&2, Nat>>, ~List<&2, Nat>, ~(x => dupv(x)), ~(a => b => vadd(a, b)), ~AddL(), ~LinL())

def fy(-P: Type, -Y: Type, -S: Type, r: T.Fwd<P, Y, S>) -> Y:
  match r:
    case T.Fwd{p, y, s}:
      y

def fsv(-P: Type, -Y: Type, -S: Type, r: T.Fwd<P, Y, S>) -> S:
  match r:
    case T.Fwd{p, y, s}:
      s

def bdx(-P: Type, -X: Type, r: T.Bwd<P, X>) -> X:
  match r:
    case T.Bwd{p, dx, g}:
      dx

# the dot product of two pairs of vectors
def pdot(a: T.Two<List<&2, Nat>, List<&2, Nat>>, b: T.Two<List<&2, Nat>, List<&2, Nat>>) -> Nat:
  match a b:
    case T.Two{a1, a2} T.Two{b1, b2}:
      Nat.add(dotn(a1, b1), dotn(a2, b2))

# ---- The order of a row's additions (the law block_sum). A sum here is the tree of additions it makes over the cells'
# positions, not a number: two ways of summing that build the same tree add the same pairs in the same order, so they
# give the same bits in F32, or in any arithmetic, with nothing assumed about rounding. ----

type Sum is Data:
  SCell{i: Nat}
  SAdd{a: Sum, b: Sum}

# 2^p
def pw(p: Nat) -> Nat:
  match p:
    case 0n:
      1n
    case 1n+q:
      +h = pw(q)
      Nat.add(h, h)

# the 2^p cells from cell i summed as a balanced tree of pairs: how a GPU leaf sums its block (the device's MSE and
# cross-entropy block sums)
def bal(+p: Nat, +i: Nat) -> Sum:
  match p:
    case 0n:
      SCell{i}
    case 1n+q:
      SAdd{bal(q, i), bal(q, Nat.add(i, pw(q)))}

# k blocks of 2^p cells, the first from cell lo, each summed by bal; with p = 0, a row of k cells
def blocks(+p: Nat, k: Nat, +lo: Nat) -> List<&2, Sum>:
  match k:
    case 0n:
      Nil{}
    case 1n+j:
      bal(p, lo) <> blocks(p, j, Nat.add(lo, pw(p)))

# one pass of main.bend's tree_sum (ts_pass): neighbours added in pairs, an odd last cell carried up as it is
def spass(xs: List<&2, Sum>) -> List<&2, Sum>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{a, Nil{}}:
      a <> Nil{}
    case Con{a, Con{b, t}}:
      SAdd{a, b} <> spass(t)

# r passes of tree_sum; tree_sum (ts_go) stops once one cell is left, and a pass leaves one cell as it is, so the sum of
# m cells is passes(r, row) for any r at least the number of passes it makes
def passes(r: Nat, xs: List<&2, Sum>) -> List<&2, Sum>:
  match r:
    case 0n:
      xs
    case 1n+q:
      spass(passes(q, xs))
