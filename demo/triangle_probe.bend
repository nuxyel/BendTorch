# triangle probe (BendTorch #28): Attention's causal products over the triangle (tmul, tacc) against the full sums
# FullAttention keeps (pmul, pacc), cell by cell in their bits, on the P and G the causal path really builds (probs,
# grows) from fixed numbers of both signs. Four shapes: one with query heads sharing a key/value head away from column
# 0, the ViT block's head (t = 16, hd = 64), a single position, and SmolLM2-135M's (t = 128, 9 heads over 3). Each
# line is a product's cells and how many differ; dK and dV start from nonzero sums (their += is tested too). The last
# line is a control: the same comparison on a P that is not zero past the diagonal, where the two must differ.
# reference/triangle_check.py reads the lines.
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ../torch/main.bend as T

# a[i] = ((i·m + o) mod 1009) / 1009 - 0.5 for i = idx.., left + 1 of them
def fill(left: Nat, +idx: U32, +m: U32, +o: U32, a: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      Array.set(F32, a, idx, ((U32.to_f32(U32.mod(((idx * m : U32) + o : U32), 1009)) / 1009.0 : F32) - 0.5 : F32))
    case 1n+p:
      fill(p, (idx + 1 : U32), m, o, Array.set(F32, a, idx, ((U32.to_f32(U32.mod(((idx * m : U32) + o : U32), 1009)) / 1009.0 : F32) - 0.5 : F32)))

def arr(+cnt: Nat, +m: U32, +o: U32) -> Array<F32>:
  fill(Nat.sub(cnt, 1n), 0, m, o, T.zeros_a(cnt))

def b01(b: Bool) -> U32:
  match b:
    case True{}: 1
    case False{}: 0

# how many of the n cells from idx differ in their bits; ra and rb hold the cells at idx
def ndiff(left: Nat, +idx: U32, +acc: U32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> U32:
  match left ra rb:
    case 0n Tuple{a, +x} Tuple{b, +y}:
      (acc + b01(Bool.not(U32.is_eq(F32.bits(x), F32.bits(y)))) : U32)
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      ndiff(p, (idx + 1 : U32), (acc + b01(Bool.not(U32.is_eq(F32.bits(x), F32.bits(y)))) : U32), Array.get(F32, a, (idx + 1 : U32)), Array.get(F32, b, (idx + 1 : U32)))

def cmp(+n: Nat, a: Array<F32>, b: Array<F32>) -> U32:
  ndiff(Nat.sub(n, 1n), 0, 0, Array.get(F32, a, 0), Array.get(F32, b, 0))

def po_o(r: T.PO) -> Array<F32>:
  match r:
    case T.PO{a, b, o}:
      o

def qk_p(r: T.QK) -> Array<F32>:
  match r:
    case T.QK{q, k, p}:
      p

def gw_g(r: T.GW) -> Array<F32>:
  match r:
    case T.GW{d, v, p, g}:
      g

# the inputs, the same numbers every time they are made: Q (t x w), K and V (t x kvw), dO (t x w)
def mq(+t: Nat, +w: Nat) -> Array<F32>:
  arr(Nat.mul(t, w), 31, 3)

def mk(+t: Nat, +kvw: Nat) -> Array<F32>:
  arr(Nat.mul(t, kvw), 37, 11)

def mv(+t: Nat, +kvw: Nat) -> Array<F32>:
  arr(Nat.mul(t, kvw), 53, 7)

def md(+t: Nat, +w: Nat) -> Array<F32>:
  arr(Nat.mul(t, w), 41, 5)

# P of the head at columns hc (queries) and kc (its key/value head)
def pm(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, +sc: F32) -> Array<F32>:
  qk_p(T.probs(t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, mq(t, w), mk(t, kvw)))

# G of the same head: softmax's backward pass of dP = dO·Vᵀ, scaled
def gm(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, +sc: F32) -> Array<F32>:
  gw_g(T.grows(Nat.sub(t, 1n), 0n, t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, T.grow(0n, t, hd, sc, U32.from_nat(w), hc, U32.from_nat(kvw), kc, md(t, w), mv(t, kvw), pm(t, hd, w, kvw, hc, kc, sc), T.zeros_a(Nat.mul(t, t)))))

# a·b into the head's columns of a zero t x w array (O = P·V, dQ = G·K): the full sum and the triangle's
def mul_full(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  po_o(T.pmul(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(kvw), kc, U32.from_nat(w), hc, T.zeros_a(Nat.mul(t, w)), T.dot0(t, 0, kc, 1, U32.from_nat(kvw), a, b)))

def mul_tri(+t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  po_o(T.tmul(Nat.sub(t, 1n), 0n, hd, t, U32.from_nat(kvw), kc, U32.from_nat(w), hc, T.trow0(0n, hd, t, U32.from_nat(kvw), kc, U32.from_nat(w), hc, a, b, T.zeros_a(Nat.mul(t, w)))))

# aᵀ·b added into a t x hd array that already holds sums (dK += Gᵀ·Q, dV += Pᵀ·dO): the full sum and the triangle's
def acc_full(+t: Nat, +hd: Nat, +w: Nat, +hc: U32, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  po_o(T.pacc(Nat.sub(Nat.mul(t, hd), 1n), 0, U32.from_nat(hd), t, U32.from_nat(w), hc, arr(Nat.mul(t, hd), 43, 2), T.dot0(t, 0, hc, U32.from_nat(t), U32.from_nat(w), a, b)))

def acc_tri(+t: Nat, +hd: Nat, +w: Nat, +hc: U32, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  po_o(T.tacc(Nat.sub(t, 1n), 0n, hd, t, U32.from_nat(w), hc, T.tcol0(0n, Nat.sub(t, 1n), hd, t, U32.from_nat(w), hc, a, b, arr(Nat.mul(t, hd), 43, 2))))

def line(+name: String, +what: String, +n: Nat, +d: U32) -> String:
  name ++ " " ++ what ++ " cells " ++ Nat.show(n) ++ " differing " ++ U32.show(d)

# the four products of one head at one shape, each line: name, product, cells, cells that differ
def shape(+name: String, +t: Nat, +hd: Nat, +w: Nat, +kvw: Nat, +hc: U32, +kc: U32, +sc: F32) -> String:
  +o = cmp(Nat.mul(t, w), mul_full(t, hd, w, kvw, hc, kc, pm(t, hd, w, kvw, hc, kc, sc), mv(t, kvw)), mul_tri(t, hd, w, kvw, hc, kc, pm(t, hd, w, kvw, hc, kc, sc), mv(t, kvw)))
  +dq = cmp(Nat.mul(t, w), mul_full(t, hd, w, kvw, hc, kc, gm(t, hd, w, kvw, hc, kc, sc), mk(t, kvw)), mul_tri(t, hd, w, kvw, hc, kc, gm(t, hd, w, kvw, hc, kc, sc), mk(t, kvw)))
  +dk = cmp(Nat.mul(t, hd), acc_full(t, hd, w, hc, gm(t, hd, w, kvw, hc, kc, sc), mq(t, w)), acc_tri(t, hd, w, hc, gm(t, hd, w, kvw, hc, kc, sc), mq(t, w)))
  +dv = cmp(Nat.mul(t, hd), acc_full(t, hd, w, hc, pm(t, hd, w, kvw, hc, kc, sc), md(t, w)), acc_tri(t, hd, w, hc, pm(t, hd, w, kvw, hc, kc, sc), md(t, w)))
  line(name, "o", Nat.mul(t, w), o) ++ "\n" ++ line(name, "dq", Nat.mul(t, w), dq) ++ "\n" ++ line(name, "dk", Nat.mul(t, hd), dk) ++ "\n" ++ line(name, "dv", Nat.mul(t, hd), dv)

def main() -> IO(Unit):
  do IO<Unit>:
    # t = 7, two key/value heads of 3 shared by 2 query heads each; query head 3 reads columns 9.. of Q and 3.. of K, V
    IO.print(shape("gqa", 7n, 3n, 12n, 6n, 9, 3, 0.57735026))
    # the ViT block's one head: t = 16, hd = 64
    IO.print(shape("vit", 16n, 64n, 64n, 64n, 0, 0, 0.125))
    # a single position: one product a cell
    IO.print(shape("one", 1n, 4n, 4n, 4n, 0, 0, 0.5))
    # SmolLM2-135M: t = 128, 9 query heads over 3 key/value heads of 64; query head 4 reads key/value head 1
    IO.print(shape("smol", 128n, 64n, 576n, 192n, 256, 64, 0.125))
    # the control: a P with numbers past the diagonal, where the full sum and the triangle must differ
    IO.print(line("control", "o", 112n, cmp(112n, mul_full(16n, 7n, 7n, 7n, 0, 0, arr(256n, 59, 13), mv(16n, 7n)), mul_tri(16n, 7n, 7n, 7n, 0, 0, arr(256n, 59, 13), mv(16n, 7n)))))
    return Unit{}
