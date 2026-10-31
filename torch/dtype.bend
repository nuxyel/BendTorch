# BendTorch's dtypes (plan M11): BF16 and F16 stored two per U32, int8 with a scale per row stored four per U32, a
# two-F32 ("double-word") accumulator, and the layers that give a model BF16 storage around F32 arithmetic.
#
# Every number is computed in F32. A 16-bit value lives in the low 16 bits of a U32 while it is handled alone, and two
# of them share a word in storage (element 2k in the low half of word k, element 2k + 1 in the high half: the bytes of
# a little-endian torch.bfloat16 / torch.float16 tensor). The conversions from F32 round to nearest, ties to even, and
# treat NaN as PyTorch 2.14.1's CPU kernels do on this machine (tested bit for bit by reference/test_dtype.py):
#   - F32 -> BF16: every NaN becomes 0xFFFF (the vectorized kernel's answer; c10's scalar code would give 0x7FC0).
#   - F32 -> F16: a NaN keeps its sign and the top 9 bits of its payload, with the quiet bit set (F16C's vcvtps2ph).
#   - F16 -> F32: a NaN gets the quiet bit set; subnormals become normal F32 numbers, exactly.
# No @unsafe and no foreign code: plain Bend over U32 and F32 bit operations.

import Base
import ./main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

# =====================================================================================================
# Bits
# =====================================================================================================

# U32 and F32 hold the same Word(32n)
def u2f(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def f2u(x: F32) -> U32:
  match x:
    case F32{w}:
      U32{w}

def pick_u(b: Bool, +x: U32, +y: U32) -> U32:
  match b:
    case True{}:
      x
    case False{}:
      y

# =====================================================================================================
# BF16: the top 16 bits of an F32
# =====================================================================================================

# F32 -> BF16, round to nearest even: add 0x7FFF plus the lowest kept bit, keep the top half. The carry rounds up into
# the exponent when the mantissa overflows, and past the largest finite value to infinity, as IEEE asks.
def bf16_rne(+u: U32) -> U32:
  U32.shrn(((u + 32767 : U32) + (U32.shrn(u, 16n) .&. 1 : U32) : U32), 16n)

def bf16_of_bits(+u: U32) -> U32:
  pick_u(U32.is_gt((u .&. 2147483647 : U32), 2139095040), 65535, bf16_rne(u))

def BF16.of_f32(x: F32) -> U32:
  bf16_of_bits(f2u(x))

# BF16 -> F32 is exact: the 16 bits become the top half
def BF16.to_f32(+h: U32) -> F32:
  u2f(U32.shln((h .&. 65535 : U32), 16n))

# x rounded to the nearest BF16 value, as an F32 (x.to(torch.bfloat16).to(torch.float32))
def BF16.round(+x: F32) -> F32:
  BF16.to_f32(BF16.of_f32(x))

# =====================================================================================================
# F16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits
# =====================================================================================================

# a normal F16 (|x| >= 2^-14): rebias the exponent (127 - 15 = 112), round the 23-bit mantissa to 10 bits to nearest
# even; past 65504 the carry reaches the infinity pattern 0x7C00 and anything above it is infinity too
def f16_normal(+a: U32) -> U32:
  +m = (a - 939524096 : U32)
  +r = U32.shrn(((m + 4095 : U32) + (U32.shrn(m, 13n) .&. 1 : U32) : U32), 13n)
  pick_u(U32.is_gt(r, 31744), 31744, r)

# a subnormal F16 (2^-25 < |x| < 2^-14): the value in units of 2^-24 is mant / 2^sh, sh = 126 - e in 14..24, rounded to
# nearest even; a result of 0x400 is the smallest normal, which is the right answer
def f16_sub2(+q: U32, +rem: U32, +half: U32) -> U32:
  +up = Bool.or(U32.is_gt(rem, half), Bool.and(U32.is_eq(rem, half), U32.is_eq((q .&. 1 : U32), 1)))
  (q + Bool.to_u32(up) : U32)

def f16_sub(+a: U32) -> U32:
  +mant = ((a .&. 8388607 : U32) .|. 8388608 : U32)
  +sh = U32.to_nat((126 - U32.shrn(a, 23n) : U32))
  f16_sub2(U32.shrn(mant, sh), (mant .&. (U32.shln(1, sh) - 1 : U32) : U32), U32.shln(1, Nat.sub(sh, 1n)))

# a NaN keeps the top 9 bits of its payload and gets the quiet bit
def f16_nan(+a: U32) -> U32:
  (32256 .|. U32.shrn((a .&. 8388607 : U32), 13n) : U32)

def f16_mag(nan: Bool, normal: Bool, tiny: Bool, +a: U32) -> U32:
  match nan normal tiny:
    case True{} _ _:
      f16_nan(a)
    case False{} True{} _:
      f16_normal(a)
    case False{} False{} True{}:
      0
    case False{} False{} False{}:
      f16_sub(a)

# F32 -> F16, round to nearest even, with the sign kept (so -0.0 and the negative tiny numbers give 0x8000)
def f16_of_bits(+u: U32) -> U32:
  +a = (u .&. 2147483647 : U32)
  ((U32.shrn(u, 16n) .&. 32768 : U32) .|. f16_mag(U32.is_gt(a, 2139095040), U32.is_ge(a, 947912704), U32.is_le(a, 855638016), a) : U32)

def F16.of_f32(x: F32) -> U32:
  f16_of_bits(f2u(x))

# F16 -> F32, exact. A subnormal is m · 2^-24: the product of two F32 numbers that is exact and normal.
def f16_up(zero: Bool, top: Bool, +m: U32, +s: U32, +eu: U32) -> F32:
  match zero top:
    case True{} _:
      u2f((s .|. f2u((U32.to_f32(m) * 0.000000059604644775390625 : F32)) : U32))
    case False{} True{}:
      u2f((((s .|. 2139095040 : U32) .|. U32.shln(m, 13n) : U32) .|. pick_u(U32.is_eq(m, 0), 0, 4194304) : U32))
    case False{} False{}:
      u2f(((s .|. U32.shln((eu + 112 : U32), 23n) : U32) .|. U32.shln(m, 13n) : U32))

def F16.to_f32(+h: U32) -> F32:
  +eu = (U32.shrn(h, 10n) .&. 31 : U32)
  f16_up(U32.is_eq(eu, 0), U32.is_eq(eu, 31), (h .&. 1023 : U32), U32.shln((h .&. 32768 : U32), 16n), eu)

def F16.round(+x: F32) -> F32:
  F16.to_f32(F16.of_f32(x))

# =====================================================================================================
# Two halves in a word
# =====================================================================================================

def half_join(+lo: U32, +hi: U32) -> U32:
  ((lo .&. 65535 : U32) .|. U32.shln(hi, 16n) : U32)

def half_lo(+w: U32) -> U32:
  (w .&. 65535 : U32)

def half_hi(+w: U32) -> U32:
  U32.shrn(w, 16n)

# a pair of 16-bit values, the unit of storage
type HW is Data:
  HW{lo: U32, hi: U32}

# the values two by two; an odd last one is paired with 0
def pairs(xs: List<&2, U32>) -> List<&2, HW>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{a, Nil{}}:
      [HW{a, 0}]
    case Con{a, Con{b, t}}:
      HW{a, b} <> pairs(t)

# every value of a list of pairs, in order
def unpairs(ws: List<&2, HW>) -> List<&2, U32>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{HW{a, b}, t}:
      a <> (b <> unpairs(t))

# the first n of them: the padding of an odd count is left out
def unpairs_n(+n: Nat, ws: List<&2, HW>) -> List<&2, U32>:
  List.take(&2, U32, unpairs(ws), n)

# =====================================================================================================
# Packed 16-bit matrices: r·c values in ceil(r·c / 2) words, element 2k in the low half of word k
# =====================================================================================================

type HMat<-r: Nat, -c: Nat> is Type:
  HMat{d: Array<U32>}

type UL is Type:
  UL{a: Array<U32>, l: List<&2, U32>}

def fill_u(xs: List<&2, U32>, +i: U32, a: Array<U32>) -> Array<U32>:
  match xs:
    case Nil{}:
      a
    case Con{h, t}:
      fill_u(t, (i + 1 : U32), Array.set(U32, a, i, h))

# the k + 1 values from index i, in order (ra holds the array and the value at i, already read)
def read_u(k: Nat, +i: U32, acc: List<&2, U32>, ra: Array<U32> & U32) -> UL:
  match k ra:
    case 0n Tuple{a, v}:
      UL{a, List.reverse(&2, U32, v <> acc)}
    case 1n+p Tuple{a, +v}:
      read_u(p, (i + 1 : U32), v <> acc, Array.get(U32, a, (i + 1 : U32)))

def ul_list(r: UL) -> List<&2, U32>:
  match r:
    case UL{a, l}:
      l

def enc_l(~f: F32 -> U32, xs: List<&2, F32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      f(h) <> enc_l(~f, t)

def dec_l(~f: @+h: U32 -> F32, xs: List<&2, U32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      f(h) <> dec_l(~f, t)

def join_l(ws: List<&2, HW>) -> List<&2, U32>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{HW{+a, +b}, t}:
      half_join(a, b) <> join_l(t)

def split_l(ws: List<&2, U32>) -> List<&2, HW>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{+w, t}:
      HW{half_lo(w), half_hi(w)} <> split_l(t)

# the number of words for n halves
def hwords(+n: Nat) -> Nat:
  Nat.div(Nat.add(n, 1n), 2n)

def ml_l(-r: Nat, -c: Nat, m: TA.ML<r, c>) -> List<&2, F32>:
  match m:
    case TA.ML{d, xs}:
      xs

# the values of a 16-bit list packed two per word
def Half.words(+n: Nat, hs: List<&2, U32>) -> Array<U32>:
  fill_u(join_l(pairs(hs)), 0, Array.new(U32, T.depth(hwords(n)), 0))

# the first n 16-bit values of a packed array
def Half.values(+n: Nat, a: Array<U32>) -> List<&2, U32>:
  unpairs_n(n, split_l(ul_list(read_u(Nat.sub(hwords(n), 1n), 0, Nil{}, Array.get(U32, a, 0)))))

# a matrix encoded by f (BF16.of_f32 or F16.of_f32) and packed
def Half.pack(~f: F32 -> U32, +r: Nat, +c: Nat, m: TA.Mat<r, c>) -> HMat<r, c>:
  HMat{Half.words(Nat.mul(r, c), enc_l(~f, ml_l(r, c, TA.Mat.to_list(r, c, m))))}

# a packed matrix decoded by g (BF16.to_f32 or F16.to_f32) into F32
def Half.unpack(~g: @+h: U32 -> F32, +r: Nat, +c: Nat, h: HMat<r, c>) -> TA.Mat<r, c>:
  match h:
    case HMat{d}:
      TA.Mat.from_list(r, c, dec_l(~g, Half.values(Nat.mul(r, c), d)))

def BF16.pack(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> HMat<r, c>:
  Half.pack(~BF16.of_f32, r, c, m)

def BF16.unpack(+r: Nat, +c: Nat, h: HMat<r, c>) -> TA.Mat<r, c>:
  Half.unpack(~BF16.to_f32, r, c, h)

def F16.pack(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> HMat<r, c>:
  Half.pack(~F16.of_f32, r, c, m)

def F16.unpack(+r: Nat, +c: Nat, h: HMat<r, c>) -> TA.Mat<r, c>:
  Half.unpack(~F16.to_f32, r, c, h)

# the words of a packed matrix, for printing and tests
def HMat.words(+r: Nat, +c: Nat, h: HMat<r, c>) -> List<&2, U32>:
  match h:
    case HMat{d}:
      ul_list(read_u(Nat.sub(hwords(Nat.mul(r, c)), 1n), 0, Nil{}, Array.get(U32, d, 0)))

# =====================================================================================================
# Laws of the packing
# =====================================================================================================

# LAW (unpair_pair): packing a list two by two and reading back as many values as it had gives the list back, in
# order, for any length (an odd last value is paired with a 0 that the read leaves out). Proof: by induction two
# values at a time; the pair HW{a, b} unpacks to a then b, and the rest is the induction hypothesis.
def unpair_pair(xs: List<&2, U32>) -> {List.take(&2, U32, unpairs(pairs(xs)), List.length(&2, U32, xs)) == xs : List<&2, U32>}:
  match xs:
    case Nil{}:
      {==}
    case Con{a, Nil{}}:
      {==}
    case Con{+a, Con{+b, +t}}:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- the binder of Equal.cong's motive, used in its body not-expected-because: a law, never run
      Equal.cong(List<&2, U32>, List<&2, U32>, l => a <> (b <> l), List.take(&2, U32, unpairs(pairs(t)), List.length(&2, U32, t)), t, unpair_pair(t))

# Not proved (debt, tested instead by reference/test_dtype.py):
#   - half_lo(half_join(a, b)) == a and half_hi(half_join(a, b)) == b for 16-bit a and b: a fact about Word(32n)'s bit
#     operations for every value, which the checker does not reach by computation (`dtype_test joins` checks all 65536
#     values of each half against a second one walking every value);
#   - that fill_u and read_u give back what was written (Array.set then Array.get), and the int8 words' byte order;
#   - the rounding itself: F32 is opaque to the checker, so BF16, F16 and int8 are tested bit for bit against PyTorch.

# =====================================================================================================
# int8 with a scale per row, for inference
# =====================================================================================================
# Row i of an r x c weight matrix W gets the scale s_i = max_j |W[i, j]| / 127 and the integers
# q[i, j] = round(W[i, j] / s_i), to nearest even, in -127..127 (0 for a row of zeros). q is stored four per word,
# element 4k + b in byte b of word k, as two's complement bytes: the bytes of a torch.int8 tensor. A product with F32
# activations reads q straight from the words: y[n, i] = s_i · Σ_j x[n, j] · q[i, j], summed in order of j.

type Q8<-r: Nat, -c: Nat> is Type:
  Q8{q: Array<U32>, s: Array<F32>}

def pick_f(b: Bool, +x: F32, +y: F32) -> F32:
  match b:
    case True{}:
      x
    case False{}:
      y

# v rounded to the nearest integer, ties to even (torch.round), for |v| < 2^23 (here |v| <= 127 and a bit)
def rne_f(+v: F32) -> F32:
  +f = F32.floor(v)
  +d = (v - f : F32)
  +h = (f * 0.5 : F32)
  +up = Bool.or(F32.is_gt(d, 0.5), Bool.and(F32.is_eq(d, 0.5), F32.is_ne(F32.floor(h), h)))
  pick_f(up, (f + 1.0 : F32), f)

def q8_of(+w: F32, +s: F32) -> F32:
  pick_f(F32.is_eq(s, 0.0), 0.0, F32.clamp(rne_f((w / s : F32)), F32.neg(127.0), 127.0))

# an integer -127..127 (as an F32) to its two's complement byte, and back
def q8_byte(+q: F32) -> U32:
  ((F32.to_u32((q + 128.0 : F32)) + 128 : U32) .&. 255 : U32)

def q8_val(+b: U32) -> F32:
  (U32.to_f32(((b + 128 : U32) .&. 255 : U32)) - 128.0 : F32)

def amax_l(xs: List<&2, F32>, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      amax_l(t, F32.max(acc, F32.abs(h)))

def q8_row(xs: List<&2, F32>, +s: F32) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      q8_byte(q8_of(h, s)) <> q8_row(t, s)

type QR is Type:
  QR{q: List<&2, U32>, s: List<&2, F32>}

def qr_add(+row: List<&2, F32>, r: QR) -> QR:
  match r:
    case QR{qs, ss}:
      +s = (amax_l(row, 0.0) / 127.0 : F32)
      QR{List.append(&2, U32, q8_row(row, s), qs), s <> ss}

# the rows of a flat list of k rows of c, from the last one: each row's bytes and scale
def q8_rows(k: Nat, +c: Nat, +xs: List<&2, F32>) -> QR:
  match k:
    case 0n:
      QR{Nil{}, Nil{}}
    case 1n+p:
      qr_add(List.take(&2, F32, xs, c), q8_rows(p, c, List.drop(&2, F32, xs, c)))

# four bytes to a word, little-endian; a short last group is padded with zeros
def quads(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+a, Nil{}}:
      [a]
    case Con{+a, Con{+b, Nil{}}}:
      [(a .|. U32.shln(b, 8n) : U32)]
    case Con{+a, Con{+b, Con{+c, Nil{}}}}:
      [((a .|. U32.shln(b, 8n) : U32) .|. U32.shln(c, 16n) : U32)]
    case Con{+a, Con{+b, Con{+c, Con{+d, t}}}}:
      (((a .|. U32.shln(b, 8n) : U32) .|. U32.shln(c, 16n) : U32) .|. U32.shln(d, 24n) : U32) <> quads(t)

def fill_f(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      a
    case Con{h, t}:
      fill_f(t, (i + 1 : U32), Array.set(F32, a, i, h))

def q8_fin(-r: Nat, -c: Nat, +n: Nat, +rr: Nat, qr: QR) -> Q8<r, c>:
  match qr:
    case QR{qs, ss}:
      Q8{fill_u(quads(qs), 0, Array.new(U32, T.depth(Nat.div(Nat.add(n, 3n), 4n)), 0)), fill_f(ss, 0, Array.new(F32, T.depth(rr), 0.0))}

# W quantized by rows (the matrix is consumed)
def Q8.quantize(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> Q8<r, c>:
  q8_fin(r, c, Nat.mul(r, c), r, q8_rows(r, c, ml_l(r, c, TA.Mat.to_list(r, c, m))))

type FL is Type:
  FL{a: Array<F32>, l: List<&2, F32>}

def read_f(k: Nat, +i: U32, acc: List<&2, F32>, ra: Array<F32> & F32) -> FL:
  match k ra:
    case 0n Tuple{a, v}:
      FL{a, List.reverse(&2, F32, v <> acc)}
    case 1n+p Tuple{a, +v}:
      read_f(p, (i + 1 : U32), v <> acc, Array.get(F32, a, (i + 1 : U32)))

def fl_list(r: FL) -> List<&2, F32>:
  match r:
    case FL{a, l}:
      l

# byte e of the packed words, as its integer
def q8_at(+e: U32, +w: U32) -> F32:
  q8_val((U32.shrn(w, U32.to_nat((8 * (e .&. 3 : U32) : U32))) .&. 255 : U32))

# the words and scales, for printing and tests
type QL is Type:
  QL{q: List<&2, U32>, s: List<&2, F32>}

def Q8.lists(+r: Nat, +c: Nat, w: Q8<r, c>) -> QL:
  match w:
    case Q8{q, s}:
      QL{ul_list(read_u(Nat.sub(Nat.div(Nat.add(Nat.mul(r, c), 3n), 4n), 1n), 0, Nil{}, Array.get(U32, q, 0))), fl_list(read_f(Nat.sub(r, 1n), 0, Nil{}, Array.get(F32, s, 0)))}

# W back in F32: q[i, j] · s_i, one F32 product each
def dq_row(k: Nat, +e: U32, +s: F32, acc: List<&2, F32>, rq: Array<U32> & U32) -> UF:
  match k rq:
    case 0n Tuple{q, w}:
      UF{q, acc}
    case 1n+p Tuple{q, +w}:
      dq_row(p, (e + 1 : U32), s, (q8_at(e, w) * s : F32) <> acc, Array.get(U32, q, U32.shrn((e + 1 : U32), 2n)))

type UF is Type:
  UF{q: Array<U32>, l: List<&2, F32>}

def dq_rows(k: Nat, +c: Nat, +cu: U32, +i: U32, ss: List<&2, F32>, r: UF) -> List<&2, F32>:
  match k ss r:
    case 0n _ UF{q, acc}:
      List.reverse(&2, F32, acc)
    case 1n+p Nil{} UF{q, acc}:
      List.reverse(&2, F32, acc)
    case 1n+p Con{+s, st} UF{q, acc}:
      dq_rows(p, c, cu, (i + 1 : U32), st, dq_row(c, (i * cu : U32), s, acc, Array.get(U32, q, U32.shrn((i * cu : U32), 2n))))

def dq_fin(+r: Nat, +c: Nat, q: Array<U32>, ss: List<&2, F32>) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, dq_rows(r, c, U32.from_nat(c), 0, ss, UF{q, Nil{}}))

def dq_go(+r: Nat, +c: Nat, q: Array<U32>, sl: FL) -> TA.Mat<r, c>:
  match sl:
    case FL{s, ss}:
      dq_fin(r, c, q, ss)

def Q8.dequantize(+r: Nat, +c: Nat, w: Q8<r, c>) -> TA.Mat<r, c>:
  match w:
    case Q8{q, s}:
      dq_go(r, c, q, read_f(Nat.sub(r, 1n), 0, Nil{}, Array.get(F32, s, 0)))

# ---- the product y = x · Wᵀ (n x m) for x (n x k) in F32 and W (m x k) in int8, reading the bytes from the words ----

type QD is Type:
  QD{x: Array<F32>, q: Array<U32>, acc: F32}

# Σ_l x[xi + l] · q[qe + l] for l < k, in order, from acc
def q8_dot(k: Nat, +xi: U32, +qe: U32, +acc: F32, rx: Array<F32> & F32, rq: Array<U32> & U32) -> QD:
  match k rx rq:
    case 0n Tuple{x, v} Tuple{q, w}:
      QD{x, q, acc}
    case 1n+p Tuple{x, +v} Tuple{q, +w}:
      q8_dot(p, (xi + 1 : U32), (qe + 1 : U32), (acc + (v * q8_at(qe, w) : F32) : F32), Array.get(F32, x, (xi + 1 : U32)), Array.get(U32, q, U32.shrn((qe + 1 : U32), 2n)))

type QS is Type:
  QS{x: Array<F32>, q: Array<U32>, s: Array<F32>, y: Array<F32>}

def q8_put2(+idx: U32, +acc: F32, x: Array<F32>, q: Array<U32>, rs: Array<F32> & F32, y: Array<F32>) -> QS:
  match rs:
    case Tuple{s, +sc}:
      QS{x, q, s, Array.set(F32, y, idx, (sc * acc : F32))}

def q8_put(+idx: U32, +j: U32, d: QD, s: Array<F32>, y: Array<F32>) -> QS:
  match d:
    case QD{x, q, acc}:
      q8_put2(idx, acc, x, q, Array.get(F32, s, j), y)

# the m outputs of row i
def q8_cols(ml: Nat, +i: U32, +j: U32, +k: Nat, +ku: U32, +mu: U32, st: QS) -> QS:
  match ml st:
    case 0n QS{x, q, s, y}:
      QS{x, q, s, y}
    case 1n+p QS{x, q, s, y}:
      q8_cols(p, i, (j + 1 : U32), k, ku, mu, q8_put(((i * mu : U32) + j : U32), j, q8_dot(k, (i * ku : U32), (j * ku : U32), 0.0, Array.get(F32, x, (i * ku : U32)), Array.get(U32, q, U32.shrn((j * ku : U32), 2n))), s, y))

def q8_rowsm(nl: Nat, +i: U32, +m: Nat, +k: Nat, +ku: U32, +mu: U32, st: QS) -> QS:
  match nl:
    case 0n:
      st
    case 1n+p:
      q8_rowsm(p, (i + 1 : U32), m, k, ku, mu, q8_cols(m, i, 0, k, ku, mu, st))

type Q8Mul<-n: Nat, -k: Nat, -m: Nat> is Type:
  Q8Mul{x: TA.Mat<n, k>, w: Q8<m, k>, y: TA.Mat<n, m>}

def q8_mm_fin(-n: Nat, -k: Nat, -m: Nat, st: QS) -> Q8Mul<n, k, m>:
  match st:
    case QS{x, q, s, y}:
      Q8Mul{TA.Mat{x}, Q8{q, s}, TA.Mat{y}}

# y = x · Wᵀ; x and W come back
def Q8.matmul_nt(+n: Nat, +k: Nat, +m: Nat, x: TA.Mat<n, k>, w: Q8<m, k>) -> Q8Mul<n, k, m>:
  match x w:
    case TA.Mat{xa} Q8{q, s}:
      q8_mm_fin(n, k, m, q8_rowsm(n, 0, m, k, U32.from_nat(k), U32.from_nat(m), QS{xa, q, s, T.zeros_a(Nat.mul(n, m))}))

# =====================================================================================================
# A two-F32 ("double-word") accumulator, for long sums (an experiment: Bend 2.0.37 has no F64)
# =====================================================================================================
# The running sum is hi + lo, with |lo| <= ulp(hi) / 2. Each addition is Knuth's TwoSum (the exact error of hi + x),
# folded into lo, then renormalized; a product is first split exactly by Dekker's TwoProduct (Veltkamp's split with
# 4097 = 2^12 + 1), which needs every operation rounded on its own: Bend's emitted C turns off FMA contraction.

type DW is Data:
  DW{hi: F32, lo: F32}

def DW.zero() -> DW:
  DW{0.0, 0.0}

def DW.value(d: DW) -> F32:
  match d:
    case DW{+hi, +lo}:
      (hi + lo : F32)

# hi + lo when |hi| >= |lo| (Fast2Sum), as a renormalized pair
def dw_norm(+s: F32, +e: F32) -> DW:
  +h = (s + e : F32)
  DW{h, (e - (h - s : F32) : F32)}

# hi + x: TwoSum's s and its exact error, plus the old lo and an extra error term c
def dw_fin(+s: F32, +bb: F32, +a: F32, +x: F32, +lo: F32, +c: F32) -> DW:
  +e = ((a - (s - bb : F32) : F32) + (x - bb : F32) : F32)
  dw_norm(s, ((e + lo : F32) + c : F32))

def dw_add3(d: DW, +x: F32, +c: F32) -> DW:
  match d:
    case DW{+hi, +lo}:
      +s = (hi + x : F32)
      dw_fin(s, (s - hi : F32), hi, x, lo, c)

def DW.add(d: DW, +x: F32) -> DW:
  dw_add3(d, x, 0.0)

def split_hi(+a: F32) -> F32:
  +t = (4097.0 * a : F32)
  (t - (t - a : F32) : F32)

# a · b exactly as p + e
def two_prod_e(+a: F32, +b: F32, +p: F32) -> F32:
  +ah = split_hi(a)
  +al = (a - ah : F32)
  +bh = split_hi(b)
  +bl = (b - bh : F32)
  ((((((ah * bh : F32) - p : F32) + (ah * bl : F32) : F32) + (al * bh : F32) : F32)) + (al * bl : F32) : F32)

# acc + a · b, with the product's rounding error kept
def DW.fma(d: DW, +a: F32, +b: F32) -> DW:
  +p = (a * b : F32)
  dw_add3(d, p, two_prod_e(a, b, p))

# ---- sums over arrays: a running F32 sum, and the same in double-word ----

type AF is Type:
  AF{a: Array<F32>, v: F32}

def sum_f(k: Nat, +i: U32, +acc: F32, ra: Array<F32> & F32) -> AF:
  match k ra:
    case 0n Tuple{a, v}:
      AF{a, acc}
    case 1n+p Tuple{a, +v}:
      sum_f(p, (i + 1 : U32), (acc + v : F32), Array.get(F32, a, (i + 1 : U32)))

def sum_dw(k: Nat, +i: U32, acc: DW, ra: Array<F32> & F32) -> AF:
  match k ra:
    case 0n Tuple{a, v}:
      AF{a, DW.value(acc)}
    case 1n+p Tuple{a, +v}:
      sum_dw(p, (i + 1 : U32), DW.add(acc, v), Array.get(F32, a, (i + 1 : U32)))

type AF2 is Type:
  AF2{a: Array<F32>, b: Array<F32>, v: F32}

def dot_f(k: Nat, +i: U32, +acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> AF2:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      AF2{a, b, acc}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dot_f(p, (i + 1 : U32), (acc + (x * y : F32) : F32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def dot_dw(k: Nat, +i: U32, acc: DW, ra: Array<F32> & F32, rb: Array<F32> & F32) -> AF2:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      AF2{a, b, DW.value(acc)}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      dot_dw(p, (i + 1 : U32), DW.fma(acc, x, y), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

# the sums of the first n numbers of a (the array comes back)
def Sum.f32(+n: Nat, a: Array<F32>) -> AF:
  sum_f(n, 0, 0.0, Array.get(F32, a, 0))

def Sum.dw(+n: Nat, a: Array<F32>) -> AF:
  sum_dw(n, 0, DW.zero(), Array.get(F32, a, 0))

def Dot.f32(+n: Nat, a: Array<F32>, b: Array<F32>) -> AF2:
  dot_f(n, 0, 0.0, Array.get(F32, a, 0), Array.get(F32, b, 0))

def Dot.dw(+n: Nat, a: Array<F32>, b: Array<F32>) -> AF2:
  dot_dw(n, 0, DW.zero(), Array.get(F32, a, 0), Array.get(F32, b, 0))

# =====================================================================================================
# Mixed precision: BF16 storage around F32 arithmetic
# =====================================================================================================
# Two wrappers keep a layer's types (P, X, Y, S), so a model gets BF16 storage by wrapping its layers, with the same
# parameters, checkpoints and optimizer state:
#   - CastOut(l): l's output rounded to BF16 (as if stored in BF16), and the gradient arriving at it rounded too, which
#     is what autograd does through x.to(torch.bfloat16).to(torch.float32).
#   - CastP(l), for a layer whose parameter is one matrix: the parameter stays F32 (the master weights the optimizer
#     updates); l's forward and backward passes see its BF16 rounding, and the gradient l accumulates is rounded to
#     BF16 before the optimizer sees it, as autograd does for w.to(torch.bfloat16).to(torch.float32). The rounded copy
#     is made again in the backward pass (no saved state), so S does not change.
# Inside the layers every operation is F32. The values are BF16 values held in F32 arrays: this gives BF16's numbers,
# not yet its memory (no kernel reads packed operands; that is the next step).

def bf16_mat(+r: Nat, +c: Nat, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  T.Ew.map(~BF16.round, r, c, m)

def cast_f(-P: Type, -S: Type, +r: Nat, +c: Nat, f: T.Fwd<P, TA.Mat<r, c>, S>) -> T.Fwd<P, TA.Mat<r, c>, S>:
  match f:
    case T.Fwd{p, y, s}:
      T.Fwd{p, bf16_mat(r, c, y), s}

def cast_fwd(~P: Type, ~X: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<P, X, TA.Mat<r, c>, S>, p: P, x: X) -> T.Fwd<P, TA.Mat<r, c>, S>:
  cast_f(P, S, r, c, T.Layer.fwd(P, X, TA.Mat<r, c>, S, l, p, x))

def cast_bwd(~P: Type, ~X: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<P, X, TA.Mat<r, c>, S>, p: P, s: S, dy: TA.Mat<r, c>, g: P) -> T.Bwd<P, X>:
  T.Layer.bwd(P, X, TA.Mat<r, c>, S, l, p, s, bf16_mat(r, c, dy), g)

def CastOut(~P: Type, ~X: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<P, X, TA.Mat<r, c>, S>) -> T.Layer<P, X, TA.Mat<r, c>, S>:
  T.Layer{p => x => cast_fwd(~P, ~X, ~S, ~r, ~c, ~l, p, x), p => s => dy => g => cast_bwd(~P, ~X, ~S, ~r, ~c, ~l, p, s, dy, g), op => p => g => st => T.Layer.step(P, X, TA.Mat<r, c>, S, l, op, p, g, st), u => T.Layer.zeros(P, X, TA.Mat<r, c>, S, l), p => T.Layer.dump(P, X, TA.Mat<r, c>, S, l, p), p => xs => off => T.Layer.load(P, X, TA.Mat<r, c>, S, l, p, xs, off), p => xs => off => T.Layer.store(P, X, TA.Mat<r, c>, S, l, p, xs, off), u => T.Layer.count(P, X, TA.Mat<r, c>, S, l)}

def castp_f2(-X: Type, -Y: Type, -S: Type, -r: Nat, -c: Nat, w: TA.Mat<r, c>, f: T.Fwd<TA.Mat<r, c>, Y, S>) -> T.Fwd<TA.Mat<r, c>, Y, S>:
  match f:
    case T.Fwd{wc, y, s}:
      T.Fwd{w, y, s}

def castp_f(~X: Type, ~Y: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<TA.Mat<r, c>, X, Y, S>, d: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>, x: X) -> T.Fwd<TA.Mat<r, c>, Y, S>:
  match d:
    case T.Two{w, wc}:
      castp_f2(X, Y, S, r, c, w, T.Layer.fwd(TA.Mat<r, c>, X, Y, S, l, bf16_mat(r, c, wc), x))

def castp_b2(-X: Type, +r: Nat, +c: Nat, w: TA.Mat<r, c>, b: T.Bwd<TA.Mat<r, c>, X>) -> T.Bwd<TA.Mat<r, c>, X>:
  match b:
    case T.Bwd{wc, dx, g}:
      T.Bwd{w, dx, bf16_mat(r, c, g)}

def castp_b(~X: Type, ~Y: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<TA.Mat<r, c>, X, Y, S>, d: T.Two<TA.Mat<r, c>, TA.Mat<r, c>>, s: S, dy: Y, g: TA.Mat<r, c>) -> T.Bwd<TA.Mat<r, c>, X>:
  match d:
    case T.Two{w, wc}:
      castp_b2(X, r, c, w, T.Layer.bwd(TA.Mat<r, c>, X, Y, S, l, bf16_mat(r, c, wc), s, dy, g))

def CastP(~X: Type, ~Y: Type, ~S: Type, ~r: Nat, ~c: Nat, ~l: T.Layer<TA.Mat<r, c>, X, Y, S>) -> T.Layer<TA.Mat<r, c>, X, Y, S>:
  T.Layer{p => x => castp_f(~X, ~Y, ~S, ~r, ~c, ~l, T.Mat.dup(r, c, p), x), p => s => dy => g => castp_b(~X, ~Y, ~S, ~r, ~c, ~l, T.Mat.dup(r, c, p), s, dy, g), op => p => g => st => T.Layer.step(TA.Mat<r, c>, X, Y, S, l, op, p, g, st), u => T.Layer.zeros(TA.Mat<r, c>, X, Y, S, l), p => T.Layer.dump(TA.Mat<r, c>, X, Y, S, l, p), p => xs => off => T.Layer.load(TA.Mat<r, c>, X, Y, S, l, p, xs, off), p => xs => off => T.Layer.store(TA.Mat<r, c>, X, Y, S, l, p, xs, off), u => T.Layer.count(TA.Mat<r, c>, X, Y, S, l)}
