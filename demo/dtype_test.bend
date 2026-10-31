# torch/dtype.bend against PyTorch (reference/test_dtype.py writes the inputs and checks every output):
#
#   bend demo/dtype_test.bend -o .scratch/dtype_test
#   .scratch/dtype_test conv <file> <chunks>   # F32 words from <file>, 4096 a chunk: BF16 and F16 of each (bits)
#   .scratch/dtype_test halves                 # all 65536 16-bit patterns: to F32, and back (BF16 and F16)
#   .scratch/dtype_test pack <file>            # a 64 x 63 matrix packed in BF16 and F16: the words, then unpacked
#   .scratch/dtype_test joins                  # failures of half_lo / half_hi over every 16-bit value (must be 0 0)
#   .scratch/dtype_test q8 <w> <x>             # int8 by rows of a 65 x 32 matrix: words, scales, W back, x · Wᵀ
#   .scratch/dtype_test dw <n> <reps>          # a dot product of n numbers and a 49152-class softmax sum: F32,
#                                              # tree and double-word results, the terms, and the time of each
#
# Each number is printed as the decimal of its 32 bits (a 16-bit value in the low half of its word).

import Base
import ../torch/main.bend as T
import ../torch/dtype.bend as D
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def ws(xs: List<&2, U32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{+h, t}:
      " " ++ U32.show(h) ++ ws(t)

def fs(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ U32.show(D.f2u(h)) ++ fs(t)

def bf_l(xs: List<&2, F32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      D.BF16.of_f32(h) <> bf_l(t)

# main.bend's own copy of the rounding, for AdamW's packed moments (T.CAdamWBF16)
def bft_l(xs: List<&2, F32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      T.ph_of(h) <> bft_l(t)

def hf_l(xs: List<&2, F32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      D.F16.of_f32(h) <> hf_l(t)

# ---- conv: chunks of 4096 F32 words ----

def conv_chunk(+xs: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print("bf16" ++ ws(bf_l(xs)))
    IO.print("bf16t" ++ ws(bft_l(xs)))
    IO.print("f16" ++ ws(hf_l(xs)))

def conv_loop(k: Nat, +path: String, +i: U32) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        bs : List<&2, U32> <- T.read_at(path, (i * 16384 : U32), 16384)
        conv_chunk(T.f32s(bs))
        conv_loop(p, path, (i + 1 : U32))

# ---- halves: every 16-bit pattern ----

def hl(k: Nat, +h: U32) -> List<&2, U32>:
  match k:
    case 0n:
      Nil{}
    case 1n+p:
      h <> hl(p, (h + 1 : U32))

def up_bf(xs: List<&2, U32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      D.BF16.to_f32(h) <> up_bf(t)

def up_bft(xs: List<&2, U32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      T.ph_to(h) <> up_bft(t)

def up_hf(xs: List<&2, U32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      D.F16.to_f32(h) <> up_hf(t)

def halves() -> IO(Unit):
  do IO<Unit>:
    IO.print("bf16up" ++ fs(up_bf(hl(65536n, 0))))
    IO.print("bf16tup" ++ fs(up_bft(hl(65536n, 0))))
    IO.print("f16up" ++ fs(up_hf(hl(65536n, 0))))
    IO.print("bf16rt" ++ ws(bf_l(up_bf(hl(65536n, 0)))))
    IO.print("f16rt" ++ ws(hf_l(up_hf(hl(65536n, 0)))))

# ---- joins: half_lo(half_join(a, b)) == a and half_hi(half_join(a, b)) == b for 16-bit a and b ----

def bad(b: Bool) -> U32:
  match b:
    case True{}:
      0
    case False{}:
      1

# every a against a b that walks all 65536 values in another order, and the reverse
def joins_go(k: Nat, +a: U32, acc_lo: U32, acc_hi: U32) -> IO(Unit):
  match k:
    case 0n:
      IO.print("joins " ++ U32.show(acc_lo) ++ " " ++ U32.show(acc_hi))
    case 1n+p:
      +b = ((a * 40503 : U32) .&. 65535 : U32)
      +w1 = D.half_join(a, b)
      +w2 = D.half_join(b, a)
      +l1 = (acc_lo + bad(U32.is_eq(D.half_lo(w1), a)) : U32)
      +h1 = (acc_hi + bad(U32.is_eq(D.half_hi(w1), b)) : U32)
      joins_go(p, (a + 1 : U32), (l1 + bad(U32.is_eq(D.half_lo(w2), b)) : U32), (h1 + bad(U32.is_eq(D.half_hi(w2), a)) : U32))

# ---- pack: a 64 x 63 matrix (4032 values), read four times from the file ----

def pack(+path: String) -> IO(Unit):
  do IO<Unit>:
    m1 : TA.Mat<64n, 63n> <- T.Mat.load(64n, 63n, path, 0)
    m2 : TA.Mat<64n, 63n> <- T.Mat.load(64n, 63n, path, 0)
    m3 : TA.Mat<64n, 63n> <- T.Mat.load(64n, 63n, path, 0)
    m4 : TA.Mat<64n, 63n> <- T.Mat.load(64n, 63n, path, 0)
    IO.print("bf16words" ++ ws(D.HMat.words(64n, 63n, D.BF16.pack(64n, 63n, m1))))
    IO.print("f16words" ++ ws(D.HMat.words(64n, 63n, D.F16.pack(64n, 63n, m2))))
    T.Mat.print(64n, 63n, "bf16unpack", D.BF16.unpack(64n, 63n, D.BF16.pack(64n, 63n, m3)))
    T.Mat.print(64n, 63n, "f16unpack", D.F16.unpack(64n, 63n, D.F16.pack(64n, 63n, m4)))

# ---- q8: a 65 x 32 weight matrix (read twice) and a 16 x 32 input ----

def q8_print(ql: D.QL) -> IO(Unit):
  match ql:
    case D.QL{q, s}:
      do IO<Unit>:
        IO.print("q8words" ++ ws(q))
        IO.print("q8scales" ++ fs(s))

def q8_mm(r: D.Q8Mul<16n, 32n, 65n>) -> IO(Unit):
  match r:
    case D.Q8Mul{x, w, y}:
      T.Mat.print(16n, 65n, "q8y", y)

def q8(+wp: String, +xp: String) -> IO(Unit):
  do IO<Unit>:
    w1 : TA.Mat<65n, 32n> <- T.Mat.load(65n, 32n, wp, 0)
    w2 : TA.Mat<65n, 32n> <- T.Mat.load(65n, 32n, wp, 0)
    w3 : TA.Mat<65n, 32n> <- T.Mat.load(65n, 32n, wp, 0)
    x : TA.Mat<16n, 32n> <- T.Mat.load(16n, 32n, xp, 0)
    q8_print(D.Q8.lists(65n, 32n, D.Q8.quantize(65n, 32n, w1)))
    T.Mat.print(65n, 32n, "q8deq", D.Q8.dequantize(65n, 32n, D.Q8.quantize(65n, 32n, w2)))
    q8_mm(D.Q8.matmul_nt(16n, 32n, 65n, x, D.Q8.quantize(65n, 32n, w3)))


# ---- dw: long sums ----

# a number in [-1, 1) from the hash of i: 24 random bits · 2^-23 - 1, exact (reference/test_dtype.py computes the same)
def uval(+i: U32, +seed: U32) -> F32:
  (((U32.to_f32(U32.shrn(T.mix32((i + seed : U32)), 8n)) * 0.00000011920928955078125 : F32)) - 1.0 : F32)

def ufill(k: Nat, +i: U32, +seed: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n:
      a
    case 1n+p:
      ufill(p, (i + 1 : U32), seed, Array.set(F32, a, i, uval((i * 2 : U32), seed)))

def uarr(+n: Nat, +seed: U32) -> Array<F32>:
  ufill(n, 0, seed, T.zeros_a(n))

# the softmax terms exp(z - 15) for logits z = 6·u, with the first class at z = 15 (one dominant class): what
# cross-entropy sums over SmolLM2's 49152 classes
def efill(k: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match k:
    case 0n:
      a
    case 1n+p:
      efill(p, (i + 1 : U32), Array.set(F32, a, i, F32.exp(((6.0 * uval((i * 2 : U32), 777) : F32) - 15.0 : F32))))

def earr() -> Array<F32>:
  Array.set(F32, efill(49152n, 0, T.zeros_a(49152n)), 0, 1.0)

def af_add(+v: F32, r: D.AF) -> D.AF:
  match r:
    case D.AF{a, w}:
      D.AF{a, (v + w : F32)}

def af2_add(+v: F32, r: D.AF2) -> D.AF2:
  match r:
    case D.AF2{a, b, w}:
      D.AF2{a, b, (v + w : F32)}

type C3 is Type:
  C3{a: Array<F32>, b: Array<F32>, c: Array<F32>}

def prod_l(k: Nat, +i: U32, ra: Array<F32> & F32, rb: Array<F32> & F32, c: Array<F32>) -> C3:
  match k ra rb:
    case 0n Tuple{a, x} Tuple{b, y}:
      C3{a, b, c}
    case 1n+p Tuple{a, +x} Tuple{b, +y}:
      prod_l(p, (i + 1 : U32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)), Array.set(F32, c, i, (x * y : F32)))

def tree_fin(+n: Nat, r: C3) -> D.AF2:
  match r:
    case C3{a, b, c}:
      D.AF2{a, b, T.tree_sum(U32.from_nat(n), c)}

# the products into a new array, then T.tree_sum over it: the pairwise sum cross-entropy uses
def Dot.tree(+n: Nat, a: Array<F32>, b: Array<F32>) -> D.AF2:
  tree_fin(n, prod_l(n, 0, Array.get(F32, a, 0), Array.get(F32, b, 0), T.zeros_a(n)))

def sum_tree_fin(+n: Nat, r: Array<F32> & Array<F32>) -> D.AF:
  match r:
    case Tuple{a, c}:
      D.AF{a, T.tree_sum(U32.from_nat(n), c)}

# T.tree_sum works in place, so it sums a copy
def Sum.tree(+n: Nat, a: Array<F32>) -> D.AF:
  sum_tree_fin(n, Array.clone(F32, a))

# the methods: 0 a running F32 sum, 1 a tree, 2 double-word
def dot_m(+m: Nat, +n: Nat, a: Array<F32>, b: Array<F32>) -> D.AF2:
  match m:
    case 0n:
      D.Dot.f32(n, a, b)
    case 1n:
      Dot.tree(n, a, b)
    case _:
      D.Dot.dw(n, a, b)

def sum_m(+m: Nat, +n: Nat, a: Array<F32>) -> D.AF:
  match m:
    case 0n:
      D.Sum.f32(n, a)
    case 1n:
      Sum.tree(n, a)
    case _:
      D.Sum.dw(n, a)

# k repetitions of a method; each result is added into the total, so that none is skipped
def rep_dot(k: Nat, +m: Nat, +n: Nat, r: D.AF2) -> D.AF2:
  match k r:
    case 0n D.AF2{a, b, v}:
      D.AF2{a, b, v}
    case 1n+p D.AF2{a, b, +v}:
      rep_dot(p, m, n, af2_add(v, dot_m(m, n, a, b)))

def rep_sum(k: Nat, +m: Nat, +n: Nat, r: D.AF) -> D.AF:
  match k r:
    case 0n D.AF{a, v}:
      D.AF{a, v}
    case 1n+p D.AF{a, +v}:
      rep_sum(p, m, n, af_add(v, sum_m(m, n, a)))

def show_t(+tag: String, +x: F32, +t0: Nat, +t1: Nat, +y: F32) -> IO(Unit):
  IO.print(tag ++ " " ++ U32.show(D.f2u(x)) ++ " ms " ++ Nat.show(Nat.sub(t1, t0)) ++ " total " ++ U32.show(D.f2u(y)))

def dot3(+tag: String, +x: F32, +t0: Nat, tot: D.AF2) -> IO(D.AF2):
  match tot:
    case D.AF2{a, b, +y}:
      do IO<D.AF2>:
        t1 : Nat <- IO.now()
        show_t(tag, x, t0, t1, y)
        return D.AF2{a, b, 0.0}

def dot2(+tag: String, +m: Nat, +n: Nat, +reps: Nat, one: D.AF2) -> IO(D.AF2):
  match one:
    case D.AF2{a, b, +x}:
      do IO<D.AF2>:
        t0 : Nat <- IO.now()
        dot3(tag, x, t0, rep_dot(reps, m, n, D.AF2{a, b, 0.0}))

# one result of method m, then the time of reps more
def dw_dot(+tag: String, +m: Nat, +n: Nat, +reps: Nat, r: D.AF2) -> IO(D.AF2):
  match r:
    case D.AF2{a, b, v}:
      dot2(tag, m, n, reps, dot_m(m, n, a, b))

def sum3(+tag: String, +x: F32, +t0: Nat, tot: D.AF) -> IO(D.AF):
  match tot:
    case D.AF{a, +y}:
      do IO<D.AF>:
        t1 : Nat <- IO.now()
        show_t(tag, x, t0, t1, y)
        return D.AF{a, 0.0}

def sum2(+tag: String, +m: Nat, +n: Nat, +reps: Nat, one: D.AF) -> IO(D.AF):
  match one:
    case D.AF{a, +x}:
      do IO<D.AF>:
        t0 : Nat <- IO.now()
        sum3(tag, x, t0, rep_sum(reps, m, n, D.AF{a, 0.0}))

def dw_sum(+tag: String, +m: Nat, +n: Nat, +reps: Nat, r: D.AF) -> IO(D.AF):
  match r:
    case D.AF{a, v}:
      sum2(tag, m, n, reps, sum_m(m, n, a))

def print_terms(r: D.AF) -> IO(Unit):
  match r:
    case D.AF{a, v}:
      T.Mat.print(1n, 49152n, "terms", TA.Mat{a})

# the dot product of two n-vectors (uval(2i, 1) and uval(2i, 2)) by each method, reps times each; then the softmax sum,
# sreps times each. The F32 method runs first and last, to see the drift of the machine's speed.
def dw(+n: Nat, +reps: Nat, +sreps: Nat) -> IO(Unit):
  do IO<Unit>:
    r0 : D.AF2 <- dw_dot("dot_f32", 0n, n, reps, D.AF2{uarr(n, 1), uarr(n, 2), 0.0})
    r1 : D.AF2 <- dw_dot("dot_tree", 1n, n, reps, r0)
    r2 : D.AF2 <- dw_dot("dot_dw", 2n, n, reps, r1)
    r3 : D.AF2 <- dw_dot("dot_f32", 0n, n, reps, r2)
    s0 : D.AF <- dw_sum("sum_f32", 0n, 49152n, sreps, D.AF{earr(), 0.0})
    s1 : D.AF <- dw_sum("sum_tree", 1n, 49152n, sreps, s0)
    s2 : D.AF <- dw_sum("sum_dw", 2n, 49152n, sreps, s1)
    s3 : D.AF <- dw_sum("sum_f32", 0n, 49152n, sreps, s2)
    print_terms(s3)

# ---- the command line ----

def usage() -> IO(Unit):
  IO.die(Unit, 2, "usage: dtype_test conv <file> <chunks> | halves | pack <file> | joins | q8 <w> <x> | dw <n> <reps> <sum reps>")

def conv_m(+path: String, mk: Maybe<&2, Nat>) -> IO(Unit):
  match mk:
    case Some{k}:
      conv_loop(k, path, 0)
    case None{}:
      usage()

def dw_m(mn: Maybe<&2, Nat>, mr: Maybe<&2, Nat>, ms: Maybe<&2, Nat>) -> IO(Unit):
  match mn mr ms:
    case Some{n} Some{r} Some{s}:
      dw(n, r, s)
    case _ _ _:
      usage()

def run1(a: Bool, b: Bool) -> IO(Unit):
  match a b:
    case True{} _:
      halves()
    case False{} True{}:
      joins_go(65536n, 0, 0, 0)
    case False{} False{}:
      usage()

def run2(a: Bool, +p: String) -> IO(Unit):
  match a:
    case True{}:
      pack(p)
    case False{}:
      usage()

def run3(a: Bool, b: Bool, +x: String, +y: String) -> IO(Unit):
  match a b:
    case True{} _:
      conv_m(x, Nat.read(y))
    case False{} True{}:
      q8(x, y)
    case False{} False{}:
      usage()

def run4(a: Bool, x: String, y: String, z: String) -> IO(Unit):
  match a:
    case True{}:
      dw_m(Nat.read(x), Nat.read(y), Nat.read(z))
    case False{}:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Nil{}}}:
      run1(String.eq(mode, "halves"), String.eq(mode, "joins"))
    case Con{prog, Con{+mode, Con{a, Nil{}}}}:
      run2(String.eq(mode, "pack"), a)
    case Con{prog, Con{+mode, Con{a, Con{b, Nil{}}}}}:
      run3(String.eq(mode, "conv"), String.eq(mode, "q8"), a, b)
    case Con{prog, Con{+mode, Con{a, Con{b, Con{c, Nil{}}}}}}:
      run4(String.eq(mode, "dw"), a, b, c)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
