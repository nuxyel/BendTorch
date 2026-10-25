# The weights reader past 4 GiB and on BF16 files (bend-ml-binary-read@0.2.0.0, torch/load.bend), driven by
# reference/test_reader.py, which writes the files and knows every expected bit. Offsets are decimal byte counts of up
# to 19 digits, read with torch/load.bend's W64.read.
#
#   f32 FILE OFFSET N        the reader's read_f32_at into 2^d cells holding 9.0: "count: bits of cells 0 .. N-1"
#   bf16 FILE OFFSET N       the same with read_bf16_at
#   old FILE OFFSET N        the package's read_f32 (a U32 offset): "bytes: bits of cells 0 .. N-1"
#   load_f32 FILE OFFSET N   torch/load.bend's read_f32_at (it stops the program on a short read): "N: bits ..."
#   load_bf16 FILE OFFSET N  torch/load.bend's read_bf16_at
#   hash_bf16 FILE OFFSET N  as load_bf16, but prints "N hash" (FNV-1a over the cells' 32-bit words), for big tensors
#   read FILE N              torch/load.bend's read (an exact-size F32 file): "N hash"
#   size FILE                torch/load.bend's size, in decimal
#   w64 A K                  W64.add(A, K), W64.mul_small(A, K), A / 10 and A % 10, each as W64.show writes it
#
#   bend demo/reader_test.bend -o .scratch/reader_test   (make reader)
import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import bend-ml-binary-read@0.2.0.0/main.bend as R

# " b_0 b_1 .. b_i" (the cells' bits in decimal), built from the end so each step only prepends
def dump(k: Nat, r: Array<F32> & F32, +i: U32, acc: String) -> String:
  match k r:
    case 0n Tuple{a, x}:
      " " ++ U32.show(T.bits(x)) ++ acc
    case 1n+p Tuple{a, x}:
      dump(p, Array.get(F32, a, U32.sub(i, 1)), U32.sub(i, 1), " " ++ U32.show(T.bits(x)) ++ acc)

def cells(z: Bool, a: Array<F32>, +n: U32) -> String:
  match z:
    case True{}: ""
    case False{}: dump(U32.to_nat(U32.sub(n, 1)), Array.get(F32, a, U32.sub(n, 1)), U32.sub(n, 1), "")

# FNV-1a over the 32-bit words of cells 0 .. k
def fnv(k: Nat, r: Array<F32> & F32, +i: U32, +h: U32) -> U32:
  match k r:
    case 0n Tuple{a, x}:
      (U32.xor(h, T.bits(x)) * 16777619 : U32)
    case 1n+p Tuple{a, x}:
      fnv(p, Array.get(F32, a, (i + 1 : U32)), (i + 1 : U32), (U32.xor(h, T.bits(x)) * 16777619 : U32))

def hash(z: Bool, a: Array<F32>, +n: U32) -> U32:
  match z:
    case True{}: 2166136261
    case False{}: fnv(U32.to_nat(U32.sub(n, 1)), Array.get(F32, a, 0), 0, 2166136261)

def show(r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), +n: U32) -> IO(Unit):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Unit>:
        +got : U32 <- IO.pass(U32, res)
        File.close(f)
        IO.print(U32.show(got) ++ ":" ++ cells(U32.is_zero(n), a, n))

def nines(+n: U32) -> Array<F32>:
  Array.new(F32, T.depth(U32.to_nat(n)), 9.0)

def pick_go(f32: Bool, bf16: Bool, f: File, hi: U32, lo: U32, +n: U32, a: Array<F32>) -> IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  match f32 bf16:
    case True{} _: R.read_f32_at(f, hi, lo, n, a)
    case _ True{}: R.read_bf16_at(f, hi, lo, n, a)
    case _ _: R.read_f32(f, lo, n, a)

def pick(+kind: U32, f: File, hi: U32, lo: U32, +n: U32, a: Array<F32>) -> IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  pick_go(U32.is_eq(kind, 0), U32.is_eq(kind, 1), f, hi, lo, n, a)

def raw(+path: String, +kind: U32, off: L.W64, +n: U32) -> IO(Unit):
  match off:
    case L.W64{hi, +lo}:
      do IO<Unit>:
        f : File <- IO.try(File, File.open(path, "r"))
        r : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- pick(kind, f, hi, lo, n, nines(n))
        show(r, n)

def printed(+n: U32, a: Array<F32>) -> IO(Unit):
  IO.print(U32.show(n) ++ ":" ++ cells(U32.is_zero(n), a, n))

def hashed(+n: U32, a: Array<F32>) -> IO(Unit):
  IO.print(U32.show(n) ++ " " ++ U32.show(hash(U32.is_zero(n), a, n)))

def load(+path: String, bf16: Bool, off: L.W64, +n: U32) -> IO(Array<F32>):
  match bf16:
    case True{}: L.read_bf16_at(path, off, n)
    case False{}: L.read_f32_at(path, off, n)

def out(hashit: Bool, +n: U32, a: Array<F32>) -> IO(Unit):
  match hashit:
    case True{}: hashed(n, a)
    case False{}: printed(n, a)

def loaded(+path: String, +bf16: Bool, +off: L.W64, +n: U32, +hashit: Bool) -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> <- load(path, bf16, off, n)
    out(hashit, n, a)

def usage() -> IO(Unit):
  IO.die(Unit, 1, "usage: reader_test f32|bf16|old|load_f32|load_bf16|hash_bf16 file offset n | read file n | size file | w64 a k")

def three_go(a: Bool, b: Bool, c: Bool, d: Bool, e: Bool, g: Bool, +path: String, +off: L.W64, +n: U32) -> IO(Unit):
  match a b c d e g:
    case True{} _ _ _ _ _: raw(path, 0, off, n)
    case _ True{} _ _ _ _: raw(path, 1, off, n)
    case _ _ True{} _ _ _: raw(path, 2, off, n)
    case _ _ _ True{} _ _: loaded(path, False{}, off, n, False{})
    case _ _ _ _ True{} _: loaded(path, True{}, off, n, False{})
    case _ _ _ _ _ True{}: loaded(path, True{}, off, n, True{})
    case _ _ _ _ _ _: usage()

def three(+mode: String, +path: String, mo: Maybe<&2, L.W64>, mn: Maybe<&2, U32>) -> IO(Unit):
  match mo mn:
    case Some{+off} Some{+n}:
      three_go(String.eq(mode, "f32"), String.eq(mode, "bf16"), String.eq(mode, "old"), String.eq(mode, "load_f32"), String.eq(mode, "load_bf16"), String.eq(mode, "hash_bf16"), path, off, n)
    case _ _: usage()

def read_n(+path: String, mn: Maybe<&2, U32>) -> IO(Unit):
  match mn:
    case Some{+n}:
      do IO<Unit>:
        a : Array<F32> <- L.read(path, n)
        hashed(n, a)
    case None{}: usage()

def w64_qr(+sum: L.W64, +prod: L.W64, qr: L.W64 & U32) -> IO(Unit):
  match qr:
    case Tuple{q, r}: IO.print(L.W64.show(sum) ++ " " ++ L.W64.show(prod) ++ " " ++ L.W64.show(q) ++ " " ++ U32.show(r))

def w64(ma: Maybe<&2, L.W64>, mk: Maybe<&2, U32>) -> IO(Unit):
  match ma mk:
    case Some{+a} Some{+k}: w64_qr(L.W64.add(a, L.W64.of(k)), L.W64.mul_small(a, k), L.W64.div10(a))
    case _ _: usage()

def sized(+path: String) -> IO(Unit):
  do IO<Unit>:
    s : L.W64 <- L.size(path)
    IO.print(L.W64.show(s))

def two(is_read: Bool, is_w64: Bool, +a: String, +b: String) -> IO(Unit):
  match is_read is_w64:
    case True{} _: read_n(a, U32.read(b))
    case _ True{}: w64(L.W64.read(a), U32.read(b))
    case _ _: usage()

def one(is_size: Bool, +path: String) -> IO(Unit):
  match is_size:
    case True{}: sized(path)
    case False{}: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{mode, Con{path, Con{off, Con{n, Nil{}}}}}}: three(mode, path, L.W64.read(off), U32.read(n))
    case Con{prog, Con{+mode, Con{a, Con{b, Nil{}}}}}: two(String.eq(mode, "read"), String.eq(mode, "w64"), a, b)
    case Con{prog, Con{mode, Con{path, Nil{}}}}: one(String.eq(mode, "size"), path)
    case _: usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
