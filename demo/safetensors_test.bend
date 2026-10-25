# torch/load.bend's safetensors reader (L.tensors), driven by reference/test_safetensors.py, which writes the files and
# knows every expected bit. A layout is given as name:dims arguments, dims as rows x columns (2x3), or nothing for a
# scalar (name:).
#
#   bits FILE N name:dims ...   "N: bits of cells 0 .. N-1" (the tensors in the layout's order)
#   hash FILE N name:dims ...   "N hash" (FNV-1a over the cells' 32-bit words), for whole models
#
#   bend demo/safetensors_test.bend -o .scratch/safetensors_test
import Base
import ../torch/main.bend as T
import ../torch/load.bend as L

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

def dims(ss: List<&2, String>) -> List<&2, U32>:
  match ss:
    case Nil{}: Nil{}
    case Con{s, t}: Con{Maybe.default(&2, U32, U32.read(s), 0), dims(t)}

def shape(empty: Bool, +d: String) -> List<&2, U32>:
  match empty:
    case True{}: Nil{}
    case False{}: dims(String.split(d, 'x'))

def entry(parts: List<&2, String>) -> L.Entry:
  match parts:
    case Con{name, Con{+d, Nil{}}}: L.Entry{name, shape(String.is_empty(d), d)}
    case _: L.Entry{"", Nil{}}

def layout(ss: List<String>) -> List<&2, L.Entry>:
  match ss:
    case Nil{}: Nil{}
    case Con{s, t}: Con{entry(String.split(s, ':')), layout(t)}

def out(hashit: Bool, +n: U32, a: Array<F32>) -> IO(Unit):
  match hashit:
    case True{}: IO.print(U32.show(n) ++ " " ++ U32.show(hash(U32.is_zero(n), a, n)))
    case False{}: IO.print(U32.show(n) ++ ":" ++ cells(U32.is_zero(n), a, n))

def run(+hashit: Bool, +path: String, mn: Maybe<&2, U32>, specs: List<String>) -> IO(Unit):
  match mn:
    case Some{+n}:
      do IO<Unit>:
        a : Array<F32> <- L.tensors(path, n, layout(specs))
        out(hashit, n, a)
    case None{}: IO.die(Unit, 1, "invalid count")

def go(is_bits: Bool, is_hash: Bool, +path: String, n: String, specs: List<String>) -> IO(Unit):
  match is_bits is_hash:
    case True{} _: run(False{}, path, U32.read(n), specs)
    case _ True{}: run(True{}, path, U32.read(n), specs)
    case _ _: IO.die(Unit, 1, "usage: safetensors_test bits|hash file n name:dims ...")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{path, Con{n, specs}}}}: go(String.eq(mode, "bits"), String.eq(mode, "hash"), path, n, specs)
    case _: IO.die(Unit, 1, "usage: safetensors_test bits|hash file n name:dims ...")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
