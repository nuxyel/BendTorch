# The two attention probability matrices on a 3x3 toy: causal (probs) and full
# (fprobs), one head of 2 numbers, scale 0.5. reference/test_fullattn.py builds
# the same q and k and prints the same two matrices from PyTorch's
# scaled_dot_product_attention with is_causal on and off.
#
#   bend demo/fullattn_test.bend
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import ../torch/main.bend as T

def qv() -> List<&2, F32>:
  Con{0.1, Con{0.2, Con{0.3, Con{(0.0 - 0.4 : F32),
    Con{0.5, Con{0.6, Nil{}}}}}}}

def kv() -> List<&2, F32>:
  Con{(0.0 - 0.2 : F32), Con{0.4, Con{0.7, Con{0.1,
    Con{(0.0 - 0.3 : F32), Con{(0.0 - 0.5 : F32), Nil{}}}}}}}

def un(m: TA.Mat<3n, 2n>) -> Array<F32>:
  match m:
    case TA.Mat{d}:
      d

def arr(xs: List<&2, F32>) -> Array<F32>:
  un(TA.Mat.from_list(3n, 2n, xs))

def nums(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", nums(t)))

def dump(ml: TA.ML<3n, 3n>) -> IO(Unit):
  match ml:
    case TA.ML{m, l}:
      IO.print(nums(l))

def show(r: T.QK) -> IO(Unit):
  match r:
    case T.QK{q, k, p}:
      dump(TA.Mat.to_list(3n, 3n, TA.Mat{p}))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("causal")
    Unit <- show(T.probs(3n, 2n, 0.5, (2 : U32), (0 : U32), (2 : U32), (0 : U32), arr(qv()), arr(kv())))
    IO.print("full")
    show(T.fprobs(3n, 2n, 0.5, (2 : U32), (0 : U32), (2 : U32), (0 : U32), arr(qv()), arr(kv())))
