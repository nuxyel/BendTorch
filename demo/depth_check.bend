# torch/main.bend's depth (the capacity of the kernels' scratch arrays) against bend-ml-tensor-array's cap_depth, for
# every n from 0 to 4999: prints the number of n where they differ, which must be 0.
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def go(k: Nat, +n: Nat, bad: U32) -> U32:
  match k:
    case 0n:
      bad
    case 1n+p:
      go(p, 1n+n, (bad + Bool.pick(U32, Nat.is_eq(T.depth(n), TA.cap_depth(n)), 0, 1) : U32))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(U32.show(go(5000n, 0n, 0)))
