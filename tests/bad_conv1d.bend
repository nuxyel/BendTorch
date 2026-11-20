# Must not compile: Conv1d requires 2p < eff_k + ts so that t = eff_k + ts - 2p >= 1 and Nat.sub never truncates.
# With k=1 (k1=0), d=1 (d1=0), eff_k=1, ts=0 and p=1, 2p = 2 > 1, so True == Nat.is_lt(2, 1) computes to False.
# expect: - observed : False{}
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def BadConv1d() -> T.Layer<T.Lin<1n, 1n>, TA.Mat<0n, 1n>, TA.Mat<2n, 1n>, TA.Mat<2n, 1n>>:
  T.Conv1d(1n, 0n, 1n, 1n, 0n, 0n, 1n, 0n, 0n, 0n, {==})

def main() -> Unit:
  Unit{}
