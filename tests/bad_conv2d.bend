# Must not compile: Conv2d requires 2p < k + hs and 2p < k + ws so that cimg's
# Nat.sub never truncates a dimension to 0 cells. With k=1, hs=0, ws=0 and p=1,
# 2p = 2 > 1 = k + hs (and k + ws), so True == Nat.is_lt(2, 1) computes to False.
# expect: - observed : False{}
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def BadConv2d() -> T.Layer<T.Lin<1n, 1n>, TA.Mat<0n, 1n>, TA.Mat<4n, 1n>, TA.Mat<4n, 1n>>:
  T.Conv2d(1n, 1n, 1n, 1n, 0n, 0n, 1n, 0n, 0n, {==}, {==})

def main() -> Unit:
  Unit{}
