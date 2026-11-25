# Must not compile: the width guard is independent of the height guard.
# With k=1, hs=2, ws=0 and p=1, the height guard 2p < k + hs is 2 < 3 (True),
# but the width guard 2p < k + ws is 2 < 1 (False). The width would truncate to 0 cells.
# expect: - observed : False{}
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def BadConv2dWidth() -> T.Layer<T.Lin<1n, 1n>, TA.Mat<0n, 1n>, TA.Mat<6n, 1n>, TA.Mat<6n, 1n>>:
  T.Conv2d(1n, 1n, 1n, 1n, 2n, 0n, 1n, 0n, 0n, {==}, {==})

def main() -> Unit:
  Unit{}
