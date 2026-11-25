# Must not compile: the height guard is independent of the width guard.
# With k=1, hs=0, ws=2 and p=1, the width guard 2p < k + ws is 2 < 3 (True),
# but the height guard 2p < k + hs is 2 < 1 (False). The height would truncate to 0 cells.
# expect: - observed : False{}
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def BadConv2d() -> T.Layer<T.Lin<1n, 1n>, TA.Mat<0n, 1n>, TA.Mat<6n, 1n>, TA.Mat<6n, 1n>>:
  T.Conv2d(1n, 1n, 1n, 1n, 0n, 2n, 1n, 0n, 0n, {==}, {==})

def main() -> Unit:
  Unit{}
