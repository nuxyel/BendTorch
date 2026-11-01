# Must not compile: a 784 -> 128 layer followed by a 64 -> 10 layer. Seq needs the first layer's output type to be the
# second layer's input type; here they are Mat<100, 128> and Mat<100, 64>.
# expect: - observed : T.Layer<T.Lin<64n, 10n>, TA.Mat<100n, 64n>
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def Model() -> T.Layer<T.Two<T.Lin<784n, 128n>, T.Lin<64n, 10n>>, TA.Mat<100n, 784n>, TA.Mat<100n, 10n>, T.Two<TA.Mat<100n, 784n>, TA.Mat<100n, 64n>>>:
  T.Seq(~T.Lin<784n, 128n>, ~T.Lin<64n, 10n>, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 128n>, ~TA.Mat<100n, 10n>, ~TA.Mat<100n, 784n>, ~TA.Mat<100n, 64n>, ~T.Linear(100n, 784n, 128n, 0n), ~T.Linear(100n, 64n, 10n, 0n))

def main() -> Unit:
  Unit{}
