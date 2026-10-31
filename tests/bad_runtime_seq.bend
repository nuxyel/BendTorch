# Must not compile even though n is read at run time: a 4 -> 3 layer followed by a 2 -> 2 activation.
# expect: - observed : T.Layer<Unit, TA.Mat<n, 2n>
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def Model(+n: Nat) -> T.Layer<T.Two<T.Lin<4n, 3n>, Unit>, TA.Mat<n, 4n>, TA.Mat<n, 2n>, T.Two<TA.Mat<n, 4n>, TA.Mat<n, 2n>>>:
  T.Seq.at(T.Lin<4n, 3n>, Unit, TA.Mat<n, 4n>, TA.Mat<n, 3n>, TA.Mat<n, 2n>, TA.Mat<n, 4n>, TA.Mat<n, 2n>, T.Linear(n, 4n, 3n, 0n), T.ReLU(n, 2n))

def main() -> Unit:
  Unit{}
