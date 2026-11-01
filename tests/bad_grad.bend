# Must not compile: a hand-written backward pass for y = x·Wᵀ that returns dW as Xᵀ·dY (in x out) instead of
# dYᵀ·X (out x in). The gradient of the parameters must have the parameters' type, T.Lin<3, 2>, whose weight is 2 x 3.
# expect: - observed : TA.Mat<3n, 2n>
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def bad_b3(-n: Nat, b: TA.Mat<1n, 2n>, w: TA.Mat<2n, 3n>, dx: TA.Mat<n, 3n>, r: TA.MMulTN<3n, n, 2n>) -> T.Bwd<T.Lin<3n, 2n>, TA.Mat<n, 3n>>:
  match r:
    case TA.MMulTN{x, dy, dw}:
      T.Bwd{T.Lin{w, b}, dx, T.Lin{dw, TA.Mat.zeros(1n, 2n)}}

def main() -> Unit:
  Unit{}
