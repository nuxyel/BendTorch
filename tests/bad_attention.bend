# Must not compile: attention with 4 query heads of 8 over rows of 30 numbers. The constructor asks for a proof that
# the row width is (key/value heads) x (group) x (head size) = 2 x 2 x 8 = 32, and 30 == 32 does not hold.
# expect: - observed : 32n
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def Attn() -> T.Layer<Unit, T.Two<TA.Mat<32n, 30n>, T.Two<TA.Mat<32n, 16n>, TA.Mat<32n, 16n>>>, TA.Mat<32n, 30n>, T.AttnS>:
  T.Attention(32n, 30n, 16n, 2n, 16n, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==})

def main() -> Unit:
  Unit{}
