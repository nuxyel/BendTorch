# Must not compile: RoPE rotates pairs (x[c], x[c + half]) inside each head, so a head has 2 x half numbers. Rows of 30
# numbers cannot be 4 heads of 2 x 4: the constructor's proof 30 == 4 x (4 + 4) fails.
# expect: - observed : 32n
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def Rope() -> T.Layer<Unit, TA.Mat<32n, 30n>, TA.Mat<32n, 30n>, Unit>:
  T.RoPE(32n, 30n, 16n, 4n, 4n, 10000.0, {==})

def main() -> Unit:
  Unit{}
