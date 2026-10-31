# Must not compile even though b and t are read at run time: the rows are b * t, but attention is told its sequences
# have 16 positions. Attention asks for a proof that rows == b * (positions a sequence); with t unknown the checker
# cannot prove b * t == b * 16, so the program is refused (demo/tiny_rt.bend passes t and the proof is {==}).
# expect: - observed : Nat.mul(b, 16n)
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def Core(+b: Nat, +t: Nat) -> T.Layer<Unit, T.Two<TA.Mat<Nat.mul(b, t), 32n>, T.Two<TA.Mat<Nat.mul(b, t), 16n>, TA.Mat<Nat.mul(b, t), 16n>>>, TA.Mat<Nat.mul(b, t), 32n>, T.AttnS>:
  T.Attention(Nat.mul(b, t), 32n, 16n, b, 16n, 2n, 2n, 8n, 0.35355339059327373, {==}, {==}, {==})

def main() -> Unit:
  Unit{}
