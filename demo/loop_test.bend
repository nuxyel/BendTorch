# Reusing parameters in a loop: the same 3 -> 3 linear layer runs its own output through fwd three times.
#
# A parameter tree holds arrays, so it is affine: it cannot be used twice, and it cannot be marked + (an Array is not
# Data). fwd hands the parameters back for exactly that reason, and the loop threads them through a record. fwd's
# result is a computed value, so a then-style def matches it and calls the continuation.
#
# demo/smol.bend's ev_loop is the larger example, over a model and a batch of held-out data.
import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def P() -> Type:
  T.Lin<3n, 3n>

def X() -> Type:
  TA.Mat<2n, 3n>

def Model() -> T.Layer<P, X, X, X>:
  T.Linear(2n, 3n, 3n, 0n)

# the loop's state: the parameters fwd handed back, and the output that becomes the next input
type R is Type:
  R{p: P, x: X}

def R.then(r: T.Fwd<P, X, X>, k: P -> X -> R) -> R:
  match r:
    case T.Fwd{p, y, s}:
      k(p, y)

def loop(k: Nat, r: R) -> R:
  match k:
    case 0n:
      r
    case 1n+q:
      match r:
        case R{p, x}:
          R.then(T.Layer.fwd(P, X, X, X, Model(), p, x), p2 => y => loop(q, R{p2, y}))

def R.print(r: R) -> IO(Unit):
  match r:
    case R{p, x}:
      T.Mat.print(2n, 3n, "y", x)

def main() -> IO(Unit):
  do IO<Unit>:
    R.print(loop(3n, R{T.Layer.zeros(P, X, X, X, Model()), TA.Mat.zeros(2n, 3n)}))
