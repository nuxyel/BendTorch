# The MLP of reference/test_mlp.py: Linear(7, 6) -> ReLU -> Linear(6, 3) on a batch of 5, built with Seq.
# Prints the loss, the logits, every gradient after one backward pass, then three training steps under each optimizer
# (each number as the decimal of its 32 bits). Reads its inputs from the directory given as the first argument.
#
#   bend demo/mlp_test.bend -o .scratch/mlp_test && .scratch/mlp_test <dir>

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def P() -> Type:
  T.Two<T.Two<T.Lin<7n, 6n>, Unit>, T.Lin<6n, 3n>>

def S() -> Type:
  T.Two<T.Two<TA.Mat<5n, 7n>, TA.Mat<5n, 6n>>, TA.Mat<5n, 6n>>

def Hidden() -> T.Layer<T.Two<T.Lin<7n, 6n>, Unit>, TA.Mat<5n, 7n>, TA.Mat<5n, 6n>, T.Two<TA.Mat<5n, 7n>, TA.Mat<5n, 6n>>>:
  T.Seq(~T.Lin<7n, 6n>, ~Unit, ~TA.Mat<5n, 7n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 7n>, ~TA.Mat<5n, 6n>, ~T.Linear(5n, 7n, 6n, 1n), ~T.ReLU(5n, 6n))

def Model() -> T.Layer<P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S>:
  T.Seq(~T.Two<T.Lin<7n, 6n>, Unit>, ~T.Lin<6n, 3n>, ~TA.Mat<5n, 7n>, ~TA.Mat<5n, 6n>, ~TA.Mat<5n, 3n>, ~T.Two<TA.Mat<5n, 7n>, TA.Mat<5n, 6n>>, ~TA.Mat<5n, 6n>, ~Hidden(), ~T.Linear(5n, 6n, 3n, 1n))

def ce(z: TA.Mat<5n, 3n>, labels: List<&2, Nat>) -> T.LossOut<TA.Mat<5n, 3n>>:
  T.CrossEntropy(5n, 3n, z, labels)

def labs(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs(t)

def load_params(+dir: String) -> IO(P):
  do IO<P>:
    w1 : TA.Mat<6n, 7n> <- T.Mat.load(6n, 7n, dir ++ "/w1.bin", 0)
    b1 : TA.Mat<1n, 6n> <- T.Mat.load(1n, 6n, dir ++ "/b1.bin", 0)
    w2 : TA.Mat<3n, 6n> <- T.Mat.load(3n, 6n, dir ++ "/w2.bin", 0)
    b2 : TA.Mat<1n, 3n> <- T.Mat.load(1n, 3n, dir ++ "/b2.bin", 0)
    return T.Two{T.Two{T.Lin{w1, b1}, Unit{}}, T.Lin{w2, b2}}

def print_params(+tag: String, p: P) -> IO(Unit):
  match p:
    case T.Two{T.Two{T.Lin{w1, b1}, u}, T.Lin{w2, b2}}:
      do IO<Unit>:
        T.Mat.print(6n, 7n, tag ++ ".w1", w1)
        T.Mat.print(1n, 6n, tag ++ ".b1", b1)
        T.Mat.print(3n, 6n, tag ++ ".w2", w2)
        T.Mat.print(1n, 3n, tag ++ ".b2", b2)

# ---- one forward and backward pass ----

def grads3(r: T.Bwd<P, TA.Mat<5n, 7n>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        T.Mat.print(5n, 7n, "dx", dx)
        print_params("grad", g)

def grads2(r: T.LossOut<TA.Mat<5n, 3n>>, p: P, s: S) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      do IO<Unit>:
        T.F32.print("loss", l)
        grads3(T.Layer.bwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, s, dy, T.Layer.zeros(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model())))

def grads1(+labels: List<&2, Nat>, p: P, s: S, r: TA.MM2<5n, 3n>) -> IO(Unit):
  match r:
    case TA.MM2{y, y2}:
      do IO<Unit>:
        T.Mat.print(5n, 3n, "logits", y)
        grads2(ce(y2, labels), p, s)

def grads(+labels: List<&2, Nat>, r: T.Fwd<P, TA.Mat<5n, 3n>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      grads1(labels, p, s, TA.Mat.clone(5n, 3n, y))

# ---- two backward passes into the same gradient, without zeroing it in between (PyTorch accumulates the same way) ----

def acc4(r: T.Bwd<P, TA.Mat<5n, 7n>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      print_params("acc", g)

def acc3(g: P, r: T.LossOut<TA.Mat<5n, 3n>>, p: P, s: S) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      acc4(T.Layer.bwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, s, dy, g))

def acc2(+labels: List<&2, Nat>, g: P, r: T.Fwd<P, TA.Mat<5n, 3n>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      acc3(g, ce(y, labels), p, s)

def acc1(+dir: String, +labels: List<&2, Nat>, r: T.Bwd<P, TA.Mat<5n, 7n>>) -> IO(Unit):
  match r:
    case T.Bwd{p, dx, g}:
      do IO<Unit>:
        x : TA.Mat<5n, 7n> <- T.Mat.load(5n, 7n, dir ++ "/x.bin", 0)
        acc2(labels, g, T.Layer.fwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, x))

def acc0b(+dir: String, +labels: List<&2, Nat>, p: P, s: S, r: T.LossOut<TA.Mat<5n, 3n>>) -> IO(Unit):
  match r:
    case T.LossOut{l, dy}:
      acc1(dir, labels, T.Layer.bwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, s, dy, T.Layer.zeros(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model())))

def acc0(+dir: String, +labels: List<&2, Nat>, r: T.Fwd<P, TA.Mat<5n, 3n>, S>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      acc0b(dir, labels, p, s, ce(y, labels))

def accum(+dir: String, +labels: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params(dir)
    x : TA.Mat<5n, 7n> <- T.Mat.load(5n, 7n, dir ++ "/x.bin", 0)
    acc0(dir, labels, T.Layer.fwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, x))

# ---- training steps ----

# prints the loss of the step before (0 before the first) and runs k more steps
def train(k: Nat, +dir: String, +cfg: T.OptCfg, +tag: String, +labels: List<&2, Nat>, tr: T.Train<P>) -> IO(Unit):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        print_params(tag, p)
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        x : TA.Mat<5n, 7n> <- T.Mat.load(5n, 7n, dir ++ "/x.bin", 0)
        train(q, dir, cfg, tag, labels, T.Train.step(~P, ~TA.Mat<5n, 7n>, ~TA.Mat<5n, 3n>, ~S, ~List<&2, Nat>, ~Model(), ~ce, cfg, T.Train{p, g, st, p1, p2, l}, x, labels))

def run_opt(+dir: String, +labels: List<&2, Nat>, +cfg: T.OptCfg, +tag: String) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg)
    p : P <- load_params(dir)
    train(3n, dir, cfg, tag, labels, T.Train.new(~P, ~TA.Mat<5n, 7n>, ~TA.Mat<5n, 3n>, ~S, ~Model(), cfg, p))

def go2(+dir: String, p: P, x: TA.Mat<5n, 7n>, +ys: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    grads(labs(ys), T.Layer.fwd(P, TA.Mat<5n, 7n>, TA.Mat<5n, 3n>, S, Model(), p, x))
    accum(dir, labs(ys))
    run_opt(dir, labs(ys), T.CSGD{0.1}, "sgd")
    run_opt(dir, labs(ys), T.CMomentum{0.1, 0.9}, "momentum")
    run_opt(dir, labs(ys), T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}, "adam")
    run_opt(dir, labs(ys), T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001}, "adam_l2")
    run_opt(dir, labs(ys), T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001}, "adamw")
    run_opt(dir, labs(ys), T.CAdamWBF16{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001, False{}}, "adamw16m")
    run_opt(dir, labs(ys), T.CAdamWBF16{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001, True{}}, "adamw16mv")

def go(+dir: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params(dir)
    x : TA.Mat<5n, 7n> <- T.Mat.load(5n, 7n, dir ++ "/x.bin", 0)
    ys : List<&2, U32> <- T.read_at(dir ++ "/y.bin", 0, 5)
    go2(dir, p, x, ys)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{dir, Nil{}}}:
      go(dir)
    case _:
      IO.print("usage: mlp_test <dir>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)

