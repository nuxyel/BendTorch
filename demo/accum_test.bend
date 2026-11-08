# Gradient accumulation (Train.accumulate and Train.flush, BendTorch #36) on demo/mlp_test.bend's MLP, Linear(7, 6) ->
# ReLU -> Linear(6, 3) with cross-entropy, three optimizer steps under each optimizer:
#   step.<opt>, one.<opt>, oneat.<opt>: Train.step, then one accumulate and a flush a step, then the same through
#     Train.accumulate_at and Train.flush_at, all on the first 5 rows (k = 1: the same bits three times);
#   acc.<opt>: k = 4 micro-batches of 5 rows a step, each loss divided by 4 (LossOut.div): every micro-batch's loss,
#     the state after the three steps and the gradient after the last flush (zeros);
#   at.<opt>: acc's run through Train.accumulate_at and Train.flush_at (the same bits as acc);
#   big.<opt>: Train.step on all 20 rows at once, the batch acc splits in four;
#   gsum: the gradient after the four micro-batches of one step, before any flush.
# Each number as the decimal of its 32 bits; reference/test_accum.py compares them with PyTorch's accumulated step.
#
#   bend demo/accum_test.bend -o .scratch/accum_test && .scratch/accum_test <dir>

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def P() -> Type:
  T.Two<T.Two<T.Lin<7n, 6n>, Unit>, T.Lin<6n, 3n>>

def X(n: Nat) -> Type:
  TA.Mat<n, 7n>

def Y(n: Nat) -> Type:
  TA.Mat<n, 3n>

def S(n: Nat) -> Type:
  T.Two<T.Two<TA.Mat<n, 7n>, TA.Mat<n, 6n>>, TA.Mat<n, 6n>>

def Hidden(~n: Nat) -> T.Layer<T.Two<T.Lin<7n, 6n>, Unit>, X(n), TA.Mat<n, 6n>, T.Two<X(n), TA.Mat<n, 6n>>>:
  T.Seq(~T.Lin<7n, 6n>, ~Unit, ~X(n), ~TA.Mat<n, 6n>, ~TA.Mat<n, 6n>, ~X(n), ~TA.Mat<n, 6n>, ~T.Linear(n, 7n, 6n, 1n), ~T.ReLU(n, 6n))

def Model(~n: Nat) -> T.Layer<P, X(n), Y(n), S(n)>:
  T.Seq(~T.Two<T.Lin<7n, 6n>, Unit>, ~T.Lin<6n, 3n>, ~X(n), ~TA.Mat<n, 6n>, ~Y(n), ~T.Two<X(n), TA.Mat<n, 6n>>, ~TA.Mat<n, 6n>, ~Hidden(~n), ~T.Linear(n, 6n, 3n, 1n))

# the same model as one owned record, for the _at forms
def HiddenAt(+n: Nat) -> T.Layer<T.Two<T.Lin<7n, 6n>, Unit>, X(n), TA.Mat<n, 6n>, T.Two<X(n), TA.Mat<n, 6n>>>:
  T.Seq.at(T.Lin<7n, 6n>, Unit, X(n), TA.Mat<n, 6n>, TA.Mat<n, 6n>, X(n), TA.Mat<n, 6n>, T.Linear(n, 7n, 6n, 1n), T.ReLU(n, 6n))

def ModelAt(+n: Nat) -> T.Layer<P, X(n), Y(n), S(n)>:
  T.Seq.at(T.Two<T.Lin<7n, 6n>, Unit>, T.Lin<6n, 3n>, X(n), TA.Mat<n, 6n>, Y(n), T.Two<X(n), TA.Mat<n, 6n>>, TA.Mat<n, 6n>, HiddenAt(n), T.Linear(n, 6n, 3n, 1n))

def ce5(z: Y(5n), labels: List<&2, Nat>) -> T.LossOut<Y(5n)>:
  T.CrossEntropy(5n, 3n, z, labels)

# one of four micro-batches: PyTorch's loss / 4
def ce5q(z: Y(5n), labels: List<&2, Nat>) -> T.LossOut<Y(5n)>:
  T.LossOut.div(5n, 3n, 4, T.CrossEntropy(5n, 3n, z, labels))

def ce20(z: Y(20n), labels: List<&2, Nat>) -> T.LossOut<Y(20n)>:
  T.CrossEntropy(20n, 3n, z, labels)

# ---- the steps compared ----

def stp(tr: T.Train<P>, x: X(5n), y: List<&2, Nat>) -> T.Train<P>:
  T.Train.step(~P, ~X(5n), ~Y(5n), ~S(5n), ~List<&2, Nat>, ~Model(~5n), ~ce5, tr, x, y)

def one(tr: T.Train<P>, x: X(5n), y: List<&2, Nat>) -> T.Train<P>:
  T.Train.flush(~P, ~X(5n), ~Y(5n), ~S(5n), ~Model(~5n), T.Train.accumulate(~P, ~X(5n), ~Y(5n), ~S(5n), ~List<&2, Nat>, ~Model(~5n), ~ce5, tr, x, y))

def oneat(tr: T.Train<P>, x: X(5n), y: List<&2, Nat>) -> T.Train<P>:
  T.Train.flush_at(P, X(5n), Y(5n), S(5n), ModelAt(5n), T.Train.accumulate_at(P, X(5n), Y(5n), S(5n), List<&2, Nat>, ModelAt(5n), z => t => ce5(z, t), tr, x, y))

def acc_t(tr: T.Train<P>, x: X(5n), y: List<&2, Nat>) -> T.Train<P>:
  T.Train.accumulate(~P, ~X(5n), ~Y(5n), ~S(5n), ~List<&2, Nat>, ~Model(~5n), ~ce5q, tr, x, y)

def fl_t(tr: T.Train<P>) -> T.Train<P>:
  T.Train.flush(~P, ~X(5n), ~Y(5n), ~S(5n), ~Model(~5n), tr)

def acc_a(tr: T.Train<P>, x: X(5n), y: List<&2, Nat>) -> T.Train<P>:
  T.Train.accumulate_at(P, X(5n), Y(5n), S(5n), List<&2, Nat>, ModelAt(5n), z => t => ce5q(z, t), tr, x, y)

def fl_a(tr: T.Train<P>) -> T.Train<P>:
  T.Train.flush_at(P, X(5n), Y(5n), S(5n), ModelAt(5n), tr)

# ---- inputs and printing ----

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

# micro-batch j: rows 5j..5j+4 of x.bin (7 numbers of 4 bytes a row) and of y.bin (a byte a label)
def xs(+dir: String, +j: U32) -> IO(X(5n)):
  T.Mat.load(5n, 7n, dir ++ "/x.bin", (j * 140 : U32))

def ys(+dir: String, +j: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(dir ++ "/y.bin", (j * 5 : U32), 5)
    return labs(bs)

def print_params(+tag: String, p: P) -> IO(Unit):
  match p:
    case T.Two{T.Two{T.Lin{w1, b1}, u}, T.Lin{w2, b2}}:
      do IO<Unit>:
        T.Mat.print(6n, 7n, tag ++ ".w1", w1)
        T.Mat.print(1n, 6n, tag ++ ".b1", b1)
        T.Mat.print(3n, 6n, tag ++ ".w2", w2)
        T.Mat.print(1n, 3n, tag ++ ".b2", b2)

# the state as Train.dump gives it (parameters, the moments in use, then 1 - b1^t and 1 - b2^t for Adam), and the
# gradient the record holds
def show(+tag: String, r: T.TrainPL<P>) -> IO(Unit):
  match r:
    case T.TrainPL{T.Train{p, g, st, c, p1, p2, l}, xs}:
      do IO<Unit>:
        IO.print(tag ++ ".state" ++ T.words(xs))
        print_params(tag ++ ".g", g)

def dump(tr: T.Train<P>) -> T.TrainPL<P>:
  T.Train.dump(~P, ~X(5n), ~Y(5n), ~S(5n), ~Model(~5n), tr)

# ---- the loops: each prints the record's loss before every call (0 before the first) and after the last ----

# k steps of f on micro-batch 0
def single(~f: T.Train<P> -> X(5n) -> List<&2, Nat> -> T.Train<P>, k: Nat, +dir: String, +tag: String, tr: T.Train<P>) -> IO(Unit):
  match k tr:
    case 0n T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        show(tag, dump(T.Train{p, g, st, oc, p1, p2, l}))
    case 1n+q T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        x : X(5n) <- xs(dir, 0)
        y : List<&2, Nat> <- ys(dir, 0)
        single(~f, q, dir, tag, f(T.Train{p, g, st, oc, p1, p2, l}, x, y))

def flush_if(~fl: T.Train<P> -> T.Train<P>, last: Bool, tr: T.Train<P>) -> T.Train<P>:
  match last:
    case True{}:
      fl(tr)
    case False{}:
      tr

# left micro-batches more, j the next one (0..3); a flush after micro-batch 3
def micro(~acc: T.Train<P> -> X(5n) -> List<&2, Nat> -> T.Train<P>, ~fl: T.Train<P> -> T.Train<P>, left: Nat, +j: U32, +dir: String, +tag: String, tr: T.Train<P>) -> IO(Unit):
  match left tr:
    case 0n T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        show(tag, dump(T.Train{p, g, st, oc, p1, p2, l}))
    case 1n+q T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        x : X(5n) <- xs(dir, j)
        y : List<&2, Nat> <- ys(dir, j)
        micro(~acc, ~fl, q, U32.mod((j + 1 : U32), 4), dir, tag, flush_if(~fl, U32.is_eq(j, 3), acc(T.Train{p, g, st, oc, p1, p2, l}, x, y)))

# k steps on all 20 rows
def big(k: Nat, +dir: String, +tag: String, tr: T.Train<P>) -> IO(Unit):
  match k tr:
    case 0n T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        show(tag, dump(T.Train{p, g, st, oc, p1, p2, l}))
    case 1n+q T.Train{p, g, st, oc, p1, p2, +l}:
      do IO<Unit>:
        T.F32.print(tag ++ ".loss", l)
        x : X(20n) <- T.Mat.load(20n, 7n, dir ++ "/x.bin", 0)
        bs : List<&2, U32> <- T.read_at(dir ++ "/y.bin", 0, 20)
        big(q, dir, tag, T.Train.step(~P, ~X(20n), ~Y(20n), ~S(20n), ~List<&2, Nat>, ~Model(~20n), ~ce20, T.Train{p, g, st, oc, p1, p2, l}, x, labs(bs)))

# the four micro-batches of one step into a new record's gradient, printed before any flush
def gsum(left: Nat, +j: U32, +dir: String, tr: T.Train<P>) -> IO(Unit):
  match left tr:
    case 0n T.Train{p, g, st, oc, p1, p2, l}:
      print_params("gsum", g)
    case 1n+q t0:
      do IO<Unit>:
        x : X(5n) <- xs(dir, j)
        y : List<&2, Nat> <- ys(dir, j)
        gsum(q, (j + 1 : U32), dir, acc_t(t0, x, y))

def new(+cfg: T.OptCfg, p: P) -> T.Train<P>:
  T.Train.new(~P, ~X(5n), ~Y(5n), ~S(5n), ~Model(~5n), cfg, p)

def run_opt(+dir: String, +cfg: T.OptCfg, +tag: String) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg)
    a : P <- load_params(dir)
    single(~stp, 3n, dir, "step." ++ tag, new(cfg, a))
    b : P <- load_params(dir)
    single(~one, 3n, dir, "one." ++ tag, new(cfg, b))
    c : P <- load_params(dir)
    single(~oneat, 3n, dir, "oneat." ++ tag, new(cfg, c))
    d : P <- load_params(dir)
    micro(~acc_t, ~fl_t, 12n, 0, dir, "acc." ++ tag, new(cfg, d))
    e : P <- load_params(dir)
    micro(~acc_a, ~fl_a, 12n, 0, dir, "at." ++ tag, new(cfg, e))
    f : P <- load_params(dir)
    big(3n, dir, "big." ++ tag, new(cfg, f))

def go(+dir: String) -> IO(Unit):
  do IO<Unit>:
    p : P <- load_params(dir)
    gsum(4n, 0, dir, new(T.CSGD{0.1}, p))
    run_opt(dir, T.CSGD{0.1}, "sgd")
    run_opt(dir, T.CMomentum{0.1, 0.9}, "momentum")
    run_opt(dir, T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}, "adam")
    run_opt(dir, T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001}, "adamw")
    run_opt(dir, T.CAdamWBF16{0.01, 0.9, 0.999, 0.00000001, 0.1, 0.1, 0.001, True{}}, "adamw16mv")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{dir, Nil{}}}:
      go(dir)
    case _:
      IO.print("usage: accum_test <dir>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
