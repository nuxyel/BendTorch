# MNIST with BendTorch's general layers: LinearFirst(784, 128) -> ReLU -> Linear(128, 10), cross-entropy, any optimizer.
# The training step is the package's Train.step, the same code that trains every other model; nothing here is
# specific to MNIST but the shapes and the data.
# The linear layers are torch/fast.bend's (products over bend-ml-tensor-fork, outside the proofs): main.bend's
# layers with the same bits (demo/fast_test.bend), faster where a product splits into parallel blocks.
#
#   bend demo/mnist.bend -o .scratch/mnist && .scratch/mnist <epochs> <batches (0 = all 600)> <sgd|momentum|adam|adamw> <lr>
#
# Data in data/mnist/ (scripts/mnist.sh); initial weights from reference/mnist_ref.py --init, its PyTorch twin.

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ../torch/fast.bend as F
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def X(n: Nat) -> Type:
  TA.Mat<n, 784n>

def Y(n: Nat) -> Type:
  TA.Mat<n, 10n>

def P() -> Type:
  T.Two<T.Two<T.Lin<784n, 128n>, Unit>, T.Lin<128n, 10n>>

def S(n: Nat) -> Type:
  T.Two<T.Two<X(n), TA.Mat<n, 128n>>, TA.Mat<n, 128n>>

def Hidden(+n: Nat) -> T.Layer<T.Two<T.Lin<784n, 128n>, Unit>, X(n), TA.Mat<n, 128n>, T.Two<X(n), TA.Mat<n, 128n>>>:
  T.Seq.at(T.Lin<784n, 128n>, Unit, X(n), TA.Mat<n, 128n>, TA.Mat<n, 128n>, X(n), TA.Mat<n, 128n>, F.LinearFirst(n, 784n, 128n, 4n), T.ReLU(n, 128n))

def Model(+n: Nat) -> T.Layer<P, X(n), Y(n), S(n)>:
  T.Seq.at(T.Two<T.Lin<784n, 128n>, Unit>, T.Lin<128n, 10n>, X(n), TA.Mat<n, 128n>, Y(n), T.Two<X(n), TA.Mat<n, 128n>>, TA.Mat<n, 128n>, Hidden(n), F.Linear(n, 128n, 10n, 0n))

# ---- data ----

def labs(bs: List<&2, U32>) -> List<&2, Nat>:
  match bs:
    case Nil{}:
      Nil{}
    case Con{+b, t}:
      U32.to_nat(b) <> labs(t)

type Pixels is Type:
  Pixels{a: Array<F32>, end: U32}

# Bounded byte lists keep large run-time batches within the JavaScript stack limit too.
def pixels_fill(bs: List<&2, U32>, +i: U32, a: Array<F32>) -> Pixels:
  match bs:
    case Nil{}: Pixels{a, i}
    case Con{+b, rest}: pixels_fill(rest, (i + 1 : U32), Array.set(F32, a, i, (U32.to_f32(b) / 255.0 : F32)))

def pixels_done(+expected: U32, r: Pixels) -> IO(Array<F32>):
  match r:
    case Pixels{a, +end}:
      do IO<Array<F32>>:
        L.require(U32.is_eq(end, expected), "data/mnist: a batch is short (run scripts/mnist.sh)")
        return a

def pixels_read(fuel: Nat, +path: String, +off: U32, +total: U32, +i: U32, a: Array<F32>) -> IO(Array<F32>):
  match fuel:
    case 0n: IO.pure(Array<F32>, a)
    case 1n+q:
      do IO<Array<F32>>:
        +want : U32 = Bool.pick(U32, U32.is_le((total - i : U32), 262144), (total - i : U32), 262144)
        bs : List<&2, U32> <- T.read_at(path, (off + i : U32), want)
        a2 : Array<F32> <- pixels_done((i + want : U32), pixels_fill(bs, i, a))
        pixels_read(q, path, off, total, (i + want : U32), a2)

def load_x(+n: Nat, +path: String, +row: U32) -> IO(X(n)):
  do IO<X(n)>:
    +total : Nat = Nat.mul(n, 784n)
    a : Array<F32> <- pixels_read(Nat.div(Nat.add(total, 262143n), 262144n), path, (16 + (row * 784 : U32) : U32), U32.from_nat(total), 0, Array.new(F32, T.depth(total), 0.0))
    return TA.Mat{a}

def load_y(+n: Nat, +path: String, +row: U32) -> IO(List<&2, Nat>):
  do IO<List<&2, Nat>>:
    bs : List<&2, U32> <- T.read_at(path, (8 + row : U32), U32.from_nat(n))
    return labs(bs)

# Each full or final partial batch instantiates its own checked matrix shape at run time.
def train_batch(+n: Nat, +row: U32, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(T.Train<P>):
  do IO<T.Train<P>>:
    x : X(n) <- load_x(n, "data/mnist/train-images-idx3-ubyte", row)
    ys : List<&2, Nat> <- load_y(n, "data/mnist/train-labels-idx1-ubyte", row)
    return T.Train.step_at(P, X(n), Y(n), S(n), List<&2, Nat>, Model(n), z => targets => T.CrossEntropy(n, 10n, z, targets), cfg, tr, x, ys)

# Mean of batch losses, as before; a final partial batch is one batch in the mean.
def tloop(k: Nat, +n: Nat, +row: U32, +cfg: T.OptCfg, +acc: F32, tr: T.Train<P>) -> IO(T.Train<P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      IO.pure(T.Train<P>, T.Train{p, g, st, p1, p2, (acc + l : F32)})
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        +take : Nat = Nat.min(n, Nat.sub(60000n, U32.to_nat(row)))
        next : T.Train<P> <- train_batch(take, row, cfg, T.Train{p, g, st, p1, p2, l})
        tloop(q, n, (row + U32.from_nat(take) : U32), cfg, (acc + l : F32), next)

type Hits is Type:
  Hits{n: Nat, p: P}

def ev3(-nrows: Nat, +n: Nat, p: P, r: TA.MHits<nrows, 10n>) -> Hits:
  match r:
    case TA.MHits{z, k}: Hits{Nat.add(n, k), p}

def ev2(+nrows: Nat, +n: Nat, +ys: List<&2, Nat>, r: T.Fwd<P, Y(nrows), S(nrows)>) -> Hits:
  match r:
    case T.Fwd{p, y, s}: ev3(nrows, n, p, TA.Mat.count_correct(nrows, 10n, y, ys))

def eval_batch(+n: Nat, +row: U32, h: Hits) -> IO(Hits):
  match h:
    case Hits{hits, p}:
      do IO<Hits>:
        x : X(n) <- load_x(n, "data/mnist/t10k-images-idx3-ubyte", row)
        ys : List<&2, Nat> <- load_y(n, "data/mnist/t10k-labels-idx1-ubyte", row)
        return ev2(n, hits, ys, T.Layer.fwd(P, X(n), Y(n), S(n), Model(n), p, x))

def eloop(k: Nat, +n: Nat, +row: U32, h: Hits) -> IO(Hits):
  match k:
    case 0n: IO.pure(Hits, h)
    case 1n+q:
      do IO<Hits>:
        +take : Nat = Nat.min(n, Nat.sub(10000n, U32.to_nat(row)))
        h2 : Hits <- eval_batch(take, row, h)
        eloop(q, n, (row + U32.from_nat(take) : U32), h2)

# ceil(total / n), with n checked positive at the boundary.
def batches(+total: Nat, +n: Nat) -> Nat:
  Nat.div(Nat.add(total, Nat.sub(n, 1n)), n)

# ---- epochs ----

def ms_show(+ms: Nat) -> String:
  Nat.show(Nat.div(ms, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(ms, 1000n), 100n))

def report(+e: Nat, +nb: Nat, +secs: Nat, +sum: F32, h: Hits) -> IO(P):
  match h:
    case Hits{n, p}:
      do IO<P>:
        IO.print("epoch " ++ Nat.show(e) ++ " train_loss=" ++ F32.show((sum / F32.from_nat(nb) : F32)) ++ " test_correct=" ++ Nat.show(n) ++ "/10000 train_seconds=" ++ ms_show(secs))
        return p

# the next epoch starts from these parameters with a fresh loss sum; the gradient and the optimizer state carry over
def epoch_end(+n: Nat, +e: Nat, +nb: Nat, +secs: Nat, tr: T.Train<P>) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<P>>:
        h : Hits <- eloop(batches(10000n, n), n, 0, Hits{0n, p})
        p3 : P <- report(e, nb, secs, l, h)
        return T.Train{p3, g, st, p1, p2, 0.0}

def epochs(k: Nat, +n: Nat, +e: Nat, +nb: Nat, +cfg: T.OptCfg, tr: T.Train<P>) -> IO(Unit):
  match k:
    case 0n:
      IO.print("done")
    case 1n+q:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        tr2 : T.Train<P> <- tloop(nb, n, 0, cfg, 0.0, tr)
        t1 : Nat <- IO.now()
        tr3 : T.Train<P> <- epoch_end(n, e, nb, Nat.sub(t1, t0), tr2)
        epochs(q, n, 1n+e, nb, cfg, tr3)

def load_params() -> IO(P):
  L.weights(~P, ~X(100n), ~Y(100n), ~S(100n), ~Model(100n), "data/mnist/init.bin", T.Layer.zeros(P, X(100n), Y(100n), S(100n), Model(100n)))

def cfg_pick(a: Bool, b: Bool, c: Bool, d: Bool, +lr: F32) -> Maybe<&2, T.OptCfg>:
  match a b c d:
    case True{} _ _ _:
      Some{T.CSGD{lr}}
    case False{} True{} _ _:
      Some{T.CMomentum{lr, 0.9}}
    case False{} False{} True{} _:
      Some{T.CAdam{lr, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}}
    case False{} False{} False{} True{}:
      Some{T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}}
    case _ _ _ _:
      None{}

def cfg_of(+name: String, +lr: F32) -> Maybe<&2, T.OptCfg>:
  cfg_pick(String.eq(name, "sgd"), String.eq(name, "momentum"), String.eq(name, "adam"), String.eq(name, "adamw"), lr)

def nb_pick(all: Bool, +total: Nat, +m: Nat) -> Nat:
  match all:
    case True{}: total
    case False{}: m

def go(+n: Nat, +e: Nat, +m: Nat, mc: Maybe<&2, T.OptCfg>) -> IO(Unit):
  match mc:
    case None{}: IO.die(Unit, 1, "optimizer: sgd, momentum, adam or adamw")
    case Some{+cfg}:
      do IO<Unit>:
        L.require(Nat.is_le(1n, n), "batch size must be positive")
        L.require(Nat.is_le(n, 60000n), "batch size must be at most 60000")
        +nb : Nat = nb_pick(Nat.is_eq(m, 0n), batches(60000n, n), m)
        L.require(Nat.is_le(nb, batches(60000n, n)), "too many training batches")
        T.OptCfg.check(cfg)
        p : P <- load_params()
        epochs(e, n, 1n, nb, cfg, T.Train.new(~P, ~X(100n), ~Y(100n), ~S(100n), ~Model(100n), cfg, p))

def usage() -> IO(Unit):
  IO.die(Unit, 1, "usage: mnist <epochs> <batches (0 = all)> <sgd|momentum|adam|adamw> <lr> [batch size (100)]")

def go2(mn: Maybe<&2, Nat>, me: Maybe<&2, Nat>, mm: Maybe<&2, Nat>, +name: String, ml: Maybe<&2, F32>) -> IO(Unit):
  match mn me mm ml:
    case Some{n} Some{e} Some{m} Some{lr}: go(n, e, m, cfg_of(name, lr))
    case _ _ _ _: usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{e, Con{m, Con{o, Con{lr, Nil{}}}}}}:
      go2(Some{100n}, Nat.read(e), Nat.read(m), o, F32.read(lr))
    case Con{prog, Con{e, Con{m, Con{o, Con{lr, Con{n, Nil{}}}}}}}:
      go2(Nat.read(n), Nat.read(e), Nat.read(m), o, F32.read(lr))
    case _: usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
