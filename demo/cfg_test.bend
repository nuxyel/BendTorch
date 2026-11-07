# BendTorch #24: the optimizer configuration lives in the Train. Linear(4 -> 2) under MSE, one fixed input and target
# (reference/test_cfg.py builds the same in PyTorch). Prints each case's parameters (w, then b) or its losses, as the
# decimals of their 32 bits:
#   sweep1..3  20 AdamW steps at lr 0.1, 0.01, 0.001, the learning rate given once, to Train.new
#   warmup     20 steps, lr 0.002 (k + 1) for the first five, then 0.01, through Train.with_cfg
#   change     10 steps at 0.01, then Train.with_cfg to 0.001 and 10 more: the moments and 1 - b^t carry over
#   resume     5 steps at 0.01, the checkpoint read into a new Train at 0.01, 5 more: the same bits as change's first 10
#              steps without a change; resume_lr reads it into a Train at 0.001 (the configuration is the Train's, not
#              the checkpoint's); straight10 and change5 are what they must equal
#   refused    a step whose configuration is another kind than the state (SGD -> AdamW, AdamW -> SGD, momentum ->
#              AdamW, AdamW -> AdamW with BF16 moments) or that OptCfg.ok rejects: the losses of steps 1-3 (the first
#              is the forward pass before any update, so finite; NaN after) and the parameters after three steps
#
#   bend demo/cfg_test.bend -o .scratch/cfg_test && reference/.venv/bin/python reference/test_cfg.py .scratch/cfg_test

import Base
import ../torch/main.bend as T
import bend-ml-tensor-array@0.1.20.0/main.bend as TA

def X() -> Type:
  TA.Mat<1n, 4n>

def Y() -> Type:
  TA.Mat<1n, 2n>

def P() -> Type:
  T.Lin<4n, 2n>

def Model() -> T.Layer<P, X, Y, X>:
  T.Linear(1n, 4n, 2n, 0n)

def mse(y: Y, t: Y) -> T.LossOut<Y>:
  T.MSE(1n, 2n, y, t)

def adamw(+lr: F32) -> T.OptCfg:
  T.CAdamW{lr, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def mk(+c: T.OptCfg) -> T.Train<P>:
  T.Train.new(~P, ~X, ~Y, ~X, ~Model(), c, T.Lin{TA.Mat.from_list(2n, 4n, [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35]), TA.Mat.from_list(1n, 2n, [0.0, 0.1])})

def one(tr: T.Train<P>) -> T.Train<P>:
  T.Train.step(~P, ~X, ~Y, ~X, ~Y, ~Model(), ~mse, tr, TA.Mat.from_list(1n, 4n, [0.5, 0.25, 1.0, 0.75]), TA.Mat.from_list(1n, 2n, [1.0, 0.0]))

def plain(k: Nat, tr: T.Train<P>) -> T.Train<P>:
  match k:
    case 0n: tr
    case 1n+q: plain(q, one(tr))

def warm(b: Bool, +i: U32) -> F32:
  match b:
    case True{}: (0.002 * F32.from_nat(U32.to_nat((i + 1 : U32))) : F32)
    case False{}: 0.01

def sched(k: Nat, +i: U32, tr: T.Train<P>) -> T.Train<P>:
  match k:
    case 0n: tr
    case 1n+q: sched(q, (i + 1 : U32), one(T.Train.with_cfg(P, adamw(warm((i < 5 : U32), i)), tr)))

def dumped(-P: Type, r: T.PL<P>) -> String:
  match r:
    case T.PL{p, xs}: T.words(xs)

def params(+name: String, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, l}: IO.print(name ++ dumped(P, T.Layer.dump(P, X, Y, X, Model(), p)))

def loss(+name: String, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, c, p1, p2, l}: T.F32.print(name, l)

# ---- a checkpoint (Train.dump's numbers: parameters, both moments, 1 - b1^t and 1 - b2^t) read into a new Train ----

def resume2(+c: T.OptCfg, r: TA.Mat<1n, 32n>) -> T.Train<P>:
  match r:
    case TA.Mat{a}: plain(5n, T.Train.restore(~P, ~X, ~Y, ~X, ~Model(), mk(c), a))

def resume1(+c: T.OptCfg, r: T.TrainPL<P>) -> T.Train<P>:
  match r:
    case T.TrainPL{tr, xs}: resume2(c, TA.Mat.from_list(1n, 32n, xs))

def resume(+c: T.OptCfg) -> T.Train<P>:
  resume1(c, T.Train.dump(~P, ~X, ~Y, ~X, ~Model(), plain(5n, mk(adamw(0.01)))))

# ---- refused steps ----

def refused(+name: String, +built: T.OptCfg, +stepped: T.OptCfg) -> IO(Unit):
  do IO<Unit>:
    loss(name ++ ".l1", plain(1n, T.Train.with_cfg(P, stepped, mk(built))))
    loss(name ++ ".l2", plain(2n, T.Train.with_cfg(P, stepped, mk(built))))
    loss(name ++ ".l3", plain(3n, T.Train.with_cfg(P, stepped, mk(built))))
    params(name ++ ".p", plain(3n, T.Train.with_cfg(P, stepped, mk(built))))

def test_main() -> IO(Unit):
  do IO<Unit>:
    params("sweep1", plain(20n, mk(adamw(0.1))))
    params("sweep2", plain(20n, mk(adamw(0.01))))
    params("sweep3", plain(20n, mk(adamw(0.001))))
    params("warmup", sched(20n, 0, mk(adamw(0.002))))
    params("change", plain(10n, T.Train.with_cfg(P, adamw(0.001), plain(10n, mk(adamw(0.01))))))
    params("straight10", plain(10n, mk(adamw(0.01))))
    params("resume", resume(adamw(0.01)))
    params("change5", plain(5n, T.Train.with_cfg(P, adamw(0.001), plain(5n, mk(adamw(0.01))))))
    params("resume_lr", resume(adamw(0.001)))
    refused("sgd_adamw", T.CSGD{0.1}, adamw(0.01))
    refused("adamw_sgd", adamw(0.01), T.CSGD{0.1})
    refused("momentum_adamw", T.CMomentum{0.1, 0.9}, adamw(0.01))
    refused("adamw_bf16", adamw(0.01), T.CAdamWBF16{0.01, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001, False{}})
    refused("bad_c1", adamw(0.01), T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.01, 0.01, 0.001})
    refused("same_kind", adamw(0.01), adamw(0.001))

# a test takes no arguments: one would be a runtime option written outside +RTS ... -RTS, silently ignored
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, test_main())
