# demo/tiny_llama.bend's model and data with AdamW's moments stored as packed BF16 (T.CAdamWBF16), against
# reference/tiny_bf16opt_ref.py.
#
#   bend demo/tiny_bf16opt.bend -o .scratch/tiny_bf16opt
#   .scratch/tiny_bf16opt train <steps> <m|mv>             # the loss before each update, then the final weights
#   .scratch/tiny_bf16opt save <steps> <path> <m|mv>       # train, then write a checkpoint
#   .scratch/tiny_bf16opt resume <path> <from> <k> <m|mv>  # k more steps from a checkpoint taken after <from> steps
#
# m: the first moment packed, the second in F32; mv: both packed. The weights, the gradient and every operation stay
# F32, with tiny_llama.bend's hyperparameters, initial weights (data/tiny/init.bin) and batches.

import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import ./tiny_llama.bend as TL

def cfg(v16: Bool) -> T.OptCfg:
  T.CAdamWBF16{0.003, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001, v16}

def step_with(+cfg: T.OptCfg, +ids: List<&2, Nat>, tr: T.Train<TL.P>) -> T.Train<TL.P>:
  T.Train.step(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~List<&2, Nat>, ~TL.Model(~2n), ~TL.ce, cfg, tr, TL.xs_of(ids), TL.ys_of(ids))

# k more steps from batch `step`; prints the loss of the step before (0 before the first)
def tloop(k: Nat, +step: U32, +cfg: T.OptCfg, tr: T.Train<TL.P>) -> IO(T.Train<TL.P>):
  match k tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<TL.P>>:
        T.F32.print("loss", l)
        return T.Train{p, g, st, p1, p2, l}
    case 1n+q T.Train{p, g, st, p1, p2, +l}:
      do IO<T.Train<TL.P>>:
        T.F32.print("loss", l)
        ids : List<&2, Nat> <- TL.load_batch(step)
        tloop(q, (step + 1 : U32), cfg, step_with(cfg, ids, T.Train{p, g, st, p1, p2, l}))

def train_run(+k: Nat, +v16: Bool, p: TL.P) -> IO(T.Train<TL.P>):
  do IO<T.Train<TL.P>>:
    T.OptCfg.check(cfg(v16))
    tloop(k, 0, cfg(v16), T.Train.new(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~TL.Model(~2n), cfg(v16), p))

def train(+k: Nat, +v16: Bool) -> IO(Unit):
  do IO<Unit>:
    p : TL.P <- TL.load_params()
    tr : T.Train<TL.P> <- train_run(k, v16, p)
    TL.finish(tr)

def save(+k: Nat, +path: String, +v16: Bool) -> IO(Unit):
  do IO<Unit>:
    p : TL.P <- TL.load_params()
    tr : T.Train<TL.P> <- train_run(k, v16, p)
    tr2 : T.Train<TL.P> <- T.Train.save(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~TL.Model(~2n), path, tr)
    IO.print("saved " ++ path)

def resume(+path: String, +from: Nat, +k: Nat, +v16: Bool) -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg(v16))
    tr : T.Train<TL.P> <- L.train(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~TL.Model(~2n), path, T.Train.new(~TL.P, ~TL.X, ~TL.Y(2n), ~TL.S(2n), ~TL.Model(~2n), cfg(v16), T.Layer.zeros(TL.P, TL.X, TL.Y(2n), TL.S(2n), TL.Model(~2n))))
    tr2 : T.Train<TL.P> <- tloop(k, U32.from_nat(from), cfg(v16), tr)
    TL.finish(tr2)

# ---- the command line ----

def usage() -> IO(Unit):
  IO.die(Unit, 2, "usage: tiny_bf16opt train <steps> <m|mv> | save <steps> <path> <m|mv> | resume <path> <from> <k> <m|mv>")

def v16_of(+s: String) -> Bool:
  String.eq(s, "mv")

def known(+s: String) -> Bool:
  Bool.or(String.eq(s, "m"), String.eq(s, "mv"))

def train_m(ok: Bool, mk: Maybe<&2, Nat>, +v: String) -> IO(Unit):
  match ok mk:
    case True{} Some{k}:
      train(k, v16_of(v))
    case _ _:
      usage()

def save_m(ok: Bool, mk: Maybe<&2, Nat>, +path: String, +v: String) -> IO(Unit):
  match ok mk:
    case True{} Some{k}:
      save(k, path, v16_of(v))
    case _ _:
      usage()

def resume_m(ok: Bool, ma: Maybe<&2, Nat>, mb: Maybe<&2, Nat>, +path: String, +v: String) -> IO(Unit):
  match ok ma mb:
    case True{} Some{a} Some{b}:
      resume(path, a, b, v16_of(v))
    case _ _ _:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, Con{k, Con{+v, Nil{}}}}}:
      train_m(Bool.and(String.eq(mode, "train"), known(v)), Nat.read(k), v)
    case Con{prog, Con{+mode, Con{k, Con{+path, Con{+v, Nil{}}}}}}:
      save_m(Bool.and(String.eq(mode, "save"), known(v)), Nat.read(k), path, v)
    case Con{prog, Con{+mode, Con{+path, Con{a, Con{b, Con{+v, Nil{}}}}}}}:
      resume_m(Bool.and(String.eq(mode, "resume"), known(v)), Nat.read(a), Nat.read(b), path, v)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
