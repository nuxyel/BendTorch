# M9 infrastructure: one tied parameter, a Dropout counter, all optimizer states, loading cursors and checkpoints.
import Base
import ../torch/main.bend as T
import ../torch/load.bend as L
import bend-ml-tensor-array@0.1.18.0/main.bend as TA

def M() -> Type:
  TA.Mat<2n, 3n>

def P() -> Type:
  T.Two<TA.Mat<3n, 3n>, T.DropP>

def S() -> Type:
  T.Three<M, M, M>

def Model() -> T.Layer<P, M, M, S>:
  T.Tie(~TA.Mat<3n, 3n>, ~T.DropP, ~M, ~M, ~M, ~M, ~M, ~M, ~M, ~T.LinearNB(2n, 3n, 3n, 0n), ~T.Dropout(2n, 3n, 0.25, 7), ~T.LinearNB(2n, 3n, 3n, 0n))

def adamw() -> T.OptCfg:
  # The dispatcher validates the name before using this selection.
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}

def choose(a: Bool, b: Bool, c: Bool, d: Bool, e: Bool) -> IO(T.OptCfg):
  match a b c d e:
    case True{} _ _ _ _: IO.pure(T.OptCfg, T.CSGD{0.01})
    case _ True{} _ _ _: IO.pure(T.OptCfg, T.CMomentum{0.01, 0.9})
    case _ _ True{} _ _: IO.pure(T.OptCfg, T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001})
    case _ _ _ True{} _: IO.pure(T.OptCfg, adamw())
    # c2 written as 0.01 for b2 = 0.999: OptCfg.check must stop it, and without the check every loss after the first
    # is NaN
    case _ _ _ _ True{}: IO.pure(T.OptCfg, T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.01})
    case _ _ _ _ _: IO.die(T.OptCfg, 1, "unknown optimizer")

def initial(+c: T.OptCfg) -> T.Train<P>:
  T.Train.new(~P, ~M, ~M, ~S, ~Model(), c, T.Two{TA.Mat.from_list(3n, 3n, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), T.DropP{0}})

def loss(x: M, y: M) -> T.LossOut<M>:
  T.MSE(2n, 3n, x, y)

def steps(k: Nat, +c: T.OptCfg, tr: T.Train<P>) -> T.Train<P>:
  match k:
    case 0n: tr
    case 1n+q:
      steps(q, c, T.Train.step(~P, ~M, ~M, ~S, ~M, ~Model(), ~loss, c, tr, TA.Mat.from_list(2n, 3n, [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]), TA.Mat.from_list(2n, 3n, [0.1, 0.3, 0.2, 0.4, 0.3, 0.5])))

def finish(r: T.TrainPL<P>) -> IO(Unit):
  match r:
    case T.TrainPL{tr, xs}: IO.print("state" ++ T.words(xs))

def params_dump(r: T.PL<P>) -> IO(Unit):
  match r:
    case T.PL{p, xs}: IO.print("params" ++ T.words(xs))

def params_fin(p: P) -> IO(Unit):
  params_dump(T.Layer.dump(P, M, M, S, Model(), p))

def params_of(+path: String, tr: T.Train<P>) -> IO(Unit):
  match tr:
    case T.Train{p, g, st, p1, p2, l}:
      do IO<Unit>:
        q : P <- T.Params.save(~P, ~M, ~M, ~S, ~Model(), path, p)
        params_fin(q)

def run(mode: String, +c: T.OptCfg, +k: Nat, +path: String, load: Bool, save: Bool, params: Bool) -> IO(Unit):
  match load save params:
    case _ _ True{}:
      params_of(path, steps(k, c, initial(c)))
    case True{} _ _:
      do IO<Unit>:
        tr : T.Train<P> <- L.train(~P, ~M, ~M, ~S, ~Model(), path, initial(c))
        finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), steps(k, c, tr)))
    case _ True{} _:
      do IO<Unit>:
        tr : T.Train<P> <- T.Train.save(~P, ~M, ~M, ~S, ~Model(), path, steps(k, c, initial(c)))
        finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), tr))
    case _ _ _:
      finish(T.Train.dump(~P, ~M, ~M, ~S, ~Model(), steps(k, c, initial(c))))

# Every mode checks the configuration first, as a program should; "unchecked" runs like "run" without it.
def check_unless(skip: Bool, +c: T.OptCfg) -> IO(T.OptCfg):
  match skip:
    case True{}: IO.pure(T.OptCfg, c)
    case False{}:
      do IO<T.OptCfg>:
        T.OptCfg.check(c)
        IO.pure(T.OptCfg, c)

def go(+mode: String, +opt: String, mk: Maybe<&2, Nat>, +path: String) -> IO(Unit):
  match mk:
    case None{}: IO.die(Unit, 1, "invalid step count")
    case Some{+k}:
      do IO<Unit>:
        c0 : T.OptCfg <- choose(String.eq(opt, "sgd"), String.eq(opt, "momentum"), String.eq(opt, "adam"), String.eq(opt, "adamw"), String.eq(opt, "adamw-bad"))
        c : T.OptCfg <- check_unless(String.eq(mode, "unchecked"), c0)
        run(mode, c, k, path, String.eq(mode, "resume"), String.eq(mode, "save"), String.eq(mode, "params"))

def cursor_done(r: T.PA<TA.Mat<3n, 3n>>) -> IO(Unit):
  match r:
    case T.PA{m, a, off}:
      do IO<Unit>:
        IO.print("offset " ++ U32.show(off))
        T.Mat.print(3n, 3n, "weights", m)

def cursor(+path: String) -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> <- L.read(path, 11)
    cursor_done(T.Mat.restore(3n, 3n, TA.Mat.zeros(3n, 3n), a, 1))

def read_last(r: Array<F32> & F32, +first: F32) -> IO(Unit):
  match r:
    case Tuple{a, last}: IO.print("ends " ++ U32.show(T.bits(first)) ++ " " ++ U32.show(T.bits(last)))

def read_first(+n: U32, r: Array<F32> & F32) -> IO(Unit):
  match r:
    case Tuple{a, +first}: read_last(Array.get(F32, a, U32.sub(n, 1)), first)

def read_file(+path: String, mn: Maybe<&2, U32>) -> IO(Unit):
  match mn:
    case None{}: IO.die(Unit, 1, "invalid count")
    case Some{+n}:
      do IO<Unit>:
        L.require(U32.is_gt(n, 0), "count must be positive")
        a : Array<F32> <- L.read(path, n)
        read_first(n, Array.get(F32, a, 0))

# `ok b1 b2 c1 c2 ...`: OptCfg.ok of a CAdam and a CAdamW for each four F32s given as their U32 bits: 1 when both
# accept them, 0 when both reject them, 2 when they disagree
def word(m: Maybe<&2, U32>) -> F32:
  match m:
    case Some{u}: T.bitcast(u)
    case None{}: (0.0 / 0.0 : F32)

def both(a: Bool, b: Bool) -> String:
  match a b:
    case True{} True{}: "1"
    case False{} False{}: "0"
    case _ _: "2"

def ok_list(xs: List<String>, +acc: String) -> IO(Unit):
  match xs:
    case Con{b1, Con{b2, Con{c1, Con{c2, t}}}}:
      +x1 = word(U32.read(b1))
      +x2 = word(U32.read(b2))
      +y1 = word(U32.read(c1))
      +y2 = word(U32.read(c2))
      # bend-lint: disable-next bend-lint-rules/performance/append_in_loop -- a one-shot test's output line not-expected-because: a few dozen pieces
      ok_list(t, acc ++ " " ++ both(T.OptCfg.ok(T.CAdam{0.001, x1, x2, 0.00000001, 0.0, y1, y2}), T.OptCfg.ok(T.CAdamW{0.001, x1, x2, 0.00000001, 0.01, y1, y2})))
    case _:
      IO.print(acc)

def dispatch2(isok: Bool, mode: String, rest: List<String>) -> IO(Unit):
  match isok rest:
    case True{} _: ok_list(rest, "ok")
    case False{} Con{opt, Con{k, Con{path, Nil{}}}}: go(mode, opt, Nat.read(k), path)
    case False{} Con{path, Nil{}}: cursor(path)
    case False{} Con{path, Con{n, Nil{}}}: read_file(path, U32.read(n))
    case _ _: IO.die(Unit, 1, "usage: infra_test run|save|resume|params|unchecked opt steps file | cursor file | read file count | ok b1 b2 c1 c2 ...")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+mode, rest}}: dispatch2(String.eq(mode, "ok"), mode, rest)
    case _: IO.die(Unit, 1, "usage: infra_test run|save|resume|params|unchecked opt steps file | cursor file | read file count | ok b1 b2 c1 c2 ...")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
