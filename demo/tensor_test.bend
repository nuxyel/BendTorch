# torch/tensor.bend's operations on shapes read at run time, against reference/test_tensor.py. The directory given as the
# first argument holds case.bin (little-endian U32: the operation, an axis, then each shape as its rank and its sizes)
# and the tensors (<name>.bin, little-endian F32). Every result is printed as the decimal of its 32 bits; a shape
# condition that does not hold prints "reject", as the program would refuse such shapes where they enter.
#
#   bend demo/tensor_test.bend -o .scratch/tensor_test && .scratch/tensor_test <dir>

import Base
import ../torch/main.bend as T
import ../torch/tensor.bend as TT

# ---- reading ----

def u32s(bs: List<&2, U32>) -> List<&2, U32>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      (b0 + ((b1 * 256 : U32) + ((b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32) : U32) : U32) <> u32s(t)
    case _:
      Nil{}

def nats(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(h) <> nats(t)

# a shape (its rank, then its sizes) at the front of ws, and what follows
type Sh is Data:
  Sh{s: List<&2, Nat>, rest: List<&2, U32>}

def shape(ws: List<&2, U32>) -> Sh:
  match ws:
    case Con{+r, +t}:
      Sh{nats(List.take(&2, U32, t, U32.to_nat(r))), List.drop(&2, U32, t, U32.to_nat(r))}
    case Nil{}:
      Sh{Nil{}, Nil{}}

def fill(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}:
      a
    case Con{+h, t}:
      fill(t, (i + 1 : U32), Array.set(F32, a, i, h))

def load(+s: List<&2, Nat>, +dir: String, +name: String) -> IO(TT.Tensor<s>):
  do IO<TT.Tensor<s>>:
    bs : List<&2, U32> <- T.read_at(dir ++ "/" ++ name ++ ".bin", 0, (U32.from_nat(TT.numel(s)) * 4 : U32))
    return TT.Tensor{fill(T.f32s(bs), 0, TT.new(TT.numel(s)))}

# ---- printing ----

def words(k: Nat, +i: U32, ra: Array<F32> & F32, acc: String) -> String:
  match k ra:
    case 0n Tuple{a, v}:
      acc
    case 1n+p Tuple{a, +v}:
      # bend-lint: disable-next bend-lint-rules/performance/append_in_loop -- a one-shot test's output line not-expected-because: a few dozen pieces
      words(p, (i + 1 : U32), Array.get(F32, a, (i + 1 : U32)), acc ++ " " ++ U32.show(T.bits(v)))

def show(+s: List<&2, Nat>, +name: String, t: TT.Tensor<s>) -> IO(Unit):
  match t:
    case TT.Tensor{d}:
      IO.print(name ++ words(TT.numel(s), 0, Array.get(F32, d, 0), ""))

# ---- results: print one tensor of a pair, or of a triple ----

def pr_tp(+s: List<&2, Nat>, +u: List<&2, Nat>, +name: String, r: TT.TP<s, u>) -> IO(Unit):
  match r:
    case TT.TP{x, y}:
      show(u, name, y)

def pr3(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, r: TT.T3<a, b, o>) -> IO(Unit):
  match r:
    case TT.T3{x, y, z}:
      show(o, "z", z)

def pr_g2(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, +g: List<&2, Nat>, +name: String, r: TT.G2<a, b, o, g>) -> IO(Unit):
  match r:
    case TT.G2{dz, x, y, w}:
      show(g, name, w)

def reject(u: Unit) -> IO(Unit):
  IO.print("reject")

# ---- binary operations with broadcasting: z, then both gradients from dz ----

def run_bin(~f: @+a: F32 -> @+b: F32 -> F32, ~da: @+d: F32 -> @+p: F32 -> @+q: F32 -> F32, ~db: @+d: F32 -> @+p: F32 -> @+q: F32 -> F32, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == TT.bc_ok(a, b, o) : Bool}, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TT.Tensor<a> <- load(a, dir, "x")
    y : TT.Tensor<b> <- load(b, dir, "y")
    pr3(a, b, o, TT.map2(~f, a, b, o, pf, x, y))
    dz : TT.Tensor<o> <- load(o, dir, "dz")
    x2 : TT.Tensor<a> <- load(a, dir, "x")
    y2 : TT.Tensor<b> <- load(b, dir, "y")
    pr_g2(a, b, o, a, "ga", TT.grad2(~da, a, b, o, a, dz, x2, y2, TT.zeros(a)))
    dz3 : TT.Tensor<o> <- load(o, dir, "dz")
    x3 : TT.Tensor<a> <- load(a, dir, "x")
    y3 : TT.Tensor<b> <- load(b, dir, "y")
    pr_g2(a, b, o, b, "gb", TT.grad2(~db, a, b, o, b, dz3, x3, y3, TT.zeros(b)))

def bin_op(op: Nat, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, -pf: {True{} == TT.bc_ok(a, b, o) : Bool}, +dir: String) -> IO(Unit):
  match op:
    case 0n:
      run_bin(~TT.f32_add, ~TT.d_id, ~TT.d_id, a, b, o, pf, dir)
    case 1n:
      run_bin(~TT.f32_sub, ~TT.d_id, ~TT.d_neg, a, b, o, pf, dir)
    case 2n:
      run_bin(~TT.f32_mul, ~TT.d_mul_a, ~TT.d_mul_b, a, b, o, pf, dir)
    case _:
      run_bin(~TT.f32_div, ~TT.d_div_a, ~TT.d_div_b, a, b, o, pf, dir)

def bin_chk(op: Nat, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, +dir: String, m: Maybe<&1, {True{} == TT.bc_ok(a, b, o) : Bool}>) -> IO(Unit):
  match m:
    case Some{pf}:
      bin_op(op, a, b, o, pf, dir)
    case None{}:
      reject(Unit{})

# ---- reductions (op 4 sum over axis k, 5 the same keeping the axis, 6 mean over k, 7 sum of all, 8 mean of all) ----

def run_red(op: Nat, +s: List<&2, Nat>, +k: Nat, +dir: String) -> IO(Unit):
  match op:
    case 4n:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, TT.remove(s, k), "z", TT.sum_axis(s, k, x))
        dy : TT.Tensor<TT.remove(s, k)> <- load(TT.remove(s, k), dir, "dz")
        pr_tp(TT.remove(s, k), s, "ga", TT.sum_axis_bwd(s, k, dy))
    case 5n:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, TT.setd(s, k, 1n), "z", TT.sum_axis_keep(s, k, x))
    case 6n:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, TT.remove(s, k), "z", TT.mean_axis(s, k, x))
        dy : TT.Tensor<TT.remove(s, k)> <- load(TT.remove(s, k), dir, "dz")
        pr_tp(TT.remove(s, k), s, "ga", TT.mean_axis_bwd(s, k, dy))
    case 7n:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, [], "z", TT.sum_all(s, x))
        dy : TT.Tensor<[]> <- load([], dir, "dz")
        pr_tp([], s, "ga", TT.sum_all_bwd(s, dy))
    case _:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, [], "z", TT.mean_all(s, x))
        dy : TT.Tensor<[]> <- load([], dir, "dz")
        pr_tp([], s, "ga", TT.mean_all_bwd(s, dy))

# ---- reshape (op 9, to shape t), flatten (op 10) ----

def run_reshape(+s: List<&2, Nat>, +t: List<&2, Nat>, +dir: String, m: Maybe<&1, {TT.numel(s) == TT.numel(t) : Nat}>) -> IO(Unit):
  match m:
    case Some{e}:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        show(t, "z", TT.reshape(s, t, e, x))
        dz : TT.Tensor<t> <- load(t, dir, "dz")
        show(s, "ga", TT.reshape(t, s, Equal.sym(Nat, TT.numel(s), TT.numel(t), e), dz))
    case None{}:
      reject(Unit{})

def run_flatten(+s: List<&2, Nat>, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TT.Tensor<s> <- load(s, dir, "x")
    show([TT.numel(s)], "z", TT.flatten(s, x))

# ---- permute (op 11, by p) and transpose (op 12, axes k and j) ----

def run_perm(+s: List<&2, Nat>, +p: List<&2, Nat>, +dir: String, m: Maybe<&1, {True{} == TT.is_perm(p, TT.rank(s)) : Bool}>) -> IO(Unit):
  match m:
    case Some{pf}:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, TT.permuted(s, p), "z", TT.permute(s, p, pf, x))
        dy : TT.Tensor<TT.permuted(s, p)> <- load(TT.permuted(s, p), dir, "dz")
        pr_tp(TT.permuted(s, p), s, "ga", TT.permute_bwd(s, p, dy))
    case None{}:
      reject(Unit{})

# ---- narrow (op 13: axis k, start and length from the second shape) ----

def run_narrow(+s: List<&2, Nat>, +d: Nat, +st: Nat, +l: Nat, +dir: String, m: Maybe<&1, {True{} == TT.narrow_ok(s, d, st, l) : Bool}>) -> IO(Unit):
  match m:
    case Some{pf}:
      do IO<Unit>:
        x : TT.Tensor<s> <- load(s, dir, "x")
        pr_tp(s, TT.setd(s, d, l), "z", TT.narrow(s, d, st, l, pf, x))
        dy : TT.Tensor<TT.setd(s, d, l)> <- load(TT.setd(s, d, l), dir, "dz")
        pr_tp(TT.setd(s, d, l), s, "ga", TT.narrow_bwd(s, d, st, l, dy))
    case None{}:
      reject(Unit{})

def narrow2(+s: List<&2, Nat>, +d: Nat, b: List<&2, Nat>, +dir: String) -> IO(Unit):
  match b:
    case Con{+st, Con{+l, r}}:
      run_narrow(s, d, st, l, dir, TT.Bool.check(TT.narrow_ok(s, d, st, l)))
    case _:
      reject(Unit{})

# ---- cat (op 14, a then b along axis k) ----

def pr_catg(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, r: TT.CatG<a, b, o>) -> IO(Unit):
  match r:
    case TT.CatG{dz, ga, gb}:
      do IO<Unit>:
        show(a, "ga", ga)
        show(b, "gb", gb)

def run_cat(+a: List<&2, Nat>, +b: List<&2, Nat>, +d: Nat, +dir: String, m: Maybe<&1, {True{} == TT.cat_ok(a, b, d) : Bool}>) -> IO(Unit):
  match m:
    case Some{pf}:
      do IO<Unit>:
        x : TT.Tensor<a> <- load(a, dir, "x")
        y : TT.Tensor<b> <- load(b, dir, "y")
        pr3(a, b, TT.cat_shape(a, b, d), TT.cat(a, b, d, pf, x, y))
        dz : TT.Tensor<TT.cat_shape(a, b, d)> <- load(TT.cat_shape(a, b, d), dir, "dz")
        pr_catg(a, b, TT.cat_shape(a, b, d), TT.cat_bwd(a, b, d, dz))
    case None{}:
      reject(Unit{})

# ---- batched product (op 15, [bs, n, k] · [bs, k, m]) ----

def pr_bmm_g(+a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, +name: String, r: TT.T3<a, b, o>) -> IO(Unit):
  match r:
    case TT.T3{x, y, z}:
      show(o, name, z)

def run_bmm(+bs: Nat, +n: Nat, +k: Nat, +m: Nat, +dir: String) -> IO(Unit):
  do IO<Unit>:
    x : TT.Tensor<[bs, n, k]> <- load([bs, n, k], dir, "x")
    y : TT.Tensor<[bs, k, m]> <- load([bs, k, m], dir, "y")
    pr3([bs, n, k], [bs, k, m], [bs, n, m], TT.bmm(bs, n, k, m, x, y))
    dz : TT.Tensor<[bs, n, m]> <- load([bs, n, m], dir, "dz")
    y2 : TT.Tensor<[bs, k, m]> <- load([bs, k, m], dir, "y")
    pr_bmm_g([bs, n, m], [bs, k, m], [bs, n, k], "ga", TT.bmm_dx(bs, n, k, m, dz, y2))
    x3 : TT.Tensor<[bs, n, k]> <- load([bs, n, k], dir, "x")
    dz3 : TT.Tensor<[bs, n, m]> <- load([bs, n, m], dir, "dz")
    pr_bmm_g([bs, n, k], [bs, n, m], [bs, k, m], "gb", TT.bmm_dy(bs, n, k, m, x3, dz3))

def bmm_ok(ok: Bool, +bs: Nat, +n: Nat, +k: Nat, +m: Nat, +dir: String) -> IO(Unit):
  match ok:
    case True{}:
      run_bmm(bs, n, k, m, dir)
    case False{}:
      reject(Unit{})

def bmm2(a: List<&2, Nat>, b: List<&2, Nat>, +dir: String) -> IO(Unit):
  match a b:
    case Con{+bs, Con{+n, Con{+k, Nil{}}}} Con{+bs2, Con{+k2, Con{+m, Nil{}}}}:
      bmm_ok(Bool.and(Nat.is_eq(bs, bs2), Nat.is_eq(k, k2)), bs, n, k, m, dir)
    case _ _:
      reject(Unit{})

def head0(xs: List<&2, Nat>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{h, t}:
      h

# ---- dispatch ----

def run3(op: Nat, +k: Nat, +a: List<&2, Nat>, +b: List<&2, Nat>, +o: List<&2, Nat>, +dir: String) -> IO(Unit):
  match op:
    case 0n:
      bin_chk(0n, a, b, o, dir, TT.Bool.check(TT.bc_ok(a, b, o)))
    case 1n:
      bin_chk(1n, a, b, o, dir, TT.Bool.check(TT.bc_ok(a, b, o)))
    case 2n:
      bin_chk(2n, a, b, o, dir, TT.Bool.check(TT.bc_ok(a, b, o)))
    case 3n:
      bin_chk(3n, a, b, o, dir, TT.Bool.check(TT.bc_ok(a, b, o)))
    case 9n:
      run_reshape(a, b, dir, TT.Nat.check_eq(TT.numel(a), TT.numel(b)))
    case 10n:
      run_flatten(a, dir)
    case 11n:
      run_perm(a, b, dir, TT.Bool.check(TT.is_perm(b, TT.rank(a))))
    case 12n:
      run_perm(a, TT.transpose_p(a, k, head0(b)), dir, TT.Bool.check(TT.is_perm(TT.transpose_p(a, k, head0(b)), TT.rank(a))))
    case 13n:
      narrow2(a, k, b, dir)
    case 14n:
      run_cat(a, b, k, dir, TT.Bool.check(TT.cat_ok(a, b, k)))
    case 15n:
      bmm2(a, b, dir)
    case 4n:
      run_red(4n, a, k, dir)
    case 5n:
      run_red(5n, a, k, dir)
    case 6n:
      run_red(6n, a, k, dir)
    case 7n:
      run_red(7n, a, k, dir)
    case _:
      run_red(8n, a, k, dir)

def go3(op: Nat, +k: Nat, +a: List<&2, Nat>, +b: List<&2, Nat>, so: Sh, +dir: String) -> IO(Unit):
  match so:
    case Sh{o, r}:
      run3(op, k, a, b, o, dir)

def go2(op: Nat, +k: Nat, +a: List<&2, Nat>, sb: Sh, +dir: String) -> IO(Unit):
  match sb:
    case Sh{b, r}:
      go3(op, k, a, b, shape(r), dir)

def go1(op: Nat, +k: Nat, sa: Sh, +dir: String) -> IO(Unit):
  match sa:
    case Sh{a, r}:
      go2(op, k, a, shape(r), dir)

def start(ws: List<&2, U32>, +dir: String) -> IO(Unit):
  match ws:
    case Con{+op, Con{+k, r}}:
      go1(U32.to_nat(op), U32.to_nat(k), shape(r), dir)
    case _:
      IO.print("bad case.bin")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+dir, Nil{}}}:
      do IO<Unit>:
        bs : List<&2, U32> <- T.read_at(dir ++ "/case.bin", 0, 4096)
        start(u32s(bs), dir)
    case _:
      IO.print("usage: tensor_test <dir>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
