# Bit probe (temporary): the same q, k, v, dO through the CPU lane's full forward and backward and the device lane's
# dense ones.  b=1, t=4, kv=1, g=1, hd=4, n=4, scale 0.5.  Every matrix holds 16 distinct values (i*5 + start mod 16,
# scaled into [-1, 1)), so a wrong index or a skipped cell cannot pass by symmetry.  Run it with +RTS --gpu off -RTS:
# this probe compares bits (the CPU threads), not the card.
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.2.1/main.bend as TG
import ../torch/main.bend as T
import ../torch/gpu.bend as G

def vals(+left: Nat, +i: Nat, +start: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{((F32.from_nat(Nat.mod(Nat.add(Nat.mul(i, 5n), start), 16n)) / 8.0 : F32) - 1.0 : F32),
        vals(rest, Nat.add(i, 1n), start)}

def mat(+start: Nat) -> TA.Mat<4n, 4n>:
  TA.Mat.from_list(4n, 4n, vals(16n, 0n, start))

def un(m: TA.Mat<4n, 4n>) -> Array<F32>:
  match m:
    case TA.Mat{d}:
      d

def zero16() -> Array<F32>:
  Array.new(F32, 16n, 0.0)

# ---- the forward: y through both lanes ----

def show_cpu(r: T.Fwd<Unit, TA.Mat<4n, 4n>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      T.Mat.print(4n, 4n, "cpu y", y)

def show_gpu(r: T.Fwd<Unit, TA.Mat<4n, 4n>, T.AttnS>) -> IO(Unit):
  match r:
    case T.Fwd{_, y, _}:
      T.Mat.print(4n, 4n, "gpu y", y)

# ---- stage P and G: the CPU full path's own row functions (fprobs, fgrow, fgrows) ----

def cpu_pg3(r: T.GW) -> T.Two<TA.Mat<4n, 4n>, TA.Mat<4n, 4n>>:
  match r:
    case T.GW{d, v, p, g}: T.Two{TA.Mat{p}, TA.Mat{g}}

def cpu_pg2(r: T.QK, vs: Array<F32>, ds: Array<F32>) -> T.Two<TA.Mat<4n, 4n>, TA.Mat<4n, 4n>>:
  match r:
    case T.QK{_, _, p}: cpu_pg3(T.fgrows(3n, 0n, 3n, 4n, 0.5, (4 : U32), (0 : U32), (4 : U32), (0 : U32), T.fgrow(0n, 3n, 4n, 0.5, (4 : U32), (0 : U32), (4 : U32), (0 : U32), ds, vs, p, zero16())))

def cpu_pg(q: TA.Mat<4n, 4n>, k: TA.Mat<4n, 4n>, v: TA.Mat<4n, 4n>, d: TA.Mat<4n, 4n>) -> T.Two<TA.Mat<4n, 4n>, TA.Mat<4n, 4n>>:
  cpu_pg2(T.fprobs(3n, 4n, 0.5, (4 : U32), (0 : U32), (4 : U32), (0 : U32), un(q), un(k)), un(v), un(d))

def show_pg_p(r: T.Two<TA.Mat<4n, 4n>, TA.Mat<4n, 4n>>) -> IO(Unit):
  match r:
    case T.Two{p, g}:
      do IO<Unit>:
        Unit <- T.Mat.print(4n, 4n, "cpu P", p)
        T.Mat.print(4n, 4n, "cpu G", g)

def show_pg_cpu(q: TA.Mat<4n, 4n>, k: TA.Mat<4n, 4n>, v: TA.Mat<4n, 4n>, d: TA.Mat<4n, 4n>) -> IO(Unit):
  show_pg_p(cpu_pg(q, k, v, d))

# ---- stage P and G on the device: kᵀ and vᵀ, then lf_b1_f (P, dP), then lf_gr_f (G) ----

def gpu_b1b(r: TG.R3, q: Array<F32>, d: Array<F32>) -> TG.R2:
  match r:
    case TG.R3{k, v, x3, x4, TG.O3{kt, vt, dq}}: TG.run2_new(~G.EAC, ~G.lf_b1_f, TG.depth(16n), G.eacf(16n, 4n, 4n, 4n, 4n, 1n, 1n, 4n, 0.5), q, kt, vt, d, TA.cap_depth(16n), TA.cap_depth(16n))

def gpu_b1(q: Array<F32>, k: Array<F32>, v: Array<F32>, d: Array<F32>) -> TG.R2:
  gpu_b1b(TG.run3_new(~G.EAC, ~G.lf_tkv, TG.depth(32n), G.eacf(32n, 4n, 4n, 4n, 4n, 1n, 1n, 4n, 0.5), k, v, TG.none(Unit{}), TG.none(Unit{}), TA.cap_depth(16n), TA.cap_depth(16n), TA.cap_depth(16n)), q, d)

def show_pg_g(r: TG.R2) -> IO(Unit):
  match r:
    case TG.R2{q, kt, vt, d, TG.O2{p, g}}:
      do IO<Unit>:
        Unit <- T.Mat.print(4n, 4n, "gpu P", TA.Mat{p})
        T.Mat.print(4n, 4n, "gpu G", TA.Mat{g})

# lf_b1_f's P is the scaled scores (the causal lane's prow softmaxes after); lf_gr_f softmaxes it and writes G
def show_pg_gr(r: TG.R2) -> IO(Unit):
  match r:
    case TG.R2{q, kt, vt, d, TG.O2{p, gb}}: show_pg_g(TG.run2(~G.EAC, ~G.lf_gr_f, TG.depth(4n), G.eacf(4n, 4n, 4n, 4n, 4n, 1n, 1n, 4n, 0.5), TG.none(Unit{}), TG.none(Unit{}), TG.none(Unit{}), TG.none(Unit{}), p, gb))

def show_pg_gpu(q: TA.Mat<4n, 4n>, k: TA.Mat<4n, 4n>, v: TA.Mat<4n, 4n>, d: TA.Mat<4n, 4n>) -> IO(Unit):
  show_pg_gr(gpu_b1(un(q), un(k), un(v), un(d)))

# ---- the backward: dQ, dK and dV through both lanes ----

def show_bwd(r: T.Bwd<Unit, T.Two<TA.Mat<4n, 4n>, T.Two<TA.Mat<4n, 4n>, TA.Mat<4n, 4n>>>>, +tag: String) -> IO(Unit):
  match r:
    case T.Bwd{_, T.Two{dq, T.Two{dk, dv}}, _}:
      do IO<Unit>:
        Unit <- T.Mat.print(4n, 4n, tag ++ " dQ", dq)
        Unit <- T.Mat.print(4n, 4n, tag ++ " dK", dk)
        T.Mat.print(4n, 4n, tag ++ " dV", dv)

def show_bwd_cpu(q: TA.Mat<4n, 4n>, k: TA.Mat<4n, 4n>, v: TA.Mat<4n, 4n>, d: TA.Mat<4n, 4n>) -> IO(Unit):
  show_bwd(T.attn_bwd_full(4n, 4n, 4n, T.AttnDims{1n, 4n, 1n, 1n, 4n, 0.5}, T.AttnS{un(q), un(k), un(v)}, d), "cpu")

def show_bwd_gpu(q: TA.Mat<4n, 4n>, k: TA.Mat<4n, 4n>, v: TA.Mat<4n, 4n>, d: TA.Mat<4n, 4n>) -> IO(Unit):
  show_bwd(G.attn_b_f(4n, 4n, 4n, 1n, 4n, 1n, 1n, 4n, 0.5, T.AttnS{un(q), un(k), un(v)}, d), "gpu")

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- show_cpu(T.attn_fwd_full(4n, 4n, 4n, T.AttnDims{1n, 4n, 1n, 1n, 4n, 0.5},
      T.Two{mat(1n), T.Two{mat(2n), mat(3n)}}))
    Unit <- show_gpu(G.attn_f_f(4n, 4n, 4n, 1n, 4n, 1n, 1n, 4n, 0.5,
      T.Two{mat(1n), T.Two{mat(2n), mat(3n)}}))
    Unit <- show_pg_cpu(mat(1n), mat(2n), mat(3n), mat(4n))
    Unit <- show_pg_gpu(mat(1n), mat(2n), mat(3n), mat(4n))
    Unit <- show_bwd_cpu(mat(1n), mat(2n), mat(3n), mat(4n))
    show_bwd_gpu(mat(1n), mat(2n), mat(3n), mat(4n))
