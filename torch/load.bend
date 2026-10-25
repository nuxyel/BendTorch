# Temporary loading-only exception: bend-ml-binary-read is foreign C/JavaScript. No file with laws imports this module.
# Replace it with native Bend when the language supports dense reads (Renan, 2026-10-17).
# The reader is bend-ml-binary-read@0.2.0.0, with 64-bit offsets, BF16 files and reads into a cell offset
import Base
import bend-ml-binary-read@0.2.0.0/main.bend as R
import ./main.bend as T

def require(ok: Bool, +message: String) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.die(Unit, 1, message)

# =====================================================================================================
# W64: a 64-bit byte count or offset as two U32s, hi * 2^32 + lo. Bend (2.0.39) has no 64-bit integer, and a file past
# 4 GiB needs one (SmolLM2-1.7B's F32 weights, a 360M AdamW checkpoint). Plain Bend: none of these reaches the reader.
# =====================================================================================================

type W64 is Data:
  W64{hi: U32, lo: U32}

def W64.of(x: U32) -> W64:
  W64{0, x}

def W64.of_pair(p: U32 & U32) -> W64:
  match p:
    case Tuple{hi, lo}: W64{hi, lo}

def W64.bit(b: Bool) -> U32:
  match b:
    case True{}: 1
    case False{}: 0

def W64.eq(a: W64, b: W64) -> Bool:
  match a b:
    case W64{ah, al} W64{bh, bl}: Bool.and(U32.is_eq(ah, bh), U32.is_eq(al, bl))

def W64.add_go(+ah: U32, +bh: U32, +s: U32, +al: U32) -> W64:
  W64{((ah + bh : U32) + W64.bit(U32.is_lt(s, al)) : U32), s}

# a + b, wrapping at 2^64: the low words' sum carries one when it wrapped (it is then below either term)
def W64.add(a: W64, b: W64) -> W64:
  match a b:
    case W64{ah, +al} W64{bh, bl}: W64.add_go(ah, bh, (al + bl : U32), al)

def W64.mul_go(+h: U32, +k: U32, +p0: U32, +p1: U32) -> W64:
  W64.add(W64{((h * k : U32) + U32.shrn(p1, 16n) : U32), p0}, W64{0, U32.shln(p1, 16n)})

# a * k for k < 2^16 (enough for bytes a number and decimal digits), wrapping at 2^64: lo = l1 2^16 + l0, so
# lo * k = (l1 k) 2^16 + l0 k, where both products fit a U32
def W64.mul_small(a: W64, +k: U32) -> W64:
  match a:
    case W64{h, +l}: W64.mul_go(h, k, (U32.and(l, 65535) * k : U32), (U32.shrn(l, 16n) * k : U32))

def W64.div10_go(+qh: U32, +r1: U32, +q1: U32, +l: U32) -> W64 & U32:
  +t = (U32.shln(r1, 16n) + U32.and(l, 65535) : U32)
  Tuple{W64{qh, U32.or(U32.shln(q1, 16n), U32.div(t, 10))}, U32.mod(t, 10)}

# (a / 10, a % 10), by long division in 16-bit steps so no partial remainder overflows a U32
def W64.div10(a: W64) -> W64 & U32:
  match a:
    case W64{+h, +l}:
      +t = (U32.shln(U32.mod(h, 10), 16n) + U32.shrn(l, 16n) : U32)
      W64.div10_go(U32.div(h, 10), U32.mod(t, 10), U32.div(t, 10), l)

def W64.is_zero(a: W64) -> Bool:
  match a:
    case W64{h, l}: Bool.and(U32.is_zero(h), U32.is_zero(l))

def W64.show_go(f: Nat, z: Bool, qr: W64 & U32, acc: String) -> String:
  match f z qr:
    case _ True{} Tuple{q, r}: acc
    case 0n False{} Tuple{q, r}: acc
    case 1n+g False{} Tuple{+q, r}: W64.show_go(g, W64.is_zero(q), W64.div10(q), U32.show(r) ++ acc)

def W64.show_if(z: Bool, +a: W64) -> String:
  match z:
    case True{}: "0"
    case False{}: W64.show_go(20n, False{}, W64.div10(a), "")

# the decimal digits, as U32.show writes them
def W64.show(+a: W64) -> String:
  W64.show_if(W64.is_zero(a), a)

def W64.digits(s: String, f: Nat) -> Bool:
  match s f:
    case SNil{} _: True{}
    case SCon{c, t} 0n: False{}
    case SCon{c, t} 1n+g: Bool.and(Char.is_digit(c), W64.digits(t, g))

def W64.read_go(s: String, +acc: W64) -> W64:
  match s:
    case SNil{}: acc
    case SCon{Chr{x}, t}: W64.read_go(t, W64.add(W64.mul_small(acc, 10), W64.of(U32.sub(x, 48))))

def W64.read_if(ok: Bool, s: String) -> Maybe<&2, W64>:
  match ok:
    case True{}: Some{W64.read_go(s, W64{0, 0})}
    case False{}: None{}

# A decimal number of 1 to 19 digits (every such number fits 64 bits); anything else is None
def W64.read(s: String) -> Maybe<&2, W64>:
  match s:
    case SNil{}: None{}
    case SCon{+h, +t}: W64.read_if(W64.digits(SCon{h, t}, 19n), SCon{h, t})

# =====================================================================================================
# Reading
# =====================================================================================================

def read_done(+path: String, +n: U32, r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>)) -> IO(Array<F32>):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Array<F32>>:
        +got : U32 <- IO.pass(U32, res)
        File.close(f)
        require(U32.is_eq(got, n), path ++ ": short read (" ++ U32.show(got) ++ " of " ++ U32.show(n) ++ " numbers)")
        return a

def read_go(+path: String, +n: U32, f: File, +size: W64) -> IO(Array<F32>):
  do IO<Array<F32>>:
    require(W64.eq(size, W64.mul_small(W64.of(n), 4)), path ++ ": expected " ++ W64.show(W64.mul_small(W64.of(n), 4)) ++ " bytes, got " ++ W64.show(size))
    rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32_at(f, 0, 0, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
    read_done(path, n, rr)

def read_sized(+path: String, +n: U32, r: File & Result<&1, &1, U32 & String, U32 & U32>) -> IO(Array<F32>):
  match r:
    case Tuple{f, res}:
      do IO<Array<F32>>:
        sz : U32 & U32 <- IO.pass(U32 & U32, res)
        read_go(path, n, f, W64.of_pair(sz))

# A raw, exact-size little-endian F32 file, read once into one source array: up to 2^32 - 1 numbers (16 GiB).
def read(+path: String, +n: U32) -> IO(Array<F32>):
  do IO<Array<F32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32 & U32> <- R.size64(f)
    read_sized(path, n, rr)

# n little-endian F32 numbers at byte `off` of a file of any size (a tensor inside a larger file)
def read_f32_at(+path: String, off: W64, +n: U32) -> IO(Array<F32>):
  match off:
    case W64{hi, lo}:
      do IO<Array<F32>>:
        f : File <- IO.try(File, File.open(path, "r"))
        rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32_at(f, hi, lo, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
        read_done(path, n, rr)

# n little-endian BF16 numbers at byte `off` of a file of any size, each widened exactly to F32 (bf16.to(torch.float32))
def read_bf16_at(+path: String, off: W64, +n: U32) -> IO(Array<F32>):
  match off:
    case W64{hi, lo}:
      do IO<Array<F32>>:
        f : File <- IO.try(File, File.open(path, "r"))
        rr : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_bf16_at(f, hi, lo, n, Array.new(F32, T.depth(U32.to_nat(n)), 0.0))
        read_done(path, n, rr)

def size_done(r: File & Result<&1, &1, U32 & String, U32 & U32>) -> IO(W64):
  match r:
    case Tuple{f, res}:
      do IO<W64>:
        sz : U32 & U32 <- IO.pass(U32 & U32, res)
        File.close(f)
        return W64.of_pair(sz)

# A file's size in bytes, at any size
def size(+path: String) -> IO(W64):
  do IO<W64>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32 & U32> <- R.size64(f)
    size_done(rr)

def weights_done(-P: Type, +n: U32, r: T.PA<P>) -> IO(P):
  match r:
    case T.PA{p, xs, +off}:
      do IO<P>:
        require(U32.is_eq(off, n), "Layer.count differs from the loading cursor")
        return p

# All parameters in Layer.dump order. The source array is released after the model's arrays are filled.
def weights(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, p: P) -> IO(P):
  do IO<P>:
    +n : U32 = T.Layer.count(P, X, Y, S, m)
    a : Array<F32> <- read(path, n)
    weights_done(P, n, T.Layer.load(P, X, Y, S, m, p, a, 0))

# The checkpoint layout is determined by the already initialized optimizer state, without a header:
# SGD: parameters; momentum: parameters, buffer; Adam/AdamW: parameters, m, v, 1 - b1^t, 1 - b2^t; AdamW with packed
# BF16 moments: parameters, the moments' tree (main.bend's pt_store), 1 - b1^t, 1 - b2^t, the whole file read.
def train_read(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, +base: U32, +copies: U32, +tail: U32, tr: T.Train<P>) -> IO(T.Train<P>):
  do IO<T.Train<P>>:
    require(U32.is_le(base, U32.div(U32.sub(4294967295, tail), copies)), "checkpoint exceeds 2^32 - 1 numbers")
    +n : U32 = ((copies * base : U32) + tail : U32)
    a : Array<F32> <- read(path, n)
    return T.Train.restore(~P, ~X, ~Y, ~S, ~m, tr, a)

def cells_done(r: File & Result<&1, &1, U32 & String, U32>) -> IO(U32):
  match r:
    case Tuple{f, res}:
      do IO<U32>:
        +size : U32 <- IO.pass(U32, res)
        File.close(f)
        return U32.div(size, 4)

# the number of F32 cells in a file (a packed-moment checkpoint carries its own sizes: its length is the file's)
def cells_of(+path: String) -> IO(U32):
  do IO<U32>:
    f : File <- IO.try(File, File.open(path, "r"))
    rr : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    cells_done(rr)

def train(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, tr: T.Train<P>) -> IO(T.Train<P>):
  match tr:
    case T.Train{p, g, T.NoMom{}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 1, 0, T.Train{p, g, T.NoMom{}, p1, p2, loss})
    case T.Train{p, g, T.OneMom{mm}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 2, 0, T.Train{p, g, T.OneMom{mm}, p1, p2, loss})
    case T.Train{p, g, T.TwoMom{mm, vv}, p1, p2, loss}:
      train_read(~P, ~X, ~Y, ~S, ~m, path, T.Layer.count(P, X, Y, S, m), 3, 2, T.Train{p, g, T.TwoMom{mm, vv}, p1, p2, loss})
    case T.Train{p, g, T.Packed{t}, p1, p2, loss}:
      do IO<T.Train<P>>:
        +n : U32 <- cells_of(path)
        train_read(~P, ~X, ~Y, ~S, ~m, path, n, 1, 0, T.Train{p, g, T.Packed{t}, p1, p2, loss})

# =====================================================================================================
# Safetensors (Hugging Face): 8 bytes holding the header's length (little-endian), a JSON header naming each tensor's
# dtype, shape and byte range (data_offsets, counted from the end of the header), then the data. The header is read as
# bytes and scanned in plain Bend: a lexer into tokens, then one pass that keeps what a tensor's entry holds. It reads
# what the format writes; it is not a JSON validator: an escaped character is kept as itself (\uXXXX is not decoded),
# and only strings, whole numbers, objects and lists are understood (anything else marks the header malformed).
# =====================================================================================================

# one tensor the model expects, in Layer.dump order: its name in the file and its shape (rows, columns, ...)
type Entry is Data:
  Entry{name: String, shape: List<&2, U32>}

def St.entry_name(e: Entry) -> String:
  match e:
    case Entry{n, sh}: n

type Tok is Data:
  TStr{s: String}
  TNum{v: W64}
  TPun{c: U32}
  TBad{c: U32}

# the lexer's state between bytes: outside a token, inside a string (its characters reversed), just after a backslash
# in a string, inside a number
type Lex is Data:
  LOut{}
  LStr{acc: String}
  LEsc{acc: String}
  LNum{v: W64}

# a state and the tokens one byte completed, in order
type LexR is Data:
  LexR{st: Lex, toks: List<&2, Tok>}

def St.digit(+b: U32) -> Bool:
  U32.is_lt(U32.sub(b, 48), 10)

def St.space(+b: U32) -> Bool:
  Bool.or(Bool.or(U32.is_eq(b, 32), U32.is_eq(b, 10)), Bool.or(U32.is_eq(b, 13), U32.is_eq(b, 9)))

# { } [ ] : ,
def St.pun(+b: U32) -> Bool:
  Bool.or(Bool.or(Bool.or(U32.is_eq(b, 123), U32.is_eq(b, 125)), Bool.or(U32.is_eq(b, 91), U32.is_eq(b, 93))), Bool.or(U32.is_eq(b, 58), U32.is_eq(b, 44)))

def St.out_go(quote: Bool, digit: Bool, space: Bool, pun: Bool, +b: U32) -> LexR:
  match quote digit space pun:
    case True{} _ _ _: LexR{LStr{""}, Nil{}}
    case _ True{} _ _: LexR{LNum{W64.of(U32.sub(b, 48))}, Nil{}}
    case _ _ True{} _: LexR{LOut{}, Nil{}}
    case _ _ _ True{}: LexR{LOut{}, Con{TPun{b}, Nil{}}}
    case _ _ _ _: LexR{LOut{}, Con{TBad{b}, Nil{}}}

def St.out(+b: U32) -> LexR:
  St.out_go(U32.is_eq(b, 34), St.digit(b), St.space(b), St.pun(b), b)

def St.str_go(quote: Bool, slash: Bool, +b: U32, acc: String) -> LexR:
  match quote slash:
    case True{} _: LexR{LOut{}, Con{TStr{String.reverse(acc)}, Nil{}}}
    case _ True{}: LexR{LEsc{acc}, Nil{}}
    case _ _: LexR{LStr{SCon{Chr{b}, acc}}, Nil{}}

# a number ends at the first byte that is not a digit, and that byte is then read as outside a token
def St.num_end(+v: W64, r: LexR) -> LexR:
  match r:
    case LexR{st, ts}: LexR{st, Con{TNum{v}, ts}}

def St.num_go(digit: Bool, +b: U32, +v: W64) -> LexR:
  match digit:
    case True{}: LexR{LNum{W64.add(W64.mul_small(v, 10), W64.of(U32.sub(b, 48)))}, Nil{}}
    case False{}: St.num_end(v, St.out(b))

def St.step(st: Lex, +b: U32) -> LexR:
  match st:
    case LOut{}: St.out(b)
    case LStr{acc}: St.str_go(U32.is_eq(b, 34), U32.is_eq(b, 92), b, acc)
    case LEsc{acc}: LexR{LStr{SCon{Chr{b}, acc}}, Nil{}}
    case LNum{+v}: St.num_go(St.digit(b), b, v)

# ts onto out, which holds the tokens so far in reverse
def St.push(ts: List<&2, Tok>, out: List<&2, Tok>) -> List<&2, Tok>:
  match ts:
    case Nil{}: out
    case Con{t, r}: St.push(r, Con{t, out})

# the end of the header: a number ends there, an unfinished string is malformed
def St.fin(st: Lex, out: List<&2, Tok>) -> List<&2, Tok>:
  match st:
    case LOut{}: out
    case LNum{v}: Con{TNum{v}, out}
    case LStr{acc}: Con{TBad{0}, out}
    case LEsc{acc}: Con{TBad{0}, out}

# the header's tokens, in order; r is what the previous byte left
def St.lex(bs: List<&2, U32>, r: LexR, out: List<&2, Tok>) -> List<&2, Tok>:
  match bs r:
    case Nil{} LexR{st, ts}: List.reverse(&2, Tok, St.fin(st, St.push(ts, out)))
    case Con{+b, t} LexR{st, ts}: St.lex(t, St.step(st, b), St.push(ts, out))

# one top-level entry of the header: a tensor (or __metadata__, which no layout names)
type Rec is Data:
  Rec{name: String, dtype: String, shape: List<&2, W64>, offs: List<&2, W64>}

# the pass over the tokens: the depth ({ and [ open a level), whether the last token was a colon, the last string,
# the current entry's name and field, what the entry holds so far, the numbers of the open list, the entries read
# (last first), and the count of tokens the header may not hold
type Ps is Data:
  Ps{depth: U32, colon: Bool, pend: String, name: String, field: String, dtype: String, shape: List<&2, W64>, offs: List<&2, W64>, nums: List<&2, W64>, recs: List<&2, Rec>, bad: U32}

# a string is a value after a colon (the dtype, at depth 2) or else a key, kept until its colon
def Sp.str(colon: Bool, is_dt: Bool, ps: Ps, s: String) -> Ps:
  match colon is_dt ps:
    case True{} True{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, p, n, f, s, sh, of, ns, rs, bad}
    case True{} False{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, p, n, f, dt, sh, of, ns, rs, bad}
    case False{} _ Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, s, n, f, dt, sh, of, ns, rs, bad}

# a colon makes the last string the entry's name (depth 1) or the field's (depth 2)
def Sp.colon(d1: Bool, d2: Bool, ps: Ps) -> Ps:
  match d1 d2 ps:
    case True{} _ Ps{d, c, +p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, True{}, p, p, f, dt, sh, of, ns, rs, bad}
    case _ True{} Ps{d, c, +p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, True{}, p, n, p, dt, sh, of, ns, rs, bad}
    case _ _ Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, True{}, p, n, f, dt, sh, of, ns, rs, bad}

# { opens a level; at depth 2 an entry starts empty
def Sp.open(fresh: Bool, ps: Ps) -> Ps:
  match fresh ps:
    case True{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{(d + 1 : U32), False{}, p, n, f, "", Nil{}, Nil{}, ns, rs, bad}
    case False{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{(d + 1 : U32), False{}, p, n, f, dt, sh, of, ns, rs, bad}

# } closes a level; closing depth 2 ends an entry
def Sp.close(emit: Bool, ps: Ps) -> Ps:
  match emit ps:
    case True{} Ps{d, c, p, +n, f, dt, sh, of, ns, rs, bad}: Ps{U32.sub(d, 1), False{}, p, n, f, "", Nil{}, Nil{}, ns, Con{Rec{n, dt, sh, of}, rs}, bad}
    case False{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{U32.sub(d, 1), False{}, p, n, f, dt, sh, of, ns, rs, bad}

def Sp.lopen(ps: Ps) -> Ps:
  match ps:
    case Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{(d + 1 : U32), False{}, p, n, f, dt, sh, of, Nil{}, rs, bad}

# ] closes a list; at depth 3 it is the entry's shape or data_offsets
def Sp.lclose(is_sh: Bool, is_of: Bool, ps: Ps) -> Ps:
  match is_sh is_of ps:
    case True{} _ Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{U32.sub(d, 1), False{}, p, n, f, dt, List.reverse(&2, W64, ns), of, Nil{}, rs, bad}
    case _ True{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{U32.sub(d, 1), False{}, p, n, f, dt, sh, List.reverse(&2, W64, ns), Nil{}, rs, bad}
    case _ _ Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{U32.sub(d, 1), False{}, p, n, f, dt, sh, of, Nil{}, rs, bad}

def Sp.comma(ps: Ps) -> Ps:
  match ps:
    case Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, p, n, f, dt, sh, of, ns, rs, bad}

# a number counts inside a list at depth 3; elsewhere it is not part of a tensor's entry
def Sp.num(in_list: Bool, v: W64, ps: Ps) -> Ps:
  match in_list ps:
    case True{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, p, n, f, dt, sh, of, Con{v, ns}, rs, bad}
    case False{} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}: Ps{d, False{}, p, n, f, dt, sh, of, ns, rs, bad}

def Sp.pun(colon: Bool, ob: Bool, cb: Bool, ol: Bool, cl: Bool, +d: U32, +f: String, ps: Ps) -> Ps:
  match colon ob cb ol cl:
    case True{} _ _ _ _: Sp.colon(U32.is_eq(d, 1), U32.is_eq(d, 2), ps)
    case _ True{} _ _ _: Sp.open(U32.is_eq(d, 1), ps)
    case _ _ True{} _ _: Sp.close(U32.is_eq(d, 2), ps)
    case _ _ _ True{} _: Sp.lopen(ps)
    case _ _ _ _ True{}: Sp.lclose(Bool.and(U32.is_eq(d, 3), String.eq(f, "shape")), Bool.and(U32.is_eq(d, 3), String.eq(f, "data_offsets")), ps)
    case _ _ _ _ _: Sp.comma(ps)

def Sp.step(t: Tok, ps: Ps) -> Ps:
  match t ps:
    case TStr{s} Ps{+d, +c, p, n, +f, dt, sh, of, ns, rs, bad}:
      Sp.str(c, Bool.and(U32.is_eq(d, 2), String.eq(f, "dtype")), Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}, s)
    case TNum{v} Ps{+d, c, p, n, f, dt, sh, of, ns, rs, bad}:
      Sp.num(U32.is_eq(d, 3), v, Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad})
    case TPun{+k} Ps{+d, c, p, n, +f, dt, sh, of, ns, rs, bad}:
      Sp.pun(U32.is_eq(k, 58), U32.is_eq(k, 123), U32.is_eq(k, 125), U32.is_eq(k, 91), U32.is_eq(k, 93), d, f, Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad})
    case TBad{k} Ps{d, c, p, n, f, dt, sh, of, ns, rs, bad}:
      Ps{d, c, p, n, f, dt, sh, of, ns, rs, (bad + 1 : U32)}

def Sp.parse(ts: List<&2, Tok>, ps: Ps) -> Ps:
  match ts:
    case Nil{}: ps
    case Con{t, r}: Sp.parse(r, Sp.step(t, ps))

def St.rec_name(r: Rec) -> String:
  match r:
    case Rec{n, dt, sh, of}: n

def St.find_put(r: Rec, rest: Maybe<&2, Rec>, hit: Bool) -> Maybe<&2, Rec>:
  match hit:
    case True{}: Some{r}
    case False{}: rest

# the entry named `name`
def St.find(rs: List<&2, Rec>, +name: String) -> Maybe<&2, Rec>:
  match rs:
    case Nil{}: None{}
    case Con{+r, t}: St.find_put(r, St.find(t, name), String.eq(St.rec_name(r), name))

def St.prod(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}: 1
    case Con{x, t}: (x * St.prod(t) : U32)

def St.total(es: List<&2, Entry>) -> U32:
  match es:
    case Nil{}: 0
    case Con{Entry{n, sh}, t}: (St.prod(sh) + St.total(t) : U32)

def St.shape_eq(a: List<&2, W64>, b: List<&2, U32>) -> Bool:
  match a b:
    case Nil{} Nil{}: True{}
    case Con{x, s} Con{y, t}: Bool.and(W64.eq(x, W64.of(y)), St.shape_eq(s, t))
    case _ _: False{}

def St.show_u(xs: List<&2, U32>, sep: String) -> String:
  match xs:
    case Nil{}: ""
    case Con{x, t}: sep ++ U32.show(x) ++ St.show_u(t, ", ")

def St.show_w(xs: List<&2, W64>, sep: String) -> String:
  match xs:
    case Nil{}: ""
    case Con{x, t}: sep ++ W64.show(x) ++ St.show_w(t, ", ")

# where a tensor is and how to read it, or why it cannot be read
type Plan is Data:
  PlanOk{name: String, bf16: Bool, off: W64, n: U32}
  PlanBad{msg: String}

def St.plan_span(ok: Bool, +name: String, bf16: Bool, off: W64, +n: U32) -> Plan:
  match ok:
    case True{}: PlanOk{name, bf16, off, n}
    case False{}: PlanBad{"tensor " ++ name ++ ": data_offsets do not span its " ++ U32.show(n) ++ " numbers"}

# data_offsets must be [begin, end] with end - begin the tensor's bytes
def St.plan_of(+name: String, bf16: Bool, +start: W64, +n: U32, +width: U32, of: List<&2, W64>) -> Plan:
  match of:
    case Con{+b, Con{e, Nil{}}}:
      St.plan_span(W64.eq(W64.add(b, W64.mul_small(W64.of(n), width)), e), name, bf16, W64.add(start, b), n)
    case _:
      PlanBad{"tensor " ++ name ++ ": data_offsets is not [begin, end]"}

def St.plan_shape(ok: Bool, +name: String, f32: Bool, +start: W64, +sh: List<&2, W64>, +want: List<&2, U32>, of: List<&2, W64>) -> Plan:
  match ok f32:
    case True{} True{}: St.plan_of(name, False{}, start, St.prod(want), 4, of)
    case True{} False{}: St.plan_of(name, True{}, start, St.prod(want), 2, of)
    case False{} _: PlanBad{"tensor " ++ name ++ ": shape [" ++ St.show_w(sh, "") ++ "] in the file, [" ++ St.show_u(want, "") ++ "] expected"}

def St.plan_dtype(f32: Bool, bf16: Bool, +name: String, +dt: String, +start: W64, +sh: List<&2, W64>, +want: List<&2, U32>, of: List<&2, W64>) -> Plan:
  match f32 bf16:
    case True{} _: St.plan_shape(St.shape_eq(sh, want), name, True{}, start, sh, want, of)
    case _ True{}: St.plan_shape(St.shape_eq(sh, want), name, False{}, start, sh, want, of)
    case _ _: PlanBad{"tensor " ++ name ++ ": dtype " ++ dt ++ " (only F32 and BF16 are read)"}

def St.plan(+start: W64, e: Entry, mr: Maybe<&2, Rec>) -> Plan:
  match e mr:
    case Entry{+name, want} None{}: PlanBad{"tensor " ++ name ++ ": not in the file"}
    case Entry{+name, want} Some{Rec{rn, +dt, sh, of}}: St.plan_dtype(String.eq(dt, "F32"), String.eq(dt, "BF16"), name, dt, start, sh, want, of)

def St.read(bf16: Bool, f: File, off: W64, +n: U32, a: Array<F32>, +at: U32) -> IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  match bf16 off:
    case True{} W64{hi, lo}: R.read_bf16_into(f, hi, lo, n, a, at)
    case False{} W64{hi, lo}: R.read_f32_into(f, hi, lo, n, a, at)

def St.got(+path: String, +name: String, +n: U32, r: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), k: File -> Array<F32> -> U32 -> IO(Array<F32>)) -> IO(Array<F32>):
  match r:
    case Tuple{f, Tuple{a, res}}:
      do IO<Array<F32>>:
        +got : U32 <- IO.pass(U32, res)
        require(U32.is_eq(got, n), path ++ ": tensor " ++ name ++ ": short read (" ++ U32.show(got) ++ " of " ++ U32.show(n) ++ " numbers)")
        k(f, a, n)

# reads one planned tensor into cells at .., then goes on with k
def St.then(+path: String, p: Plan, f: File, a: Array<F32>, +at: U32, k: File -> Array<F32> -> U32 -> IO(Array<F32>)) -> IO(Array<F32>):
  match p:
    case PlanBad{msg}: IO.die(Array<F32>, 1, path ++ ": " ++ msg)
    case PlanOk{+name, bf16, off, +n}:
      IO.bind(File & (Array<F32> & Result<&1, &1, U32 & String, U32>), Array<F32>, St.read(bf16, f, off, n, a, at), r => St.got(path, name, n, r, k))

# every tensor of es, in order, into consecutive cells from at
def St.fill(es: List<&2, Entry>, +path: String, +start: W64, +recs: List<&2, Rec>, f: File, a: Array<F32>, +at: U32) -> IO(Array<F32>):
  match es:
    case Nil{}:
      do IO<Array<F32>>:
        File.close(f)
        return a
    case Con{+e, t}:
      St.then(path, St.plan(start, e, St.find(recs, St.entry_name(e))), f, a, at, f2 => a2 => +n => St.fill(t, path, start, recs, f2, a2, (at + n : U32)))

def St.le32(+a: U32, +b: U32, +c: U32, +d: U32) -> U32:
  U32.or(U32.or(a, U32.shln(b, 8n)), U32.or(U32.shln(c, 16n), U32.shln(d, 24n)))

# the header's length: the file's first 8 bytes, little-endian
def St.le64(bs: List<&2, U32>) -> Maybe<&2, W64>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, Con{b4, Con{b5, Con{b6, Con{b7, t}}}}}}}}:
      Some{W64{St.le32(b4, b5, b6, b7), St.le32(b0, b1, b2, b3)}}
    case _:
      None{}

def St.count(bs: List<&2, U32>, +acc: U32) -> U32:
  match bs:
    case Nil{}: acc
    case Con{b, t}: St.count(t, (acc + 1 : U32))

# A parsed safetensors header: where the data starts and each tensor's record. Reading it once lets a model be filled
# one piece at a time (`piece`), so that no array of the whole model exists. One array for all of SmolLM2-1.7B's
# 1.71e9 parameters takes a 2^31-cell block (8 GiB) whose memory the training step's smaller blocks never reuse.
type Header is Data:
  Header{start: W64, recs: List<&2, Rec>}

def St.parsed(+path: String, +start: W64, ps: Ps) -> IO(Header):
  match ps:
    case Ps{d, c, p, nm, fl, dt, sh, of, ns, rs, bad}:
      do IO<Header>:
        require(Bool.and(U32.is_zero(d), U32.is_zero(bad)), path ++ ": malformed safetensors header")
        return Header{start, rs}

def St.header(+path: String, +len: U32, r: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(Header):
  match r:
    case Tuple{f, res}:
      do IO<Header>:
        +bs : List<&2, U32> <- IO.pass(List<&2, U32>, res)
        File.close(f)
        require(U32.is_eq(St.count(bs, 0), len), path ++ ": the header is cut short")
        St.parsed(path, W64.add(W64.of(8), W64.of(len)), Sp.parse(St.lex(bs, LexR{LOut{}, Nil{}}, Nil{}), Ps{0, False{}, "", "", "", "", Nil{}, Nil{}, Nil{}, Nil{}, 0}))

# the format caps the header at 100 MB
def St.length(f: File, +path: String, h: Maybe<&2, W64>) -> IO(Header):
  match h:
    case None{}: IO.die(Header, 1, path ++ ": shorter than a safetensors header")
    case Some{W64{hi, +lo}}:
      do IO<Header>:
        require(Bool.and(U32.is_zero(hi), U32.is_le(lo, 100000000)), path ++ ": a safetensors header past 100 MB")
        r : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 8, lo)
        St.header(path, lo, r)

def St.first8(+path: String, r: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(Header):
  match r:
    case Tuple{f, res}:
      do IO<Header>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, res)
        St.length(f, path, St.le64(bs))

# A safetensors file's header, parsed; the file is closed after it
def header(+path: String) -> IO(Header):
  do IO<Header>:
    f : File <- IO.try(File, File.open(path, "r"))
    r : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 8)
    St.first8(path, r)

# The tensors `layout` names, read from the file whose header is h, in the layout's order, into one array of n numbers
def tensors_from(+path: String, h: Header, +n: U32, +layout: List<&2, Entry>) -> IO(Array<F32>):
  match h:
    case Header{start, recs}:
      do IO<Array<F32>>:
        require(U32.is_eq(St.total(layout), n), path ++ ": the layout holds " ++ U32.show(St.total(layout)) ++ " numbers, the model " ++ U32.show(n))
        f : File <- IO.try(File, File.open(path, "r"))
        St.fill(layout, path, start, recs, f, Array.new(F32, T.depth(U32.to_nat(n)), 0.0), 0)

# The tensors `layout` names, read from a safetensors file in its order into one array of n numbers (F32 as they are,
# BF16 widened exactly). Each must be in the file with the shape given, as F32 or BF16, and the shapes must hold n
# numbers in all; otherwise the program stops with a message naming the tensor. Other tensors in the file are ignored.
def tensors(+path: String, +n: U32, +layout: List<&2, Entry>) -> IO(Array<F32>):
  do IO<Array<F32>>:
    require(U32.is_eq(St.total(layout), n), path ++ ": the layout holds " ++ U32.show(St.total(layout)) ++ " numbers, the model " ++ U32.show(n))
    h : Header <- header(path)
    tensors_from(path, h, n, layout)

# One layer's parameters from the file whose header is h: `layout` names its tensors in Layer.dump order, and
# Layer.load fills them; the array they were read into is gone once this answers
def piece(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, h: Header, +layout: List<&2, Entry>, p: P) -> IO(P):
  do IO<P>:
    +n : U32 = T.Layer.count(P, X, Y, S, m)
    a : Array<F32> <- tensors_from(path, h, n, layout)
    weights_done(P, n, T.Layer.load(P, X, Y, S, m, p, a, 0))

# A pair of parameters filled one after the other: the first piece's source array is released before the second is read
def two(-P1: Type, -P2: Type, p: T.Two<P1, P2>, k1: P1 -> IO(P1), k2: P2 -> IO(P2)) -> IO(T.Two<P1, P2>):
  match p:
    case T.Two{a, b}:
      do IO<T.Two<P1, P2>>:
        x : P1 <- k1(a)
        y : P2 <- k2(b)
        return T.Two{x, y}

# All parameters from a safetensors file: `layout` names the model's tensors in Layer.dump order (a tied output head
# is not in it, as SmolLM2's is not in its file), and Layer.load fills the parameters from them as `weights` does
def safetensors(~P: Type, ~X: Type, ~Y: Type, ~S: Type, ~m: T.Layer<P, X, Y, S>, +path: String, +layout: List<&2, Entry>, p: P) -> IO(P):
  do IO<P>:
    h : Header <- header(path)
    piece(~P, ~X, ~Y, ~S, ~m, path, h, layout, p)
