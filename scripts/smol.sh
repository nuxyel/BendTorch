#!/usr/bin/env bash
# Writes demo/smol.bend's stack of blocks (the defs between its "stack" markers) and its sizes for one SmolLM2 model.
# A template takes only arguments fixed at compile time and Bend instantiates templates 64 levels deep at most, so the
# stack of LAYERS() blocks is spelled out as a balanced tree of Seq, one def per subtree size, with the cached step
# (KStack), the cache's reader (KGet) and the safetensors loader (LStack, block i from model.layers.i) of each. 135m rewrites
# demo/smol.bend in place (the committed file); 360m writes .scratch/smol_360m.bend.
#
#   scripts/smol.sh 135m | 360m | 1.7b   (1.7b: .scratch/smol_1b7.bend; `train <steps> <model.safetensors>` loads it)
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  135m) layers=30; d=576; kvw=192; ff=1536; kv=3; g=3; data=data/smol135/; out=demo/smol.bend ;;
  360m) layers=32; d=960; kvw=320; ff=2560; kv=5; g=3; data=data/smol360/; out=.scratch/smol_360m.bend ;;
  1.7b) layers=24; d=2048; kvw=2048; ff=8192; kv=32; g=1; data=data/smol135/; out=.scratch/smol_1b7.bend ;;
  *) echo "usage: scripts/smol.sh 135m | 360m | 1.7b"; exit 2 ;;
esac
mkdir -p .scratch
python3 - "$layers" "$d" "$kvw" "$ff" "$kv" "$g" "$data" "$out" <<'PY'
import re, sys
layers, d, kvw, ff, kv, g, data, out = sys.argv[1:]
s = open("demo/smol.bend").read()
for name, v in [("D", d), ("KVW", kvw), ("FF", ff), ("HKV", kv), ("GRP", g), ("LAYERS", layers)]:
    s, n = re.subn(rf"def {name}\(\) -> Nat:\n  \d+n", f"def {name}() -> Nat:\n  {v}n", s)
    assert n == 1, name
s, n = re.subn(r'def DATA\(\) -> String:\n  "[^"]*"', f'def DATA() -> String:\n  "{data}"', s)
assert n == 1
def sizes(n, out):
    if n > 1 and n not in out:
        sizes(n // 2, out); sizes(n - n // 2, out); out.append(n)
    return out
defs = ["def StP1() -> Type:\n  Boxed<BP>\n", "def StS1(b: Nat) -> Type:\n  Boxed<BS(b)>\n",
        "def Stack1(~b: Nat) -> T.Layer<StP1, A(b, D()), A(b, D()), StS1(b)>:\n"
        "  Box(~BP, ~A(b, D()), ~A(b, D()), ~BS(b), ~Block(~b))\n",
        "def KStack1(p: StP1, +pos: Nat, s: KS) -> KR<StP1>:\n  kbox(kblock(Boxed.get(BP, p), pos, s))\n",
        "def KGet1(s: StS1(1n), acc: List<&1, KV>) -> List<&1, KV>:\n  kget(Boxed.get(BS(1n), s), acc)\n",
        "def LStack1(+path: String, +h: L.Header, +i: Nat, p: StP1) -> IO(StP1):\n"
        "  L.piece(~StP1, ~A(2n, D()), ~A(2n, D()), ~StS1(2n), ~Stack1(~2n), path, h, st_layer(i), p)\n"]
for n in sizes(int(layers), []):
    a, c = n // 2, n - n // 2
    defs.append(f"def StP{n}() -> Type:\n  T.Two<StP{a}, StP{c}>\n")
    defs.append(f"def StS{n}(b: Nat) -> Type:\n  T.Two<StS{a}(b), StS{c}(b)>\n")
    defs.append(f"def Stack{n}(~b: Nat) -> T.Layer<StP{n}, A(b, D()), A(b, D()), StS{n}(b)>:\n"
                f"  T.Seq(~StP{a}, ~StP{c}, ~A(b, D()), ~A(b, D()), ~A(b, D()), ~StS{a}(b), ~StS{c}(b), ~Stack{a}(~b), ~Stack{c}(~b))\n")
    defs.append(f"def KStack{n}(p: StP{n}, +pos: Nat, s: KS) -> KR<StP{n}>:\n  match p:\n    case T.Two{{a, c}}:\n"
                f"      kseq(StP{a}, StP{c}, KStack{a}(a, pos, s), s2 => KStack{c}(c, pos, s2))\n")
    defs.append(f"def KGet{n}(s: StS{n}(1n), acc: List<&1, KV>) -> List<&1, KV>:\n  match s:\n    case T.Two{{a, c}}:\n"
                f"      KGet{c}(c, KGet{a}(a, acc))\n")
    defs.append(f"def LStack{n}(+path: String, +h: L.Header, +i: Nat, p: StP{n}) -> IO(StP{n}):\n"
                f"  L.two(StP{a}, StP{c}, p, x => LStack{a}(path, h, i, x), y => LStack{c}(path, h, Nat.add(i, {a}n), y))\n")
defs += [f"def StackP() -> Type:\n  StP{layers}\n", f"def StackS(b: Nat) -> Type:\n  StS{layers}(b)\n",
         f"def Stack(~b: Nat) -> T.Layer<StackP, A(b, D()), A(b, D()), StackS(b)>:\n  Stack{layers}(~b)\n",
         f"def KStack(p: StackP, +pos: Nat, s: KS) -> KR<StackP>:\n  KStack{layers}(p, pos, s)\n",
         f"def KGet(s: StackS(1n), acc: List<&1, KV>) -> List<&1, KV>:\n  KGet{layers}(s, acc)\n",
         f"def LStack(+path: String, +h: L.Header, p: StackP) -> IO(StackP):\n  LStack{layers}(path, h, 0n, p)\n"]
begin, end = "# ---- stack: begin ----\n", "# ---- stack: end ----\n"
i, j = s.index(begin) + len(begin), s.index(end)
s = s[:i] + "\n".join(defs) + s[j:]
open(out, "w").write(s)
print(out)
PY
# SmolLM2-1.7B's RoPE base is 130000 (config.json's rope_theta); 135M's and 360M's are 100000. Its AdamW keeps m and v
# packed as BF16 (T.CAdamWBF16 with v16): the state is 19.5 GiB instead of 26.0, and a step fits in 32 GB only so.
if [ "$1" = 1.7b ]; then
  sed -i 's/, 100000\.0, {==})/, 130000.0, {==})/g' .scratch/smol_1b7.bend
  sed -i 's/^  T\.CAdamW{0\.0001, 0\.9, 0\.999, 0\.00000001, 0\.01, 0\.1, 0\.001}$/  T.CAdamWBF16{0.0001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001, True{}}/' .scratch/smol_1b7.bend
  grep -q 'T.CAdamWBF16{' .scratch/smol_1b7.bend
fi
