#!/usr/bin/env bash
# Assemble torch/ into BendTorch's four BendHub packages, check them, and test them against the repository's code.
# Nothing here publishes: publishing is `bend <entry> --publish <name>@<version>` from each package's folder, by hand.
#
#   scripts/package.sh <core> <version> [out]   # make package CORE=<core> VERSION=<version>
#
# <core> is the core's hub name (bendtorch, or bend-ml-torch); the others are <core>-fast, <core>-gpu and <core>-load.
# Each package lands in <out>/<name>/ (default build/packages/):
#
#   <core>/        main.bend spec.bend tensor.bend dtype.bend LAWS.bend PROOF.bend package.bend LICENSE (SPDX line on top)
#                  laws, no @unsafe, no foreign code; package.bend is the entry to publish: --publish sends a file and
#                  what it imports, so the entry imports every file (and the laws) for them to ship
#   <core>-fast/   main.bend (torch/fast.bend) LICENSE    over bend-ml-tensor-fork (@unsafe in Base)
#   <core>-gpu/    main.bend (torch/gpu.bend)  LICENSE    over bend-ml-tensor-gpu (@unsafe in Base)
#   <core>-load/   main.bend (torch/load.bend) LICENSE    over bend-ml-binary-read (foreign C and JavaScript)
#
# The only change to the code is in import lines: `./main.bend` becomes `<core>@<version>/main.bend` in the three
# satellites, and `./torch/x.bend` becomes `./x.bend` in LAWS.bend and PROOF.bend. A user imports every file they use,
# `import <core>@<version>/main.bend as T` beside `import <core>-fast@<version>/main.bend as F`, at the same version.
#
# Then, with the cross-package imports pointed at local copies (<out>-local/), it checks the core with --verdict,
# checks each satellite, builds test programs over the assembled packages and compares them with the repository's:
# fast_test and gpu_test (main.bend's bits), infra_test and dtype_test (their PyTorch tests), the tiny Llama's
# training (bench/bits.txt's digest).
set -euo pipefail
cd "$(dirname "$0")/.."
export BEND_NO_TELEMETRY=1 PATH="$HOME/.bend/bin:$PATH"
[ $# -ge 2 ] || { echo "usage: scripts/package.sh <core> <version> [out]"; exit 2; }
CORE=$1 VERSION=$2 OUT=${3:-build/packages}
LOCAL=$OUT-local
PY=reference/.venv/bin/python
fail() { echo "package: $*"; exit 1; }

# the hub's rules (Bend 2.0.37, bend2/bend.ts NAMED and the import loader's path rule)
for n in "$CORE" "$CORE-fast" "$CORE-gpu" "$CORE-load"; do
  [[ $n =~ ^[a-z][a-z0-9-]{0,63}$ ]] || fail "'$n' is not a hub name (a-z, 0-9 and -, a letter first, at most 64)"
done
[[ $VERSION =~ ^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){3}$ ]] || fail "'$VERSION' is not four numbers like 0.1.0.0"

rm -rf "$OUT" "$LOCAL"
mkdir -p "$OUT/$CORE" "$OUT/$CORE-fast" "$OUT/$CORE-gpu" "$OUT/$CORE-load"

# the core: its files as they are; the laws with their imports moved up one folder
for f in main spec tensor dtype; do cp torch/$f.bend "$OUT/$CORE/$f.bend"; done
for f in LAWS PROOF; do sed 's|^import \./torch/\([a-z]*\)\.bend as |import ./\1.bend as |' $f.bend > "$OUT/$CORE/$f.bend"; done
cat > "$OUT/$CORE/package.bend" <<'EOF'
# The entry to publish: `bend package.bend --publish <name>@<version>` sends this file and everything it imports, so
# every file of the package and its laws are here. Import the files themselves: <name>@<version>/main.bend, spec.bend,
# tensor.bend, dtype.bend.
import Base
import ./main.bend as T
import ./spec.bend as Spec
import ./tensor.bend as TT
import ./dtype.bend as D
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
EOF
# the satellites: one file each, importing the core by name
for s in fast gpu load; do
  sed "s|^import \./main\.bend as T$|import $CORE@$VERSION/main.bend as T|" torch/$s.bend > "$OUT/$CORE-$s/main.bend"
done
# the license with an SPDX line on top, which bend --publish reads to name it (the text below it is LICENSE unchanged)
for p in "$CORE" "$CORE-fast" "$CORE-gpu" "$CORE-load"; do { echo "SPDX-License-Identifier: MIT"; echo; cat LICENSE; } > "$OUT/$p/LICENSE"; done

# only import lines changed
for f in main spec tensor dtype; do cmp -s torch/$f.bend "$OUT/$CORE/$f.bend" || fail "$f.bend changed"; done
for f in LAWS PROOF; do
  diff <(grep -v '^import ' $f.bend) <(grep -v '^import ' "$OUT/$CORE/$f.bend") > /dev/null || fail "$f.bend changed beyond its imports"
done
for s in fast gpu load; do
  diff <(grep -v '^import ' torch/$s.bend) <(grep -v '^import ' "$OUT/$CORE-$s/main.bend") > /dev/null || fail "$s.bend changed beyond its imports"
  grep -qx "import $CORE@$VERSION/main.bend as T" "$OUT/$CORE-$s/main.bend" || fail "$s.bend does not import the core by name"
done

# every import of every package: Base, a named hub package, or a file of the same package (no climbing out of it)
for p in "$CORE" "$CORE-fast" "$CORE-gpu" "$CORE-load"; do
  for f in "$OUT/$p"/*.bend; do
    while read -r line; do
      path=$(echo "$line" | sed -n 's/^import \([^ ]*\)\( as .*\)\{0,1\}$/\1/p')
      case $path in
        Base) ;;
        ./*) [ -f "$OUT/$p/${path#./}" ] || fail "$f imports $path, which is not in the package" ;;
        *@*/*) [[ ${path%%@*} =~ ^[a-z][a-z0-9-]{0,63}$ ]] || fail "$f: bad package name in $path" ;;
        *) fail "$f: import '$line' is neither Base, a hub package nor a file of the package" ;;
      esac
    done < <(grep '^import ' "$f")
    # path segments: letters, digits, _ and -, not starting with a digit or -; no folder named license
    rel=${f#"$OUT/$p/"}
    [[ ${rel%.bend} =~ ^([A-Za-z_][A-Za-z0-9_-]*/)*[A-Za-z_][A-Za-z0-9_-]*$ ]] || fail "$f: not a hub path"
  done
done

# the core's guarantees: no @unsafe, no ?TODO, no foreign code, and only nat-lemmas and tensor-array from the hub
! grep -n -E '^[^#]*(@unsafe|\?TODO)' "$OUT/$CORE"/*.bend || fail "the core has @unsafe or ?TODO"
! grep -n -E '^\s*import\s+"' "$OUT/$CORE"/*.bend || fail "the core has foreign code"
! grep -h '^import [a-z].*@' "$OUT/$CORE"/*.bend | grep -v -E '^import bend-ml-(tensor-array|nat-lemmas)@' || fail "the core imports a hub package other than tensor-array and nat-lemmas"

# local copies: the satellites import the core as ../<core>/main.bend
cp -r "$OUT" "$LOCAL"
for s in fast gpu load; do
  sed -i "s|^import $CORE@$VERSION/main.bend as T$|import ../$CORE/main.bend as T|" "$LOCAL/$CORE-$s/main.bend"
done
( cd "$LOCAL/$CORE" && bend package.bend --check-only | tail -2 | head -1 | grep -qx "ALL PROOFS CHECK" ) || fail "the core does not check"
( cd "$LOCAL/$CORE" && bend package.bend --verdict | tail -1 | grep -qx "ALL PROOFS CHECK" ) || fail "the core's --verdict is not ALL PROOFS CHECK"
echo "package: $CORE checks, and --verdict passes (no @unsafe, no foreign code)"
# a satellite's check lists its defs outside the proofs (the ones make laws pins) and must say nothing else: the same
# output as the repository's file
for s in fast gpu load; do
  got=$(cd "$LOCAL/$CORE-$s" && bend main.bend --check-only 2>&1 || true)
  want=$(bend torch/$s.bend --check-only 2>&1 || true)
  [ "$got" = "$want" ] || fail "$CORE-$s checks differently from torch/$s.bend"
  echo "$got" | sed -n 2p | grep -qE '^Error: [0-9]+ defs rely on unsafe or foreign code:$' || fail "$CORE-$s: an error other than its unsafe or foreign defs"
done
echo "package: $CORE-fast, $CORE-gpu, $CORE-load check over the local core as torch/ does (only their pinned defs outside the proofs)"

# test programs over the assembled packages, against the repository's
T=$LOCAL/tests
mkdir -p "$T" .scratch
over() {  # demo/$1.bend with its torch/ imports pointed at the local packages
  sed -e "s|^import \.\./torch/main\.bend as |import ../$CORE/main.bend as |" \
      -e "s|^import \.\./torch/dtype\.bend as |import ../$CORE/dtype.bend as |" \
      -e "s#^import \.\./torch/\(fast\|gpu\|load\)\.bend as #import ../$CORE-\1/main.bend as #" demo/$1.bend > "$T/$1.bend"
  ! grep -q '^import \.\./torch/' "$T/$1.bend" || fail "demo/$1.bend has a torch/ import left"
  bend "$T/$1.bend" -o "$T/$1" > /dev/null || fail "demo/$1.bend does not build over the packages"
}
for d in fast_test gpu_test infra_test dtype_test tiny_llama; do over $d; done
for t in 1 16; do
  test "$("$T/fast_test" +RTS --threads $t -RTS | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " || fail "fast_test differs ($t threads)"
  test "$("$T/gpu_test" +RTS --threads $t -RTS | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 " || fail "gpu_test differs ($t CPU threads)"
done
echo "package: $CORE-fast's and $CORE-gpu's layers give the core's bits (1 and 16 threads)"
$PY reference/test_infra.py "$T/infra_test" > "$T/infra.txt" || { cat "$T/infra.txt"; fail "test_infra fails over the packages"; }
$PY reference/test_dtype.py "$T/dtype_test" > "$T/dtype.txt" || { cat "$T/dtype.txt"; fail "test_dtype fails over the packages"; }
echo "package: test_infra and test_dtype pass over the packages"
digest() { sed -E 's/ train_seconds=[0-9.]+//' | sha256sum | cut -c1-16; }
for t in 1 16; do
  want=$(sed -n "s/^tiny_llama train 50, $t threads: //p" bench/bits.txt)
  [ -n "$want" ] || fail "no tiny_llama train 50 line for $t threads in bench/bits.txt"
  test "$("$T/tiny_llama" +RTS --threads $t -RTS train 50 | digest)" = "$want" || fail "the tiny Llama's training differs from bench/bits.txt ($t threads)"
done
echo "package: the tiny Llama trains over the packages with bench/bits.txt's bits (1 and 16 threads)"
echo "package: $OUT/{$CORE,$CORE-fast,$CORE-gpu,$CORE-load} at $VERSION"
