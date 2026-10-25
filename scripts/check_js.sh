#!/usr/bin/env bash
# The JavaScript lane (plan M12): the test programs compiled to JavaScript (`bend x.bend -o x.js`; `-o x.mjs` builds a
# module of the non-IO defs, which does not run `main`), run with Bun, and compared with the same PyTorch oracles and the
# same tolerance as the native programs (reference/oracle.py). Bun, not node: Bend 2.0.37's JavaScript file reads call
# libc's read/pread through `bun:ffi`, so under node every program that reads a file fails. A small wrapper per program
# lets the tests call it as they call a native binary, with a 1 GB stack: the generated JavaScript keeps a call frame
# per step of every recursion that is not a tail call (a list built head first, as the loaders do), and the default
# stack holds about 40,000 of them (8 million with this one).
#
#   scripts/check_js.sh       # make check-js
set -euo pipefail
cd "$(dirname "$0")/.."
PY=reference/.venv/bin/python
BUN=${BUN:-$(command -v bun || echo "$HOME/.bun/bin/bun")}
[ -x "$BUN" ] || { echo "bun is not installed (https://bun.sh)"; exit 2; }
mkdir -p .scratch/js
for d in mlp_test ops_test mnist tiny_llama tensor_test imap_check fast_test infra_test reader_test safetensors_test; do
  bend "demo/$d.bend" -o ".scratch/js/$d.js"
  printf '#!/bin/sh\nulimit -s 1048576\nBUN_JSC_maxPerThreadStackUsage=1000000000 exec %s "$(dirname "$0")/%s.js" "$@"\n' "$BUN" "$d" > ".scratch/js/$d"
  chmod +x ".scratch/js/$d"
done
echo "bun $("$BUN" --version)"
$PY reference/test_infra.py .scratch/js/infra_test --js
$PY reference/test_reader.py .scratch/js/reader_test --js
$PY reference/test_safetensors.py .scratch/js/safetensors_test --js
$PY reference/test_ops.py .scratch/js/ops_test
$PY reference/test_tensor.py .scratch/js/tensor_test
$PY reference/check_imap.py .scratch/js/imap_check
$PY reference/test_mlp.py .scratch/js/mlp_test
$PY reference/test_mnist.py .scratch/js/mnist
$PY reference/test_tiny.py .scratch/js/tiny_llama 20
test "$(.scratch/js/fast_test | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 0 0 0 0 " && echo "fast: torch/fast.bend's layers and optimizer step give main.bend's bits (JavaScript)" || { echo "fast: the bits differ (JavaScript)"; exit 1; }
# the exact tier, on the whole programs: the JavaScript build must give the native build's bits (bench/bits.txt)
digest() { sed -E 's/ train_seconds=[0-9.]+//' | sha256sum | cut -c1-16; }
P=$'ROMEO:\nBut soft, wh'
{
  echo "tiny_llama train 50, 1 threads: $(.scratch/js/tiny_llama train 50 | digest)"
  echo "tiny_llama sample 100 64, 1 threads: $(.scratch/js/tiny_llama sample 100 64 "$P" | digest)"
  echo "mnist 1 epoch 50 batches adam, 1 threads: $(.scratch/js/mnist 1 50 adam 0.001 | digest)"
} > .scratch/js/bits.txt
diff <(grep ", 1 threads:" bench/bits.txt) .scratch/js/bits.txt && echo "js bits: the same outputs as the native build"
