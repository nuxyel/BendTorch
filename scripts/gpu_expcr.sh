#!/usr/bin/env bash
# make gpu's exactness check: demo/gpu_layers_test.bend built with CUDA against a copy of torch/ in .scratch/gpu_expcr/
# where every F32.exp of main.bend and gpu.bend is demo/exp_cr.bend's exp_cr (the host's F32.exp bits, in plain F32 and
# U32 arithmetic), each case run alone so that its bangs run on the device. Prints the 24 cases' ulps; with the
# exponential the same on the host and the device, every case must be 0, which shows that the device code differs from
# main.bend's only through CUDA's float exp. torch/ itself is not touched.
# With `accum`, the same for demo/accum_lanes.bend (gradient accumulation): its copy whose F is gpu.bend, built with
# CUDA and each run alone, against its copy whose F is main.bend, built for the CPU, both over the swapped torch/;
# prints reference/accum_lanes.py's line (five models and k = 1), which must be all 0.
#
#   scripts/gpu_expcr.sh [accum]  # needs a CUDA toolkit: CUDA_HOME, or the first of ~/.local/cuda12,
#                                  ~/.local/cuda-nvrtc, $CUDA_PATH, /usr/local/cuda, /opt/cuda with nvrtc.h;
#                                  run it under the heavy-run lock
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${CUDA_HOME:-}" ]; then
  for d in "$HOME/.local/cuda12" "$HOME/.local/cuda-nvrtc" "${CUDA_PATH:-}" /usr/local/cuda /opt/cuda; do
    if [ -n "$d" ] && [ -f "$d/include/nvrtc.h" ]; then CUDA_HOME=$d; break; fi
  done
fi
CUDA_HOME=${CUDA_HOME:-}
CUDA_LIB=$CUDA_HOME/lib64
[ -d "$CUDA_LIB" ] || CUDA_LIB=$CUDA_HOME/lib
test -f "$CUDA_HOME/include/nvrtc.h" && test -d "$CUDA_LIB" || { echo "gpu_expcr: no CUDA toolkit (CUDA_HOME=$CUDA_HOME)"; exit 1; }
D=.scratch/gpu_expcr
rm -rf "$D" && mkdir -p "$D/demo"
cp -r torch "$D/torch"
cp demo/exp_cr.bend "$D/torch/exp_cr.bend"
for f in main gpu; do
  sed -i 's/F32\.exp(/XE.exp_cr(/g; 0,/^import Base$/s//import Base\nimport .\/exp_cr.bend as XE/' "$D/torch/$f.bend"
  grep -q 'import ./exp_cr.bend as XE' "$D/torch/$f.bend" && ! grep -q 'F32\.exp(' "$D/torch/$f.bend" || { echo "gpu_expcr: the swap failed in $f.bend"; exit 1; }
done
if [ "${1:-}" = accum ]; then
  sed 's|^import ../torch/fast.bend as F$|import ../torch/main.bend as F|' demo/accum_lanes.bend > "$D/demo/accum_lanes_main.bend"
  sed 's|^import ../torch/fast.bend as F$|import ../torch/gpu.bend as F|' demo/accum_lanes.bend > "$D/demo/accum_lanes_gpu.bend"
  grep -q '^import ../torch/main.bend as F$' "$D/demo/accum_lanes_main.bend" && grep -q '^import ../torch/gpu.bend as F$' "$D/demo/accum_lanes_gpu.bend" || { echo "gpu_expcr: the import swap failed in accum_lanes"; exit 1; }
  env -u CUDA_HOME bend "$D/demo/accum_lanes_main.bend" -o "$D/accum_lanes_main"
  CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$CUDA_LIB bend "$D/demo/accum_lanes_gpu.bend" -o "$D/accum_lanes_gpu_cuda"
  "$D/accum_lanes_main" > "$D/accum_main.txt"
  for m in 0 1 2 3 4; do for r in 0 1 2 3; do LD_LIBRARY_PATH=$CUDA_LIB "$D/accum_lanes_gpu_cuda" $m $r; done; done > "$D/accum_cuda.txt"
  reference/.venv/bin/python reference/accum_lanes.py "$D/accum_main.txt" "$D/accum_cuda.txt"
  exit
fi
cp demo/gpu_layers_test.bend "$D/demo/gpu_layers_test.bend"
CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$CUDA_LIB bend "$D/demo/gpu_layers_test.bend" -o "$D/gpu_layers_test_cuda"
for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do LD_LIBRARY_PATH=$CUDA_LIB "$D/gpu_layers_test_cuda" $k; done | tr '\n' ' '
echo
