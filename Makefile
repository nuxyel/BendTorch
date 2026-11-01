# BendTorch: setup and verification shortcuts. Run `make setup` and `make data` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
DEMOS := mlp_test ops_test mnist tiny_llama tiny_rt tensor_test imap_check fast_test gpu_test gpu_layers_test depth_check smol_block infra_test dtype_test tiny_mixed reader_test safetensors_test tiny_bf16opt loop_test
# torch/load.bend's defs that reach the foreign binary reader (bend-ml-binary-read@0.2.0.0): the list may change only on purpose
LOAD_DEFS := read_go read_sized read read_f32_at read_bf16_at size weights train_read train St.read St.then St.fill tensors_from tensors piece safetensors
# torch/gpu.bend's defs that rely on bend-ml-tensor-gpu's @unsafe Array.fork (its leaves do not: they are plain Bend)
GPU_DEFS := lin_fwd lin_b2 lin_bwd Mat.dup Mat.sum2 silu_f silu_b SiLU mul2_f mul2_b Mul2 add2_f Add2 Residual opt_mat mat_step lin_step2 lin_step LinearF linf_bwd LinearFirstF lnb_b2 LinearNBF rms_f rms_b2 rms_b RMSNorm gn_f gn_b2 gn_b ln_step2 ln_step GroupNorm rope_go2 rope_go RoPE af2 af1 af0 attn_f ab3 ab2 ab1 attn_b af2_f af1_f af0_f attn_f_f ab3_f ab2_f ab1_f attn_b_f Attention FullAttention mse_go MSE emb_f2 emb_f emb_b2 emb_bwd Embedding cx_4 cx_3 cx_2 cx_1 CrossEntropy Linear LinearFirst LinearNB
# a file's defs outside the proofs, as the checker lists them ("defs that depend on @unsafe: a, b"), without the hub's
# (0x<hash>/main.f): each package pins its own
OUTSIDE = sed -n 's/^defs that depend on [^:]*: //p' | tr ',' '\n' | sed 's/^ *//' | grep -E '^[A-Za-z0-9_.]+$$'
# the GPU build: bend -o builds the device program when CUDA_HOME names a CUDA toolkit
# where the toolkit is: the first candidate that holds nvrtc.h, the same file cli_build looks for
# (set CUDA_HOME to override); empty when none does, which gpu: refuses with a message instead of
# building the CPU lane and calling it the GPU
CUDA_CANDIDATES := $(HOME)/.local/cuda12 $(HOME)/.local/cuda-nvrtc $(CUDA_PATH) /usr/local/cuda /opt/cuda
CUDA_HOME ?= $(patsubst %/include/nvrtc.h,%,$(firstword $(wildcard $(addsuffix /include/nvrtc.h,$(CUDA_CANDIDATES)))))
# the libraries live in lib64 (what cli_build links first) or lib, as nix lays them: never assume one
CUDA_LIB := $(if $(wildcard $(CUDA_HOME)/lib64),$(CUDA_HOME)/lib64,$(CUDA_HOME)/lib)
# llama.bend's checkout, where scripts/smol_data.sh finds the Hugging Face checkpoints (make smol reads SmolLM2-135M's)
LLAMA_BEND ?= ../llama.bend
ST135 = $(LLAMA_BEND)/data/smollm2-135m/hf/model.safetensors
FAST_DEFS := ph_leafs ph_mat opt_mat mat_step lin_step2 lin_step map zip Mat.dup Mat.sum2 lin_fwd lin_b2 lin_bwd LinearF linf_bwd LinearFirstF lnb_b2 LinearNBF acti_f3 ActIn SiLU add2_f Add2 mul2_f2 mul2_f mul2_b2 Mul2 Residual Embedding Linear LinearFirst LinearNB

.PHONY: setup data laws build check check-js reader bits mutate bench lint smol finetune smol-gen smol-gpu mutate-smol cuda-toolkit gpu mixed-long adamw16-long dw memory corpus bits-smol package own

setup:            ## install the pinned Bend and Lean if missing, create the Python venv
	scripts/setup.sh

data:             ## MNIST and Tiny Shakespeare into data/, the twins' initial weights and batches (seeded)
	scripts/mnist.sh
	scripts/tiny.sh

laws:             ## the proofs (LAWS.bend, PROOF.bend) by bend and again by --verdict; tests/bad_*.bend rejected with their expected error
	$(PY) scripts/check_rts.py --self-test
	$(PY) scripts/check_rts.py
	! grep -n -E '^[^#]*(@unsafe|\?TODO)' torch/*.bend LAWS.bend PROOF.bend demo/*.bend
	bend torch/main.bend --check-only
	bend PROOF.bend --check-only | tail -2 | head -1 | grep -qx "ALL PROOFS CHECK"
	bend PROOF.bend --verdict | tee /dev/stderr | tail -1 | grep -qx "ALL PROOFS CHECK"
	bend torch/dtype.bend --verdict | tail -1 | grep -qx "ALL PROOFS CHECK"
	test "$$(bend torch/fast.bend 2>&1 | $(OUTSIDE) | tr '\n' ' ')" = "$(FAST_DEFS) " || { echo "torch/fast.bend: its defs outside the proofs are not FAST_DEFS"; exit 1; }
	test "$$(bend torch/load.bend 2>&1 | $(OUTSIDE) | tr '\n' ' ')" = "$(LOAD_DEFS) " || { echo "torch/load.bend: its defs outside the proofs are not LOAD_DEFS"; exit 1; }
	! grep -n -E 'load.bend|bend-ml-binary-read|binary-read/' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend torch/dtype.bend torch/fast.bend torch/gpu.bend
	! grep -n 'fast.bend' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend
	test "$$(bend torch/gpu.bend 2>&1 | $(OUTSIDE) | grep -v '^fast\.' | tr '\n' ' ')" = "$(GPU_DEFS) " || { echo "torch/gpu.bend: its defs outside the proofs are not GPU_DEFS"; exit 1; }
	! grep -n -E 'gpu.bend|tensor-gpu' LAWS.bend PROOF.bend torch/main.bend torch/spec.bend torch/tensor.bend torch/fast.bend
	! grep -n 'exp_cr' LAWS.bend PROOF.bend torch/*.bend
	for f in tests/bad_*.bend; do want=$$(sed -n 's/^# expect: //p' $$f); bend $$f --check-only 2>&1 | grep -qF -- "$$want" && echo "rejected as it must be: $$f" || { echo "NOT REJECTED AS EXPECTED: $$f (wanted: $$want)"; exit 1; }; done

build:            ## compile the test and demo programs into .scratch/
	mkdir -p .scratch
	for d in $(DEMOS); do bend demo/$$d.bend -o .scratch/$$d || exit 1; done

check: laws build ## the laws, then every layer, the tensor operations, the MLP, MNIST, the tiny Llama and a SmolLM2-sized block against PyTorch
	$(PY) reference/test_infra.py .scratch/infra_test
	$(PY) reference/test_reader.py .scratch/reader_test
	$(PY) reference/test_safetensors.py .scratch/safetensors_test
	$(PY) reference/test_ops.py .scratch/ops_test
	$(PY) reference/test_tensor.py .scratch/tensor_test
	$(PY) reference/check_imap.py .scratch/imap_check
	$(PY) reference/test_mlp.py .scratch/mlp_test
	$(PY) reference/test_mnist.py .scratch/mnist
	$(PY) reference/test_tiny.py .scratch/tiny_llama 20
	$(PY) reference/test_tiny_rt.py .scratch/tiny_rt 20
	$(PY) reference/test_dtype.py .scratch/dtype_test
	$(PY) reference/test_tiny_mixed.py .scratch/tiny_mixed .scratch/tiny_llama 20
	$(PY) reference/test_tiny_bf16opt.py .scratch/tiny_bf16opt .scratch/tiny_llama 20
	$(PY) reference/test_block.py .scratch/smol_block 4
	$(PY) reference/test_block.py .scratch/smol_block 4 135m sgd
	for t in 1 16; do test "$$(.scratch/fast_test +RTS --threads $$t -RTS | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " && echo "fast: torch/fast.bend's layers and optimizer step give main.bend's bits ($$t threads)" || { echo "fast: the bits differ ($$t threads)"; exit 1; }; done
	.scratch/fast_test --threads 16 > /dev/null 2>&1; test $$? = 2 && echo "fast: fast_test refuses a bare --threads 16 (a runtime option outside +RTS ... -RTS)" || { echo "fast_test accepted a bare --threads"; exit 1; }
	$(PY) scripts/rts_probe.py --min 1.8 .scratch/fast_test
	for t in 1 16; do test "$$(.scratch/gpu_test +RTS --threads $$t -RTS | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 " && echo "gpu: torch/gpu.bend's layers give main.bend's bits ($$t CPU threads)" || { echo "gpu: the bits differ ($$t threads)"; exit 1; }; done
	for t in 1 16; do test "$$(.scratch/gpu_layers_test +RTS --gpu off --threads $$t -RTS | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " && echo "gpu: torch/gpu.bend's element-wise layers, RMSNorm, GroupNorm, RoPE, attention, full attention, optimizer step, MSE, embedding and cross-entropy give main.bend's bits ($$t CPU threads)" || { echo "gpu: the bits differ ($$t threads)"; exit 1; }; done
	test "$$(.scratch/depth_check)" = "0" && echo "depth: the scratch arrays' capacity is TA.cap_depth's for n = 0..4999" || { echo "depth: differs from TA.cap_depth"; exit 1; }

corpus:           ## release time only (M14): every layer, loss, optimizer step and tensor operation against bench/corpus/'s frozen PyTorch results (corpus-2: float64 and the sum of |terms|), 1 and 16 threads
	mkdir -p .scratch
	bend demo/corpus_test.bend -o .scratch/corpus_test
	for t in 1 16; do $(PY) reference/test_corpus.py .scratch/corpus_test bench/corpus --threads $$t > .scratch/corpus_$$t.txt; s=$$?; cat .scratch/corpus_$$t.txt; test $$s = 0 || exit $$s; done
	test "$$(grep sha256 .scratch/corpus_1.txt)" = "$$(grep sha256 .scratch/corpus_16.txt)" && echo "corpus: the same bits on 1 and 16 threads" || { echo "corpus: the bits differ between 1 and 16 threads"; exit 1; }

bits-smol:        ## SmolLM2-135M's loss and 3 AdamW steps, 1 and 16 threads, against bench/bits.txt's smol lines (needs data/smol135; ~5 GB, ~3 min)
	mkdir -p .scratch
	bend demo/smol.bend -o .scratch/smol
	scripts/bits.sh smol check

mixed-long:       ## the tiny Llama with BF16 storage, 2000 steps, against the twin's BF16 storage in float64 (M11's criterion)
	$(PY) reference/test_tiny_mixed.py .scratch/tiny_mixed .scratch/tiny_llama 2000

adamw16-long:     ## the tiny Llama with AdamW's moments in packed BF16 (T.CAdamWBF16, m and m+v), 2000 steps, against the twin with the same storage in float64 (M11's criterion)
	$(PY) reference/test_tiny_bf16opt.py .scratch/tiny_bf16opt .scratch/tiny_llama 2000

dw:               ## the double-word accumulator against an exact sum, and its cost (one thread on CPU 3)
	taskset -c 3 $(PY) reference/test_dtype.py .scratch/dtype_test dw 1048576 200 4000

memory:           ## SmolLM2's training state under each storage choice (arithmetic)
	$(PY) reference/smol_memory.py 1.7b

cuda-toolkit:     ## refuse the GPU lane when no CUDA toolkit is found
	@test -f "$(CUDA_HOME)/include/nvrtc.h" && test -d "$(CUDA_LIB)" || { echo "cuda-toolkit: no CUDA toolkit: $(CUDA_HOME)/include/nvrtc.h or $(CUDA_LIB) is missing"; echo "cuda-toolkit: set CUDA_HOME (candidates: $(CUDA_CANDIDATES)); without a toolkit the build is CPU-only and the gate would pass on it"; exit 1; }

gpu: cuda-toolkit  ## torch/gpu.bend on the GPU (needs a CUDA toolkit in CUDA_HOME): its bits, then MNIST, the tiny Llama and the SmolLM2-sized block on the GPU against PyTorch
	mkdir -p .scratch
	CUDA_HOME=$(CUDA_HOME) LD_LIBRARY_PATH=$(CUDA_LIB) bend demo/gpu_test.bend -o .scratch/gpu_test_cuda
	ldd .scratch/gpu_test_cuda | grep -q libnvrtc && echo "gpu: the test binary links libnvrtc" || { echo "gpu: gpu_test_cuda does not link libnvrtc; the build is CPU-only"; exit 1; }
	test "$$(LD_LIBRARY_PATH=$(CUDA_LIB) .scratch/gpu_test_cuda | tr '\n' ' ')" = "0 0 0 0 0 0 0 0 " && echo "gpu: torch/gpu.bend's layers give main.bend's bits on the GPU" || { echo "gpu: the bits differ on the GPU"; exit 1; }
	CUDA_HOME=$(CUDA_HOME) LD_LIBRARY_PATH=$(CUDA_LIB) bend demo/gpu_layers_test.bend -o .scratch/gpu_layers_test_cuda
	for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do LD_LIBRARY_PATH=$(CUDA_LIB) .scratch/gpu_layers_test_cuda $$k || exit 1; done | tr '\n' ' ' | tee .scratch/gpu_layers.txt; echo
	set -- $$(cat .scratch/gpu_layers.txt); test "$$1" -le 4 && test "$${12}" -le 4 && test "$$2 $$3 $$4 $$5 $$7 $$8 $$9 $${10} $${11} $${14}" = "0 0 0 0 0 0 0 0 0 0" && echo "gpu: the same bits on the GPU, but SiLU ($$1 ulps), cross-entropy ($${12}) and attention ($$6 ulps, at small sizes $${13}, full $${15} and full at small sizes $${16}: no bound, see the next check) through CUDA's float exp" || { echo "gpu: the layers differ on the GPU beyond CUDA's exp"; exit 1; }
	CUDA_HOME=$(CUDA_HOME) scripts/gpu_expcr.sh | tee .scratch/gpu_expcr.txt
	test "$$(tail -1 .scratch/gpu_expcr.txt)" = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 " && echo "gpu: with exp_cr (the host's exp bits) in place of F32.exp, every case gives main.bend's bits on the device: the device code is exact but for CUDA's exp" || { echo "gpu: with exp_cr the device still differs from main.bend"; exit 1; }
	for d in mnist tiny_llama smol_block; do sed 's|^import ../torch/fast.bend as F$$|import ../torch/gpu.bend as F|' demo/$$d.bend > .scratch/$${d}_gpu.bend && grep -q 'torch/gpu.bend' .scratch/$${d}_gpu.bend && CUDA_HOME=$(CUDA_HOME) LD_LIBRARY_PATH=$(CUDA_LIB) bend .scratch/$${d}_gpu.bend -o .scratch/$${d}_gpu || exit 1; done
	LD_LIBRARY_PATH=$(CUDA_LIB) $(PY) reference/test_mnist.py .scratch/mnist_gpu
	LD_LIBRARY_PATH=$(CUDA_LIB) $(PY) reference/test_tiny.py .scratch/tiny_llama_gpu 20
	LD_LIBRARY_PATH=$(CUDA_LIB) $(PY) reference/test_block.py .scratch/smol_block_gpu 4
	LD_LIBRARY_PATH=$(CUDA_LIB) scripts/bits.sh gpu

reader:           ## the weights reader: offsets past 4 GiB (a sparse 8 GiB file in .scratch/reader/) and BF16 files, C lane; MODEL= a BF16 safetensors file adds two of its tensors
	mkdir -p .scratch
	bend demo/reader_test.bend -o .scratch/reader_test
	$(PY) reference/test_reader.py .scratch/reader_test $(if $(MODEL),--model $(MODEL))

check-js:         ## the test programs compiled to JavaScript and run with Bun, against the same oracles; their outputs against bench/bits.txt
	scripts/check_js.sh

bits: build      ## the exact outputs of the Llama (training, generation) and MNIST, 1 and 16 threads, against bench/bits.txt
	scripts/bits.sh check
	scripts/bits.sh rt

mutate:           ## break the code and the specification on purpose; every mutant must be caught (scripts/mutate.py)
	$(PY) scripts/mutate.py

bench:            ## MNIST epoch and tiny-Llama step times, Bend against PyTorch, on 1 and 16 threads (scripts/bench.sh)
	scripts/bench.sh

smol:             ## SmolLM2-135M and 360M against their twins (scripts/smol_data.sh 135m and 360m first; ~13 GB at 360M)
	mkdir -p .scratch
	bend demo/smol.bend -o .scratch/smol
	sed 's|^  T.CAdamW{0.0001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}$$|  T.CSGD{0.01}|' demo/smol.bend > .scratch/smol_sgd.bend && grep -q 'T.CSGD{0.01}' .scratch/smol_sgd.bend && bend .scratch/smol_sgd.bend -o .scratch/smol_sgd
	scripts/smol.sh 360m && bend .scratch/smol_360m.bend -o .scratch/smol_360m
	sed 's|^  T.CAdamW{0.0001, 0.9, 0.999, 0.00000001, 0.01, 0.1, 0.001}$$|  T.CSGD{0.01}|' .scratch/smol_360m.bend > .scratch/smol_360m_sgd.bend && grep -q 'T.CSGD{0.01}' .scratch/smol_360m_sgd.bend && bend .scratch/smol_360m_sgd.bend -o .scratch/smol_360m_sgd
	$(PY) reference/test_smol.py .scratch/smol 135m .scratch/smol_sgd
	$(PY) reference/test_smol.py .scratch/smol_360m 360m .scratch/smol_360m_sgd
	bend demo/reader_test.bend -o .scratch/reader_test && bend demo/safetensors_test.bend -o .scratch/safetensors_test
	$(PY) reference/test_safetensors.py .scratch/safetensors_test --model $(ST135) --init data/smol135/init.bin --reader .scratch/reader_test
	test "$$(.scratch/smol +RTS --threads 16 -RTS loss $(ST135) | sed -E '/^train_ms /d' | sha256sum | cut -c1-16)" = "$$(sed -n 's/^smol loss, threads 16: //p' bench/bits.txt)" && echo "smol: the loss with the weights from model.safetensors, read piece by piece, is bench/bits.txt's (init.bin's)" || { echo "smol: the loss from model.safetensors differs"; exit 1; }
	test "$$(.scratch/smol +RTS --threads 16 -RTS train 2 $(ST135) | sed -E '/^train_ms /d' | sha256sum)" = "$$(.scratch/smol +RTS --threads 16 -RTS train 2 | sed -E '/^train_ms /d' | sha256sum)" && echo "smol: 2 AdamW steps from model.safetensors give the bits of 2 steps from init.bin" || { echo "smol: training from model.safetensors differs from init.bin"; exit 1; }

finetune:         ## 400 steps of SmolLM2-135M on Tiny Shakespeare in Bend and the twin (float32, float64): the held-out loss (~1 h)
	mkdir -p .scratch
	bend demo/smol.bend -o .scratch/smol
	.scratch/smol +RTS --threads 16 -RTS finetune 400 50 > .scratch/finetune_bend.txt
	THREADS=16 $(PY) reference/smol_ref.py finetune 400 50 > .scratch/finetune_torch32.txt
	THREADS=16 $(PY) reference/smol_ref.py finetune64 400 50 > .scratch/finetune_torch64.txt
	$(PY) reference/check_finetune.py .scratch/finetune_bend.txt .scratch/finetune_torch32.txt .scratch/finetune_torch64.txt

smol-gpu: cuda-toolkit  ## SmolLM2-135M with torch/gpu.bend's layers (the import line switched) on the GPU, against its twin as make smol (~5 GB)
	mkdir -p .scratch
	sed 's|^import ../torch/fast.bend as F$$|import ../torch/gpu.bend as F|' demo/smol.bend > .scratch/smol_gpu.bend && grep -q 'torch/gpu.bend' .scratch/smol_gpu.bend && CUDA_HOME=$(CUDA_HOME) LD_LIBRARY_PATH=$(CUDA_LIB) bend .scratch/smol_gpu.bend -o .scratch/smol_gpu
	LD_LIBRARY_PATH=$(CUDA_LIB) $(PY) reference/test_smol.py .scratch/smol_gpu 135m

smol-gen:         ## SmolLM2-135M continues a prompt: greedy against the twin, then drawn with a seed; both again with the key/value cache (WEIGHTS= a finetune <out> file; N=60)
	mkdir -p .scratch
	scripts/smol_gen.sh $(or $(WEIGHTS),data/smol135/init.bin) $(or $(N),60)

own:              ## the tiny Llama on your own text: make own TEXT=file.txt [STEPS=2000] [PROMPT="at least 16 characters"]
	@test -n "$(TEXT)" || { echo "usage: make own TEXT=file.txt [STEPS=2000] [PROMPT=...]"; exit 2; }
	mkdir -p .scratch
	test .scratch/tiny_llama -nt demo/tiny_llama.bend || bend demo/tiny_llama.bend -o .scratch/tiny_llama
	$(PY) scripts/own_text.py "$(TEXT)" --steps $(or $(STEPS),2000) $(if $(PROMPT),--prompt "$(PROMPT)")

mutate-smol:      ## mutants against make smol's test, in a worktree (scripts/mutate_smol.py; ~30 min)
	$(PY) scripts/mutate_smol.py

package:          ## the four BendHub packages into build/packages/, checked and tested over local copies; publishes nothing (CORE=<hub name> VERSION=<a.b.c.d>)
	@test -n "$(CORE)" && test -n "$(VERSION)" || { echo "make package CORE=<the core's hub name> VERSION=<a.b.c.d>"; exit 2; }
	scripts/package.sh $(CORE) $(VERSION)

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate; measure a finding before taking it
	bunx bend-lint --jobs max
