# Contributing to BendTorch

Thanks for looking at BendTorch. I'm Renan Vinícius (@nuxyel) and I maintain it alone, so I read every issue and pull
request myself. Bug reports, numbers from other machines, new layers, new laws and fixes are all welcome.

## Branches

- **Open pull requests against `devel`.** All work happens there; `main` only receives releases, through a pull
  request from `devel`, and the tags live on `main`. GitHub proposes `main` by default, so change the base branch when
  you open the PR. If you forget, I'll retarget it.
- Branch from `devel` in your fork, not from `main`.

## Before you start

- For anything beyond a small fix (a change to the layer protocol, a new kind of tensor, performance work, a new
  dependency), open an issue first so we can agree on the approach before you spend time on it.
- Read [`NOTES.md`](NOTES.md): it records what was tried, measured and decided, including what did not work.
  [`docs/plan.md`](docs/plan.md) has the milestones, [`docs/design.md`](docs/design.md) the reasons behind the design.
- Bend 2 is young and differs a lot from Bend 1. The "pitfalls" section of [`CLAUDE.md`](CLAUDE.md) lists the traps I
  hit in practice; read it before writing Bend.

## Setup and checks

```bash
make setup   # the pinned Bend 2.0.39, Lean 4.34.0 and the Python venv of reference/
make data    # MNIST and Tiny Shakespeare (not committed), the twins' weights and batches
make laws    # the proofs, by bend and again by --verdict; tests/bad_*.bend must be rejected with their expected error
make check   # the laws, then every layer, the MLP, MNIST and the tiny Llama against PyTorch
make mutate  # deliberate faults, each caught by a law or a test (or listed with the reason it is not)
make bench   # BendTorch against PyTorch on MNIST and the tiny Llama, 1 and 16 threads
```

A pull request should pass `make check`. If it touches `torch/`, `LAWS.bend` or the tests, also run `make mutate`.

## Rules for code

- **Bend stays pinned to 2.0.37.** A version bump is its own change, with an entry in `NOTES.md`.
- **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet goes into an issue, not
  into `LAWS.bend`. The exceptions are three files outside the laws, each with its own package below: `torch/fast.bend`
  and `torch/gpu.bend` (Base's `@unsafe` `Array.fork`, inside bend-ml's packages) and `torch/load.bend` (foreign C and
  JavaScript to read weights); no file with laws imports them, and `make laws` pins their defs.
- **Prove structure, test numbers.** Shapes and composition belong in the types and the laws; `F32` results are checked
  against PyTorch by the scripts in `reference/`, with the one tolerance in `reference/oracle.py`. Never widen it to
  make a test pass; find out why the numbers differ.
- **A new layer comes with its test**: a section in `demo/ops_test.bend` and the same computation in
  `reference/test_ops.py`, written with PyTorch's own functions.
- **Every proved law is explained in plain language** in [`torch/README.md`](torch/README.md).

## Benchmarks

Performance claims need numbers someone else can reproduce: the hardware, the Bend and compiler versions, an idle
machine, runs alternated between the old and the new version, one-thread timings pinned to a single core. Report the
losses and the noise too, not only the best run. A speedup that changes a test's result is a bug until explained.

## Packages

The code lives in `torch/` as one folder, which the Makefile, the laws, the mutants and the tests use. On BendHub it is
four packages, split by what each one guarantees:

| Package | From | What it guarantees |
|---|---|---|
| `bendtorch` | `torch/main.bend`, `spec.bend`, `tensor.bend`, `dtype.bend`, with `LAWS.bend` and `PROOF.bend` | laws checked by `--verdict`; no `@unsafe`, no foreign code; depends only on bend-ml-tensor-array and bend-ml-nat-lemmas |
| `bendtorch-fast` | `torch/fast.bend` | the same bits as the core, tested; uses bend-ml-tensor-fork (`@unsafe` in Base) |
| `bendtorch-gpu` | `torch/gpu.bend` | the core's bits on the CPU, and on the GPU except CUDA's float `exp`; uses bend-ml-tensor-gpu |
| `bendtorch-load` | `torch/load.bend` | loading only; uses bend-ml-binary-read (foreign C and JavaScript) |

`make package CORE=<name> VERSION=<a.b.c.d>` runs `scripts/package.sh`, which assembles them into `build/packages/`
without changing any code: the satellites' `import ./main.bend` becomes `import <core>@<version>/main.bend`, and the
laws' `./torch/x.bend` becomes `./x.bend`. Then it checks the hub's name and path rules, that the core has no `@unsafe`
and no foreign code (`--verdict` on the whole core), and that the satellites check exactly as `torch/` does; and with
the cross-package imports pointed at local copies it builds `fast_test`, `gpu_test`, `infra_test`, `dtype_test` and the
tiny Llama over the packages and compares them with the repository's results.

The core's entry is a generated `package.bend` that imports all of its files and the laws, because `bend --publish`
sends one file and what it imports. Users import the files themselves (`bendtorch@<version>/main.bend`, and each
satellite's `main.bend`), all at the same version; the README's "Use it" has the imports. The script never publishes; that is done by hand from each folder.

## Commits and pull requests

- Small commits, one logical change each, with clear messages in English.
- Keep a pull request to one topic.
- Don't commit datasets, `data/`, `.scratch/` or large logs.
- Using AI tools is fine. You are still the author: say in the PR what you ran yourself and what you did not, and keep
  the description short enough for a person to read.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE) of this repository.
