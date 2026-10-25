"""torch/dtype.bend against PyTorch: BF16 and F16 conversions bit for bit, the packed storage, int8 by rows, and the
double-word accumulator against an exact sum.

    reference/.venv/bin/python reference/test_dtype.py .scratch/dtype_test          # the conversions, packing, int8
    reference/.venv/bin/python reference/test_dtype.py .scratch/dtype_test dw [n] [reps] [sreps]   # long sums

The conversions are judged bit for bit (the exact tier of docs/design.md: one operation, a copy of bits); int8's words,
scales and dequantized weights too (a division, a rounding and one product each); int8's product with F32 activations
sums in another order than PyTorch's, so it is judged by oracle.close.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from oracle import close, parse, run, write_f32

OUT = Path(".scratch")
CHUNK = 4096


def sample() -> np.ndarray:
    """Every F32 class: all 65536 high halves with the low halves that decide BF16's rounding, every F16 exponent with
    the low bits that decide F16's rounding (normal and subnormal ties), random words, and named edge cases."""
    rng = np.random.default_rng(20261018)
    parts = []
    hi = np.arange(65536, dtype=np.uint32) << 16
    for lo in [0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF]:
        parts.append(hi | lo)
    parts.append(hi | rng.integers(0, 65536, 65536, dtype=np.uint32))
    m10 = np.arange(1024, dtype=np.uint32) << 13
    for sign in [0, 0x80000000]:
        for e in range(101, 144):
            base = np.uint32(sign | (e << 23)) | m10
            for lo in [0, 1, 0xFFF, 0x1000, 0x1001, 0x1FFF]:
                parts.append(base | np.uint32(lo))
            parts.append(base | rng.integers(0, 0x2000, 1024, dtype=np.uint32))
        for e in range(102, 113):  # F16 subnormals: the value is mant >> (126 - e) units of 2^-24; ties at its half
            sh = 126 - e
            q = np.arange((1 << 23) >> sh, max(1, (1 << 24) >> sh), dtype=np.uint32)
            for d in [-1, 0, 1]:
                mant = (q << sh) | np.uint32((1 << (sh - 1)) + d)
                parts.append(np.uint32(sign | (e << 23)) | (mant & 0x7FFFFF))
    parts.append(rng.integers(0, 2**32, 1 << 18, dtype=np.uint64).astype(np.uint32))
    edges = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001,
             0x7FBFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FA00000, 0x7F802000, 0x00000001, 0x80000001, 0x007FFFFF,
             0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0x477FE000, 0x477FEFFF, 0x477FF000,
             0x477FF001, 0x47800000, 0x33000000, 0x33000001, 0xB3000000, 0x33800000, 0x387FC000, 0x387FE000,
             0x38800000, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000, 0x3F800001, 0x3F807FFF]
    parts.append(np.array(edges, dtype=np.uint32))
    xs = np.concatenate(parts)
    pad = (-len(xs)) % CHUNK
    return np.concatenate([xs, np.zeros(pad, dtype=np.uint32)])


def u16(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16).astype(np.uint32)


def f32_words(xs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(xs.view(np.float32).copy())


def same_bits(name: str, got: np.ndarray, want: np.ndarray, failures: list[str]) -> None:
    got = np.asarray(got).reshape(-1).view(np.uint32) if got.dtype != np.uint32 else got.reshape(-1)
    want = np.asarray(want, dtype=np.uint32).reshape(-1)
    if got.shape != want.shape:
        failures.append(f"{name}: {got.size} values, expected {want.size}")
        return
    bad = np.nonzero(got != want)[0]
    print(f"{name}: {want.size - bad.size} of {want.size} bit-identical")
    if bad.size:
        i = int(bad[0])
        failures.append(f"{name}: {bad.size} differ, first [{i}]: bend {got[i]:#x}, torch {want[i]:#x}")


def lines(binary: str, *args: str) -> dict[str, list[np.ndarray]]:
    import subprocess
    out = subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout
    got: dict[str, list[np.ndarray]] = {}
    for line in out.splitlines():
        parts = line.split()
        if parts:
            words = [int(w) for w in parts[1:] if w.isdigit()]  # the dw lines also hold the words "ms" and "total"
            got.setdefault(parts[0], []).append(np.array(words, dtype=np.uint64).astype(np.uint32))
    return got


def check_conv(binary: str, failures: list[str]) -> None:
    xs = sample()
    path = OUT / "dtype_in.bin"
    path.write_bytes(xs.astype("<u4").tobytes())
    got = lines(binary, "conv", str(path), str(len(xs) // CHUNK))
    t = f32_words(xs)
    print(f"conv: {len(xs)} F32 words")
    same_bits("F32 -> BF16", np.concatenate(got["bf16"]), u16(t.to(torch.bfloat16)), failures)
    same_bits("F32 -> BF16 (main.bend's packed AdamW moments)", np.concatenate(got["bf16t"]), u16(t.to(torch.bfloat16)), failures)
    same_bits("F32 -> F16", np.concatenate(got["f16"]), u16(t.to(torch.float16)), failures)


def check_halves(binary: str, failures: list[str]) -> None:
    got = lines(binary, "halves")
    h = torch.from_numpy(np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.int16))
    for tag, dt in [("bf16", torch.bfloat16), ("f16", torch.float16)]:
        up = h.view(dt).to(torch.float32)
        same_bits(f"{tag.upper()} -> F32 (all 65536)", got[f"{tag}up"][0], up.numpy().view(np.uint32), failures)
        if tag == "bf16":
            same_bits("BF16 -> F32 (all 65536, main.bend's packed AdamW moments)", got["bf16tup"][0], up.numpy().view(np.uint32), failures)
        same_bits(f"{tag.upper()} -> F32 -> {tag.upper()} (all 65536)", got[f"{tag}rt"][0], u16(up.to(dt)), failures)


def check_pack(binary: str, failures: list[str]) -> None:
    rng = np.random.default_rng(7)
    xs = sample()
    m = np.concatenate([xs[-CHUNK:][:2000], rng.integers(0, 2**32, 64 * 63 - 2000, dtype=np.uint64).astype(np.uint32)])
    path = OUT / "dtype_pack.bin"
    path.write_bytes(m.astype("<u4").tobytes())
    got = lines(binary, "pack", str(path))
    t = f32_words(m)
    for tag, dt in [("bf16", torch.bfloat16), ("f16", torch.float16)]:
        h = t.to(dt).contiguous()
        words = h.view(torch.int32).numpy().view(np.uint32)  # the tensor's bytes: element 2k low, 2k + 1 high
        same_bits(f"{tag.upper()} packed words (64 x 63)", got[f"{tag}words"][0], words, failures)
        same_bits(f"{tag.upper()} packed and unpacked", got[f"{tag}unpack"][0], h.to(torch.float32).numpy().view(np.uint32), failures)


def q8_ref(w: torch.Tensor, x: torch.Tensor):
    s = w.abs().amax(dim=1) / 127.0
    q = torch.where(s[:, None] == 0, torch.zeros_like(w), torch.round(w / s[:, None])).clamp(-127, 127).to(torch.int8)
    pad = (-q.numel()) % 4
    qb = torch.cat([q.reshape(-1), torch.zeros(pad, dtype=torch.int8)])
    words = qb.view(torch.int32).numpy().view(np.uint32)
    deq = q.to(torch.float32) * s[:, None]
    y = (x @ q.to(torch.float32).T) * s
    return words, s, deq, y


def check_q8(binary: str, failures: list[str]) -> None:
    g = torch.Generator().manual_seed(11)
    w = torch.randn(65, 32, generator=g) * 0.02
    w[0] = 0.0                                            # a row of zeros: scale 0, every q 0
    w[1] = torch.tensor([127.0] + [k + 0.5 for k in range(-15, 16)])  # scale 1: every other entry a tie
    w[2, 5] = -1.0                                        # the largest magnitude negative
    x = torch.randn(16, 32, generator=g)
    write_f32(OUT / "q8_w.bin", w)
    write_f32(OUT / "q8_x.bin", x)
    got = lines(binary, "q8", str(OUT / "q8_w.bin"), str(OUT / "q8_x.bin"))
    words, s, deq, y = q8_ref(w, x)
    same_bits("int8 words (65 x 32)", got["q8words"][0], words, failures)
    same_bits("int8 scales", got["q8scales"][0], s.numpy().view(np.uint32), failures)
    same_bits("int8 dequantized", got["q8deq"][0], deq.numpy().view(np.uint32), failures)
    worst = close("int8 x · Wᵀ", got["q8y"][0].view(np.float32), y, failures)
    print(f"int8 x · Wᵀ (16 x 65): worst {worst:.3f} of the tolerance")


# ---- the double-word accumulator ----

def mix32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) & 0xFFFFFFFF
    a = x ^ (x >> 16)
    b = (a * 0x7FEB352D) & 0xFFFFFFFF
    c = b ^ (b >> 15)
    d = (c * 0x846CA68B) & 0xFFFFFFFF
    return (d ^ (d >> 16)).astype(np.uint32)


def uval(n: int, seed: int) -> np.ndarray:
    i = np.arange(n, dtype=np.uint64) * 2 + seed
    h = (mix32(i) >> 8).astype(np.float32)
    return (h * np.float32(2.0**-23) - np.float32(1.0)).astype(np.float32)


def ulps(x: float, exact: float) -> float:
    return abs(x - exact) / np.spacing(np.float32(abs(exact)))


def check_dw(binary: str, n: int, reps: int, sreps: int) -> None:
    got = lines(binary, "+RTS", "--threads", "1", "-RTS", "dw", str(n), str(reps), str(sreps))
    a, b = uval(n, 1), uval(n, 2)
    exact = math.fsum((a.astype(np.float64) * b.astype(np.float64)).tolist())  # each product is exact in float64
    terms = got["terms"][0].view(np.float32)
    sexact = math.fsum(terms.astype(np.float64).tolist())
    ta, tb, tt = torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(terms.copy())
    print(f"dot product of {n}: exact {exact!r}")
    rows = []
    for key, label in [("dot_f32", "Bend, running F32"), ("dot_tree", "Bend, tree (pairwise)"), ("dot_dw", "Bend, double-word")]:
        for i, w in enumerate(got[key]):
            v = float(np.array([w[0]], dtype=np.uint32).view(np.float32)[0])
            ms = int(w[1])
            rows.append((label + (" (again)" if i else ""), v, ms))
    print_rows(rows, exact, reps)
    pt = float(torch.dot(ta, tb))
    print(f"  PyTorch float32 torch.dot: {pt!r}, {ulps(pt, exact):.1f} ulps")
    print(f"softmax sum over 49152 classes: exact {sexact!r}")
    rows = []
    for key, label in [("sum_f32", "Bend, running F32"), ("sum_tree", "Bend, tree (pairwise)"), ("sum_dw", "Bend, double-word")]:
        for i, w in enumerate(got[key]):
            v = float(np.array([w[0]], dtype=np.uint32).view(np.float32)[0])
            rows.append((label + (" (again)" if i else ""), v, int(w[1])))
    print_rows(rows, sexact, sreps)
    ps = float(tt.sum())
    print(f"  PyTorch float32 sum: {ps!r}, {ulps(ps, sexact):.1f} ulps")


def print_rows(rows, exact: float, reps: int) -> None:
    for label, v, ms in rows:
        print(f"  {label}: {v!r}, error {abs(v - exact) / abs(exact):.3e} relative, {ulps(v, exact):.1f} ulps; "
              f"{ms} ms for {reps} ({ms / reps * 1000:.1f} us each)")


def main() -> None:
    binary = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    if sys.argv[2:3] == ["dw"]:
        n, reps, sreps = (int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 20), (int(sys.argv[4]) if len(sys.argv) > 4 else 20), (int(sys.argv[5]) if len(sys.argv) > 5 else 400)
        check_dw(binary, n, reps, sreps)
        return
    failures: list[str] = []
    check_conv(binary, failures)
    check_halves(binary, failures)
    check_pack(binary, failures)
    check_q8(binary, failures)
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("dtype: every conversion bit-exact against PyTorch; int8 within the tolerance")


if __name__ == "__main__":
    main()
