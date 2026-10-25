"""The weights reader past 4 GiB and on BF16 files (bend-ml-binary-read@0.2.0.0, torch/load.bend), through demo/reader_test.bend.

  (a) F32 and BF16 numbers at offsets just below and above 2^31, 2^32 and 2^33 of a sparse file of 8 GiB + 6 bytes
      (only a few pages are written), a read cut short by the end of the file, a read past it, a hole, the file's size;
      the package's read_f32 with its U32 offset still reads up to 4 GiB; torch/load.bend's loaders stop on short reads.
  (b) a BF16 file written by PyTorch (tensor.to(torch.bfloat16), raw bytes) at offset 0 and at an offset that is not a
      multiple of 4: every value read equals bf16.to(torch.float32) bit for bit.
  (c) with --model FILE (a BF16 safetensors file), the tensor at the largest offset and one past 2^31 are read through
      torch/load.bend's read_bf16_at and compared, by a hash of every bit, with PyTorch's widening of the same bytes.
  Also W64 (torch/load.bend's 64-bit byte counts in two U32s): parsing, sums, small products, division by 10, printing.

  reference/.venv/bin/python reference/test_reader.py .scratch/reader_test [--js] [--model FILE]   (make reader)
"""

import json
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

GIB = 1 << 30
NINE = 0x41100000  # 9.0, the value the raw probes' cells start with


def runner(binary):
    def run(*args, ok=True):
        p = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
        if ok:
            assert p.returncode == 0, (args, p.returncode, p.stdout, p.stderr)
        else:
            assert p.returncode != 0, (args, "should have failed", p.stdout)
        return p.stdout.strip()
    return run


def parse(line):
    """'count: b0 b1 ..' -> (count, [b0, b1, ..])"""
    head, _, rest = line.partition(":")
    return int(head), [int(w) for w in rest.split()]


def js_bits(bits, js):
    # Bend's JavaScript F32 is a Number, and this runtime canonicalizes NaN payloads (as test_infra.py records); native
    # keeps them. Every other bit, -0 and subnormals included, must survive on both lanes.
    if not js:
        return bits
    return [0x7FC00000 if (b & 0x7F800000) == 0x7F800000 and b & 0x7FFFFF else b for b in bits]


def cells(n):
    return 1 << max(0, (n - 1).bit_length())


def expect_f32(data, off, n):
    """read_f32_at's answer on `data` (the file's bytes as a function of offset and length)"""
    raw = data(off, 4 * n)
    whole = len(raw) // 4
    out = list(struct.unpack(f"<{whole}I", raw[:4 * whole])) + [NINE] * (n - whole)
    part = len(raw) % 4
    if part:  # a last partial number writes only its bytes over the cell
        out[whole] = struct.unpack("<I", raw[4 * whole:] + struct.pack("<I", NINE)[part:])[0]
    return whole, out


def expect_bf16(data, off, n):
    raw = data(off, 2 * n)
    whole = len(raw) // 2
    return whole, [h << 16 for h in struct.unpack(f"<{whole}H", raw[:2 * whole])] + [NINE] * (n - whole)


def fnv(words):
    h = 2166136261
    for w in words:
        h = ((h ^ int(w)) * 16777619) & 0xFFFFFFFF
    return h


def check_w64(run):
    rng = random.Random(2026)
    cases = [(0, 0), (9, 1), (4294967295, 1), (4294967296, 65535), (18446744073709551615, 0), (10 ** 18 + 7, 10),
             (4294967295, 65535), (18446744069414584320, 65535)]
    cases += [(rng.randrange(1 << rng.randrange(1, 64)), rng.randrange(65536)) for _ in range(60)]
    for a, k in cases:
        if len(str(a)) > 19:  # 20 digits: W64.read refuses them (every 19-digit number fits; that is enough)
            run("w64", a, k, ok=False)
            continue
        want = f"{(a + k) % 2 ** 64} {(a * k) % 2 ** 64} {a // 10} {a % 10}"
        assert run("w64", a, k) == want, (a, k, run("w64", a, k), want)
    for bad in ["", "12a", "-1", "1 2"]:
        run("w64", bad, 1, ok=False)
    print(f"reader: W64 parses, adds, multiplies, divides by 10 and prints {len(cases)} cases as Python's integers")


def check_sparse(run, scratch, js):
    big = scratch / "big.bin"
    size = 8 * GIB + 6
    rng = random.Random(4)
    special = [0x3F800000, 0xC0200000, 0x80000000, 0x7F800000, 0xFF800000, 0x00000001, 0x007FFFFF, 0x7FC12345,
               0x7F800001, 0x3DCCCCCD]
    pieces = {2 * GIB - 8: 16, 4 * GIB - 8: 16, 4 * GIB + 4097: 12, 4 * GIB + 8192: 40, 8 * GIB - 10: 16}
    written = {}
    with open(big, "wb") as f:
        f.truncate(size)  # sparse: only the pieces below take disk space
        for at, length in pieces.items():
            words = special[: length // 4] if at == 4 * GIB + 8192 else [rng.getrandbits(32) for _ in range(length // 4)]
            raw = struct.pack(f"<{len(words)}I", *words)[:length]
            f.seek(at)
            f.write(raw)
            written[at] = raw

    def data(off, length):  # the file's bytes, from what was written; the rest is the sparse file's zeros
        end = min(off + length, size)
        out = bytearray(max(0, end - off))
        for at, raw in written.items():
            lo, hi = max(at, off), min(at + len(raw), end)
            if lo < hi:
                out[lo - off:hi - off] = raw[lo - at:hi - at]
        return bytes(out)

    with open(big, "rb") as f:  # what was written is what the file holds
        for at, raw in written.items():
            f.seek(at)
            assert f.read(len(raw)) == raw
    assert run("size", big) == str(size), run("size", big)
    probes = [
        ("f32", 2 * GIB - 8, 4), ("f32", 4 * GIB - 8, 4), ("f32", 4 * GIB - 4, 1), ("f32", 4 * GIB, 2),
        ("f32", 4 * GIB + 4097, 3), ("f32", 4 * GIB + 8192, 10), ("f32", 8 * GIB - 8, 4), ("f32", 8 * GIB - 8, 3),
        ("f32", 3 * GIB, 5), ("f32", 16 * GIB, 2), ("f32", 4 * GIB - 8, 0),
        ("bf16", 2 * GIB - 6, 6), ("bf16", 4 * GIB - 6, 6), ("bf16", 4 * GIB + 4097, 6), ("bf16", 4 * GIB + 8192, 20),
        ("bf16", 8 * GIB - 9, 8), ("bf16", 8 * GIB - 10, 8), ("bf16", 3 * GIB, 3), ("bf16", 16 * GIB, 2),
    ]
    for kind, off, n in probes:
        whole, bits = (expect_f32 if kind == "f32" else expect_bf16)(data, off, n)
        got, cells_ = parse(run(kind, big, off, n))
        assert (got, cells_) == (whole, js_bits(bits, js)), (kind, off, n, got, cells_, whole, bits)
    # the package's read_f32 still reads with its U32 offset, up to the 4 GiB boundary
    got, cells_ = parse(run("old", big, 4 * GIB - 8, 2))
    assert (got, cells_) == (8, js_bits(expect_f32(data, 4 * GIB - 8, 2)[1], js)), (got, cells_)
    # torch/load.bend's loaders: a whole read prints the numbers, a short one stops the program
    assert parse(run("load_f32", big, 4 * GIB - 8, 4)) == (4, js_bits(expect_f32(data, 4 * GIB - 8, 4)[1], js))
    assert parse(run("load_bf16", big, 4 * GIB - 6, 6)) == (6, js_bits(expect_bf16(data, 4 * GIB - 6, 6)[1], js))
    run("load_f32", big, 8 * GIB - 8, 4, ok=False)
    run("load_bf16", big, 8 * GIB - 9, 8, ok=False)
    run("load_f32", big, 16 * GIB, 1, ok=False)
    run("read", big, 1, ok=False)  # not an exact-size file
    print(f"reader: {len(probes) + 3} reads around 2 GiB, 4 GiB and 8 GiB of a sparse {size}-byte file give the file's"
          " bits (short reads, a hole, past the end, the old U32 offset); the loaders stop on short reads")


def check_bf16(run, scratch, js):
    g = torch.Generator().manual_seed(1021)
    x = torch.cat([torch.randn(10007, generator=g) * 10.0 ** torch.randint(-30, 30, (10007,), generator=g),
                   torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.0e38, -3.0e38, 1.0, -1.0,
                                 1e-45, 9.2e-41])])
    b = x.to(torch.bfloat16)
    raw = b.view(torch.int16).numpy().tobytes()
    want = (b.to(torch.float32).view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF).tolist()
    n = len(want)
    for prefix in [0, 6]:
        path = scratch / f"bf16_{prefix}.bin"
        path.write_bytes(b"\x55" * prefix + raw)
        for mode in ["bf16", "load_bf16"]:
            got, cells_ = parse(run(mode, path, prefix, n))
            assert got == n and cells_ == js_bits(want, js), (mode, prefix, got, [
                (i, c, w) for i, (c, w) in enumerate(zip(cells_, want)) if c != w][:5])
    # the whole file as F32 words, through read: the exact-size loader over a file of 2 BF16 numbers a word
    words = n // 2
    (scratch / "bf16_words.bin").write_bytes(raw[:4 * words])
    expect = js_bits(np.frombuffer(raw[:4 * words], "<u4").tolist(), js)  # an inf next to a nonzero half is a NaN word
    assert run("read", scratch / "bf16_words.bin", words) == f"{words} {fnv(expect)}"
    print(f"reader: {n} BF16 numbers written by PyTorch read at offsets 0 and 6 equal bf16.to(torch.float32) bit for bit"
          " (raw and through torch/load.bend)")


def check_model(run, path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    tensors = sorted(((v["data_offsets"][0] + base, k, v) for k, v in header.items() if k != "__metadata__"))
    size = Path(path).stat().st_size
    assert run("size", path) == str(size)
    picks = [tensors[-1]]  # the largest offset
    past = [t for t in tensors if t[0] >= 2 * GIB and t[0] != tensors[-1][0]]
    if past:
        picks.append(past[0])  # the first tensor past 2^31 (where a signed 32-bit offset would go negative)
    for at, name, v in picks:
        assert v["dtype"] == "BF16", (name, v["dtype"])
        n = int(np.prod(v["shape"]))
        with open(path, "rb") as f:
            f.seek(at)
            raw = f.read(2 * n)
        want = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float32).view(torch.int32).numpy()
        out = run("hash_bf16", path, at, n)
        assert out == f"{n} {fnv(want.view(np.uint32))}", (name, out)
        print(f"reader: {name} ({n} BF16 numbers at byte {at} of a {size}-byte safetensors file) equals PyTorch's"
              " widening, hashed over every bit")


def main():
    binary = sys.argv[1]
    js = "--js" in sys.argv[2:]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
    scratch = Path(".scratch/reader")
    scratch.mkdir(parents=True, exist_ok=True)
    run = runner(binary)
    check_w64(run)
    check_sparse(run, scratch, js)
    check_bf16(run, scratch, js)
    if model:
        check_model(run, model)
    print("reader: PASS" + (" (JavaScript)" if js else ""))


if __name__ == "__main__":
    main()
