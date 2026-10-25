"""torch/load.bend's safetensors reader (L.tensors) through demo/safetensors_test.bend.

  Small files written here by hand (the venv has no safetensors package; the format is 8 bytes of header length, a JSON
  header and the data): F32 and BF16 tensors from PyTorch, a scalar, an extra tensor and names the layout does not use,
  headers of odd lengths padded with spaces, indented JSON, the metadata first or last, data stored in another order
  than the header's. Every number read must be the file's F32, or bf16.to(torch.float32), bit for bit. Then the cases
  that must stop the program with a message naming the tensor: a wrong shape, a missing name, an unread dtype, offsets
  that do not span the tensor, data cut short, a layout whose total differs from the count, a malformed header.
  With --model FILE (SmolLM2-135M's model.safetensors) and --init FILE (data/smol135/init.bin), the whole model read
  through reference/smol_layout.txt's layout hashes to the same value as init.bin read by L.read (reader_test).

  reference/.venv/bin/python reference/test_safetensors.py .scratch/safetensors_test [--js] [--model F --init F --reader R]
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def run(binary, *args):
    p = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def write(path, tensors, order=None, meta="first", indent=None, pad=0, extra=b""):
    """tensors: [(name, dtype, shape, raw bytes)]; order: the order of the data blocks (names); pad: spaces after the
    JSON; extra: bytes after the data. Returns the header's length."""
    order = order or [t[0] for t in tensors]
    by = {t[0]: t for t in tensors}
    offs, at = {}, 0
    for name in order:
        offs[name] = (at, at + len(by[name][3]))
        at += len(by[name][3])
    head = {name: {"dtype": dt, "shape": list(shape), "data_offsets": list(offs[name])} for name, dt, shape, _ in tensors}
    items = list(head.items())
    if meta == "first":
        items = [("__metadata__", {"format": "pt", "note": "a \"quoted\" value, a \\ and a: colon"})] + items
    elif meta == "last":
        items = items + [("__metadata__", {"format": "pt"})]
    text = json.dumps(dict(items), indent=indent).encode() + b" " * pad
    Path(path).write_bytes(struct.pack("<Q", len(text)) + text + b"".join(by[n][3] for n in order) + extra)
    return len(text)


def f32(name, shape, g):
    t = torch.randn(*shape, generator=g) if shape else torch.randn((), generator=g)
    raw = t.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    return (name, "F32", shape, raw), np.frombuffer(raw, "<u4").tolist()


def bf16(name, shape, g):
    t = (torch.randn(*shape, generator=g) * 3).to(torch.bfloat16)
    raw = t.contiguous().view(torch.int16).numpy().tobytes()
    bits = (t.to(torch.float32).view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF).reshape(-1).tolist()
    return (name, "BF16", shape, raw), bits


def spec(name, shape):
    return f"{name}:{'x'.join(map(str, shape))}"


def nan_js(bits, js):
    # as test_infra.py records: the JavaScript lane's F32 is a Number, and a NaN loses its payload there
    return [0x7FC00000 if (b & 0x7F800000) == 0x7F800000 and b & 0x7FFFFF else b for b in bits] if js else bits


def check_good(binary, tmp, js):
    g = torch.Generator().manual_seed(1021)
    a, abits = f32("model.a.weight", (3, 5), g)
    b, bbits = bf16("model.b.weight", (7, 2), g)
    c, cbits = f32("model.c", (), g)
    d, dbits = bf16("model.d.bias", (9,), g)
    x, _ = f32("unused \"name\" with \\ and : signs", (4,), g)
    layout = [(d, dbits), (a, abits), (c, cbits), (b, bbits)]  # not the file's order
    n = sum(len(bits) for _, bits in layout)
    want = [w for _, bits in layout for w in bits]
    specs = [spec(t[0], t[2]) for t, _ in layout]
    cases = 0
    lens = set()
    for meta in ["first", "last", None]:
        for indent in [None, 2]:
            for pad in [0, 1, 3, 6]:
                path = tmp / "good.safetensors"
                order = [b[0], d[0], x[0], a[0], c[0]]
                lens.add(write(path, [a, x, b, c, d], order=order, meta=meta, indent=indent, pad=pad, extra=b"\0" * 5))
                code, out, err = run(binary, "bits", path, n, *specs)
                assert code == 0, (meta, indent, pad, code, out, err)
                got = out.partition(":")
                assert int(got[0]) == n and list(map(int, got[2].split())) == nan_js(want, js), (meta, indent, pad, out)
                cases += 1
    odd = sum(1 for h in lens if h % 2)
    print(f"safetensors: {cases} files (metadata first, last or absent; compact or indented JSON; {len(lens)} header "
          f"lengths, {odd} odd; data in another order, an extra tensor, escaped names) give {n} numbers bit for bit, "
          "F32 as stored and BF16 as bf16.to(torch.float32)")
    return a, b, c, d, abits


def check_bad(binary, tmp, a, b, c, d):
    def fails(name, tensors, specs, n, words, **kw):
        path = tmp / f"bad_{name}.safetensors"
        if isinstance(tensors, bytes):
            path.write_bytes(tensors)
        else:
            write(path, tensors, **kw)
        code, out, err = run(binary, "bits", path, n, *specs)
        msg = out + err
        assert code != 0 and all(w in msg for w in words), (name, code, msg)
        return msg

    sa, sb = spec(a[0], a[2]), spec(b[0], b[2])
    msgs = [
        fails("shape", [a, b], [spec(a[0], (5, 3)), sb], 29, [a[0], "shape [3, 5] in the file, [5, 3] expected"]),
        fails("missing", [a, b], [sa, "model.z.weight:2x2"], 19, ["model.z.weight", "not in the file"]),
        fails("dtype", [a, ("model.h", "F16", (2,), b"\x00" * 4)], [sa, "model.h:2"], 17, ["model.h", "dtype F16"]),
        fails("span", [a, (b[0], b[1], b[2], b[3][:14])], [sa, sb], 29, [b[0], "data_offsets do not span"]),
        fails("total", [a, b], [sa, sb], 30, ["layout holds 29 numbers, the model 30"]),
    ]
    # data cut short: the last tensor's bytes end early
    path = tmp / "bad_short.safetensors"
    write(path, [a, b])
    raw = path.read_bytes()
    msgs.append(fails("short", raw[:-6], [sa, sb], 29, [b[0], "short read"]))
    # a header cut in the middle of a string, a header that is not an object, a file shorter than 8 bytes
    head = b'{"model.a.weight":{"dtype":"F32","shape":[3,5],"data_offsets":[0,60]}'
    msgs.append(fails("open", struct.pack("<Q", len(head)) + head + b"\x00" * 60, [sa], 15, ["malformed"]))
    head = b'{"model.a.weight":{"dtype":"F32","shape":[3,5],"data_offsets":[0,60],"x":true}}'
    msgs.append(fails("literal", struct.pack("<Q", len(head)) + head + b"\x00" * 60, [sa], 15, ["malformed"]))
    msgs.append(fails("tiny", b"\x05\x00\x00", [sa], 15, ["shorter than a safetensors header"]))
    msgs.append(fails("cut", struct.pack("<Q", 1000) + b"{}", [sa], 15, ["header is cut short"]))
    print(f"safetensors: {len(msgs)} bad files or layouts stop the program with a message naming the cause:")
    for m in msgs:
        print("  " + m.splitlines()[-1])


def check_model(binary, model, init, reader):
    specs = [f"{l.split(chr(9))[0]}:{l.split(chr(9))[1]}" for l in Path("reference/smol_layout.txt").read_text().splitlines()
             if not l.startswith("#")]
    total = int(Path("reference/smol_layout.txt").read_text().splitlines()[-1].split("\t")[1])
    code, out, err = run(binary, "hash", model, total, *specs)
    assert code == 0, (code, out, err)
    code2, out2, err2 = run(reader, "read", init, total)
    assert code2 == 0 and out == out2, (out, out2, err2)
    print(f"safetensors: SmolLM2-135M's model.safetensors through smol_layout.txt's {len(specs)} tensors hashes as "
          f"init.bin does ({out})")


def main():
    binary = sys.argv[1]
    js = "--js" in sys.argv[2:]
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    tmp = Path(".scratch/safetensors")
    tmp.mkdir(parents=True, exist_ok=True)
    a, b, c, d, _ = check_good(binary, tmp, js)
    check_bad(binary, tmp, a, b, c, d)
    if arg("--model"):
        check_model(binary, arg("--model"), arg("--init"), arg("--reader"))
    print("safetensors: PASS" + (" (JavaScript)" if js else ""))


if __name__ == "__main__":
    main()
