"""The training state of SmolLM2 in BendTorch's arrays, from the parameter layout (reference/smol_layout.txt's pattern),
under several storage choices (plan M11). Arithmetic, not a measurement.

    reference/.venv/bin/python reference/smol_memory.py [135m | 360m | 1.7b]

A Bend array is one block of 2^k cells (NOTES, 2026-10-17), so each tensor takes the next power of two of its count;
an F32 cell is 4 bytes, and a packed BF16 matrix (torch/dtype.bend's HMat) holds two values a word, so it takes the
next power of two of half its count, 4 bytes a word. The head is tied to the embedding, as in demo/smol.bend.
"""

import sys

SIZES = {  # width, layers, query heads, key/value heads, head size, feed-forward, vocabulary
    "135m": (576, 30, 9, 3, 64, 1536, 49152),
    "360m": (960, 32, 15, 5, 64, 2560, 49152),
    "1.7b": (2048, 24, 32, 32, 64, 8192, 49152),
}


def layout(size):
    d, layers, h, kv, hd, ff, v = SIZES[size]
    out = [("model.embed_tokens.weight", v * d)]
    for i in range(layers):
        p = f"model.layers.{i}."
        out += [(p + "input_layernorm.weight", d), (p + "self_attn.q_proj.weight", h * hd * d),
                (p + "self_attn.k_proj.weight", kv * hd * d), (p + "self_attn.v_proj.weight", kv * hd * d),
                (p + "self_attn.o_proj.weight", d * h * hd), (p + "post_attention_layernorm.weight", d),
                (p + "mlp.gate_proj.weight", ff * d), (p + "mlp.up_proj.weight", ff * d),
                (p + "mlp.down_proj.weight", d * ff)]
    return out + [("model.norm.weight", d)]


def cap(n):
    return 1 << (n - 1).bit_length()


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "1.7b"
    t = layout(size)
    n = sum(c for _, c in t)
    f32 = sum(cap(c) * 4 for _, c in t)       # one F32 buffer of every parameter, at the arrays' capacity
    bf16 = sum(cap((c + 1) // 2) * 4 for _, c in t)  # one packed BF16 buffer
    gib = 2.0**30
    print(f"SmolLM2-{size}: {len(t)} tensors, {n:,} parameters; one F32 buffer {f32 / gib:.2f} GiB "
          f"({f32 / (4 * n):.3f}x the numbers), one packed BF16 buffer {bf16 / gib:.2f} GiB")
    rows = [
        ("F32 everything (today): weights, gradient, two moments", 4 * f32),
        ("BF16 values in F32 arrays (tiny_mixed's CastP/CastOut): the same buffers", 4 * f32),
        ("F32 master weights and moments, packed BF16 gradient", 3 * f32 + bf16),
        ("CAdamWBF16 (v16 false): F32 weights, gradient and v, packed BF16 m", 3 * f32 + bf16),
        ("CAdamWBF16 (v16 true): F32 weights and gradient, packed BF16 m and v", 2 * f32 + 2 * bf16),
        ("F32 master weights, packed BF16 gradient and moments", f32 + 3 * bf16),
        ("packed BF16 weights, gradient and moments (no F32 master)", 4 * bf16),
    ]
    print("AdamW state (parameters, gradient, m, v), before activations and the runtime's other memory:")
    for name, b in rows:
        print(f"  {name}: {b / gib:.2f} GiB")
    print(f"weights file: {4 * n / 1e9:.2f} GB in F32, {2 * n / 1e9:.2f} GB in BF16 "
          f"(the binary reader's byte offsets are U32: at most {2**32 / 1e9:.2f} GB)")
    print(f"loading: the source array of the F32 file takes {cap(n) * 4 / gib:.2f} GiB next to the parameters")


if __name__ == "__main__":
    main()
