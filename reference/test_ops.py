"""demo/ops_test.bend against torch.autograd: every layer on its own, forward and backward.

For each seed, random inputs (with zeros, negative zeros and saturating values mixed in) go to the Bend program and to
PyTorch's own functions; the output, dX and every parameter gradient must agree within the shared tolerance (oracle.py),
and the results in EXACT (copies, selections, one or two F32 operations in PyTorch's order) bit for bit.

    reference/.venv/bin/python reference/test_ops.py .scratch/ops_test
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import erf_model
from oracle import bits_equal, close, run, write_f32, write_u8

EPS = 1e-5
EDGE_OPS = {"relu", "sigmoid", "tanh", "silu", "gelu", "gelu_erf"}
# the row-wise layers, softmax inside attention and the losses meet the edge cases a row at a time (ROW_EDGE)
ROW_OPS = {"softmax", "logsoftmax", "rmsnorm", "layernorm"}
# The exact tier: results that are a copy, a selection or one or two F32 operations in PyTorch's order, so they must be
# the same bits, NaN included (oracle.py's tolerance still applies to everything else)
EXACT = {"relu.y", "relu.dx", "embedding.y", "mse.dy", "nll.dy", "dropout0.y", "dropout0.dx", "dropout3.y", "dropout3.dx", "dropout4.y", "dropout4.dx"}


def rmsnorm(x, g):
    var = x.pow(2).mean(-1, keepdim=True)
    return g * (x * torch.rsqrt(var + EPS))


def groupnorm(x, g, b, n=2, groups=3):
    """nn.GroupNorm on BendTorch's channels-last layout: (n·hw) x c in and out, (n, c, hw) inside."""
    c = x.shape[1]
    xt = x.reshape(n, -1, c).permute(0, 2, 1)
    return F.group_norm(xt, groups, g, b, EPS).permute(0, 2, 1).reshape(x.shape)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(x, t=4, heads=2, hd=4, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = (torch.arange(x.shape[0]) % t).float()
    freqs = (inv[:, None] @ pos[None, :]).T  # transformers' outer product
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    xh = x.reshape(x.shape[0], heads, hd)
    return (xh * cos + rotate_half(xh) * sin).reshape(x.shape)


def attention_full(q, k, v, b=2, t=4, h=4, kv=2, hd=4, scale=0.5):
    """The same as attention() without the mask: every position sees every key."""
    q = q.reshape(b, t, h, hd).transpose(1, 2)
    k = k.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    v = v.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    s = (q @ k.transpose(-1, -2)) * scale
    return (s.softmax(-1) @ v).transpose(1, 2).reshape(b * t, h * hd)


def attention(q, k, v, b=2, t=4, h=4, kv=2, hd=4, scale=0.5):
    q = q.reshape(b, t, h, hd).transpose(1, 2)
    k = k.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    v = v.reshape(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    o = s.softmax(-1) @ v
    return o.transpose(1, 2).reshape(b * t, h * hd)


# every IEEE class a layer can meet: NaN, both infinities, both zeros, subnormals, values where exp overflows or
# underflows in F32 (|x| > 88.7), and ordinary values
EDGE = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1e-40, -1e-40, 1.5e-45, 89.0, -89.0, 104.0, -104.0,
        1e6, -1e6, 20.0, -20.0, 1e-7, -1e-7, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0, 0.1, -0.1, 2.0, -2.0]


INF, NAN = float("inf"), float("nan")
# two sets of five rows: "A" the non-finite classes (a NaN, a +inf, -inf among finite values, a row of -inf, two +inf),
# "B" the finite extremes (sums of squares and exp that overflow, subnormals, signed zeros, exp's overflow and underflow)
ROW_EDGE = {
    "A": [[NAN, 1.0, 2.0, 3.0, 4.0, 5.0], [INF, 1.0, -1.0, 0.0, 2.0, -2.0], [-INF, 0.5, -INF, 2.0, -INF, -1.0],
          [-INF] * 6, [INF, INF, 0.0, 1.0, 2.0, 3.0]],
    "B": [[3e38, -3e38, 1e38, 2e38, -1e38, 1.0], [1e-40, -1e-40, 1.5e-45, 0.0, -0.0, 1e-39], [-0.0] * 6,
          [89.0, -89.0, 104.0, -104.0, 1e6, -1e6], [1e-7, -1e-7, 20.0, -20.0, 0.5, -0.5]],
}


def edge_attention(g, kind):
    """Attention inputs with the edge cases in fixed places: (q 8x16, k 8x8, v 8x8); 2 sequences of 4 positions."""
    q, k, v = torch.randn(8, 16, generator=g), torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)
    if kind == "A":
        q[0, 0] = NAN       # sequence 0, position 0, query head 0
        k[2, 1] = INF       # sequence 0, position 2: an infinite score for the later positions of heads 0 and 1
        v[5, 3] = -INF      # sequence 1, position 1
        k[7, 4] = NAN       # sequence 1, position 3, key head 1
    else:
        q[1] = 1e20         # scores of 1e20: a softmax that is one-hot
        k[3, :4] = 3e38     # q·k overflows to infinity
        v[4] = 1e-40        # subnormal values
        q[6] = -0.0
        k[6] = 0.0
    return q, k, v


def mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(2146121005)
    x ^= x >> np.uint32(15)
    x *= np.uint32(2221713035)
    x ^= x >> np.uint32(16)
    return x


def dropout_factors(n, seed, step, p):
    """BendTorch's dropout mask: number i is kept (times 1/(1-p)) when ((mix32(key + i) >> 8) / 2^24) >= p."""
    key = mix32(np.uint32(seed) ^ mix32(np.uint32(step)))
    with np.errstate(over="ignore"):
        h = mix32(key + np.arange(n, dtype=np.uint32))
    u = (h >> np.uint32(8)).astype(np.float32) / np.float32(16777216.0)
    scale = np.float32(1.0) / (np.float32(1.0) - np.float32(p))
    return torch.from_numpy(np.where(u < np.float32(p), np.float32(0.0), scale).astype(np.float32))


def erf_inputs(g):
    """4096 inputs for erf: the IEEE classes, both sides of every boundary between erf's pieces (0.84375, 1.25, 1.75,
    2.25, 2.75, 3.25, 3.92, where erf starts to round to 1), then normal values, uniform values over the pieces and
    magnitudes from 2^-40 to 2^4 with random signs."""
    cuts = [0.84375, 1.25, 1.75, 2.25, 2.75, 3.25, 3.92, 3.9192047, 3.91922]
    special = [NAN, INF, -INF, 0.0, -0.0, 1.5e-45, -1.5e-45, 1e-40, -1e-40, 2.0 ** -126, -(2.0 ** -126), 1e-30, -1e-30,
               1e-10, -1e-10, 1e-7, -1e-7, 3.4e38, -3.4e38]
    for c in cuts:
        c32 = np.float32(c)
        for v in (np.nextafter(c32, np.float32(0)), c32, np.nextafter(c32, np.float32(9))):
            special += [float(v), -float(v)]
    n = 4096 - len(special)
    q = n // 4
    mags = torch.pow(2.0, torch.rand(n - 3 * q, generator=g) * 44 - 40)
    signs = torch.where(torch.rand(n - 3 * q, generator=g) < 0.5, -1.0, 1.0)
    rest = torch.cat([torch.randn(2 * q, generator=g) * 2, torch.rand(q, generator=g) * 9 - 4.5, mags * signs])
    return torch.cat([torch.tensor(special, dtype=torch.float32), rest.float()]).reshape(64, 64)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distance between two F32 arrays in units in the last place (NaN against NaN is 0, NaN against a number is
    2^40); -0 and +0 are one apart, so a lost sign of zero counts."""
    def line(u):
        u = u.astype(np.int64)
        return np.where(u & 0x80000000, -(u & 0x7fffffff) - 1, u)
    a, b = np.asarray(a, np.float32).reshape(-1), np.asarray(b, np.float32).reshape(-1)
    d = np.abs(line(a.view(np.uint32)) - line(b.view(np.uint32)))
    d[np.isnan(a) & np.isnan(b)] = 0
    d[np.isnan(a) ^ np.isnan(b)] = 1 << 40
    return d


def spiced(g, shape):
    """Normal values with a few exact zeros, a negative zero and large values (saturation) in fixed places."""
    x = torch.randn(*shape, generator=g)
    flat = x.view(-1)
    flat[0], flat[1], flat[2], flat[3] = 0.0, -0.0, 12.0, -12.0
    return x
# GroupNorm's (image, group) blocks of 8 numbers (4 pixels x 2 channels, in the order they are summed) take ROW_EDGE's
# classes, a block each: blocks 0-2 are image 0's groups 0-2, blocks 3-5 image 1's
GN_EDGE = {
    "A": [[NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [INF, 1.0, -1.0, 0.0, 2.0, -2.0, 0.5, 3.0],
          [-INF, 0.5, -INF, 2.0, -INF, -1.0, 1.0, 1.0], [-INF] * 8, [INF, INF, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
          [0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0]],
    "B": [[3e38, -3e38, 1e38, 2e38, -1e38, 1.0, 0.5, -0.5], [1e-40, -1e-40, 1.5e-45, 0.0, -0.0, 1e-39, 1e-40, -1.5e-45],
          [-0.0] * 8, [89.0, -89.0, 104.0, -104.0, 1e6, -1e6, 20.0, -20.0], [1e-7, -1e-7, 20.0, -20.0, 0.5, -0.5, 0.1, -0.1],
          [0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0]],
}


def gn_edge(kind):
    """GN_EDGE's blocks laid out channels last: block 3s + k at rows 4s..4s+3, columns 2k, 2k+1."""
    x = torch.zeros(8, 6)
    for blk, vals in enumerate(GN_EDGE[kind]):
        s, k = divmod(blk, 3)
        x[4 * s:4 * s + 4, 2 * k:2 * k + 2] = torch.tensor(vals).reshape(4, 2)
    return x


def case(binary: str, seed: int, failures: list[str], edge: str = "") -> tuple[float, int, int]:
    """edge "A" or "B": ROW_EDGE's rows into the row-wise layers and the losses, edge_attention's inputs, and (A) EDGE
    into the element-wise activations; the other layers get the usual random inputs."""
    g = torch.Generator().manual_seed(seed)
    files: dict[str, torch.Tensor] = {}
    ref: dict[str, torch.Tensor] = {}

    def leaf(name, t):
        files[name] = t
        return t.clone().requires_grad_(True)

    def unary(tag, fn, x_shape=(5, 6), **params):
        if edge == "A" and tag in EDGE_OPS:
            x0 = torch.tensor(EDGE).reshape(5, 6)
        elif edge and tag in ROW_OPS:
            x0 = torch.tensor(ROW_EDGE[edge])
        elif edge and tag == "groupnorm":
            x0 = gn_edge(edge)
        else:
            x0 = spiced(g, x_shape)
        x = leaf(f"{tag}.x", x0)
        ps = {k: leaf(f"{tag}.{k}", v) for k, v in params.items()}
        y = fn(x, **ps)
        dy = torch.randn(*y.shape, generator=g)
        files[f"{tag}.dy"] = dy
        y.backward(dy)
        ref[f"{tag}.y"], ref[f"{tag}.dx"] = y.detach(), x.grad
        for k, p in ps.items():
            ref[f"{tag}.g.{k}" if len(ps) > 1 else f"{tag}.g"] = p.grad

    unary("relu", torch.relu)
    unary("sigmoid", torch.sigmoid)
    unary("tanh", torch.tanh)
    unary("silu", F.silu)
    unary("gelu", lambda x: F.gelu(x, approximate="tanh"))
    unary("softmax", lambda x: F.softmax(x, -1))
    unary("logsoftmax", lambda x: F.log_softmax(x, -1))
    unary("rmsnorm", rmsnorm, g=torch.randn(6, generator=g))
    unary("layernorm", lambda x, g, b: F.layer_norm(x, (6,), g, b, EPS),
          g=torch.randn(6, generator=g), b=torch.randn(6, generator=g))
    if edge == "B":
        # Accepted for now (Renan, 2026-10-18; the trade-offs are to be tested and compared later): row 0 of set B is
        # [3e38, -3e38, 1e38, 2e38, -1e38, 1], whose squares overflow. RMSNorm: mean(x^2) = inf, r = 1/sqrt(inf) = 0, and
        # BendTorch's dx = r (g dy - x r^2 mean(g dy x)) is 0, where PyTorch's autograd forms 0 * inf = NaN. (y is 0 in
        # both.) BendTorch's row 0 of dx is expected, every other number stays PyTorch's.
        ref["rmsnorm.dx"] = ref["rmsnorm.dx"].clone()
        ref["rmsnorm.dx"][0] = 0.0
        # LayerNorm: the variance overflows to inf, r = 0, and BendTorch's row 0 is y = b and dx = 0, so row 0 adds
        # nothing to g.g, where PyTorch gives NaN for all three. Rows are independent: the other rows, and g.g over them,
        # are PyTorch's on rows 1-4 alone; g.b (the sum of dy) is PyTorch's own.
        x1, g1, b1 = (files[f"layernorm.{k}"].clone().requires_grad_(True) for k in ("x", "g", "b"))
        y1 = F.layer_norm(x1[1:], (6,), g1, b1, EPS)
        y1.backward(files["layernorm.dy"][1:])
        ref["layernorm.y"] = torch.cat([files["layernorm.b"][None, :], y1.detach()])
        ref["layernorm.dx"] = torch.cat([torch.zeros(1, 6), x1.grad[1:]])
        ref["layernorm.g.g"] = g1.grad
    unary("linear_nb", lambda x, w: F.linear(x, w), w=torch.randn(4, 6, generator=g) * 0.5)
    unary("swiglu", lambda x, wg, wu, wd: F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd),
          wg=torch.randn(8, 6, generator=g) * 0.5, wu=torch.randn(8, 6, generator=g) * 0.5,
          wd=torch.randn(6, 8, generator=g) * 0.5)
    unary("residual", lambda x, g: x + rmsnorm(x, g), g=torch.randn(6, generator=g))
    unary("rope", rope, x_shape=(8, 8))
    for step in (0, 3, 4):
        k = dropout_factors(30, 7, step, 0.25).reshape(5, 6)
        unary(f"dropout{step}", lambda x, k=k: x * k)

    if edge:
        q0, k0, v0 = edge_attention(g, edge)
    else:
        q0, k0, v0 = torch.randn(8, 16, generator=g), torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)
    q, k, v = leaf("attn.q", q0), leaf("attn.k", k0), leaf("attn.v", v0)
    y = attention(q, k, v)
    dy = torch.randn(8, 16, generator=g)
    files["attn.dy"] = dy
    y.backward(dy)
    ref.update({"attn.y": y.detach(), "attn.dq": q.grad, "attn.dk": k.grad, "attn.dv": v.grad})
    # the same q, k, v through FullAttention: no mask, so every row sums over all t positions. The edge inputs are not
    # used here (an -inf in a visible key makes the reference itself NaN in ways the causal case documents), so the
    # full blocks are compared only on the random inputs.
    if not edge:
        qf = leaf("attn.q", q0.detach().clone())
        kf = leaf("attn.k", k0.detach().clone())
        vf = leaf("attn.v", v0.detach().clone())
        yf = attention_full(qf, kf, vf)
        yf.backward(dy)
        ref.update({"attnf.y": yf.detach(), "attnf.dq": qf.grad, "attnf.dk": kf.grad, "attnf.dv": vf.grad})
    if edge == "A":
        # Accepted for now (Renan, 2026-10-18; to be tested and compared later): v[5, 3] = -inf sits at position 1 of
        # sequence 1, which row 4 (position 0) cannot see. Row 4 has one visible position, so dS = P (dP - P dP) = 0 and
        # BendTorch's dq there is 0 for the two query heads of key/value head 0 (columns 0-7); PyTorch also multiplies the
        # masked probability 0 by dP = dy·v5 = inf and gets NaN. Expected: PyTorch's own dq for row 4 with v[5, 3] made
        # finite (row 4 does not depend on it). (Masked keys are not ignored the same way: k[7, 4] = NaN makes BendTorch's
        # dq[4, 8] and dq[4, 12] NaN, as PyTorch's, and that stays compared.)
        v2 = v0.clone()
        v2[5, 3] = 0.0
        q2, k2, v2 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True), v2.requires_grad_(True)
        attention(q2, k2, v2).backward(dy)
        assert torch.isnan(ref["attn.dq"][4, :8]).all()
        ref["attn.dq"] = ref["attn.dq"].clone()
        ref["attn.dq"][4, :8] = q2.grad[4, :8]

    e = leaf("embedding.e", torch.randn(7, 4, generator=g))
    ids = torch.tensor([3, 0, 3, 6, 3])  # repeated ids: their gradients add up
    y = F.embedding(ids, e)
    dy = torch.randn(5, 4, generator=g)
    files["embedding.dy"] = dy
    y.backward(dy)
    ref.update({"embedding.y": y.detach(), "embedding.g": e.grad})

    e, w = leaf("tied.e", torch.randn(7, 4, generator=g)), leaf("tied.w", torch.randn(4, 4, generator=g))
    y = F.linear(F.linear(F.embedding(ids, e), w), e)  # the table is used twice: its gradients add up
    dy = torch.randn(5, 7, generator=g)
    files["tied.dy"] = dy
    y.backward(dy)
    ref.update({"tied.y": y.detach(), "tied.g.e": e.grad, "tied.g.w": w.grad})

    labels = torch.randint(0, 6, (5,), generator=g)
    if edge:
        labels = torch.tensor([0, 1, 2, 3, 4])  # row r's label is column r: the edge value at a known place
    for tag, fn, x0, t in [("mse", F.mse_loss, "y", torch.randn(5, 6, generator=g)),
                           ("nll", F.nll_loss, "x", labels), ("ce", F.cross_entropy, "x", labels)]:
        x = leaf(f"{tag}.{x0}", torch.tensor(ROW_EDGE[edge]) if edge else torch.randn(5, 6, generator=g))
        if tag == "mse":
            files["mse.t"] = t
        loss = fn(x, t)
        loss.backward()
        ref[f"{tag}.loss"], ref[f"{tag}.dy"] = loss.detach().reshape(1), x.grad

    # exact GELU (PyTorch's default, approximate="none") and erf on its own, drawn last so that the layers above keep
    # the inputs they had before these were added
    unary("gelu_erf", lambda x: F.gelu(x, approximate="none"))
    if edge == "A" and torch.isnan(ref["gelu_erf.y"][0, 1]):
        # GELU(+inf) is +inf (x times a probability that reaches 1), as on my machine; the same PyTorch (2.14.1, the CPU
        # wheel, AVX2 kernels) gives NaN on some CI runners, now and then. Expected: +inf, in that cell only.
        ref["gelu_erf.y"] = ref["gelu_erf.y"].clone()
        ref["gelu_erf.y"][0, 1] = float("inf")
    files["erf.x"] = erf_inputs(g)
    ref["erf.y"] = torch.special.erf(files["erf.x"].double())
    # GroupNorm, drawn after erf for the same reason: 2 images of 4 pixels, channels last (8 x 6), 3 groups of 2 channels
    unary("groupnorm", groupnorm, x_shape=(8, 6), g=torch.randn(6, generator=g), b=torch.randn(6, generator=g))
    if edge == "B":
        # LayerNorm's accepted case (Renan, 2026-10-18; extended to GroupNorm, 2026-10-30), on GroupNorm's kernel: block 0 (image 0, channels 0-1) holds
        # 3e38-sized values whose squares overflow, so its variance is inf and r = 0: BendTorch's y there is b and its dx
        # 0, and the block adds nothing to g.g, where PyTorch gives NaN for all three. The other blocks are PyTorch's,
        # and g.g is PyTorch's with block 0's dy at 0 (its x made finite, which no other block reads); g.b is PyTorch's own.
        m = torch.zeros(8, 6, dtype=torch.bool)
        m[0:4, 0:2] = True
        x1 = torch.where(m, 1.0, files["groupnorm.x"]).requires_grad_(True)
        g1, b1 = (files[f"groupnorm.{k}"].clone().requires_grad_(True) for k in ("g", "b"))
        y1 = groupnorm(x1, g1, b1)
        y1.backward(torch.where(m, 0.0, files["groupnorm.dy"]))
        ref["groupnorm.y"] = torch.where(m, files["groupnorm.b"].expand(8, 6), ref["groupnorm.y"])
        ref["groupnorm.dx"] = torch.where(m, 0.0, ref["groupnorm.dx"])
        ref["groupnorm.g.g"] = g1.grad

    with tempfile.TemporaryDirectory() as d:
        for name, t in files.items():
            write_f32(Path(d) / f"{name}.bin", t)
        write_u8(Path(d) / "embedding.ids.bin", ids.tolist())
        write_u8(Path(d) / "nll.labels.bin", labels.tolist())
        got = run(binary, d)

    # erf is held to more than the shared tolerance: within 1 ulp of float64's erf rounded to F32 (its measured bound
    # over all 2^32 inputs is 0.93 ulp) and NaN for NaN; and, as an exact tier, the bits of erf_model.py, the same
    # operations in numpy (a change below an ulp, or a lost sign of zero, shows there)
    if "erf.y" in got:
        xs = files["erf.x"].reshape(-1)
        d = ulps(got["erf.y"], ref["erf.y"].float().numpy())
        if d.max() > 1:
            i = int(d.argmax())
            failures.append(f"seed {seed} erf.y[{i}]: erf({xs[i].item()!r}) is {d[i]} ulps from float64's")
        d = ulps(got["erf.y"], erf_model.erf(xs.numpy()))
        if d.max() > 0:
            i = int(d.argmax())
            failures.append(f"seed {seed} erf.y: {int((d > 0).sum())} of {d.size} numbers not erf_model.py's bits (exact "
                            f"tier), the first at erf({xs[int(np.flatnonzero(d)[0])].item()!r})")

    worst, same, total = 0.0, 0, 0
    for name, r in ref.items():
        bend_name = name
        if bend_name not in got:
            failures.append(f"seed {seed}: the Bend program printed no {bend_name}")
            continue
        worst = max(worst, close(f"seed {seed} {name}", got[bend_name], r, failures))
        if name in EXACT:
            s_, t_ = bits_equal(got[bend_name], r)
            if s_ != t_:
                failures.append(f"seed {seed} {name}: {t_ - s_} of {t_} numbers not bit-identical (exact tier)")
        s, t = bits_equal(got[bend_name], r)
        same, total = same + s, total + t
    return worst, same, total


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    worst, same, total = 0.0, 0, 0
    for seed in range(10):
        w, s, t = case(binary, seed, failures)
        worst, same, total = max(worst, w), same + s, total + t
    edge_failures: list[str] = []
    case(binary, 100, edge_failures, edge="A")
    case(binary, 101, edge_failures, edge="B")
    print(f"ops: 10 seeds, {total} numbers, {same} bit-identical to torch, worst error {worst:.3f} of the tolerance")
    print(f"ops: IEEE edge cases (NaN, infinities, signed zeros, subnormals, overflow) on {sorted(EDGE_OPS)}, "
          f"{sorted(ROW_OPS)}, groupnorm, attention and the losses (mse, nll, ce): "
          f"{'PASS' if not edge_failures else str(len(edge_failures)) + ' differences'}")
    failures += edge_failures
    if failures:
        print("\n".join(failures[:30]))
        sys.exit(1)
    print("ops: PASS")


if __name__ == "__main__":
    main()
