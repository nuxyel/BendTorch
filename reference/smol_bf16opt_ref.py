"""SmolLM2's twin (reference/smol_ref.py) trained by AdamW with its moments stored in BF16, as T.CAdamWBF16 does.

    reference/.venv/bin/python reference/smol_bf16opt_ref.py <m|mv> train <steps>    # float32: "loss" lines in bits
    reference/.venv/bin/python reference/smol_bf16opt_ref.py <m|mv> train64 <steps>  # float64 arithmetic, BF16 moments

smol_ref.py's own loop, with torch.optim.AdamW replaced by test_mlp.py's AdamW16 (torch.optim.AdamW's single-tensor
step with each moment rounded to BF16 after its update; v too for mv). Compare with a Bend build of demo/smol.bend whose
cfg() is T.CAdamWBF16{..., v16} (make smol's sed, with CAdamWBF16 in place of CSGD).
"""

import sys

import smol_ref as R
from test_mlp import AdamW16

which = sys.argv.pop(1)
assert which in ("m", "mv"), which
R.torch.optim.AdamW = lambda params, **kw: AdamW16(params, v16=(which == "mv"), **kw)
R.main()
