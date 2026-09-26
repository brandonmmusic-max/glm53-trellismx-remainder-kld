#!/usr/bin/env python3
"""D-x2-RP2 smoke (inside the image with b12x-dx2rp2 mounted): every M path compiles and runs;
RP2 differs from prod and from RP on direct paths (M<=16); M>16 paths are bit-identical to prod."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402
from dx2_timing import inputs  # noqa: E402

layer = int(os.environ.get("RP2_LAYER", "8"))
torch.cuda.set_device(th.DEV)
overlay = th.load_overlay("/checkpoint")


def make(env):
    if env:
        os.environ["B12X_P8_DOWN_REMAINDER"] = env
    try:
        return th.runtime(overlay, layer, env or "prod")
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)


rts = {"prod": make(None), "rp": make("rp"), "rp2": make("rp2")}
print("flags", {k: (getattr(v, "p8_dx2_rowpack", False), getattr(v, "p8_input_rowpack", False)) for k, v in rts.items()}, flush=True)
for m in (1, 4, 16, 64, 512):
    x, w, i, _ = inputs(layer, m, 0)
    o = {k: rt(x, w, i).float() for k, rt in rts.items()}
    again = rts["rp2"](x, w, i).float()
    torch.cuda.synchronize()
    rel = lambda a, b: round(float((o[a] - o[b]).norm() / o[b].norm()), 5)
    print(json.dumps({"m": m, "rp2_deterministic": bool(torch.equal(again, o["rp2"])), "finite": bool(torch.isfinite(o["rp2"]).all()),
                      "rel_rp_vs_prod": rel("rp", "prod"), "rel_rp2_vs_prod": rel("rp2", "prod"), "rel_rp2_vs_rp": rel("rp2", "rp"),
                      "rp2_bit_identical_prod": bool(torch.equal(o["rp2"], o["prod"]))}), flush=True)
