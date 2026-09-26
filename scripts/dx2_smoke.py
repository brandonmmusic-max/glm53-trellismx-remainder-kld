#!/usr/bin/env python3
"""D-x2 smoke (inside the image with b12x-dx2 mounted): every M path compiles and runs;
D-x2 differs from prod by a plausible activation-error-sized amount; flag-off == prod."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/work/scripts")
import timing_harness as th  # records selected kernel classes, checks cache flags

layer = int(os.environ.get("DX2_LAYER", "8"))
torch.cuda.set_device(th.DEV)
overlay = th.load_overlay("/checkpoint")
os.environ.pop("B12X_P8_DOWN_REMAINDER", None)
prod = th.runtime(overlay, layer, "prod")
os.environ["B12X_P8_DOWN_REMAINDER"] = "1"
dx2 = th.runtime(overlay, layer, "dx2")
os.environ.pop("B12X_P8_DOWN_REMAINDER", None)
print("flags", prod.p8_down_remainder, dx2.p8_down_remainder, flush=True)
for m in (1, 4, 64, 512):
    x, w, i, start = th.inputs(layer, m if m <= 16 else 16, 0)
    if m > 16:  # consecutive rows from one fit window
        root = f"{th.CAPTURE}/layers/layer-{layer:03d}"
        hid = np.memmap(f"{root}/hidden.bf16.bin", dtype="<u2", mode="r", shape=(th.ROWS, th.H))
        ids = np.memmap(f"{root}/topk_ids.u16le.bin", dtype="<u2", mode="r", shape=(th.ROWS, th.TOPK))
        wts = np.memmap(f"{root}/topk_weights.f32le.bin", dtype="<f4", mode="r", shape=(th.ROWS, th.TOPK))
        s0 = th.FIT["fit_window_indices"][5] * 2048
        x = torch.from_numpy(np.array(hid[s0:s0 + m], copy=True)).view(torch.bfloat16).to(th.DEV)
        i = torch.from_numpy(np.array(ids[s0:s0 + m], copy=True).astype(np.int32)).to(th.DEV)
        w = torch.from_numpy(np.array(wts[s0:s0 + m], copy=True)).to(th.DEV)
    before = len(th.SELECTED)
    a = prod(x, w, i).float(); b = dx2(x, w, i).float(); a2 = prod(x, w, i).float(); b2 = dx2(x, w, i).float()
    torch.cuda.synchronize()
    rel = float((b - a).norm() / a.norm())
    print(json.dumps({"m": m, "prod_deterministic": bool(torch.equal(a, a2)), "dx2_deterministic": bool(torch.equal(b, b2)),
                      "rel_diff_dx2_vs_prod": round(rel, 5), "finite": bool(torch.isfinite(b).all()),
                      "kernels": sorted({"/".join(p) for p in th.SELECTED[before:]})}), flush=True)
