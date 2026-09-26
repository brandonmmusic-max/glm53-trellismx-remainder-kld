#!/usr/bin/env python3
"""Post-audit RP2 smoke plus guard test (run inside the image with the guarded tree, b12x-dx2rp2-guarded, mounted).

Part 1 repeats rp2_smoke.py and saves its output (the original run printed to the console only). The guarded tree's
kernels are byte-identical to the measured b12x-dx2rp2 tree; only host-side configuration checks were added.
Part 2 checks that the guards reject the flag combinations the remainder code does not support.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402
from dx2_timing import inputs  # noqa: E402

LAYER = int(os.environ.get("RP2_LAYER", "8"))
OUT = os.environ.get("RP2_OUT", "/work/results/rp2_guard_smoke.json")


def build(overlay, env, **over):
    kw = dict(topk=8, hidden=4096, intermediate=512, swiglu_limit=10.0, small_m_scheduler=True, fc1_tile_n=128,
              fuse_scratch_zero=True, prefill_chunk_tokens=0, grid_policy=True, fc1_warp_quant=False, fc1_broadcast_a=True)
    kw.update(over)
    record = overlay.records[LAYER, 0]
    if env:
        os.environ["B12X_P8_DOWN_REMAINDER"] = env
    try:
        return th.P8NativeTPMoE(overlay.sidecar(LAYER, 0), device=th.DEV, tp_rank=0, world_size=4, layer=LAYER,
                                expected_design_sha256=record["source_design_sha256"],
                                expected_transform_sha256=overlay.transform_hash, **kw)
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    out = {"gpu": torch.cuda.get_device_name(0), "layer": LAYER, "smoke": [], "guards": []}
    rts = {"prod": build(overlay, None), "rp": build(overlay, "rp"), "rp2": build(overlay, "rp2")}
    for m in (1, 4, 16, 64, 512):
        x, w, i, start = inputs(LAYER, m, 0)
        o = {k: rt(x, w, i).float() for k, rt in rts.items()}
        again = rts["rp2"](x, w, i).float()
        torch.cuda.synchronize()
        rel = lambda a, b: float((o[a] - o[b]).norm() / o[b].norm())  # noqa: E731
        row = {"m": m, "start_row": int(start), "rp2_deterministic": bool(torch.equal(again, o["rp2"])),
               "finite": bool(torch.isfinite(o["rp2"]).all()), "rel_rp_vs_prod": rel("rp", "prod"),
               "rel_rp2_vs_prod": rel("rp2", "prod"), "rel_rp2_vs_rp": rel("rp2", "rp"),
               "rp_bit_identical_prod": bool(torch.equal(o["rp"], o["prod"])),
               "rp2_bit_identical_prod": bool(torch.equal(o["rp2"], o["prod"]))}
        out["smoke"].append(row)
        print(json.dumps(row), flush=True)
    del rts
    torch.cuda.empty_cache()

    x, w, i, _ = inputs(LAYER, 1, 0)
    cases = (("rp2 + fc1_warp_quant", "rp2", {"fc1_warp_quant": True}, "fc1_warp_quant"),
             ("rp2 without fc1_broadcast_a", "rp2", {"fc1_broadcast_a": False}, "fc1_broadcast_a"),
             ("rp + tile_major_tasks", "rp", {"tile_major_tasks": True}, "one route per direct tile"),
             ("plane D-x2 + fc1_warp_quant", "1", {"fc1_warp_quant": True}, "fc1_warp_quant"))
    for name, env, over, expect in cases:
        try:
            rt = build(overlay, env, **over)
            rt(x, w, i)
            torch.cuda.synchronize()
            got = "no error"
        except Exception as exc:  # the guard raises RuntimeError at the first call on the direct path
            got = f"{type(exc).__name__}: {exc}"
        row = {"case": name, "expected_substring": expect, "result": got[:300], "rejected": expect in got}
        out["guards"].append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    smoke_ok = all(r["rp2_deterministic"] and r["finite"] and (r["rp2_bit_identical_prod"] == (r["m"] > 16)) for r in out["smoke"])
    out["status"] = "PASS" if smoke_ok and all(r["rejected"] for r in out["guards"]) else "FAIL"
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"status": out["status"]}), flush=True)
    if out["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
