#!/usr/bin/env python3
"""Pre-flight for the RP2 release image: its kernels must reproduce the measured kernels bit for bit.

Mode "save" runs in the r27 image with the MEASURED RP2 tree (b12x-dx2rp2) mounted. It saves the prod and rp2
outputs of real fit-capture routes. Mode "compare" runs in the RELEASE image (its own b12x, no mount). It rebuilds
four arms and requires:
- final (rp2 + EPI-PAR, the image default) == measured rp2;
- rp2 (EPI-PAR off) == measured rp2;
- prod (all flags off) == measured prod;
- prod + EPI-PAR == measured prod.

Everything is compared with torch.equal on layers x ranks x M paths. Both variables are set explicitly for every
arm, so the image's ENV defaults cannot leak into an arm.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402
from dx2_timing import inputs  # noqa: E402

MODE = sys.argv[1]
LAYERS = [int(v) for v in os.environ.get("FI_LAYERS", "3,8,23,43").split(",")]
RANKS = [int(v) for v in os.environ.get("FI_RANKS", "0,3").split(",")]
CELLS = [(1, 0), (1, 1), (1, 2), (1, 3), (4, 0), (16, 0), (64, 0)]
STORE = "/work/results/final_identity_measured.pt"
OUT = "/work/results/final_identity_check.json"
ARMS = {"save": {"prod": ("", ""), "rp2": ("rp2", "")},
        "compare": {"prod": ("", ""), "prodpar": ("", "1"), "rp2": ("rp2", ""), "final": ("rp2", "1")}}[MODE]


def build(overlay, layer, rank, rem, par):
    os.environ["B12X_P8_DOWN_REMAINDER"] = rem
    os.environ["B12X_P8_EPI_PAR"] = par
    record = overlay.records[layer, rank]
    return th.P8NativeTPMoE(overlay.sidecar(layer, rank), device=th.DEV, tp_rank=rank, world_size=4, layer=layer,
                            expected_design_sha256=record["source_design_sha256"],
                            expected_transform_sha256=overlay.transform_hash, topk=8, hidden=4096, intermediate=512,
                            swiglu_limit=10.0, small_m_scheduler=True, fc1_tile_n=128, fuse_scratch_zero=True,
                            prefill_chunk_tokens=0, grid_policy=True, fc1_warp_quant=False, fc1_broadcast_a=True)


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    saved = torch.load(STORE) if MODE == "compare" else {}
    rows, ok = [], True
    for layer in LAYERS:
        for rank in RANKS:
            rts = {arm: build(overlay, layer, rank, *env) for arm, env in ARMS.items()}
            for m, s in CELLS:
                x, w, i, start = inputs(layer, m, s)
                outs = {arm: rt(x, w, i).detach().float().cpu() for arm, rt in rts.items()}
                torch.cuda.synchronize()
                key = f"L{layer}_R{rank}_M{m}_S{s}"
                if MODE == "save":
                    for arm, o in outs.items():
                        saved[f"{key}_{arm}"] = o
                    row = {"cell": key, "rp2_eq_prod": bool(torch.equal(outs["rp2"], outs["prod"]))}
                else:
                    mp, mr = saved[f"{key}_prod"], saved[f"{key}_rp2"]
                    row = {"cell": key, "final_eq_measured_rp2": bool(torch.equal(outs["final"], mr)),
                           "rp2_eq_measured_rp2": bool(torch.equal(outs["rp2"], mr)),
                           "prod_eq_measured_prod": bool(torch.equal(outs["prod"], mp)),
                           "prodpar_eq_measured_prod": bool(torch.equal(outs["prodpar"], mp)),
                           "rp2_differs_from_prod": not bool(torch.equal(mr, mp))}
                    good = all(row[k] for k in ("final_eq_measured_rp2", "rp2_eq_measured_rp2",
                                                "prod_eq_measured_prod", "prodpar_eq_measured_prod"))
                    good = good and (row["rp2_differs_from_prod"] == (m <= 16))
                    row["pass"] = good
                    ok = ok and good
                rows.append(row)
                print(json.dumps(row), flush=True)
            del rts
            torch.cuda.empty_cache()
    if MODE == "save":
        torch.save(saved, STORE)
        print(json.dumps({"saved": len(saved)}), flush=True)
    else:
        json.dump({"cells": rows, "status": "PASS" if ok else "FAIL", "layers": LAYERS, "ranks": RANKS}, open(OUT, "w"), indent=1)
        print(json.dumps({"status": "PASS" if ok else "FAIL", "cells": len(rows)}), flush=True)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
