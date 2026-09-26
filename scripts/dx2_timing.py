#!/usr/bin/env python3
"""A2 (PREREG amendment 2): real D-x2 kernel timing vs production kernels.

Same runtime construction and CUDA-graph interleaved-block timing as Test A (rank 0 sidecars,
real routes and routed-block inputs from the fit capture). Arms: prod (flag off) and dx2
(B12X_P8_DOWN_REMAINDER=1 at construction).
Decode M in {1, 4, 16} uses 4 input sets. Prefill M in {64, 512, 2048} uses 2 input sets of
consecutive rows inside one fit window. Blocks alternate the arm order.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402

MS = [int(v) for v in os.environ.get("A2_MS", "1,4,16,64,512,2048").split(",")]
LAYERS = [int(v) for v in os.environ.get("A2_LAYERS", "8,3").split(",")]
BLOCKS = int(os.environ.get("A2_BLOCKS", "60"))
REPLAYS = int(os.environ.get("A2_REPLAYS", "200"))
OUT = os.environ.get("A2_OUT", "/work/results/a2_timing_raw.json")
ARM_ENV = os.environ.get("A2_ARM", "1")  # "1" = plane D-x2 (A2), "rp" = row-packed D-x2-RP (A3)


def make_runtime(overlay, layer, dx2):
    if dx2:
        os.environ["B12X_P8_DOWN_REMAINDER"] = ARM_ENV
    try:
        return th.runtime(overlay, layer, "dx2" if dx2 else "prod")
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)


def inputs(layer, m, s):
    if m <= 16:
        x, w, i, start = th.inputs(layer, m, s)
        return x, w, i, start
    root = f"{th.CAPTURE}/layers/layer-{layer:03d}"
    hid = np.memmap(f"{root}/hidden.bf16.bin", dtype="<u2", mode="r", shape=(th.ROWS, th.H))
    ids = np.memmap(f"{root}/topk_ids.u16le.bin", dtype="<u2", mode="r", shape=(th.ROWS, th.TOPK))
    wts = np.memmap(f"{root}/topk_weights.f32le.bin", dtype="<f4", mode="r", shape=(th.ROWS, th.TOPK))
    window = th.FIT["fit_window_indices"][(7 * s + 3) % len(th.FIT["fit_window_indices"])]
    start = window * 2048 + (0 if m == 2048 else 97 * (s + 1))
    x = torch.from_numpy(np.array(hid[start:start + m], copy=True)).view(torch.bfloat16).to(th.DEV)
    i = torch.from_numpy(np.array(ids[start:start + m], copy=True).astype(np.int32)).to(th.DEV)
    w = torch.from_numpy(np.array(wts[start:start + m], copy=True)).to(th.DEV)
    return x, w, i, start


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    out = {"gpu": torch.cuda.get_device_name(0), "blocks": BLOCKS, "replays": REPLAYS, "layers": {}}
    for layer in LAYERS:
        prod = make_runtime(overlay, layer, False)
        dx2 = make_runtime(overlay, layer, True)
        flag = "p8_dx2_rowpack" if ARM_ENV == "rp" else "p8_down_remainder"
        if not (getattr(dx2, flag) and not getattr(prod, flag)):
            raise RuntimeError("arm flags wrong")
        cells = []
        for m in MS:
            for s in range(4 if m <= 16 else 2):
                x, w, i, start = inputs(layer, m, s)
                a, b = prod(x, w, i).float(), dx2(x, w, i).float()
                torch.cuda.synchronize()
                rel = float((b - a).norm() / a.norm())
                gp, _ = th.graph(prod, x, w, i)
                gd, _ = th.graph(dx2, x, w, i)
                reps = REPLAYS  # PREREG: >= 60 blocks x 200 replays per cell, every M
                t_end = time.time() + th.WARMUP_S
                while time.time() < t_end:
                    th.timed(gp, reps); th.timed(gd, reps)
                tp, td = [], []
                for blk in range(BLOCKS):
                    if blk % 2 == 0:
                        tp.append(th.timed(gp, reps)); td.append(th.timed(gd, reps))
                    else:
                        td.append(th.timed(gd, reps)); tp.append(th.timed(gp, reps))
                ratio = float(np.median(np.asarray(td) / np.asarray(tp)))
                cells.append({"m": m, "input_set": s, "start_row": int(start), "replays": reps, "prod_us": tp,
                              "dx2_us": td, "median_ratio": ratio, "rel_diff_dx2_vs_prod": rel})
                print(json.dumps({"layer": layer, "m": m, "set": s, "prod_us": round(float(np.median(tp)), 2),
                                  "dx2_us": round(float(np.median(td)), 2), "ratio": round(ratio, 4),
                                  "rel_diff": round(rel, 4)}), flush=True)
                del gp, gd
                torch.cuda.empty_cache()
        out["layers"][str(layer)] = {"bits": overlay.records[layer, 0]["bits"], "cells": cells}
        del prod, dx2
        torch.cuda.empty_cache()
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps({"status": "done"}), flush=True)


if __name__ == "__main__":
    main()
