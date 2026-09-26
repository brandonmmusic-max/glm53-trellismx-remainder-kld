#!/usr/bin/env python3
"""EXPLORATORY (after Test A failed; not preregistered): split the zero-remainder cost.

Same runtime construction, inputs and graph timing as timing_harness.py. Arms, selected by
B12X_P8_ZERO_REMAINDER at compile time:
  base (unset) | zr_both (1) | zr_fc2 (3, down GEMM only) | zr_fc1 (4, gate/up GEMM only)
  zru_fc2 (5): positive control for the FC2-only arm, checked for output change only.
Each block times every timed arm once, in a rotating order; ratios are paired per block.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timing_harness as th  # noqa: E402  (records selected kernel classes; checks cache flags)

ARMS = {"base": None, "zr_both": "1", "zr_fc2": "3", "zr_fc1": "4", "zru_fc2": "5"}
TIMED = ["base", "zr_both", "zr_fc2", "zr_fc1"]


def compile_arm(rt, code, layer):
    if code is None:
        os.environ.pop("B12X_P8_ZERO_REMAINDER", None)
    else:
        os.environ["B12X_P8_ZERO_REMAINDER"] = code
    before = len(th.SELECTED)
    try:
        for m in sorted(set(th.MS)):
            x, w, i, _ = th.inputs(layer, m, 0)
            rt(x, w, i)
            torch.cuda.synchronize()
    finally:
        os.environ.pop("B12X_P8_ZERO_REMAINDER", None)
    return sorted({"/".join(p) for p in th.SELECTED[before:]})


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    out = {"gpu": torch.cuda.get_device_name(0), "blocks": th.BLOCKS, "replays": th.REPLAYS, "layers": {}}
    for layer in th.LAYERS:
        rts, kinds = {}, {}
        for arm, code in ARMS.items():
            rts[arm] = th.runtime(overlay, layer, arm)
            kinds[arm] = compile_arm(rts[arm], code, layer)
            print(json.dumps({"layer": layer, "arm": arm, "kernels": kinds[arm]}), flush=True)
        cells = []
        for m in th.MS:
            for s in range(th.INPUT_SETS):
                x, w, i, start = th.inputs(layer, m, s)
                outs = {arm: rts[arm](x, w, i).clone() for arm in ARMS}
                torch.cuda.synchronize()
                checks = {arm: bool(torch.equal(outs[arm], outs["base"])) for arm in ("zr_both", "zr_fc2", "zr_fc1")}
                control = not bool(torch.equal(outs["zru_fc2"], outs["base"]))
                graphs = {arm: th.graph(rts[arm], x, w, i)[0] for arm in TIMED}
                t_end = time.time() + th.WARMUP_S
                while time.time() < t_end:
                    for arm in TIMED:
                        th.timed(graphs[arm], th.REPLAYS)
                times = {arm: [] for arm in TIMED}
                for b in range(th.BLOCKS):
                    order = TIMED[b % len(TIMED):] + TIMED[:b % len(TIMED)]
                    for arm in order:
                        times[arm].append(th.timed(graphs[arm], th.REPLAYS))
                ratios = {arm: float(np.median(np.asarray(times[arm]) / np.asarray(times["base"]))) for arm in TIMED[1:]}
                cells.append({"m": m, "input_set": s, "exact": checks, "zru_fc2_differs": control, "times_us": times,
                              "median_ratio": ratios})
                print(json.dumps({"layer": layer, "m": m, "set": s, "exact": checks, "control": control,
                                  "base_us": round(float(np.median(times["base"])), 2),
                                  "ratio": {k: round(v, 4) for k, v in ratios.items()}}), flush=True)
                del graphs
        out["layers"][str(layer)] = {"bits": overlay.records[layer, 0]["bits"], "kernels": kinds, "cells": cells}
        del rts
        torch.cuda.empty_cache()
        with open(os.environ.get("ZR_OUT", "/work/results/timing_split_raw.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps({"status": "done"}), flush=True)


if __name__ == "__main__":
    main()
