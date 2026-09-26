#!/usr/bin/env python3
"""Test A (PREREG): is a second FP8 MMA per decoded weight fragment free at decode?

Runs inside the production image with the scratch b12x bind-mounted over
/opt/glm53-flash/b12x. Builds P8NativeTPMoE exactly as vLLM's TrellisMXMoEMethod does,
for one TP rank of one routed layer, and feeds it real routed-block inputs and top-8
routes from the fit capture (consecutive tokens, like an MTP verify step).

Arms (selected by B12X_P8_ZERO_REMAINDER at compile time, then frozen):
  base : production kernels
  zr   : + second mxfp8 MMA per fragment, activation fragment zeroed at runtime
  zru  : positive control, second MMA consumes the real activation fragment
Checks: zr output == base output exactly; zru output != base output.
Timing: CUDA-graph replay, interleaved ABBA blocks, per-block Z/base ratio.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from b12x.moe._shared.kernels import route_hoist_dynamic as rhd
from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE
from vllm.utils.trellismx import load_overlay

# Record which phase-kernel classes each backend instance actually selected.
SELECTED: list[tuple[str, str]] = []
_backend_init = rhd.MoEDynamicKernelBackend.__init__


def _recording_init(self, *args, **kwargs):
    _backend_init(self, *args, **kwargs)
    SELECTED.append((type(getattr(self, "materialized_phase1_kernel", None)).__name__,
                     type(getattr(self, "materialized_phase2_kernel", None)).__name__))


rhd.MoEDynamicKernelBackend.__init__ = _recording_init
for _flag in ("B12X_COMPILE_MEMORY_CACHE", "B12X_COMPILE_DISK_CACHE"):
    if os.environ.get(_flag) != "0":
        sys.exit(f"{_flag}=0 is required: compile caches key on the spec, not the ZR arm")

CAPTURE = "<data-volume>/p8-remainder-kvrot-20260923/capture"
FIT = json.load(open("/work/fit_ranges.json"))
ROWS, H, TOPK = 1310720, 4096, 8
LAYERS = [int(v) for v in os.environ.get("ZR_LAYERS", "8,3").split(",")]
MS = [int(v) for v in os.environ.get("ZR_MS", "1,4,16").split(",")]
BLOCKS = int(os.environ.get("ZR_BLOCKS", "30"))
REPLAYS = int(os.environ.get("ZR_REPLAYS", "50"))
INPUT_SETS = int(os.environ.get("ZR_INPUT_SETS", "4"))
OUT = os.environ.get("ZR_OUT", "/work/results/timing_raw.json")
WARMUP_S = float(os.environ.get("ZR_WARMUP_S", "2.0"))
DEV = torch.device("cuda:0")


def inputs(layer: int, m: int, set_idx: int):
    windows = FIT["fit_window_indices"]
    window = windows[(set_idx * 17 + m) % len(windows)]
    start = window * 2048 + 64 + 97 * set_idx
    root = f"{CAPTURE}/layers/layer-{layer:03d}"
    hid = np.memmap(f"{root}/hidden.bf16.bin", dtype="<u2", mode="r", shape=(ROWS, H))
    ids = np.memmap(f"{root}/topk_ids.u16le.bin", dtype="<u2", mode="r", shape=(ROWS, TOPK))
    wts = np.memmap(f"{root}/topk_weights.f32le.bin", dtype="<f4", mode="r", shape=(ROWS, TOPK))
    x = torch.from_numpy(np.array(hid[start:start + m], copy=True)).view(torch.bfloat16).to(DEV)
    i = torch.from_numpy(np.array(ids[start:start + m], copy=True).astype(np.int32)).to(DEV)
    w = torch.from_numpy(np.array(wts[start:start + m], copy=True)).to(DEV)
    return x, w, i, start


def runtime(overlay, layer: int, arm: str):
    record = overlay.records[layer, 0]
    return P8NativeTPMoE(
        overlay.sidecar(layer, 0), device=DEV, tp_rank=0, world_size=4, layer=layer,
        expected_design_sha256=record["source_design_sha256"],
        expected_transform_sha256=overlay.transform_hash,
        topk=8, hidden=4096, intermediate=512, swiglu_limit=10.0, small_m_scheduler=True,
        fc1_tile_n=128, fuse_scratch_zero=True, prefill_chunk_tokens=0, grid_policy=True,
        fc1_warp_quant=False, fc1_broadcast_a=True,
    )


def compile_arm(rt, arm: str, layer: int):
    """First call per M path compiles; the env var picks the kernel class at that moment."""
    env = {"base": None, "zr": "1", "zru": "2"}[arm]
    if env is None:
        os.environ.pop("B12X_P8_ZERO_REMAINDER", None)
    else:
        os.environ["B12X_P8_ZERO_REMAINDER"] = env
    classes = {}
    before = len(SELECTED)
    try:
        for m in sorted(set(MS)):
            x, w, i, _ = inputs(layer, m, 0)
            t0 = time.time()
            rt(x, w, i)
            torch.cuda.synchronize()
            classes[m] = round(time.time() - t0, 1)
    finally:
        os.environ.pop("B12X_P8_ZERO_REMAINDER", None)
    return classes, sorted({"/".join(pair) for pair in SELECTED[before:]})


def graph(rt, x, w, i):
    rt(x, w, i)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        rt(x, w, i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = rt(x, w, i)
    torch.cuda.synchronize()
    return g, out


def timed(g, n: int) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / n  # microseconds per MoE call


def main() -> None:
    torch.cuda.set_device(DEV)
    overlay = load_overlay("/checkpoint")
    results = {"gpu": torch.cuda.get_device_name(0), "blocks": BLOCKS, "replays": REPLAYS,
               "input_sets": INPUT_SETS, "warmup_s": WARMUP_S, "layers": {}}
    for layer in LAYERS:
        rts, meta = {}, {}
        for arm in ("base", "zr", "zru"):
            rts[arm] = runtime(overlay, layer, arm)
            meta[arm] = compile_arm(rts[arm], arm, layer)
            print(json.dumps({"layer": layer, "arm": arm, "compile_s": meta[arm][0], "kernels": meta[arm][1]}), flush=True)
        layer_res = {"bits": overlay.records[layer, 0]["bits"], "compile": meta, "cells": []}
        for m in MS:
            for s in range(INPUT_SETS):
                x, w, i, start = inputs(layer, m, s)
                outs = {arm: rts[arm](x, w, i).clone() for arm in ("base", "zr", "zru")}
                repeat = rts["base"](x, w, i).clone()
                torch.cuda.synchronize()
                deterministic = bool(torch.equal(repeat, outs["base"]))
                exact = bool(torch.equal(outs["zr"], outs["base"]))
                control_differs = not bool(torch.equal(outs["zru"], outs["base"]))
                control_rel = float((outs["zru"].float() - outs["base"].float()).norm()
                                    / outs["base"].float().norm().clamp_min(1e-30))
                gb, _ = graph(rts["base"], x, w, i)
                gz, _ = graph(rts["zr"], x, w, i)
                # Let clocks leave idle before timing: alternate both graphs for WARMUP_S seconds.
                t_end = time.time() + WARMUP_S
                while time.time() < t_end:
                    timed(gb, REPLAYS); timed(gz, REPLAYS)
                base_t, zr_t = [], []
                for b in range(BLOCKS):
                    if b % 2 == 0:
                        base_t.append(timed(gb, REPLAYS)); zr_t.append(timed(gz, REPLAYS))
                    else:
                        zr_t.append(timed(gz, REPLAYS)); base_t.append(timed(gb, REPLAYS))
                cell = {"m": m, "input_set": s, "start_row": int(start), "base_deterministic": deterministic,
                        "exact_zr_equals_base": exact, "positive_control_differs": control_differs,
                        "positive_control_rel_diff": control_rel,
                        "base_us": base_t, "zr_us": zr_t,
                        "median_ratio": float(np.median(np.asarray(zr_t) / np.asarray(base_t)))}
                layer_res["cells"].append(cell)
                print(json.dumps({"layer": layer, "m": m, "set": s, "deterministic": deterministic, "exact": exact,
                                  "control_differs": control_differs, "control_rel": round(control_rel, 4),
                                  "base_us_med": round(float(np.median(base_t)), 2),
                                  "zr_us_med": round(float(np.median(zr_t)), 2),
                                  "ratio_med": round(cell["median_ratio"], 4)}), flush=True)
                del gb, gz
        results["layers"][str(layer)] = layer_res
        del rts
        torch.cuda.empty_cache()
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)
    print(json.dumps({"output": OUT, "status": "done"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
