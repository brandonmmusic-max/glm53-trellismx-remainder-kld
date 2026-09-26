#!/usr/bin/env python3
"""K2 (PREREG amendment 2): real-kernel numerics closure for D-x2.

Runs inside the production image with b12x-dx2 mounted. For each layer, all four TP4 rank
sidecars run one rank at a time, prod (flag off) and D-x2 (flag on), and their outputs are
summed across ranks (what the all-reduce computes). Tokens are the Test B fit tokens, fed
in chunks that exercise every path:
  M1 (first 96 tokens = 2 windows), M16, M64 (grouped-M32 prefill), M3072 (M64 prefill).
Damage = sum_t ||kernel_t - base_t||^2 against the BF16-source routed output.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402  (overlay loader, runtime(), selected-kernel record)

LAYERS = [int(v) for v in os.environ.get("K2_LAYERS", "3,8").split(",")]
PATHS = {"M1": (1, 96), "M16": (16, None), "M64": (64, None), "M3072": (3072, None)}
ARMS = [a for a in os.environ.get("K2_ARMS", "prod,dx2").split(",")]
BUILD = os.environ.get("K2_BUILD", "dx2build")
TPW = 48


ENV = {"prod": None, "dx2": "1", "rp": "rp"}


def runtime(overlay, layer, rank, arm):
    if ENV[arm] is None:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)
    else:
        os.environ["B12X_P8_DOWN_REMAINDER"] = ENV[arm]
    try:
        record = overlay.records[layer, rank]
        return th.P8NativeTPMoE(
            overlay.sidecar(layer, rank), device=th.DEV, tp_rank=rank, world_size=4, layer=layer,
            expected_design_sha256=record["source_design_sha256"],
            expected_transform_sha256=overlay.transform_hash,
            topk=8, hidden=4096, intermediate=512, swiglu_limit=10.0, small_m_scheduler=True,
            fc1_tile_n=128, fuse_scratch_zero=True, prefill_chunk_tokens=0, grid_policy=True,
            fc1_warp_quant=False, fc1_broadcast_a=True,
        )
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)


def run_path(rt, x, w, i, chunk, limit):
    n = x.shape[0] if limit is None else limit
    outs = []
    for s in range(0, n, chunk):
        outs.append(rt(x[s:s + chunk], w[s:s + chunk], i[s:s + chunk]).float().cpu())
    torch.cuda.synchronize()
    return torch.cat(outs)


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    result = {}
    for layer in LAYERS:
        data = torch.load(f"/work/results/k2/base-layer-{layer:03d}.pt")
        x = data["hidden"].to(th.DEV)
        i = data["ids"].to(th.DEV)
        w = data["weights"].to(th.DEV)
        base = data["base"]
        sums = {p: {a: None for a in ARMS} for p in PATHS}
        for rank in range(4):
            for arm in ARMS:
                rt = runtime(overlay, layer, rank, arm)
                if (getattr(rt, "p8_down_remainder", False) != (arm == "dx2")
                        or getattr(rt, "p8_dx2_rowpack", False) != (arm == "rp")):
                    raise RuntimeError(f"{arm}: runtime flag mismatch")
                for path, (chunk, limit) in PATHS.items():
                    out = run_path(rt, x, w, i, chunk, limit)
                    sums[path][arm] = out if sums[path][arm] is None else sums[path][arm] + out
                del rt
                torch.cuda.empty_cache()
            print(json.dumps({"layer": layer, "rank": rank, "done": True}), flush=True)
        layer_res = {}
        for path, (chunk, limit) in PATHS.items():
            n = base.shape[0] if limit is None else limit
            b = base[:n].double()
            windows = torch.arange(n) // TPW
            cell = {"tokens": n}
            for arm in ARMS:
                per_tok = (sums[path][arm].double() - b).square().sum(dim=-1)
                cell[f"D_{arm}"] = float(per_tok.sum())
                cell[f"per_window_{arm}"] = torch.zeros(int(windows.max()) + 1, dtype=torch.float64).index_add_(0, windows, per_tok).tolist()
            for arm in ARMS:
                if arm == "prod":
                    continue
                cell[f"rel_diff_{arm}_vs_prod"] = float((sums[path][arm] - sums[path]["prod"]).norm() / sums[path]["prod"].norm())
                cell[f"ratio_{arm}"] = cell[f"D_{arm}"] / cell["D_prod"]
                cell[f"bit_identical_{arm}_vs_prod"] = bool(torch.equal(sums[path][arm], sums[path]["prod"]))
            layer_res[path] = cell
            print(json.dumps({k: v for k, v in cell.items() if not k.startswith("per_window")} | {"layer": layer, "path": path}), flush=True)
        # save the rank-summed prod M16 output for the bit-identity check against the image's own b12x
        torch.save({"M16_prod": sums["M16"]["prod"], "M3072_prod": sums["M3072"]["prod"]},
                   f"/work/results/k2/kernel-prod-{BUILD}-layer-{layer:03d}.pt")
        result[str(layer)] = layer_res
        with open(f"/work/results/k2/closure-{BUILD}.json", "w") as fh:
            json.dump(result, fh, indent=1)
    print(json.dumps({"status": "done"}), flush=True)


if __name__ == "__main__":
    main()
