#!/usr/bin/env python3
"""PREREG like-for-like check: rerun the 09-04 activation screen's ordinary-e4m3 arm.

Frozen plan (activation-screen-v1/analysis-plan.json): layer 3, its 16 experts, fit role,
domain-balanced, offset 32, count 32, BF16 weights, E4M3/UE8M0-K32 at FC1 input and
post-SwiGLU. Must reproduce geometric-mean NMSE 7.3725e-4 within 1%. The same plan also
gets the two-term arm (hi + lo in FP32 at both hops).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAMPAIGN = Path("<workspace>/trellismx-review-20260922/glm53-hadamard-shapleymcg-kld")
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm53_nvfp4.canary_mxfp6_reap import _metrics  # noqa: E402
from glm53_nvfp4.capture import LayerCapture  # noqa: E402
from glm53_nvfp4.screen_p8_h128_activation import _ordinary  # noqa: E402
from glm53_nvfp4.shard_index import IndexedCheckpoint, sha256_file  # noqa: E402
from remainder_damage import ROLES, SOURCE, carrier  # noqa: E402

SCREEN = Path("<data-volume>/bmxfp4-glm53/codec-v2/p8-h128/activation-screen-v1")
CAPTURE_FULL = Path("<data-volume>/bmxfp4-glm53/teacher/calibration/main-ep4-full")
PUBLISHED = 0.0007372527043926335


def two_term(hidden, gate, up, down):
    x = carrier(hidden.float(), "a8x2")
    gate_out = F.linear(x, gate.float()).clamp(max=10.0)
    up_out = F.linear(x, up.float()).clamp(-10.0, 10.0)
    middle = carrier(F.silu(gate_out) * up_out, "a8x2")
    return F.linear(middle, down.float())


def gmean(values) -> float:
    return float(np.exp(np.log(np.asarray(values, dtype=np.float64)).mean()))


def main() -> None:
    out_path = Path(sys.argv[1])
    if out_path.exists():
        raise FileExistsError(out_path)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    plan = json.loads((SCREEN / "analysis-plan.json").read_text())
    raw = {int(c["expert"]): c for c in json.loads((SCREEN / "raw.json").read_text())["cells"]}
    index = SOURCE / "model.safetensors.index.json"
    if sha256_file(index) != plan["inputs"]["source_index"]["sha256"]:
        raise RuntimeError("BF16 index differs from the screen's")
    if sha256_file(CAPTURE_FULL / "capture-manifest.json") != plan["inputs"]["capture_manifest"]["sha256"]:
        raise RuntimeError("capture manifest differs from the screen's")
    source = IndexedCheckpoint(SOURCE, index)
    s = plan["sampling"]
    capture = LayerCapture(CAPTURE_FULL, plan["layer"], ROLES, max_samples=s["count"], sample_offset=s["offset"],
                           data_role=s["role"], sampling_strategy=s["strategy"])
    prefix = source.expert_prefix(plan["layer"], plan["experts"][0]).split(f"layers.{plan['layer']}.")[0]
    cells = []
    with torch.inference_mode():
        for expert in plan["experts"]:
            hidden_cpu, route_cpu = capture.samples(expert)
            hidden = hidden_cpu.to(device).float()
            route = route_cpu.to(device).float()
            base = f"{prefix}layers.{plan['layer']}.mlp.experts.{expert}"
            gate, up, down = (source.get(f"{base}.{p}.weight").to(device).float() for p in ("gate_proj", "up_proj", "down_proj"))
            reference = _ordinary(hidden, gate, up, down, quantized=False) * route[:, None]
            ordinary = _ordinary(hidden, gate, up, down, quantized=True) * route[:, None]
            doubled = two_term(hidden, gate, up, down) * route[:, None]
            cell = {"expert": expert, "samples": int(hidden.shape[0]),
                    "ordinary_e4m3": _metrics(ordinary, reference)["nmse"],
                    "two_term_e4m3": _metrics(doubled, reference)["nmse"],
                    "published_ordinary_e4m3": raw[expert]["ordinary_e4m3"]["nmse"]}
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    ordinary_g = gmean([c["ordinary_e4m3"] for c in cells])
    two_g = gmean([c["two_term_e4m3"] for c in cells])
    payload = {
        "plan_sha256": sha256_file(SCREEN / "analysis-plan.json"), "cells": cells,
        "geometric_mean_nmse": {"ordinary_e4m3": ordinary_g, "two_term_e4m3": two_g},
        "published_ordinary_e4m3": PUBLISHED, "reproduction_rel_diff": ordinary_g / PUBLISHED - 1.0,
        "reproduced_within_1pct": abs(ordinary_g / PUBLISHED - 1.0) <= 0.01,
        "two_term_over_ordinary_geomean": two_g / ordinary_g,
        "two_term_wins": int(sum(c["two_term_e4m3"] < c["ordinary_e4m3"] for c in cells)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
