#!/usr/bin/env python3
"""PREREG amendment 2 analysis: B2 (fresh-layer numerics) and A2 (real-kernel timing)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_BC import REPS, SEED, bca  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def b2(rng) -> dict:
    layers = [json.loads(f.read_text()) for f in sorted((HERE / "results/testB2").glob("layer-*.json"))]
    if not layers:
        return {"decision": "missing"}
    arm = lambda name: np.array([l["arms"][name]["per_window_damage"] for l in layers])  # noqa: E731
    theta, lo, hi = bca(arm("P-A8x2-downonly"), arm("P-A8"), rng)
    per_layer = [{"layer": l["layer"], "bits": l["bits"],
                  "ratio_downonly": l["arms"]["P-A8x2-downonly"]["damage_sum"] / l["arms"]["P-A8"]["damage_sum"],
                  "ratio_both_hops": l["arms"]["P-A8x2"]["damage_sum"] / l["arms"]["P-A8"]["damage_sum"],
                  "activation_share": l["arms"]["B-A8"]["damage_sum"] / l["arms"]["P-A8"]["damage_sum"],
                  "relative_damage_prod": l["arms"]["P-A8"]["relative_to_routed_output"],
                  "closure_B-Aexact": l["arms"]["B-Aexact"]["relative_to_routed_output"]} for l in layers]
    improve = sum(p["ratio_downonly"] < 1 for p in per_layer)
    reduction = 1 - float(np.exp(theta))
    passed = len(layers) == 6 and reduction >= 0.20 and improve >= 5 and hi < 0
    return {"layers": [l["layer"] for l in layers], "pooled_ratio_downonly": float(np.exp(theta)),
            "pooled_reduction": reduction, "ratio_ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "layers_improved": improve, "per_layer": per_layer,
            "decision": "PASS" if passed else ("incomplete" if len(layers) < 6 else "FAIL")}


def a2(rng) -> dict:
    path = HERE / "results/a2_timing_raw.json"
    if not path.exists():
        return {"decision": "missing"}
    raw = json.loads(path.read_text())
    rows, decode_ok, prefill_ok = [], [], []
    for layer, res in raw["layers"].items():
        for m in sorted({c["m"] for c in res["cells"]}):
            cells = [c for c in res["cells"] if c["m"] == m]
            ratios = np.concatenate([np.asarray(c["dx2_us"]) / np.asarray(c["prod_us"]) for c in cells])
            boot = np.median(ratios[rng.integers(0, len(ratios), size=(REPS, len(ratios)))], axis=1)
            med = float(np.median(ratios))
            rows.append({"layer": int(layer), "bits": res["bits"], "m": m, "blocks": int(len(ratios)),
                         "prod_us": float(np.median(np.concatenate([c["prod_us"] for c in cells]))),
                         "dx2_us": float(np.median(np.concatenate([c["dx2_us"] for c in cells]))),
                         "ratio_median": med, "ratio_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]})
            if m in (1, 4):
                decode_ok.append(med <= 1.03)
            if m in (512, 2048):
                prefill_ok.append(med <= 1.05)
    return {"rows": rows,
            "decode_decision": "PASS" if len(decode_ok) == 4 and all(decode_ok) else "FAIL",
            "prefill_decision": "PASS" if len(prefill_ok) == 4 and all(prefill_ok) else "FAIL"}


def main() -> None:
    rng = np.random.default_rng(SEED)
    payload = {"B2": b2(rng), "A2": a2(rng)}
    (HERE / "results/analysis_amendment2.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
