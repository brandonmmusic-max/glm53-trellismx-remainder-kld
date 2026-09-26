#!/usr/bin/env python3
"""Summarize a row-pack timing file written by dx2_timing.py (A3: A2_ARM=rp, A4: A2_ARM=rp2).

Per (layer, M): the median of the paired per-block ratios arm/prod, and a percentile bootstrap
CI of that median over blocks. This is the estimator analyze_B2_A2.py uses for A2 (20,000
resamples, seed 20260923), drawn from one generator in file order: layers as stored, then
M ascending. Decision (Amendments 3 and 6): PASS iff the median <= 1.03 at M1 and M4 on both
layers.

usage: summarize_rowpack_timing.py RAW.json OUT.json [LABEL]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPS, SEED, LIMIT = 20_000, 20260923, 1.03


def main() -> None:
    raw_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "arm/prod"
    raw = json.loads(raw_path.read_text())
    rng = np.random.default_rng(SEED)
    rows, decisive = [], []
    for layer, res in raw["layers"].items():
        for m in sorted({c["m"] for c in res["cells"]}):
            cells = [c for c in res["cells"] if c["m"] == m]
            ratios = np.concatenate([np.asarray(c["dx2_us"]) / np.asarray(c["prod_us"]) for c in cells])
            boot = np.median(ratios[rng.integers(0, len(ratios), size=(REPS, len(ratios)))], axis=1)
            med = float(np.median(ratios))
            rows.append({"layer": int(layer), "bits": res["bits"], "m": m, "blocks": int(len(ratios)),
                         "prod_us_median": float(np.median(np.concatenate([c["prod_us"] for c in cells]))),
                         "arm_us_median": float(np.median(np.concatenate([c["dx2_us"] for c in cells]))),
                         "ratio_median": med, "ratio_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]})
            if m in (1, 4):
                decisive.append(med <= LIMIT)
    payload = {"raw": raw_path.name, "ratio": label, "gpu": raw.get("gpu"), "blocks_per_input_set": raw.get("blocks"),
               "replays_per_block": raw.get("replays"),
               "estimator": "median of paired per-block ratios; percentile bootstrap of the median over blocks, "
                            f"{REPS} resamples, seed {SEED}, one generator in file order (layers as stored, M ascending)",
               "rows": rows,
               "decision_rule": f"PASS iff the median ratio <= {LIMIT} at M1 and M4 on both layers",
               "decision": "PASS" if len(decisive) == 4 and all(decisive) else "FAIL"}
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    for r in rows:
        print(f"layer {r['layer']:2d} K{r['bits']} M{r['m']:<2d} blocks {r['blocks']}  {label} {r['ratio_median']:.4f} "
              f"[{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]")
    print("decision:", payload["decision"])


if __name__ == "__main__":
    main()
