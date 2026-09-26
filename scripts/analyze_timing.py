#!/usr/bin/env python3
"""Test A analysis (PREREG): paired per-block Z/baseline ratio, median + bootstrap CI.

Per (layer, M), the per-block ratios from all input sets are pooled. The CI is a
percentile bootstrap of the median over blocks (20,000 replicates, seed 20260923).
Decision: PASS iff the median ratio <= 1.03 at M1 and M4 for both layers and every
exactness check held; M16 is reported only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "results/timing_raw.json")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/timing_analysis.json")
REPS, SEED, LIMIT = 20_000, 20260923, 1.03


def main() -> None:
    raw = json.loads(RAW.read_text())
    rng = np.random.default_rng(SEED)
    rows, checks_ok, decisive = [], True, []
    for layer, res in raw["layers"].items():
        for m in sorted({c["m"] for c in res["cells"]}):
            cells = [c for c in res["cells"] if c["m"] == m]
            ratios = np.concatenate([np.asarray(c["zr_us"]) / np.asarray(c["base_us"]) for c in cells])
            base = np.concatenate([np.asarray(c["base_us"]) for c in cells])
            zr = np.concatenate([np.asarray(c["zr_us"]) for c in cells])
            idx = rng.integers(0, len(ratios), size=(REPS, len(ratios)))
            boot = np.median(ratios[idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ok = all(c["exact_zr_equals_base"] and c["positive_control_differs"] and c["base_deterministic"]
                     for c in cells)
            checks_ok &= ok
            row = {"layer": int(layer), "bits": res["bits"], "m": m, "blocks": int(len(ratios)),
                   "base_us_median": float(np.median(base)), "zr_us_median": float(np.median(zr)),
                   "ratio_median": float(np.median(ratios)), "ratio_ci95": [float(lo), float(hi)],
                   "exact_and_controls_ok": ok}
            rows.append(row)
            if m in (1, 4):
                decisive.append(row["ratio_median"] <= LIMIT)
    passed = checks_ok and len(decisive) == 4 and all(decisive)
    payload = {"raw": str(RAW), "gpu": raw["gpu"], "blocks_per_input_set": raw["blocks"],
               "replays_per_block": raw["replays"], "cells": rows, "all_checks_exact": checks_ok,
               "decision_rule": "PASS iff median Z/base <= 1.03 at M1 and M4 for both layers and all outputs exact",
               "decision": "PASS" if passed else "FAIL"}
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    for r in rows:
        print(f"layer {r['layer']:2d} K{r['bits']} M{r['m']:<2d} base {r['base_us_median']:8.2f} us  "
              f"Z {r['zr_us_median']:8.2f} us  ratio {r['ratio_median']:.4f} "
              f"[{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]  exact/controls {r['exact_and_controls_ok']}")
    print("decision:", payload["decision"])


if __name__ == "__main__":
    main()
