#!/usr/bin/env python3
"""EXPLORATORY split-timing summary: per (layer, M, arm) median paired ratio vs base + bootstrap CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "results/timing_split_raw.json")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/timing_split_analysis.json")


def main() -> None:
    raw = json.loads(RAW.read_text())
    rng = np.random.default_rng(20260923)
    rows = []
    for layer, res in raw["layers"].items():
        for m in sorted({c["m"] for c in res["cells"]}):
            cells = [c for c in res["cells"] if c["m"] == m]
            base = np.concatenate([np.asarray(c["times_us"]["base"]) for c in cells])
            for arm in ("zr_both", "zr_fc2", "zr_fc1"):
                arm_t = np.concatenate([np.asarray(c["times_us"][arm]) for c in cells])
                ratios = arm_t / base
                boot = np.median(ratios[rng.integers(0, len(ratios), size=(20_000, len(ratios)))], axis=1)
                rows.append({"layer": int(layer), "bits": res["bits"], "m": m, "arm": arm,
                             "base_us": float(np.median(base)), "arm_us": float(np.median(arm_t)),
                             "ratio_median": float(np.median(ratios)),
                             "ratio_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                             "exact": all(c["exact"][arm] for c in cells),
                             "fc2_control_differs": all(c["zru_fc2_differs"] for c in cells)})
    OUT.write_text(json.dumps({"exploratory": True, "rows": rows}, indent=1) + "\n")
    for r in rows:
        print(f"layer {r['layer']:2d} K{r['bits']} M{r['m']:<2d} {r['arm']:8s} base {r['base_us']:7.2f} us  "
              f"arm {r['arm_us']:7.2f} us  ratio {r['ratio_median']:.4f} [{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}]"
              f"  exact {r['exact']}  control {r['fc2_control_differs']}")


if __name__ == "__main__":
    main()
