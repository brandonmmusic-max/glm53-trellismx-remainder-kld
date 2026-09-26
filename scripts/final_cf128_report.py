#!/usr/bin/env python3
"""Final report for window 3 (PREREG amendment 7): preregistered primaries from kld-cf128/analysis.json
plus reported-only extras (row-level robust statistics, per-domain means, original-32 subset against the
earlier runs, speed against the earlier windows, KV capacity). Writes kld-cf128/final-report.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
# Works from the testing workspace (kld-*/ at the top level) and from the published repository (results/kld-*/).
RES = HERE / "results" if (HERE / "results" / "kld-cf128" / "analysis.json").exists() else HERE
W3 = RES / "kld-cf128"
PUB = HERE / "results" / "reference-20260909"  # copies of the 09-09 published comparison files


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def rows_of(arm: str, ids: list[str]) -> np.ndarray:
    return np.stack([np.load(W3 / arm / "scores" / f"{w}.npz")["kld"][1:] for w in ids])


def main() -> None:
    a = json.loads((W3 / "analysis.json").read_text())
    ids = [r["window_id"] for r in a["arms"]["control-fp8"]["per_window"]]
    out = {"preregistered": {k: a[k] for k in ("rp2_minus_control", "rp2_minus_tr3", "control_minus_tr3", "reading_rp2_vs_control")},
           "arm_means": {arm: {"mean": v["mean_true_decode_kld"], "window_bca95": v["window_bca95"]} for arm, v in a["arms"].items()},
           "first32_subset": a["first32_subset"], "per_domain_mean": a["per_domain_mean"]}

    print("== Arm means (128 windows, FP8 MLA KV, true decode)")
    for arm, v in out["arm_means"].items():
        print(f"  {arm:12s} {v['mean']:.7f}  window BCa95 [{v['window_bca95'][0]:.5f}, {v['window_bca95'][1]:.5f}]")
    print("== Preregistered paired primaries")
    for key, name in (("rp2_minus_control", "rp2 - control"), ("rp2_minus_tr3", "rp2 - tr3"), ("control_minus_tr3", "control - tr3")):
        p = a[key]
        print(f"  {name:15s} {p['diff']:+.6f} ({pct(p['relative'])})  BCa95 [{p['bca95'][0]:+.6f}, {p['bca95'][1]:+.6f}]"
              f" = [{pct(p['relative_bca95'][0])}, {pct(p['relative_bca95'][1])}]  lower {p['a_lower_windows']}/{p['windows']}"
              f"  CI incl 0: {p['ci_includes_0']}  equiv 5%: {p['equivalent_5pct']}  equiv 10%: {p['equivalent_10pct']}")
    print("  reading rp2 vs control:", a["reading_rp2_vs_control"])

    # Reported only: row-level robust statistics.
    rows = {arm: rows_of(arm, ids) for arm in ("control-fp8", "rp2-fp8", "tr3-fp8")}
    robust = {}
    for x, y in (("rp2-fp8", "control-fp8"), ("rp2-fp8", "tr3-fp8"), ("control-fp8", "tr3-fp8")):
        X, Y = rows[x], rows[y]
        thr = np.quantile(np.concatenate([X.ravel(), Y.ravel()]), 0.995)
        keep = (X < thr) & (Y < thr)
        robust[f"{x}_vs_{y}"] = {
            "row_mean_ratio": float(X.mean() / Y.mean()), "row_median_ratio": float(np.median(X) / np.median(Y)),
            "q90_ratio": float(np.quantile(X, 0.9) / np.quantile(Y, 0.9)), "q99_ratio": float(np.quantile(X, 0.99) / np.quantile(Y, 0.99)),
            "q999_ratio": float(np.quantile(X, 0.999) / np.quantile(Y, 0.999)),
            "trimmed_mean_ratio_pooled_q995": float(X[keep].mean() / Y[keep].mean()),
            "rows_lower_fraction": float((X < Y).mean()),
            "window_medians_lower": int(sum(np.median(xx) < np.median(yy) for xx, yy in zip(X, Y)))}
    out["row_level_reported_only"] = robust
    print("== Row-level (reported only)")
    for k, v in robust.items():
        print(f"  {k:26s} mean {pct(v['row_mean_ratio'] - 1)}  median {pct(v['row_median_ratio'] - 1)}  q90 {pct(v['q90_ratio'] - 1)}"
              f"  q99 {pct(v['q99_ratio'] - 1)}  q99.9 {pct(v['q999_ratio'] - 1)}  trimmed {pct(v['trimmed_mean_ratio_pooled_q995'] - 1)}"
              f"  rows lower {v['rows_lower_fraction']:.1%}  window medians lower {v['window_medians_lower']}/128")

    # Reported only: the original 32 windows against the earlier runs.
    by = lambda arm: {r["window_id"]: r["true_decode_mean_kld"] for r in a["arms"][arm]["per_window"]}  # noqa: E731
    first32 = ids[:32]
    w2 = json.loads((RES / "kld-fp8-20260925/analysis.json").read_text())["arms"]
    tr3_0909 = {r["window_id"]: r["true_decode_mean_kld"] for r in json.loads((PUB / "kld-tr3-20260909/comparison.json").read_text())["arms"]["fp8"]["per_window"]}
    w2c = {r["window_id"]: r["true_decode_mean_kld"] for r in w2["control-fp8"]["per_window"]}
    subset = {"control_fp8_window3": float(np.mean([by("control-fp8")[w] for w in first32])),
              "control_fp8_window2": float(np.mean([w2c[w] for w in first32])),
              "tr3_fp8_window3": float(np.mean([by("tr3-fp8")[w] for w in first32])),
              "tr3_fp8_20260909": float(np.mean([tr3_0909[w] for w in first32])),
              "rp2_fp8_window3": float(np.mean([by("rp2-fp8")[w] for w in first32]))}
    subset["control_run_to_run"] = subset["control_fp8_window3"] / subset["control_fp8_window2"] - 1
    subset["tr3_run_to_run"] = subset["tr3_fp8_window3"] / subset["tr3_fp8_20260909"] - 1
    out["first32_vs_earlier_runs"] = subset
    print("== Original 32 windows vs earlier runs (reported only)")
    print("  " + json.dumps({k: round(v, 6) for k, v in subset.items()}))

    # Speed (descriptive): rp2-fp8 vs earlier windows' cells.
    speed = {}
    for cell in ("ctx0-c1", "ctx0-c4", "ctx8192-c1", "ctx8192-c4"):
        row = {}
        for label, path in (("control_nvfp4_w1", RES / "kld-rp-20260925/speed/control"), ("rp_nvfp4_w1", RES / "kld-rp-20260925/speed/rp"),
                            ("rp_fp8_w2", RES / "kld-fp8-20260925/speed/rp-fp8"), ("rp2_fp8_w3", W3 / "speed/rp2-fp8")):
            f = path / f"{cell}.json"
            if f.exists():
                r = json.loads(f.read_text())["results"][0]
                row[label] = {"tps": r["aggregate_tps"], "mtp_accept": r.get("server_spec_accept_rate")}
        speed[cell] = row
    out["speed"] = speed
    print("== Speed, tok/s (MTP acceptance), single runs, cooling gate before each cell")
    for cell, row in speed.items():
        print(f"  {cell:11s} " + "  ".join(f"{k} {v['tps']:.1f} ({v['mtp_accept']:.3f})" for k, v in row.items()))

    # KV capacity from the engine startup logs.
    cap = {}
    for label, f in (("capture_control_fp8", W3 / "control-fp8/startup.log"), ("capture_tr3_fp8", W3 / "tr3-fp8/startup.log"),
                     ("serving_rp2_fp8", W3 / "speed/rp2-fp8/startup.log")):
        if f.exists():
            cap[label] = [line.split("GPU KV cache size:")[1].strip() for line in f.read_text().splitlines() if "GPU KV cache size:" in line][:1]
    out["kv_capacity"] = cap
    print("== KV capacity:", json.dumps(cap))

    (W3 / "final-report.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", W3 / "final-report.json")


if __name__ == "__main__":
    main()
