#!/usr/bin/env python3
"""PREREG analysis for Tests B and C: pooled log ratios with a paired window BCa bootstrap.

Windows are the same 64 fit token windows in every layer, so one resample of window indices
is applied to every layer at once (paired across layers). Statistic: mean over layers of
log(sum_w num_new / sum_w num_ref); its exp is the pooled geometric-mean ratio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPS, SEED = 20_000, 20260923
PHI = NormalDist()


def pooled(new: np.ndarray, ref: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray | float:
    """new/ref: [layers, windows]. idx: [reps, windows] resample, or None for the point estimate."""
    if idx is None:
        return float(np.mean(np.log(new.sum(axis=1) / ref.sum(axis=1))))
    return np.mean(np.log(new[:, idx].sum(axis=2) / ref[:, idx].sum(axis=2)), axis=0)


def bca(new: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n = new.shape[1]
    theta = pooled(new, ref)
    boot = pooled(new, ref, rng.integers(0, n, size=(REPS, n)))
    z0 = PHI.inv_cdf(min(max(float(np.mean(boot < theta)), 1e-6), 1 - 1e-6))
    jack = np.array([pooled(np.delete(new, i, axis=1), np.delete(ref, i, axis=1)) for i in range(n)])
    d = jack.mean() - jack
    a = float((d ** 3).sum() / (6.0 * ((d ** 2).sum()) ** 1.5)) if (d ** 2).sum() > 0 else 0.0
    bounds = []
    for alpha in (0.025, 0.975):
        z = PHI.inv_cdf(alpha)
        bounds.append(float(np.quantile(boot, PHI.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))
    return theta, bounds[0], bounds[1]


def test_b(rng) -> dict:
    files = sorted((HERE / "results/testB").glob("layer-*.json"))
    layers = [json.loads(f.read_text()) for f in files]
    arm = lambda name: np.array([l["arms"][name]["per_window_damage"] for l in layers])  # noqa: E731
    base = arm("P-A8")
    theta, lo, hi = bca(arm("P-A8x2"), base, rng)
    per_layer = []
    for l in layers:
        d = {k: v["damage_sum"] for k, v in l["arms"].items()}
        per_layer.append({
            "layer": l["layer"], "bits": l["bits"], "relative_damage_P-A8": l["arms"]["P-A8"]["relative_to_routed_output"],
            "ratio_P-A8x2": d["P-A8x2"] / d["P-A8"], "activation_share_B-A8": d["B-A8"] / d["P-A8"],
            "weight_share_P-Aexact": d["P-Aexact"] / d["P-A8"], "ratio_fc1only": d["P-A8x2-fc1only"] / d["P-A8"],
            "ratio_downonly": d["P-A8x2-downonly"] / d["P-A8"],
            "closure_B-Aexact_relative": l["arms"]["B-Aexact"]["relative_to_routed_output"],
            "weight_nmse_mean": l["weight_nmse_mean"]})
    domains = {}
    for l in layers:
        for w, dom in enumerate(l["window_domains"]):
            e = domains.setdefault(dom, {"P-A8": 0.0, "P-A8x2": 0.0})
            e["P-A8"] += l["arms"]["P-A8"]["per_window_damage"][w]
            e["P-A8x2"] += l["arms"]["P-A8x2"]["per_window_damage"][w]
    improve = sum(p["ratio_P-A8x2"] < 1 for p in per_layer)
    reduction = 1 - np.exp(theta)
    passed = len(layers) == 6 and reduction >= 0.05 and improve >= 5 and hi < 0
    post = {}
    for name in ("P-A8x2-downonly", "P-A8x2-fc1only"):
        t, l_, h_ = bca(arm(name), base, rng)
        post[name] = {"pooled_ratio": float(np.exp(t)), "ci95": [float(np.exp(l_)), float(np.exp(h_))]}
    return {"layers": [l["layer"] for l in layers], "pooled_ratio_P-A8x2_over_P-A8": float(np.exp(theta)),
            "pooled_reduction": float(reduction), "pooled_log_ratio_ci95_bca": [lo, hi],
            "ratio_ci95": [float(np.exp(lo)), float(np.exp(hi))], "layers_improved": improve,
            "decision": "worth-end-to-end" if passed else ("incomplete" if len(layers) < 6 else "not-worth"),
            "per_layer": per_layer, "per_domain_ratio": {k: v["P-A8x2"] / v["P-A8"] for k, v in sorted(domains.items())},
            "post_hoc_reported_only": post}


def test_c(rng) -> dict:
    decisive = [(7, 6), (15, 14), (19, 18), (27, 26), (35, 34), (43, 42)]
    loaded = {}
    for f in sorted((HERE / "results/testC").glob("A*-C*.json")):
        for r in json.loads(f.read_text())["results"]:
            loaded[(r["attn_layer"], r["capture_layer"])] = r
    have = [c for c in decisive if c in loaded]
    out = {"decisive_configs_present": [f"{a}:{c}" for a, c in have], "metrics": {}}
    ok = len(have) == 6
    for metric in ("latent", "logit"):
        ref = np.array([loaded[c][f"{metric}_num"]["U"] for c in have])
        res = {}
        for arm in ("H", "HD"):
            new = np.array([loaded[c][f"{metric}_num"][arm] for c in have])
            theta, lo, hi = bca(new, ref, rng)
            res[arm] = {"pooled_ratio": float(np.exp(theta)), "ci95": [float(np.exp(lo)), float(np.exp(hi))],
                        "per_layer": {f"{a}:{c}": float(sum(loaded[(a, c)][f"{metric}_num"][arm]) /
                                                        sum(loaded[(a, c)][f"{metric}_num"]["U"])) for a, c in have}}
        res["U_relative_error"] = {f"{a}:{c}": loaded[(a, c)][f"{metric}_relative"]["U"] for a, c in have}
        out["metrics"][metric] = res
        ok = ok and res["H"]["pooled_ratio"] <= 0.8 and res["H"]["ci95"][1] < 1.0
    out["sensitivity_same_layer_standin"] = {
        f"{a}:{c}": {m: {arm: loaded[(a, c)][f"{m}_relative"][arm] / loaded[(a, c)][f"{m}_relative"]["U"]
                         for arm in ("H", "HD")} for m in ("latent", "logit")}
        for (a, c) in loaded if (a, c) not in decisive}
    out["decision"] = "worth-end-to-end" if ok else ("incomplete" if len(have) < 6 else "not-worth")
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    which = sys.argv[1] if len(sys.argv) > 1 else "BC"
    payload = {}
    if "B" in which:
        payload["test_b"] = test_b(rng)
    if "C" in which:
        payload["test_c"] = test_c(rng)
    (HERE / "results/analysis_BC.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
