#!/usr/bin/env python3
"""Reported-only numbers for Amendment 7 that the harness does not produce. CPU only; reads result files, writes one JSON.

Everything here is reported only (not preregistered):
- subsets: original 32, new 96, and the interim look (rp2 and control complete, TR3 at its first 47 windows);
- per-domain paired differences, and a domain-balanced sensitivity (equal weight per domain);
- the fraction of the control-TR3 gap that RP2 closes, with a paired bootstrap interval;
- run-to-run repeat differences for identical configurations (same night and across days);
- per-window spread of the rp2/control ratio against same-configuration reruns;
- row-level rerun agreement between the two control-fp8 runs, and top-1 agreement with the BF16 teacher per arm.

Estimators: paired window differences with BCa 95% (scipy.stats.bootstrap, 20,000 resamples, seed 20260902), as in
the harness. The domain-balanced mean and the gap fraction use percentile intervals from a paired (stratified, for
the domain-balanced mean) bootstrap over windows with the same seed. Relative values divide by the comparator's
observed mean.

Usage: python3 scripts/reported_extras_cf128.py [ROOT]. ROOT is the repository root (layout results/kld-cf128/...);
it defaults to the parent of this script's directory. The testing workspace layout (kld-cf128/... at the top level)
is also accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import bootstrap

SEED = 20260902
N = 20000


def find(root: Path, rel: str) -> Path:
    for cand in (root / "results" / rel, root / rel):
        if cand.exists():
            return cand
    raise FileNotFoundError(rel)


def per_window(analysis: dict, arm: str) -> dict:
    return {r["window_id"]: r for r in analysis["arms"][arm]["per_window"]}


def paired(a: dict, b: dict, ids: list) -> dict:
    x = np.array([a[w] for w in ids])
    y = np.array([b[w] for w in ids])
    d = x - y
    ci = bootstrap((d,), np.mean, method="BCa", n_resamples=N, random_state=SEED).confidence_interval
    m = y.mean()
    return {"windows": len(ids), "a_mean": float(x.mean()), "b_mean": float(y.mean()), "diff": float(d.mean()),
            "relative": float(d.mean() / m), "bca95": [float(ci.low), float(ci.high)],
            "relative_bca95": [float(ci.low / m), float(ci.high / m)], "a_lower_windows": int((d < 0).sum()),
            "ci_includes_0": bool(ci.low <= 0 <= ci.high)}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    a7 = json.loads(find(root, "kld-cf128/analysis.json").read_text())
    w2 = json.loads(find(root, "kld-fp8-20260925/analysis.json").read_text())
    w1 = json.loads(find(root, "kld-rp-20260925/analysis.json").read_text())
    ref = json.loads(find(root, "reference-20260909/kld-reference-20260909/comparison.json").read_text())
    tr3ref = json.loads(find(root, "reference-20260909/kld-tr3-20260909/comparison.json").read_text())

    arms = {k: per_window(a7, k) for k in ("control-fp8", "rp2-fp8", "tr3-fp8")}
    kld = {k: {w: r["true_decode_mean_kld"] for w, r in v.items()} for k, v in arms.items()}
    ids = [r["window_id"] for r in a7["arms"]["control-fp8"]["per_window"]]
    dom = {w: arms["control-fp8"][w]["domain"] for w in ids}
    out = {"versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
           "estimator": f"paired window BCa95, {N} resamples, seed {SEED}"}
    pairs = (("rp2-fp8", "control-fp8"), ("rp2-fp8", "tr3-fp8"), ("control-fp8", "tr3-fp8"))

    # Subsets. Manifest order: the original 32 first, then the 96 new windows in id order. TR3 scored in that order,
    # so its first 47 windows are what the interim look (23:47 EDT, TR3 at 47/128) could see.
    out["subsets"] = {}
    for name, sub in (("original_32", ids[:32]), ("new_96", ids[32:]), ("interim_look_first_47", ids[:47])):
        out["subsets"][name] = {f"{x}_minus_{y}": paired(kld[x], kld[y], sub) for x, y in pairs}

    # Robustness of the two preregistered primaries: Bonferroni-adjusted (97.5%) paired BCa, and a paired BCa interval
    # for the ratio of means (relative difference with the denominator resampled too).
    out["primary_robustness"] = {}
    for x, y in (("rp2-fp8", "control-fp8"), ("rp2-fp8", "tr3-fp8")):
        xa = np.array([kld[x][w] for w in ids])
        ya = np.array([kld[y][w] for w in ids])
        d = xa - ya
        bon = bootstrap((d,), np.mean, method="BCa", n_resamples=N, random_state=SEED, confidence_level=0.975).confidence_interval
        rom = bootstrap((xa, ya), lambda a, b: a.mean() / b.mean() - 1.0, paired=True, vectorized=False, method="BCa",
                        n_resamples=N, random_state=SEED).confidence_interval
        out["primary_robustness"][f"{x}_minus_{y}"] = {
            "bonferroni_bca97_5_relative": [float(bon.low / ya.mean()), float(bon.high / ya.mean())],
            "ratio_of_means_bca95": [float(rom.low), float(rom.high)]}

    # Per-domain paired differences (12 intervals, no multiplicity correction).
    domains = sorted(set(dom.values()))
    out["per_domain"] = {d: {f"{x}_minus_{y}": paired(kld[x], kld[y], [w for w in ids if dom[w] == d]) for x, y in pairs}
                         for d in domains}

    # Domain-balanced sensitivity: equal weight per domain, windows resampled within domain.
    rng = np.random.default_rng(SEED)
    by_dom = {d: [w for w in ids if dom[w] == d] for d in domains}
    balanced = {}
    for x, y in pairs:
        dd = {d: np.array([kld[x][w] - kld[y][w] for w in by_dom[d]]) for d in domains}
        yy = {d: np.array([kld[y][w] for w in by_dom[d]]) for d in domains}
        point = float(np.mean([dd[d].mean() for d in domains]))
        base = float(np.mean([yy[d].mean() for d in domains]))
        boots = np.empty(N)
        for i in range(N):
            boots[i] = np.mean([dd[d][rng.integers(0, len(dd[d]), len(dd[d]))].mean() for d in domains])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        balanced[f"{x}_minus_{y}"] = {"diff": point, "relative": point / base, "percentile95": [float(lo), float(hi)],
                                      "relative_percentile95": [float(lo / base), float(hi / base)],
                                      "ci_includes_0": bool(lo <= 0 <= hi)}
    out["domain_balanced_equal_weights"] = balanced

    # Window dependence. Windows that share token spans are not independent draws. Clusters are the connected
    # components of the graph that links two windows when they share at least SPAN_LINK distinct 32-token spans
    # (from the sha-verified token arrays, when present). The primaries are re-estimated with a percentile
    # bootstrap over clusters (the mean over windows of each resample).
    tok_root = find(root, "kld-cf128/verified-inputs.json")
    vin = json.loads(tok_root.read_text())
    toks = {}
    for rec in vin["windows"]:
        p = Path(rec["token_path"])
        if p.exists():
            toks[rec["id"]] = np.load(p)
    if len(toks) == len(ids):
        spans = {}
        for w, t in toks.items():
            for i in range(len(t) - 31):
                spans.setdefault(t[i:i + 32].tobytes(), set()).add(w)
        shared = {}
        for s in spans.values():
            if len(s) > 1:
                s = sorted(s)
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        shared[(s[i], s[j])] = shared.get((s[i], s[j]), 0) + 1
        clusterings = {}
        for link in (1, 64):
            parent = {w: w for w in ids}

            def root_of(w):
                while parent[w] != w:
                    parent[w] = parent[parent[w]]
                    w = parent[w]
                return w
            for (x, y), n in shared.items():
                if n >= link:
                    parent[root_of(x)] = root_of(y)
            groups = {}
            for w in ids:
                groups.setdefault(root_of(w), []).append(w)
            clusterings[f"share_ge_{link}_spans"] = list(groups.values())
        out["window_dependence"] = {"pairs_sharing_a_32_token_span": len(shared),
                                    "pairs_sharing_ge_64_spans": sum(1 for n in shared.values() if n >= 64)}
        for name, groups in clusterings.items():
            rng = np.random.default_rng(SEED)
            res = {"clusters": len(groups), "largest_cluster": max(len(g) for g in groups)}
            for x, y in pairs:
                gd = [np.array([kld[x][w] - kld[y][w] for w in g]) for g in groups]
                base = np.mean([kld[y][w] for w in ids])
                boots = np.empty(N)
                for i in range(N):
                    pick = rng.integers(0, len(groups), len(groups))
                    boots[i] = np.concatenate([gd[k] for k in pick]).mean()
                lo, hi = np.percentile(boots, [2.5, 97.5])
                res[f"{x}_minus_{y}"] = {"relative": float(np.concatenate(gd).mean() / base),
                                         "relative_percentile95": [float(lo / base), float(hi / base)],
                                         "ci_includes_0": bool(lo <= 0 <= hi)}
            out["window_dependence"][name] = res

    # Fraction of the control-TR3 gap closed by RP2, paired bootstrap over windows.
    c = np.array([kld["control-fp8"][w] for w in ids])
    r = np.array([kld["rp2-fp8"][w] for w in ids])
    t = np.array([kld["tr3-fp8"][w] for w in ids])
    rng = np.random.default_rng(SEED)
    fr = np.empty(N)
    for i in range(N):
        k = rng.integers(0, len(ids), len(ids))
        fr[i] = (c[k].mean() - r[k].mean()) / (c[k].mean() - t[k].mean())
    out["gap_closed_fraction"] = {"point": float((c.mean() - r.mean()) / (c.mean() - t.mean())),
                                  "percentile95": [float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))]}

    # Run-to-run repeats of identical configurations (original 32 windows).
    first32 = ids[:32]
    rw2 = {k: {w: r["true_decode_mean_kld"] for w, r in per_window(w2, k).items()} for k in ("control-fp8", "rp-fp8")}
    rw1 = {k: {w: r["true_decode_mean_kld"] for w, r in per_window(w1, k).items()} for k in ("control", "rp")}
    r09 = {k: {w: r["true_decode_mean_kld"] for w, r in per_window(ref, k).items()} for k in ("fp8", "nvfp4")}
    t09 = {k: {w: r["true_decode_mean_kld"] for w, r in per_window(tr3ref, k).items()} for k in ("fp8", "nvfp4")}
    repeats = {
        "control_fp8_final_vs_fp8_window_same_night": (kld["control-fp8"], rw2["control-fp8"]),
        "control_fp8_final_vs_20260909": (kld["control-fp8"], r09["fp8"]),
        "control_fp8_fp8_window_vs_20260909": (rw2["control-fp8"], r09["fp8"]),
        "control_nvfp4_first_window_vs_20260909": (rw1["control"], r09["nvfp4"]),
        "tr3_fp8_final_vs_20260909": (kld["tr3-fp8"], t09["fp8"]),
    }
    out["run_to_run"] = {}
    for name, (x, y) in repeats.items():
        p = paired(x, y, first32)
        rel = np.array([x[w] / y[w] - 1 for w in first32])
        p["per_window_relative_sd"] = float(rel.std(ddof=1))
        p["per_window_relative_max_abs"] = float(np.abs(rel).max())
        p["per_window_log_ratio_sd"] = float(np.log([x[w] / y[w] for w in first32]).std(ddof=1))
        out["run_to_run"][name] = p
    out["primary_margins"] = {"rp2_minus_control_upper_relative": a7["rp2_minus_control"]["relative_bca95"][1],
                              "rp2_minus_tr3_lower_relative": a7["rp2_minus_tr3"]["relative_bca95"][0]}

    # Per-window spread of the treatment effect.
    lr = np.log(r / c)
    out["per_window_rp2_over_control"] = {"min_ratio": float(np.exp(lr.min())), "max_ratio": float(np.exp(lr.max())),
                                          "median_ratio": float(np.exp(np.median(lr))), "log_ratio_sd": float(lr.std(ddof=1))}

    # Row level: the two control-fp8 runs on the original 32, and top-1 agreement with the teacher per arm.
    def npz(rel: str) -> Path:
        return find(root, rel)
    same_rows = diff_rows = agree = total = identical_windows = 0
    for w in first32:
        za = np.load(npz(f"kld-cf128/control-fp8/scores/{w}.npz"))
        zb = np.load(npz(f"kld-fp8-20260925/control-fp8/scores/{w}.npz"))
        eq = za["kld"][1:] == zb["kld"][1:]
        same_rows += int(eq.sum())
        diff_rows += int((~eq).sum())
        identical_windows += int(eq.all())
        agree += int((za["student_top1"][1:] == zb["student_top1"][1:]).sum())
        total += eq.size
    out["control_fp8_rerun_rows"] = {"windows_bit_identical": identical_windows, "windows": 32,
                                     "rows_differing_fraction": diff_rows / total,
                                     "student_top1_agreement_between_runs": agree / total}
    top1 = {}
    for arm in ("control-fp8", "rp2-fp8", "tr3-fp8"):
        hit = n = 0
        for w in ids:
            z = np.load(npz(f"kld-cf128/{arm}/scores/{w}.npz"))
            hit += int((z["student_top1"][1:] == z["teacher_top1"][1:]).sum())
            n += z["kld"][1:].size
        top1[arm] = hit / n
    out["top1_agreement_with_teacher"] = top1

    dest = find(root, "kld-cf128/analysis.json").parent / "reported-extras.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("domain_balanced_equal_weights", "gap_closed_fraction", "per_window_rp2_over_control",
                                         "control_fp8_rerun_rows", "top1_agreement_with_teacher")}, indent=1))
    for name, p in out["run_to_run"].items():
        print(f"{name:45s} {p['relative']:+.2%} [{p['relative_bca95'][0]:+.2%}, {p['relative_bca95'][1]:+.2%}]"
              f"  per-window rel SD {p['per_window_relative_sd']:.2%}  max |rel| {p['per_window_relative_max_abs']:.1%}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
