#!/usr/bin/env python3
"""Summarize window 4 (control vs candidate): decode cells, standalone prefill, and the three behaviour profiles.
Writes summary.json next to this script and prints markdown tables. Single runs per cell; descriptive only."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("control", "candidate")
DECODE = [("8", "0"), ("8", "8192"), ("8", "16384"), ("8", "32768"),
          ("1", "0"), ("1", "8192"), ("1", "16384"), ("1", "32768"), ("1", "65536"), ("1", "131072")]
PROFILES = ("estonia", "lavd-test", "hotel-lights")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out = {"decode": {}, "prefill": {}, "profiles": {}, "runtime": {}}
    for arm in ARMS:
        out["runtime"][arm] = load(ROOT / arm / "runtime-audit.json")
    for conc, ctx in DECODE:
        row = {}
        for arm in ARMS:
            d = load(ROOT / arm / f"decode-ctx{ctx}-c{conc}.json")
            if d and d.get("results"):
                r = d["results"][0]
                row[arm] = {"tps": r.get("aggregate_tps"), "mtp_accept": r.get("server_spec_accept_rate"),
                            "errors": r.get("num_errors"), "underfilled": r.get("underfilled"),
                            "context_tokens": r.get("context_tokens")}
        if all(a in row and row[a]["tps"] for a in ARMS):
            row["ratio_candidate_over_control"] = row["candidate"]["tps"] / row["control"]["tps"]
        out["decode"][f"C{conc} {int(ctx) // 1024}K"] = row
    for arm in ARMS:
        d = load(ROOT / arm / "prefill.json")
        if d:
            for k, v in (d.get("prefill") or {}).items():
                if isinstance(v, dict):
                    out["prefill"].setdefault(k, {})[arm] = v.get("tok_per_sec")
    for k, row in out["prefill"].items():
        if all(row.get(a) for a in ARMS):
            row["ratio_candidate_over_control"] = row["candidate"] / row["control"]
    for prof in PROFILES:
        for arm in ARMS:
            d = load(ROOT / arm / f"profile-{prof}.json")
            if not d:
                continue
            s = d.get("selected_summary") or d.get("all_summary") or {}
            ct = s.get("completion_tokens") or {}
            out["profiles"].setdefault(prof, {})[arm] = {
                "correct": s.get("correct"), "attempted": s.get("attempted"), "completed": s.get("completed"),
                "errors": s.get("errors"), "truncated": s.get("truncated"), "hit_max_tokens": s.get("hit_max_tokens"),
                "wrong": s.get("wrong"), "score_counts": s.get("score_counts"),
                "completion_tokens_p50": ct.get("p50"), "completion_tokens_p90": ct.get("p90"), "completion_tokens_max": ct.get("max"),
                "aggregate_gen_tok_s": s.get("aggregate_gen_tok_s"),
                "max_tokens": (d.get("metadata") or {}).get("max_tokens"),
                "concurrency": d.get("selected_concurrency")}
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")

    print("| cell | control tok/s (MTP accept) | candidate tok/s (MTP accept) | candidate/control |")
    print("|---|---|---|---|")
    for cell, row in out["decode"].items():
        f = lambda a: (f"{row[a]['tps']:.1f} ({row[a]['mtp_accept']:.3f})" if a in row and row[a]["tps"] and row[a]["mtp_accept"] is not None  # noqa: E731
                       else (f"{row[a]['tps']:.1f}" if a in row and row[a]["tps"] else "n/a"))
        ratio = row.get("ratio_candidate_over_control")
        print(f"| {cell} | {f('control')} | {f('candidate')} | {ratio:.3f} |" if ratio else f"| {cell} | {f('control')} | {f('candidate')} | n/a |")
    print("\n| prefill context | control tok/s | candidate tok/s | candidate/control |")
    print("|---|---|---|---|")
    for k, row in out["prefill"].items():
        r = row.get("ratio_candidate_over_control")
        print(f"| {k} | {row.get('control') or 'n/a'} | {row.get('candidate') or 'n/a'} | {f'{r:.3f}' if r else 'n/a'} |")
    print("\n| profile | control correct/attempted (p50 tokens) | candidate correct/attempted (p50 tokens) |")
    print("|---|---|---|")
    for prof, row in out["profiles"].items():
        f = lambda a: (f"{row[a]['correct']}/{row[a]['attempted']} ({row[a]['completion_tokens_p50']})" if a in row else "n/a")  # noqa: E731
        print(f"| {prof} | {f('control')} | {f('candidate')} |")


if __name__ == "__main__":
    main()
