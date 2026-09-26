#!/usr/bin/env python3
"""EPI-PAR check (run inside the image with b12x-rp2par mounted).

The bit-identity gate runs first:
- prodpar must equal prod, and rp2par must equal rp2, on every M path, both layers and every
  input set.
- M > 16 must be identical across all four arms.
Timing is only read if that gate passes.

The timing uses the A-series method: one rank-0 layer per runtime, real fit-capture routes, CUDA
graphs, EP_BLOCKS blocks of EP_REPLAYS replays per arm, and the arm order rotated every block.
Per-block ratios against prod (and rp2par against rp2) are reduced to medians with a BCa-free
percentile interval over blocks (descriptive only).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402
from dx2_timing import inputs  # noqa: E402

LAYERS = [int(v) for v in os.environ.get("EP_LAYERS", "8,3").split(",")]
MS = [int(v) for v in os.environ.get("EP_MS", "1,4,16").split(",")]
BLOCKS = int(os.environ.get("EP_BLOCKS", "60"))
REPLAYS = int(os.environ.get("EP_REPLAYS", "200"))
OUT = os.environ.get("EP_OUT", "/work/results/epipar_check.json")
ARMS = {"prod": (None, None), "prodpar": (None, "1"), "rp2": ("rp2", None), "rp2par": ("rp2", "1")}


def make(overlay, layer: int, name: str):
    rem, par = ARMS[name]
    for key, val in (("B12X_P8_DOWN_REMAINDER", rem), ("B12X_P8_EPI_PAR", par)):
        if val:
            os.environ[key] = val
    try:
        return th.runtime(overlay, layer, name)
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)
        os.environ.pop("B12X_P8_EPI_PAR", None)


def main() -> None:
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    out = {"gpu": torch.cuda.get_device_name(0), "blocks": BLOCKS, "replays": REPLAYS, "identity": [], "timing": []}
    gate_ok = True
    for layer in LAYERS:
        rts = {name: make(overlay, layer, name) for name in ARMS}
        flags = {k: (bool(getattr(v, "p8_input_rowpack", False)), bool(getattr(v, "p8_epi_par", False))) for k, v in rts.items()}
        want = {"prod": (False, False), "prodpar": (False, True), "rp2": (True, False), "rp2par": (True, True)}
        if flags != want:
            raise RuntimeError(f"arm flags wrong: {flags}")
        for m in (1, 4, 16, 64, 512):
            for s in range(4 if m <= 16 else 1):
                x, w, i, start = inputs(layer, m, s)
                o = {k: rt(x, w, i).float() for k, rt in rts.items()}
                again = rts["rp2par"](x, w, i).float()
                torch.cuda.synchronize()
                row = {"layer": layer, "m": m, "set": s, "start_row": int(start),
                       "prodpar_eq_prod": bool(torch.equal(o["prodpar"], o["prod"])),
                       "rp2par_eq_rp2": bool(torch.equal(o["rp2par"], o["rp2"])),
                       "rp2par_deterministic": bool(torch.equal(again, o["rp2par"])),
                       "rp2_eq_prod": bool(torch.equal(o["rp2"], o["prod"])),
                       "finite": bool(all(torch.isfinite(v).all() for v in o.values()))}
                ok = row["prodpar_eq_prod"] and row["rp2par_eq_rp2"] and row["rp2par_deterministic"] and row["finite"]
                ok = ok and (row["rp2_eq_prod"] if m > 16 else not row["rp2_eq_prod"])
                row["pass"] = bool(ok)
                gate_ok = gate_ok and ok
                out["identity"].append(row)
                print(json.dumps(row), flush=True)
        if not gate_ok:
            break
        for m in MS:
            for s in range(4):
                x, w, i, start = inputs(layer, m, s)
                graphs = {k: th.graph(rt, x, w, i)[0] for k, rt in rts.items()}
                names = list(graphs)
                t_end = time.time() + th.WARMUP_S
                while time.time() < t_end:
                    for k in names:
                        th.timed(graphs[k], REPLAYS)
                times = {k: [] for k in names}
                for blk in range(BLOCKS):
                    order = names[blk % 4:] + names[:blk % 4]
                    for k in order:
                        times[k].append(th.timed(graphs[k], REPLAYS))
                t = {k: np.asarray(v) for k, v in times.items()}
                ratio = {f"{a}_over_{b}": t[a] / t[b] for a, b in (("rp2", "prod"), ("rp2par", "prod"), ("prodpar", "prod"), ("rp2par", "rp2"))}
                cell = {"layer": layer, "m": m, "set": s, "start_row": int(start),
                        "median_us": {k: round(float(np.median(v)), 3) for k, v in t.items()},
                        "median_ratio": {k: round(float(np.median(v)), 4) for k, v in ratio.items()},
                        "raw_us": {k: [round(float(u), 3) for u in v] for k, v in t.items()}}
                out["timing"].append(cell)
                print(json.dumps({k: v for k, v in cell.items() if k != "raw_us"}), flush=True)
                del graphs
                torch.cuda.empty_cache()
        del rts
        torch.cuda.empty_cache()
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=1)
    out["identity_gate"] = "PASS" if gate_ok else "FAIL"
    if out["timing"]:
        summary = {}
        for layer in LAYERS:
            for m in MS:
                cells = [c for c in out["timing"] if c["layer"] == layer and c["m"] == m]
                if cells:
                    summary[f"L{layer}_M{m}"] = {k: round(float(np.median([c["median_ratio"][k] for c in cells])), 4)
                                                 for k in cells[0]["median_ratio"]}
        out["summary_median_of_set_medians"] = summary
        print(json.dumps({"summary": summary}), flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"status": "done", "identity_gate": out["identity_gate"]}), flush=True)
    if out["identity_gate"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
