#!/usr/bin/env python3
"""EXPLORATORY (after A2 failed; not preregistered): where does real D-x2 cost come from?

Arms (rank 0, same graphs/inputs as A2): prod | dx2 (both) | fc1 (epilogue writes the lo plane,
FC2 ignores it) | fc2 (FC2 stages the zero lo plane + extra MMA). fc1 and fc2 must be
bit-identical to prod. Each block times every arm once in a rotating order.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/work/scripts")
import timing_harness as th  # noqa: E402
from dx2_timing import inputs  # noqa: E402

ARMS = {"prod": None, "dx2": "both", "fc1": "fc1", "fc2": "fc2"}
MS = [int(v) for v in os.environ.get("S_MS", "1,4,16,512").split(",")]
LAYERS = [int(v) for v in os.environ.get("S_LAYERS", "8,3").split(",")]
BLOCKS, REPLAYS = int(os.environ.get("S_BLOCKS", "40")), int(os.environ.get("S_REPLAYS", "200"))


def make(overlay, layer, phases):
    if phases is not None:
        os.environ["B12X_P8_DOWN_REMAINDER"] = "1"
        os.environ["B12X_P8_DOWN_REMAINDER_PHASES"] = phases
    try:
        return th.runtime(overlay, layer, phases or "prod")
    finally:
        os.environ.pop("B12X_P8_DOWN_REMAINDER", None)
        os.environ.pop("B12X_P8_DOWN_REMAINDER_PHASES", None)


def main():
    torch.cuda.set_device(th.DEV)
    overlay = th.load_overlay("/checkpoint")
    out = {"layers": {}}
    names = list(ARMS)
    for layer in LAYERS:
        rts = {a: make(overlay, layer, p) for a, p in ARMS.items()}
        cells = []
        for m in MS:
            for s in range(2):
                x, w, i, _ = inputs(layer, m, s)
                o = {a: rts[a](x, w, i).clone() for a in names}
                torch.cuda.synchronize()
                exact = {a: bool(torch.equal(o[a], o["prod"])) for a in ("fc1", "fc2")}
                graphs = {a: th.graph(rts[a], x, w, i)[0] for a in names}
                t_end = time.time() + th.WARMUP_S
                while time.time() < t_end:
                    for a in names:
                        th.timed(graphs[a], REPLAYS)
                times = {a: [] for a in names}
                for b in range(BLOCKS):
                    for a in names[b % 4:] + names[:b % 4]:
                        times[a].append(th.timed(graphs[a], REPLAYS))
                ratio = {a: float(np.median(np.asarray(times[a]) / np.asarray(times["prod"]))) for a in names[1:]}
                cells.append({"m": m, "set": s, "exact_vs_prod": exact, "times_us": times, "ratio": ratio})
                print(json.dumps({"layer": layer, "m": m, "set": s, "exact": exact,
                                  "prod_us": round(float(np.median(times["prod"])), 2),
                                  "ratio": {k: round(v, 4) for k, v in ratio.items()}}), flush=True)
                del graphs
                torch.cuda.empty_cache()
        out["layers"][str(layer)] = cells
        del rts
        torch.cuda.empty_cache()
        with open("/work/results/dx2_split_timing_raw.json", "w") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps({"status": "done"}), flush=True)


if __name__ == "__main__":
    main()
