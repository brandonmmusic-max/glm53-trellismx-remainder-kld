#!/usr/bin/env python3
"""Run Test B layer by layer as each layer's BF16 shards and fit capture land on disk."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = "<home>/klc-env/bin/python"
BF16 = Path("<data-volume>/p8-remainder-kvrot-20260923/bf16")
CAPTURE = Path("<data-volume>/p8-remainder-kvrot-20260923/capture/layers")
LAYERS = [18, 23, 33, 43, 3, 8]


def ready(layer: int) -> bool:
    index = json.loads((BF16 / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {v for k, v in index.items() if f"layers.{layer}.mlp.experts." in k}
    if not all((BF16 / s).is_file() for s in shards):
        return False
    root = CAPTURE / f"layer-{layer:03d}"
    return (root / "hidden.bf16.bin").is_symlink() or (root / "fit-ranges-receipt.json").is_file()


def run(cmd: list[str], log: Path) -> None:
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    with log.open("a") as fh:
        rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if rc:
        sys.exit(f"{cmd} failed rc={rc}; see {log}")


def main() -> None:
    pending = list(LAYERS)
    screen_done = (HERE / "results/testB/screen-repro.json").exists()
    while pending or not screen_done:
        layer = next((l for l in pending if ready(l)), None)
        if layer is not None:
            out = HERE / f"results/testB/layer-{layer:03d}.json"
            if not out.exists():
                print(f"{time.strftime('%H:%M:%S')} layer {layer}: start", flush=True)
                run([PY, "scripts/remainder_damage.py", "--layer", str(layer), "--gpu", "0", "--output", str(out)],
                    HERE / f"logs/testB-layer-{layer:03d}.log")
            print(f"{time.strftime('%H:%M:%S')} layer {layer}: done", flush=True)
            pending.remove(layer)
            continue
        if not screen_done and ready(3):
            run([PY, "scripts/screen_repro.py", str(HERE / "results/testB/screen-repro.json")],
                HERE / "logs/testB-screen-repro.log")
            screen_done = True
            print(f"{time.strftime('%H:%M:%S')} screen repro: done", flush=True)
            continue
        time.sleep(30)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
