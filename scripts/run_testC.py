#!/usr/bin/env python3
"""Run Test C config by config as each capture's fit windows land on disk."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CAPTURE = Path("<data-volume>/p8-remainder-kvrot-20260923/capture/layers")
# decisive (attention layer : capture layer), then reported-only same-layer stand-ins
CONFIGS = [(19, 18), (3, 3), (23, 23), (7, 6), (15, 14), (27, 26), (35, 34), (43, 42), (43, 43)]


def ready(capture: int) -> bool:
    root = CAPTURE / f"layer-{capture:03d}"
    return (root / "hidden.bf16.bin").is_symlink() or (root / "fit-ranges-receipt.json").is_file()


def main() -> None:
    pending = list(CONFIGS)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    while pending:
        item = next((c for c in pending if ready(c[1])), None)
        if item is None:
            time.sleep(30)
            continue
        attn, cap = item
        out = HERE / f"results/testC/A{attn:02d}-C{cap:02d}.json"
        if not out.exists():
            print(f"{time.strftime('%H:%M:%S')} A{attn} C{cap}: start", flush=True)
            with (HERE / f"logs/testC-A{attn:02d}-C{cap:02d}.log").open("a") as fh:
                rc = subprocess.call(["<home>/klc-env/bin/python", "scripts/kv_hadamard.py", "--configs",
                                      f"{attn}:{cap}", "--gpu", "0", "--output", str(out)],
                                     cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
            if rc:
                sys.exit(f"A{attn} C{cap} failed rc={rc}")
        print(f"{time.strftime('%H:%M:%S')} A{attn} C{cap}: done", flush=True)
        pending.remove(item)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
