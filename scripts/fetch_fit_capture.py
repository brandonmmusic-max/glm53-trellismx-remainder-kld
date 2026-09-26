#!/usr/bin/env python3
"""Fetch ONLY the roles-v3 fit windows of chosen layers' routed-block captures.

Selection / confirmation / final windows are never downloaded (protected roles stay
unopened). Each fit window is written at its true byte offset in a sparse file with
the published apparent size, so the existing LayerCapture size + materialization
checks apply unchanged. Routing files (topk ids/weights) are small and fetched whole.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO = "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
HERE = Path(__file__).resolve().parents[1]
OUT = Path("<data-volume>/p8-remainder-kvrot-20260923/capture")
MANIFEST = Path("<data-volume>/bmxfp4-glm53/teacher/calibration/main-ep4-full/capture-manifest.json")


def _get(url: str, headers: dict, rng: tuple[int, int] | None, sink, attempts: int = 6) -> None:
    for attempt in range(attempts):
        try:
            h = dict(headers)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with requests.get(url, headers=h, stream=True, timeout=120, allow_redirects=True) as r:
                if rng and r.status_code != 206:
                    raise RuntimeError(f"expected 206 for a range, got {r.status_code}")
                r.raise_for_status()
                sink(r)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient transport failure
            if attempt == attempts - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt + 1} after {type(exc).__name__}: {str(exc)[:120]} (sleep {wait}s)", flush=True)
            time.sleep(wait)


def main() -> None:
    layers = [int(x) for x in sys.argv[1:]]
    manifest = json.loads(MANIFEST.read_text())
    fit = json.loads((HERE / "fit_ranges.json").read_text())
    headers = build_hf_headers()
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for layer in layers:
        info = manifest["files"][str(layer)]
        root = OUT / f"layers/layer-{layer:03d}"
        root.mkdir(parents=True, exist_ok=True)
        base = f"calibration/main-ep4-full/layers/layer-{layer:03d}"
        # routing files, whole
        for name, key in (("topk_ids.u16le.bin", "topk_ids_u16le"), ("topk_weights.f32le.bin", "topk_weights_f32le")):
            path = root / name
            if path.exists() and path.stat().st_size == info[key]["bytes"]:
                continue
            url = hf_hub_url(REPO, f"{base}/{name}", repo_type="dataset", revision=REVISION)
            tmp = path.with_suffix(path.suffix + ".part")
            with open(tmp, "wb") as fh:
                _get(url, headers, None, lambda r: [fh.write(c) for c in r.iter_content(1 << 22)])
            if tmp.stat().st_size != info[key]["bytes"]:
                raise RuntimeError(f"{name} size mismatch")
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest != info[key]["sha256"]:
                raise RuntimeError(f"{name} sha256 mismatch against capture manifest")
            os.replace(tmp, path)
        # hidden states: fit windows only, at their true offsets in a sparse file
        path = root / "hidden.bf16.bin"
        size = info["hidden_bf16"]["bytes"]
        if not path.exists():
            with open(path, "wb") as fh:
                fh.truncate(size)
        url = hf_hub_url(REPO, f"{base}/hidden.bf16.bin", repo_type="dataset", revision=REVISION)
        ranges = []
        started = time.time()
        with open(path, "r+b") as fh:
            for i, (a, b) in enumerate(fit["byte_ranges"]):
                fh.seek(a)
                h = hashlib.sha256()

                def sink(r, fh=fh, h=h):
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        h.update(chunk)

                _get(url, headers, (a, b), sink)
                if fh.tell() != b + 1:
                    raise RuntimeError(f"short range write at {a}")
                ranges.append({"start": a, "end": b, "bytes": b - a + 1, "sha256": h.hexdigest()})
                if (i + 1) % 16 == 0:
                    print(f"layer {layer}: {i + 1}/64 fit windows ({time.time() - started:.0f}s)", flush=True)
        receipts[str(layer)] = {"hidden_ranges": ranges, "apparent_bytes": size,
                                "published_full_sha256": info["hidden_bf16"]["sha256"]}
        (root / "fit-ranges-receipt.json").write_text(json.dumps(receipts[str(layer)], indent=1))
        print(f"layer {layer}: done", flush=True)
    print(json.dumps({"layers": layers, "status": "ok"}))


if __name__ == "__main__":
    main()
