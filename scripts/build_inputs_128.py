#!/usr/bin/env python3
"""Verified inputs for all 128 conditional-fit windows (same record format as the 09-09 CF32 run).
The original 32 keep their order and records; the other 96 follow in window-id order."""
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, "<workspace>/trellismx-package-20260907")
from glm53_nvfp4 import p8_decode_protocol as protocol

ROOT = Path("<data-volume>/bmxfp4-glm53/teacher")
ARRAYS = Path("<workspace>/glm53-flash-kld-eval/data/teacher/calibration/panel-v1/arrays")
cf32 = json.load(open("<workspace>/p8-remainder-kvrot-20260923/kld-rp-20260925/verified-inputs.json"))
manifest = json.load(open(ROOT / "logits/full-panel/full-panel-manifest.json"))
entries = {Path(x["path"]).stem: x for x in manifest["logit_files"] if x["role"] == "conditional-fit"}
sha = lambda p: hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()


def record(wid):
    e = entries[wid]
    tp = ARRAYS / f"{wid}.tokens.npy"
    tokens = np.load(tp, allow_pickle=False)
    protocol.validate_tokens(tokens, 2048)
    teacher_ok = sha(ROOT / e["path"]) == e["sha256"]
    return {"domain": e["domain"], "id": wid, "input_sha256": sha(tp), "prediction_positions": 2047,
            "source_partition": "previously-used-conditional-fit" if wid in old else "conditional-fit-first-scored-20260925",
            "teacher_bytes_verified": teacher_ok, "teacher_path": e["path"], "teacher_sha256": e["sha256"],
            "teacher_source_role": "conditional-fit", "token_path": str(tp), "token_values_sha256": protocol.tokens_sha(tokens)}


old = {w["id"]: w for w in cf32["windows"]}
order = [w["id"] for w in cf32["windows"]] + sorted(w for w in entries if w not in old)
with ThreadPoolExecutor(8) as ex:
    recs = list(ex.map(record, order))
bad = [r["id"] for r in recs if not r["teacher_bytes_verified"]]
mismatch = [r["id"] for r in recs if r["id"] in old and any(r[k] != old[r["id"]][k] for k in ("input_sha256", "token_values_sha256", "teacher_sha256", "domain"))]
out = {k: v for k, v in cf32.items() if k != "windows"}
out.update({"windows": recs, "window_count": len(recs), "panel": "all 128 conditional-fit windows (32 previously used + 96 first scored 2026-09-25)"})
Path("<workspace>/p8-remainder-kvrot-20260923/kld-cf128/verified-inputs.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"windows": len(recs), "teacher_hash_failures": bad, "cf32_record_mismatches": mismatch,
                  "domains": {d: sum(r["domain"] == d for r in recs) for d in sorted({r["domain"] for r in recs})}}))
