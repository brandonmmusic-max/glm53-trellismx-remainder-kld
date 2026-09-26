#!/usr/bin/env python3
"""Production window 2 (2026-09-25, PREREG amendment 5): FP8 MLA KV, control-fp8 vs rp-fp8.

1. Decode-scored KLD with the unchanged 09-09 reference protocol, `control` then `rp`
   (D-x2-RP: patched b12x mounted + B12X_P8_DOWN_REMAINDER=rp), one fresh capture server each.
2. Descriptive C1/C4 speed with the production serving config on private port 8038.
3. ALWAYS restores the unchanged production container and verifies /health and /v1/models.

Capture/score/analysis logic is copied from trellismx-reference-release-20260909/kld/run.py.
"""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
REF = Path("<workspace>/trellismx-reference-release-20260909/kld")
sys.path.insert(0, "<workspace>/trellismx-package-20260907")
from glm53_nvfp4 import p8_decode_protocol as protocol  # noqa: E402

PRODUCTION = "trellismx-reference-production-20260909"
CAPTURE = "p8-fp8-cf32-20260925"
SPEED = "p8-fp8-speed-20260925"
MODEL = "glm53-flash-trellismx-p8-k45"
KLD_BASE = "http://127.0.0.1:8037"
SPEED_PORT = 8038
PROD_BASE = "http://127.0.0.1:8000"
CAPTURE_IMAGE = "sha256:0405a1c0dc128b51069798a5d00b346257bbd006c0deb7e6530a3d005d75de71"
PROD_IMAGE = "verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf"
DX2 = "<workspace>/p8-remainder-kvrot-20260923/b12x-dx2"
PROD_JIT = "trellismx-r27-dcp4_trellismx-r27-jit"
RP_JIT = "p8-rp-jit-20260925"
SERVE = "<workspace>/trellismx-klc-daily-driver/reference-20260909/serve-r27-production.sh"
BENCH = "<workspace>/trellismx-performance-audit-20260908/llm_decode_bench.py"
ARMS = ["control-fp8", "rp-fp8"]
IS_RP = {"control-fp8": False, "rp-fp8": True}
RP_MARKER = "P8_DX2_ROWPACK_ACTIVE"
STOP = threading.Event()
FAIL: list[str] = []
LAST_BUSY = [time.time()]


def sha(p):
    return hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(x, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (ROOT / "window.log").open("a") as f:
        f.write(line + "\n")


def cmd(argv, **kw):
    return subprocess.run(argv, text=True, capture_output=True, check=True, **kw)


def get(url, data=None, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        b = f.read()
        return json.loads(b) if b else None


def container_state(name):
    r = subprocess.run(["docker", "inspect", name, "--format", "{{json .State}}"], text=True, capture_output=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def remove_container(name):
    if container_state(name) is not None:
        logs = subprocess.run(["docker", "logs", name], text=True, capture_output=True)
        (ROOT / f"{name}.last.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "stop", "-t", "30", name], capture_output=True)
        subprocess.run(["docker", "rm", name], capture_output=True)


def telemetry():
    with (ROOT / "telemetry.jsonl").open("a", buffering=1) as f:
        while not STOP.wait(2):
            try:
                raw = cmd(["nvidia-smi", "--query-gpu=index,temperature.gpu,memory.used,power.draw,clocks.sm,clocks.mem,utilization.gpu",
                           "--format=csv,noheader,nounits"]).stdout
                free = shutil.disk_usage(ROOT).free
                f.write(json.dumps({"time": time.time(), "gpu": raw, "root_free": free}) + "\n")
                rows = [line.split(",") for line in raw.strip().splitlines()]
                if any(float(r[6]) > 5 for r in rows):
                    LAST_BUSY[0] = time.time()
                if any(float(r[1]) >= 87 for r in rows) or free < 10 * 1024 ** 3:
                    FAIL.append("temperature>=87C or filesystem free<10GiB")
                    for name in (CAPTURE, SPEED):
                        subprocess.run(["docker", "stop", "-t", "20", name], capture_output=True)
                    return
            except Exception as e:  # noqa: BLE001
                FAIL.append(repr(e))
                return


def healthy(container, url, deadline):
    while time.time() < deadline:
        if FAIL:
            raise RuntimeError(FAIL)
        state = container_state(container)
        if state is None or not state["Running"]:
            raise RuntimeError(f"{container} exited: {state}")
        try:
            get(url + "/health")
            models = get(url + "/v1/models")
            if MODEL not in [x["id"] for x in models["data"]]:
                raise RuntimeError("wrong model identity")
            return models
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"{container}: startup deadline exceeded")


def kld_argv(arm):
    ref = json.loads((REF / "launch-argv.json").read_text())["nvfp4"]
    argv = list(ref)

    def swap(old, new):
        idx = [k for k, v in enumerate(argv) if v == old]
        if len(idx) != 1:
            raise RuntimeError(f"launch argv: expected one {old!r}, found {len(idx)}")
        argv[idx[0]] = new

    swap("trellismx-reference-cf32-20260909", CAPTURE)
    swap("KV_CACHE_DTYPE=nvfp4_ds_mla", "KV_CACHE_DTYPE=fp8")
    swap("<model-volume>/trellismx-reference-kld-20260909/nvfp4/captures:/p8-captures",
         f"{ROOT / arm / 'captures'}:/p8-captures")
    if IS_RP[arm]:
        swap(f"{PROD_JIT}:/cache/jit", f"{RP_JIT}:/cache/jit")
        image_at = argv.index(CAPTURE_IMAGE)
        argv[image_at:image_at] = ["-v", f"{DX2}:/opt/glm53-flash/b12x:ro", "-e", "B12X_P8_DOWN_REMAINDER=rp"]
    return argv


def speed_argv(arm):
    env = {"PORT": str(SPEED_PORT), "SERVED_MODEL_NAME": MODEL, "MODEL_ROOT": "/model",
           "VLLM_TRELLISMX_CHECKPOINT": "/checkpoint", "TP": "4", "DCP": "4", "CACHE_MODE": "vram",
           "SPECULATOR": "mtp", "NUM_SPECULATIVE_TOKENS": "3", "KV_CACHE_DTYPE": "fp8",
           "MAX_MODEL_LEN": "1000000", "MAX_NUM_SEQS": "48", "MAX_NUM_BATCHED_TOKENS": "8192",
           "GPU_MEMORY_UTILIZATION": "0.88", "CP_KV_CACHE_INTERLEAVE_SIZE": "4", "DCP_CKV_GATHER": "auto",
           "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
           "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "131072", "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "86016",
           "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "4096", "VLLM_NO_USAGE_STATS": "1"}
    if IS_RP[arm]:
        env["B12X_P8_DOWN_REMAINDER"] = "rp"
    argv = ["docker", "run", "-d", "--name", SPEED, "--gpus", "all", "--network", "host", "--ipc", "host",
            "--workdir", "/", "--entrypoint", "/bin/bash"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    argv += ["-v", "<home>/models/GLM-5.3-Flash-NVFP4:/model:ro",
             "-v", "<model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1:/checkpoint:ro",
             "-v", f"{SERVE}:/release/serve-r27-production.sh:ro",
             "-v", f"{RP_JIT if IS_RP[arm] else PROD_JIT}:/cache/jit"]
    if IS_RP[arm]:
        argv += ["-v", f"{DX2}:/opt/glm53-flash/b12x:ro"]
    argv += [PROD_IMAGE, "/release/serve-r27-production.sh", "--host", "127.0.0.1"]
    return argv


def score(window, armdir, teacher_root):
    wid = window["id"]
    tokens = np.load(window["token_path"], allow_pickle=False)
    assert sha(window["token_path"]) == window["input_sha256"]
    assert protocol.tokens_sha(tokens) == window["token_values_sha256"]
    student, meta = protocol.load_capture((armdir / "captures").resolve(), wid, tokens)
    teacher_path = Path(teacher_root) / window["teacher_path"]
    assert sha(teacher_path) == window["teacher_sha256"]
    with safe_open(str(teacher_path), framework="numpy") as h:
        teacher = h.get_tensor("logits")
    scores = protocol.score_aligned(teacher, student, tokens)
    assert len(scores.kld) == 2047 and np.isfinite(scores.kld).all()
    score_path = armdir / "scores" / f"{wid}.npz"
    with score_path.open("wb") as f:
        np.savez_compressed(f, **dataclasses.asdict(scores))
        f.flush()
        os.fsync(f.fileno())
    record = {"window_id": wid, "domain": window["domain"], "prediction_rows": 2047, "true_decode_rows": 2046,
              "true_decode_mean_kld": float(scores.kld[1:].mean()), "including_prefill_mean_kld": float(scores.kld.mean()),
              "prefill_row_kld": float(scores.kld[0]), "raw_sha256": meta["raw_sha256"], "score_sha256": sha(score_path),
              "teacher_sha256": window["teacher_sha256"], "token_sha256": window["token_values_sha256"],
              "raw_retirement": "after durable score and verified raw hash", "raw_bytes": meta["raw_bytes"]}
    save(armdir / "scores" / f"{wid}.json", record)
    del teacher, student, scores
    raw = armdir / "captures" / f"{wid}.logits.f32"
    assert sha(raw) == meta["raw_sha256"] and raw.stat().st_size == 1268157440
    raw.unlink()
    save(armdir / "scores" / f"{wid}.retirement.json", {"raw_sha256": meta["raw_sha256"], "score_sha256": sha(score_path),
                                                        "retired_bytes": 1268157440, "time": time.time()})
    return record


def cache_hits():
    raw = urllib.request.urlopen(KLD_BASE + "/metrics", timeout=10).read().decode()
    rows = [float(line.split()[-1]) for line in raw.splitlines() if line.startswith("vllm:prefix_cache_hits_total{")]
    if len(rows) != 1:
        raise RuntimeError("prefix cache hit counter missing/ambiguous")
    return rows[0], raw


def run_kld_arm(arm, inputs):
    armdir = ROOT / arm
    assert not armdir.exists(), f"{armdir} exists; preserve it and use a new attempt"
    (armdir / "captures").mkdir(parents=True)
    (armdir / "scores").mkdir()
    (armdir / "requests").mkdir()
    (armdir / "captures").chmod(0o777)
    argv = kld_argv(arm)
    save(armdir / "launch.json", argv)
    log(f"KLD {arm}: starting capture server")
    cmd(argv)
    healthy(CAPTURE, KLD_BASE, time.time() + (3600 if IS_RP[arm] else 1200))
    logs = cmd(["docker", "logs", CAPTURE])
    startup = logs.stdout + logs.stderr
    (armdir / "startup.log").write_text(startup)
    for marker in ["Using V2 Model Runner", "GLM53_P8_DECODE_CAPTURE_V2_READY", "WARMUP_SCOPE_CLOSED",
                   "'attention_backend': 'B12X'", "kv_cache_dtype=fp8"]:
        assert marker in startup, marker
    if "Traceback (most recent call last)" in startup:
        raise RuntimeError("startup traceback")
    rp_lines = startup.count(RP_MARKER)
    if "kv_cache_dtype=nvfp4_ds_mla" in startup:
        raise RuntimeError("fp8 arm started with NVFP4 KV")
    if IS_RP[arm] and rp_lines == 0:
        raise RuntimeError("rp arm: P8_DX2_ROWPACK_ACTIVE missing, D-x2-RP not active")
    if not IS_RP[arm] and rp_lines:
        raise RuntimeError("control arm shows P8_DX2_ROWPACK_ACTIVE")
    kv_tokens = [line.split("GPU KV cache size:")[1].strip() for line in startup.splitlines() if "GPU KV cache size:" in line]
    save(armdir / "runtime-audit.json", {"startup_sha256": sha(armdir / "startup.log"), "capture_image_id": CAPTURE_IMAGE,
                                         "kv_cache": "fp8", "kv_cache_size": kv_tokens, "rp_marker_lines": rp_lines, "started": time.time()})
    log(f"KLD {arm}: server healthy ({rp_lines} {RP_MARKER} lines); capturing 32 windows")
    start = time.time()
    for window in inputs["windows"]:
        if FAIL:
            raise RuntimeError(FAIL)
        if time.time() - start > 3600:
            raise TimeoutError("arm deadline 60 min")
        wid = window["id"]
        tokens = np.load(window["token_path"], allow_pickle=False)
        req = protocol.completion_request(tokens, wid, MODEL)
        save(armdir / "requests" / f"{wid}.request.json", req)
        before_hits, before_metrics = cache_hits()
        (armdir / "requests" / f"{wid}.metrics-before.txt").write_text(before_metrics)
        t = time.time()
        response = get(KLD_BASE + "/v1/completions", req, timeout=300)
        save(armdir / "requests" / f"{wid}.response.json", response)
        after_hits, after_metrics = cache_hits()
        (armdir / "requests" / f"{wid}.metrics-after.txt").write_text(after_metrics)
        assert after_hits == before_hits == 0, "nonzero prefix-cache hits invalidate this window"
        check = protocol.verify_response(response, req)
        save(armdir / "requests" / f"{wid}.alignment.json", check)
        cmd(["docker", "exec", CAPTURE, "chown", f"{os.getuid()}:{os.getgid()}", f"/p8-captures/{wid}.logits.f32",
             f"/p8-captures/{wid}.capture.json"])
        record = score(window, armdir, inputs["teacher_root"])
        log(json.dumps({"arm": arm, "window": wid, "kld": record["true_decode_mean_kld"], "seconds": round(time.time() - t, 1)}))
    logs = cmd(["docker", "logs", CAPTURE])
    (armdir / "server.log").write_text(logs.stdout + logs.stderr)
    cmd(["docker", "stop", "-t", "30", CAPTURE])
    save(armdir / "state.json", container_state(CAPTURE))
    cmd(["docker", "rm", CAPTURE])
    save(armdir / "COMPLETE.json", {"windows": 32, "time": time.time()})
    log(f"KLD {arm}: complete")


def cooling_gate():
    t0 = time.time()
    cool_since = None
    while time.time() - t0 < 900:
        raw = cmd(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]).stdout.split()
        now = time.time()
        if all(float(x) <= 55 for x in raw):
            cool_since = cool_since or now
        else:
            cool_since = None
        if cool_since and now - cool_since >= 30 and now - LAST_BUSY[0] >= 90:
            return {"waited_s": round(now - t0, 1), "passed": True, "temps": raw}
        time.sleep(2)
    return {"waited_s": 900, "passed": False, "temps": raw}


def run_speed_arm(arm):
    sdir = ROOT / "speed" / arm
    sdir.mkdir(parents=True, exist_ok=False)
    argv = speed_argv(arm)
    save(sdir / "launch.json", argv)
    log(f"speed {arm}: starting production-config server on :{SPEED_PORT}")
    cmd(argv)
    healthy(SPEED, f"http://127.0.0.1:{SPEED_PORT}", time.time() + (3600 if IS_RP[arm] else 1800))
    logs = cmd(["docker", "logs", SPEED])
    startup = logs.stdout + logs.stderr
    (sdir / "startup.log").write_text(startup)
    rp_lines = startup.count(RP_MARKER)
    if "kv_cache_dtype=fp8" not in startup:
        raise RuntimeError(f"speed {arm}: not started with FP8 KV")
    if IS_RP[arm] != (rp_lines > 0):
        raise RuntimeError(f"speed {arm}: {RP_MARKER} lines = {rp_lines}")
    for ctx in ("0", "8192"):
        for conc in ("1", "4"):
            gate = cooling_gate()
            out = sdir / f"ctx{ctx}-c{conc}.json"
            bench = ["python3", BENCH, "--host", "127.0.0.1", "--port", str(SPEED_PORT), "--model", MODEL,
                     "--skip-prefill", "--contexts", ctx, "--concurrency", conc, "--duration", "30",
                     "--max-tokens", "2048", "--cell-warmup-timeout-seconds", "180", "--display-mode", "plain",
                     "--output", str(out)]
            save(sdir / f"ctx{ctx}-c{conc}-command.json", {"argv": bench, "cooling_gate": gate})
            r = subprocess.run(bench, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=900)
            (sdir / f"ctx{ctx}-c{conc}.log").write_text(r.stdout + r.stderr)
            LAST_BUSY[0] = time.time()
            tps = None
            if out.exists():
                cells = json.loads(out.read_text()).get("results", [])
                tps = [c.get("aggregate_tps") for c in cells if isinstance(c, dict)]
            log(json.dumps({"speed": arm, "ctx": ctx, "conc": conc, "rc": r.returncode, "aggregate_tps": tps, "gate": gate}))
    logs = cmd(["docker", "logs", SPEED])
    (sdir / "server.log").write_text(logs.stdout + logs.stderr)
    cmd(["docker", "stop", "-t", "30", SPEED])
    cmd(["docker", "rm", SPEED])
    log(f"speed {arm}: complete")


def analyze(inputs):
    from scipy.stats import bootstrap
    arms, means = {}, {}
    for arm in ARMS:
        rows = [json.loads((ROOT / arm / "scores" / f"{w['id']}.json").read_text()) for w in inputs["windows"]]
        values = np.array([r["true_decode_mean_kld"] for r in rows])
        means[arm] = values
        ci = bootstrap((values,), np.mean, method="BCa", n_resamples=20000, random_state=20260902)
        arms[arm] = {"mean_true_decode_kld": float(values.mean()),
                     "window_bca95": [float(ci.confidence_interval.low), float(ci.confidence_interval.high)], "per_window": rows}
    delta = means["rp-fp8"] - means["control-fp8"]
    ci = bootstrap((delta,), np.mean, method="BCa", n_resamples=20000, random_state=20260902)
    ref = json.loads(Path("<workspace>/trellismx-reference-release-20260909/hf/results/"
                          "kld-reference-20260909/comparison.json").read_text())["arms"]["fp8"]["per_window"]
    ref_by = {r["window_id"]: r["true_decode_mean_kld"] for r in ref}
    ctrl_by = {r["window_id"]: r["true_decode_mean_kld"] for r in arms["control-fp8"]["per_window"]}
    repro = [ctrl_by[w] - ref_by[w] for w in ctrl_by]
    lo, hi = float(ci.confidence_interval.low), float(ci.confidence_interval.high)
    reading = "improvement" if (delta.mean() < 0 and hi < 0) else ("harm" if (delta.mean() > 0 and lo > 0) else "no detectable change")
    result = {"schema": "p8-dx2-rp-kld-cf32-fp8kv.v1",
              "evidence": "development comparison on already-opened conditional-fit; FP8 MLA KV; one server per arm, fixed order control-fp8->rp-fp8; not final qualification",
              "arms": arms, "rp_minus_control": float(delta.mean()), "paired_window_bca95": [lo, hi],
              "relative_change": float(delta.mean() / means["control-fp8"].mean()),
              "rp_lower_windows": int((delta < 0).sum()), "reading": reading,
              "control_vs_published_20260909": {"published_mean": 0.03194517316654619,
                                                 "max_abs_window_diff": float(np.max(np.abs(repro))),
                                                 "identical_windows": int(sum(abs(x) == 0.0 for x in repro))},
              "limits": ["MTP off, maxseq1 forced decode: every token uses the M1 direct path, where D-x2-RP is active.",
                         "Prefill (M>16) is unchanged by design; the prefill row is excluded from the score.",
                         "Fixed order, one server preparation per arm; window intervals do not capture server-run variability."]}
    save(ROOT / "analysis.json", result)
    log(json.dumps({k: v for k, v in result.items() if k != "arms"}))


SERVICE = "trellismx-klc.service"
USER_ENV = dict(os.environ, XDG_RUNTIME_DIR=f"/run/user/{os.getuid()}",
                DBUS_SESSION_BUS_ADDRESS=f"unix:path=/run/user/{os.getuid()}/bus")


def service(action):
    return subprocess.run(["systemctl", "--user", action, SERVICE], text=True, capture_output=True, env=USER_ENV)


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    inputs = json.loads((ROOT / "verified-inputs.json").read_text())
    assert not (ROOT / "execution.json").exists(), "window already started; preserve it and prepare a new attempt"
    assert service("is-active").stdout.strip() == "active", "production service is not active; nothing to take over"
    locks = []
    stopped_service = False
    save(ROOT / "execution.json", {"status": "starting", "started": time.time(), "order": ARMS + ["speed-rp-fp8"]})
    th = threading.Thread(target=telemetry, daemon=True)
    th.start()
    outcome = {"status": "failed"}
    try:
        for s in ["SIGTERM", "SIGINT"]:
            signal.signal(getattr(signal, s), lambda signum, frame: (_ for _ in ()).throw(RuntimeError(f"signal{signum}")))
        # Drain: production idle for 30 s before stopping it.
        idle_since, t0 = None, time.time()
        while True:
            m = urllib.request.urlopen(PROD_BASE + "/metrics", timeout=10).read().decode()
            busy = sum(float(line.split()[-1]) for line in m.splitlines()
                       if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"))
            idle_since = (idle_since or time.time()) if busy == 0 else None
            if idle_since and time.time() - idle_since >= 30:
                break
            if time.time() - t0 > 1800:
                raise TimeoutError("production did not drain within 30 min")
            time.sleep(3)
        save(ROOT / "production-before.json", container_state(PRODUCTION))
        log(f"production idle 30 s; stopping {SERVICE} for the approved window")
        stopped_service = True
        r = service("stop")
        if r.returncode:
            raise RuntimeError(f"systemctl stop failed: {r.stderr}")
        t0 = time.time()
        while container_state(PRODUCTION)["Running"]:
            if time.time() - t0 > 180:
                raise TimeoutError("production container still running 180 s after service stop")
            time.sleep(3)
        # The service released the model-stack lease; hold both leases so no recovery can start mid-window.
        locks = [open(p, "a") for p in ["/run/lock/klc/llm-workload.lock", "/run/lock/klc/model-stack.lock"]]
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("production stopped; model-stack and llm-workload leases held")
        assert not cmd(["docker", "ps", "-q"]).stdout.strip(), "GPU window requires stopped services"
        for arm in ARMS:
            run_kld_arm(arm, inputs)
        analyze(inputs)
        run_speed_arm("rp-fp8")
        outcome = {"status": "complete", "time": time.time()}
    except BaseException as e:  # noqa: BLE001
        outcome = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc(), "time": time.time()}
        log("FAILED: " + repr(e))
    finally:
        STOP.set()
        th.join(timeout=5)
        for name in (CAPTURE, SPEED):
            remove_container(name)
        restoration = {"status": "failed"}
        for lock in locks:
            lock.close()  # release the leases first: run_dcp4.sh must take the model-stack lease
        try:
            if stopped_service or service("is-active").stdout.strip() != "active":
                log(f"restoring production via systemctl --user start {SERVICE} (adopts the unchanged container)")
                r = service("start")
                if r.returncode:
                    raise RuntimeError(f"systemctl start failed: {r.stderr}")
            deadline = time.time() + 2400
            while time.time() < deadline:
                try:
                    get(PROD_BASE + "/health")
                    models = get(PROD_BASE + "/v1/models")
                    if MODEL in [x["id"] for x in models["data"]]:
                        restoration = {"status": "healthy", "models": models, "time": time.time(), "container": PRODUCTION}
                        break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(10)
        except Exception as e:  # noqa: BLE001
            restoration = {"status": "failed", "error": repr(e)}
        restoration["service_active"] = service("is-active").stdout.strip()
        save(ROOT / "restoration.json", restoration)
        log(f"production restoration: {restoration['status']} (service {restoration['service_active']})")
        save(ROOT / "execution-final.json", outcome)
    if outcome["status"] != "complete" or restoration["status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    main()
