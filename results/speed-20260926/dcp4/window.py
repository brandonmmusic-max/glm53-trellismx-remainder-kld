#!/usr/bin/env python3
"""Production window 4 (2026-09-26, Brandon's go-ahead): speed and behaviour of the RP2 release image against the
current production configuration, with llm-inference-bench in the usual way.

Arms, fixed order, one fresh server each on private port 8038, production serving profile (TP4/DCP4, MTP3,
48 sequences, 8192 batched tokens, GMU 0.88):
1. control:   r27 reference image ca6b8018 + serve-r27-production.sh, NVFP4 MLA KV (production today).
2. candidate: release image verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926 (RP2 + EPI-PAR M1 + guards),
              its serve-rp2.sh, FP8 MLA KV.

Per arm, in this order, with the cooling gate (>= 90 s idle, all GPUs <= 55 C for 30 s) before every bench call:
- decode, --skip-prefill, 30 s per cell: C8 at 0/8K/16K/32K; C1 at 0/8K/16K/32K/64K/128K;
- prefill: --prefill-only --prefill-contexts 8k,16k,32k,64k,128k --prefill-duration 20;
- behaviour: estonia, lavd-test, hotel-lights, each --profile-concurrency 10 --profile-runs 10 --reasoning-effort high,
  with each profile's own max-token default.

ALWAYS restores the unchanged production container via systemd and verifies /health and /v1/models.
"""
from __future__ import annotations

import fcntl
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

ROOT = Path(__file__).resolve().parent
PRODUCTION = "trellismx-reference-production-20260909"
SERVER = "p8-bench-20260926"
MODEL = "glm53-flash-trellismx-p8-k45"
PORT = 8038
PROD_BASE = "http://127.0.0.1:8000"
PROD_IMAGE = "verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf"
REL_IMAGE = "verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926"
PROD_JIT = "trellismx-r27-dcp4_trellismx-r27-jit"
REL_JIT = "trellismx-rp2-jit-20260926"
SERVE = "<workspace>/trellismx-klc-daily-driver/reference-20260909/serve-r27-production.sh"
BENCH = "<workspace>/trellismx-performance-audit-20260908/llm_decode_bench.py"
ARMS = ["control", "candidate"]
DECODE = [("8", "0"), ("8", "8192"), ("8", "16384"), ("8", "32768"),
          ("1", "0"), ("1", "8192"), ("1", "16384"), ("1", "32768"), ("1", "65536"), ("1", "131072")]
PROFILES = ["estonia", "lavd-test", "hotel-lights"]
RP_MARKER = "P8_DX2_ROWPACK_ACTIVE"
EPI_MARKER = "P8_EPI_PAR_ACTIVE"
STOP = threading.Event()
FAIL: list[str] = []
LAST_BUSY = [time.time()]
SERVICE = "trellismx-klc.service"
# Production switch (Brandon 2026-09-26: "yes switch"): if SWITCH_TO_RP2 exists and the candidate arm completed,
# production is restored on the RP2 build via a systemd drop-in; if that does not come up healthy, the drop-in is
# removed and the unchanged reference container is restored instead.
SWITCH_FLAG = ROOT / "SWITCH_TO_RP2"
DROPIN_SRC = Path("<workspace>/trellismx-klc-daily-driver/rp2-20260926/50-rp2.conf")
DROPIN_DST = Path.home() / ".config/systemd/user/trellismx-klc.service.d/50-rp2.conf"
USER_ENV = dict(os.environ, XDG_RUNTIME_DIR=f"/run/user/{os.getuid()}",
                DBUS_SESSION_BUS_ADDRESS=f"unix:path=/run/user/{os.getuid()}/bus")


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


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as f:
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


def service(action):
    return subprocess.run(["systemctl", "--user", action, SERVICE], text=True, capture_output=True, env=USER_ENV)


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
                    subprocess.run(["docker", "stop", "-t", "20", SERVER], capture_output=True)
                    return
            except Exception as e:  # noqa: BLE001
                FAIL.append(repr(e))
                return


def cooling_gate():
    t0 = time.time()
    cool_since = None
    raw = []
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


def server_argv(arm):
    env = {"PORT": str(PORT), "SERVED_MODEL_NAME": MODEL, "MODEL_ROOT": "/model",
           "VLLM_TRELLISMX_CHECKPOINT": "/checkpoint", "TP": "4", "DCP": "4", "CACHE_MODE": "vram",
           "SPECULATOR": "mtp", "NUM_SPECULATIVE_TOKENS": "3",
           "KV_CACHE_DTYPE": "nvfp4_ds_mla" if arm == "control" else "fp8",
           "MAX_MODEL_LEN": "1000000", "MAX_NUM_SEQS": "48", "MAX_NUM_BATCHED_TOKENS": "8192",
           "GPU_MEMORY_UTILIZATION": "0.88", "CP_KV_CACHE_INTERLEAVE_SIZE": "4", "DCP_CKV_GATHER": "auto",
           "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
           "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "131072", "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "86016",
           "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "4096", "VLLM_NO_USAGE_STATS": "1"}
    if arm == "candidate":
        env.update({"B12X_P8_DOWN_REMAINDER": "rp2", "B12X_P8_EPI_PAR": "1"})
    argv = ["docker", "run", "-d", "--name", SERVER, "--gpus", "all", "--network", "host", "--ipc", "host", "--workdir", "/"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    argv += ["-v", "<home>/models/GLM-5.3-Flash-NVFP4:/model:ro",
             "-v", "<model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1:/checkpoint:ro"]
    if arm == "control":
        argv += ["-v", f"{SERVE}:/release/serve-r27-production.sh:ro", "-v", f"{PROD_JIT}:/cache/jit",
                 "--entrypoint", "/bin/bash", PROD_IMAGE, "/release/serve-r27-production.sh"]
    else:
        argv += ["-v", f"{REL_JIT}:/cache/jit", REL_IMAGE]  # image ENTRYPOINT = /release/serve-rp2.sh
    return argv


def healthy(deadline):
    url = f"http://127.0.0.1:{PORT}"
    while time.time() < deadline:
        if FAIL:
            raise RuntimeError(FAIL)
        state = container_state(SERVER)
        if state is None or not state["Running"]:
            raise RuntimeError(f"{SERVER} exited: {state}")
        try:
            get(url + "/health")
            models = get(url + "/v1/models")
            if MODEL in [x["id"] for x in models["data"]]:
                return models
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError("startup deadline exceeded")


def bench(adir, name, extra, timeout):
    gate = cooling_gate()
    out = adir / f"{name}.json"
    argv = ["python3", BENCH, "--host", "127.0.0.1", "--port", str(PORT), "--model", MODEL, "--display-mode", "plain",
            "--output", str(out)] + extra
    save(adir / f"{name}-command.json", {"argv": argv, "cooling_gate": gate, "started": time.time()})
    t0 = time.time()
    try:
        r = subprocess.run(argv, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout)
        rc, text = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else "timeout"
    (adir / f"{name}.log").write_text(text if isinstance(text, str) else "")
    LAST_BUSY[0] = time.time()
    return {"name": name, "rc": rc, "seconds": round(time.time() - t0, 1), "gate_passed": gate["passed"],
            "output": out.name if out.exists() else None}


def run_arm(arm):
    adir = ROOT / arm
    adir.mkdir(parents=True, exist_ok=False)
    if arm == "candidate":
        inspect = cmd(["docker", "image", "inspect", REL_IMAGE, "--format", "{{.Id}}"]).stdout.strip()
        save(adir / "image.json", {"reference": REL_IMAGE, "local_id": inspect})
    argv = server_argv(arm)
    save(adir / "launch.json", argv)
    log(f"{arm}: starting server on :{PORT}")
    cmd(argv)
    healthy(time.time() + 3600)
    logs = cmd(["docker", "logs", SERVER])
    startup = logs.stdout + logs.stderr
    (adir / "startup.log").write_text(startup)
    rp = sum(1 for line in startup.splitlines() if RP_MARKER in line and "input_hop=1" in line)
    epi = startup.count(EPI_MARKER)
    kv_ok = ("kv_cache_dtype=nvfp4_ds_mla" if arm == "control" else "kv_cache_dtype=fp8") in startup
    kv = [line.split("GPU KV cache size:")[1].strip() for line in startup.splitlines() if "GPU KV cache size:" in line]
    audit = {"rp2_markers": rp, "epi_par_markers": epi, "kv_dtype_ok": kv_ok, "kv_cache_size": kv}
    save(adir / "runtime-audit.json", audit)
    if not kv_ok:
        raise RuntimeError(f"{arm}: wrong KV dtype")
    if arm == "candidate" and (rp != 168 or epi != 168):
        raise RuntimeError(f"candidate: expected 168 RP2 and 168 EPI-PAR markers, saw {rp}/{epi}")
    if arm == "control" and (rp or epi):
        raise RuntimeError(f"control shows remainder markers {rp}/{epi}")
    log(f"{arm}: healthy; {json.dumps(audit)}")
    results = []
    for conc, ctx in DECODE:
        res = bench(adir, f"decode-ctx{ctx}-c{conc}", ["--skip-prefill", "--contexts", ctx, "--concurrency", conc,
                                                        "--duration", "30", "--max-tokens", "2048"], 1200)
        tps = None
        if res["output"]:
            cells = json.loads((adir / res["output"]).read_text()).get("results", [])
            tps = [c.get("aggregate_tps") for c in cells if isinstance(c, dict)]
        res["aggregate_tps"] = tps
        results.append(res)
        log(json.dumps({"arm": arm, **res}))
    res = bench(adir, "prefill", ["--prefill-only", "--prefill-contexts", "8k,16k,32k,64k,128k", "--prefill-duration", "20"], 2400)
    results.append(res)
    log(json.dumps({"arm": arm, **res}))
    for prof in PROFILES:
        res = bench(adir, f"profile-{prof}", ["--test-profile", prof, "--profile-concurrency", "10", "--profile-runs", "10",
                                              "--reasoning-effort", "high"], 5400)
        results.append(res)
        log(json.dumps({"arm": arm, **res}))
    save(adir / "results-index.json", results)
    logs = cmd(["docker", "logs", SERVER])
    (adir / "server.log").write_text(logs.stdout + logs.stderr)
    cmd(["docker", "stop", "-t", "30", SERVER])
    cmd(["docker", "rm", SERVER])
    log(f"{arm}: complete")


def wait_prod_healthy(deadline):
    while time.time() < deadline:
        try:
            get(PROD_BASE + "/health")
            models = get(PROD_BASE + "/v1/models")
            if MODEL in [x["id"] for x in models["data"]]:
                return models
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(10)
    return None


def main():
    assert not (ROOT / "execution.json").exists(), "window already started; preserve it and prepare a new attempt"
    assert service("is-active").stdout.strip() == "active", "production service is not active"
    subprocess.run(["docker", "volume", "create", REL_JIT], capture_output=True)
    locks, stopped_service = [], False
    save(ROOT / "execution.json", {"status": "starting", "started": time.time(), "order": ARMS,
                                   "decode": DECODE, "profiles": PROFILES})
    th = threading.Thread(target=telemetry, daemon=True)
    th.start()
    outcome = {"status": "failed"}
    restoration = {"status": "failed"}
    try:
        for s in ["SIGTERM", "SIGINT"]:
            signal.signal(getattr(signal, s), lambda signum, frame: (_ for _ in ()).throw(RuntimeError(f"signal{signum}")))
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
        locks = [open(p, "a") for p in ["/run/lock/klc/llm-workload.lock", "/run/lock/klc/model-stack.lock"]]
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("production stopped; model-stack and llm-workload leases held")
        assert not cmd(["docker", "ps", "-q"]).stdout.strip(), "GPU window requires stopped services"
        errors = {}
        for arm in ARMS:
            try:
                run_arm(arm)
            except Exception as e:  # noqa: BLE001  (one arm failing must not skip the other or the restore)
                errors[arm] = repr(e)
                log(f"{arm}: FAILED {e!r}")
                remove_container(SERVER)
                if FAIL:
                    break
        outcome = {"status": "complete" if not errors else "partial", "errors": errors, "time": time.time()}
    except BaseException as e:  # noqa: BLE001
        outcome = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc(), "time": time.time()}
        log("FAILED: " + repr(e))
    finally:
        STOP.set()
        th.join(timeout=5)
        remove_container(SERVER)
        for lock in locks:
            lock.close()
        switched = False
        try:
            candidate_ok = (ROOT / "candidate" / "results-index.json").exists() and "candidate" not in outcome.get("errors", {})
            if stopped_service and SWITCH_FLAG.exists() and candidate_ok:
                DROPIN_DST.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DROPIN_SRC, DROPIN_DST)
                subprocess.run(["systemctl", "--user", "daemon-reload"], env=USER_ENV, capture_output=True)
                switched = True
                log("switching production to the RP2 build (drop-in 50-rp2.conf installed)")
            if stopped_service or service("is-active").stdout.strip() != "active":
                log(f"restoring production via systemctl --user start {SERVICE}")
                r = service("start")
                if r.returncode:
                    raise RuntimeError(f"systemctl start failed: {r.stderr}")
            models = wait_prod_healthy(time.time() + 2400)
            if models is None and switched:
                log("RP2 production did not become healthy; rolling back to the reference container")
                service("stop")
                DROPIN_DST.unlink(missing_ok=True)
                subprocess.run(["systemctl", "--user", "daemon-reload"], env=USER_ENV, capture_output=True)
                subprocess.run(["docker", "stop", "-t", "60", "trellismx-rp2-production-20260926"], capture_output=True)
                switched = False
                r = service("start")
                models = wait_prod_healthy(time.time() + 2400)
            if models is not None:
                restoration = {"status": "healthy", "models": models, "time": time.time(),
                               "switched_to_rp2": switched,
                               "container": "trellismx-rp2-production-20260926" if switched else PRODUCTION}
        except Exception as e:  # noqa: BLE001
            restoration = {"status": "failed", "error": repr(e), "switched_to_rp2": switched}
        restoration["service_active"] = service("is-active").stdout.strip()
        save(ROOT / "restoration.json", restoration)
        log(f"production restoration: {restoration['status']} (service {restoration['service_active']})")
        save(ROOT / "execution-final.json", outcome)
    if outcome["status"] == "failed" or restoration["status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    main()
