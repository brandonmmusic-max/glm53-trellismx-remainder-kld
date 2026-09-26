#!/usr/bin/env python3
"""DCP1 speed check (Brandon 2026-09-26 ~12:30): the RP2 release image, then the r27 reference image, both at
TP4/DCP1 with MTP3, 24 sequences, 4,096 batched tokens, GMU 0.97 (his Sept 9 scheduler settings).

Method = his Sept 9 published method:
- decode via fixed-workload-candidate-comparison-01/bench_fixed.py (greedy, temperature 0, fixed inputs, exact
  token targeting, 20 s, max 8,192 tokens);
- prefill via llm_decode_bench v0.4.29 --prefill-only.
The cooling gate (>= 90 s idle, all GPUs <= 55 C for 30 s) runs before every cell.

Production stays stopped during the run, and PaddleOCR (GPU3) is paused. At the end both are restored.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = "p8-dcp1-speed-20260926"
MODEL = "glm53-flash-trellismx-p8-k45"
PORT = 8038
BENCH = "<workspace>/trellismx-performance-audit-20260908/llm_decode_bench.py"
FIXED = "<workspace>/trellismx-tp4-swarm-20260909/fixed-workload-candidate-comparison-01/bench_fixed.py"
ARMS = {
    "rp2": {"image": "verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926", "jit": "trellismx-rp2-jit-20260926",
            "env": {"KV_CACHE_DTYPE": "fp8", "B12X_P8_DOWN_REMAINDER": "rp2", "B12X_P8_EPI_PAR": "1"}},
    "control": {"image": "verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf",
                "jit": "trellismx-r27-dcp4_trellismx-r27-jit", "env": {"KV_CACHE_DTYPE": "nvfp4_ds_mla"}},
}
DECODE = [("1", "0"), ("1", "8k"), ("1", "16k"), ("1", "32k"), ("1", "64k"), ("1", "128k"),
          ("8", "0"), ("8", "8k"), ("8", "16k"), ("8", "32k")]
STOP = threading.Event()
LAST_BUSY = [time.time()]
USER_ENV = dict(os.environ, XDG_RUNTIME_DIR=f"/run/user/{os.getuid()}",
                DBUS_SESSION_BUS_ADDRESS=f"unix:path=/run/user/{os.getuid()}/bus")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (ROOT / "run.log").open("a") as f:
        f.write(line + "\n")


def sh(argv, **kw):
    return subprocess.run(argv, text=True, capture_output=True, **kw)


def telemetry(path):
    with open(path, "a", buffering=1) as f:
        while not STOP.wait(1):
            raw = sh(["nvidia-smi", "--query-gpu=index,clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                      "--format=csv,noheader,nounits"]).stdout
            f.write(json.dumps({"t": time.time(), "gpu": raw}) + "\n")
            if any(float(line.split(",")[4]) > 5 for line in raw.strip().splitlines()):
                LAST_BUSY[0] = time.time()


def cooling_gate():
    t0, cool_since = time.time(), None
    while time.time() - t0 < 900:
        temps = sh(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]).stdout.split()
        now = time.time()
        cool_since = (cool_since or now) if all(float(x) <= 55 for x in temps) else None
        if cool_since and now - cool_since >= 30 and now - LAST_BUSY[0] >= 90:
            return {"waited_s": round(now - t0, 1), "passed": True, "temps": temps}
        time.sleep(2)
    return {"waited_s": 900, "passed": False, "temps": temps}


def server_argv(arm):
    a = ARMS[arm]
    env = {"PORT": str(PORT), "SERVED_MODEL_NAME": MODEL, "MODEL_ROOT": "/model", "VLLM_TRELLISMX_CHECKPOINT": "/checkpoint",
           "TP": "4", "DCP": "1", "CACHE_MODE": "vram", "SPECULATOR": "mtp", "NUM_SPECULATIVE_TOKENS": "3",
           "MAX_MODEL_LEN": "1000000", "MAX_NUM_SEQS": "24", "MAX_NUM_BATCHED_TOKENS": "4096",
           "GPU_MEMORY_UTILIZATION": "0.97", "DCP_CKV_GATHER": "auto", "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
           "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "131072", "VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "86016",
           "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "4096", "VLLM_NO_USAGE_STATS": "1"}
    env.update(a["env"])
    argv = ["docker", "run", "-d", "--name", SERVER, "--gpus", "all", "--network", "host", "--ipc", "host", "--workdir", "/"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    argv += ["-v", "<home>/models/GLM-5.3-Flash-NVFP4:/model:ro",
             "-v", "<model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1:/checkpoint:ro",
             "-v", f"{a['jit']}:/cache/jit", "--entrypoint", "/bin/bash", a["image"],
             "-c", "exec /usr/local/bin/serve-glm53-flash.sh /model"]
    return argv


def healthy(deadline):
    while time.time() < deadline:
        st = sh(["docker", "inspect", SERVER, "--format", "{{.State.Running}}"]).stdout.strip()
        if st != "true":
            raise RuntimeError("server exited")
        try:
            models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read())
            if MODEL in [m["id"] for m in models["data"]]:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError("startup deadline")


def run_arm(arm):
    adir = ROOT / arm
    adir.mkdir(parents=True, exist_ok=False)
    argv = server_argv(arm)
    (adir / "launch.json").write_text(json.dumps(argv, indent=1))
    log(f"{arm}: starting DCP1 server")
    r = sh(argv)
    if r.returncode:
        raise RuntimeError(r.stderr)
    healthy(time.time() + 3600)
    startup = sh(["docker", "logs", SERVER]).stdout + sh(["docker", "logs", SERVER]).stderr
    (adir / "startup.log").write_text(startup)
    kv = [x.split("GPU KV cache size:")[1].strip() for x in startup.splitlines() if "GPU KV cache size:" in x]
    rp2 = sum(1 for x in startup.splitlines() if "P8_DX2_ROWPACK_ACTIVE" in x and "input_hop=1" in x)
    epi = startup.count("P8_EPI_PAR_ACTIVE")
    dcp_line = [x for x in startup.splitlines() if "decode_context_parallel_size" in x or "dcp_size" in x][:2]
    log(f"{arm}: healthy; kv {kv[:1]}; rp2 markers {rp2}; epi markers {epi}; dcp lines {len(dcp_line)}")
    for conc, ctx in DECODE:
        gate = cooling_gate()
        out = adir / f"decode-{ctx}-c{conc}.json"
        cmdv = ["/usr/bin/python3", FIXED, "--host", "127.0.0.1", "--port", str(PORT), "--model", MODEL,
                "--duration", "20", "--max-tokens", "8192", "--token-targeting", "exact", "--display-mode", "plain",
                "--output", str(out), "--contexts", ctx, "--concurrency", conc, "--skip-prefill",
                "--cell-warmup-timeout-seconds", "180", "--temperature", "0"]
        (adir / f"decode-{ctx}-c{conc}-command.json").write_text(json.dumps({"argv": cmdv, "cooling_gate": gate}))
        t0 = time.time()
        r = sh(cmdv, stdin=subprocess.DEVNULL, timeout=1500)
        (adir / f"decode-{ctx}-c{conc}.log").write_text(r.stdout + r.stderr)
        LAST_BUSY[0] = time.time()
        tps = acc = None
        if out.exists():
            cell = json.loads(out.read_text())["results"][0]
            tps, acc = cell.get("aggregate_tps"), cell.get("server_spec_accept_rate")
        log(json.dumps({"arm": arm, "cell": f"C{conc} {ctx}", "rc": r.returncode, "tps": tps, "mtp_accept": acc,
                        "seconds": round(time.time() - t0, 1), "gate": gate["passed"]}))
    gate = cooling_gate()
    cmdv = ["python3", BENCH, "--host", "127.0.0.1", "--port", str(PORT), "--model", MODEL, "--display-mode", "plain",
            "--output", str(adir / "prefill.json"), "--prefill-only", "--prefill-contexts", "8k,16k,32k,64k,128k",
            "--prefill-duration", "20"]
    r = sh(cmdv, stdin=subprocess.DEVNULL, timeout=2400)
    (adir / "prefill.log").write_text(r.stdout + r.stderr)
    LAST_BUSY[0] = time.time()
    pre = {}
    if (adir / "prefill.json").exists():
        pre = {k: v.get("tok_per_sec") for k, v in (json.loads((adir / "prefill.json").read_text()).get("prefill") or {}).items()
               if isinstance(v, dict)}
    log(json.dumps({"arm": arm, "prefill": pre, "rc": r.returncode, "gate": gate["passed"]}))
    logs = sh(["docker", "logs", SERVER])
    (adir / "server.log").write_text(logs.stdout + logs.stderr)
    sh(["docker", "stop", "-t", "30", SERVER])
    sh(["docker", "rm", SERVER])
    log(f"{arm}: complete")


def main():
    assert sh(["systemctl", "--user", "is-active", "trellismx-klc.service"], env=USER_ENV).stdout.strip() != "active", \
        "production is running; this harness expects it stopped"
    locks = [open(p, "a") for p in ["/run/lock/klc/llm-workload.lock", "/run/lock/klc/model-stack.lock"]]
    for lk in locks:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sh(["systemctl", "--user", "stop", "klc-paddleocr-gpu.service"], env=USER_ENV)
    log("leases held; PaddleOCR paused; GPU3 gpc offset " + sh(["sudo", "-n", "<home>/klc-env/bin/python", "-c",
        "import pynvml as n; n.nvmlInit(); print([n.nvmlDeviceGetGpcClkVfOffset(n.nvmlDeviceGetHandleByIndex(i)) for i in range(4)])"]).stdout.strip())
    th = threading.Thread(target=telemetry, args=(ROOT / "telemetry.jsonl",), daemon=True)
    th.start()
    try:
        for arm in ARMS:
            try:
                run_arm(arm)
            except Exception as e:  # noqa: BLE001
                log(f"{arm}: FAILED {e!r}")
                sh(["docker", "rm", "-f", SERVER])
    finally:
        STOP.set()
        sh(["docker", "rm", "-f", SERVER])
        for lk in locks:
            lk.close()
        sh(["systemctl", "--user", "start", "klc-paddleocr-gpu.service"], env=USER_ENV)
        log("restoring production (systemctl --user start trellismx-klc.service) and PaddleOCR")
        sh(["systemctl", "--user", "start", "trellismx-klc.service"], env=USER_ENV)
        t0 = time.time()
        while time.time() - t0 < 2400:
            try:
                models = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5).read())
                if MODEL in [m["id"] for m in models["data"]]:
                    log("production healthy")
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(10)
        else:
            log("production NOT healthy after 40 min")


if __name__ == "__main__":
    main()
