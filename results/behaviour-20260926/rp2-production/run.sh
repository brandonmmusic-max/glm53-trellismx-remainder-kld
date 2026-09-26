#!/bin/bash
# LAVD / Estonia / Hotel Lights, 10 runs each at concurrency 10, reasoning effort high, against production :8000
# (now the RP2 build). llm-inference-bench upstream commit 42c38fd (adds --reasoning-effort for profiles), unmodified.
cd <workspace>/p8-remainder-kvrot-20260923/bench-20260926/profiles-rp2-production
for p in hotel-lights lavd-test estonia; do
  echo "$(date '+%F %T') start $p" >> progress.log
  python3 <workspace>/glm53-exl3-k4-r10-rebase/tooling/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 8000 --model glm53-flash-trellismx-p8-k45 --display-mode plain     --test-profile $p --profile-concurrency 10 --profile-runs 10 --reasoning-effort high --output profile-$p.json > profile-$p.log 2>&1
  echo "$(date '+%F %T') end $p rc=$?" >> progress.log
done
