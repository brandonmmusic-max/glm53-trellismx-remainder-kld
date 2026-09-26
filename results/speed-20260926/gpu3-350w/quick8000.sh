#!/bin/bash
# quick.sh ARM CTX_LIST : C1 greedy decode (Sept 9 bench_fixed method) + prefill 8k,32k,128k against :8038.
ARM=$1; CTXS=$2; W=<workspace>/p8-remainder-kvrot-20260923/dcp1-speed-20260926; D=<workspace>/p8-remainder-kvrot-20260923/prod-speed-gpu3-20260926/$ARM; mkdir -p $D
FIXED=<workspace>/trellismx-tp4-swarm-20260909/fixed-workload-candidate-comparison-01/bench_fixed.py
BENCH=<workspace>/trellismx-performance-audit-20260908/llm_decode_bench.py
gate() { for i in $(seq 1 60); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1); [ "$t" -le 55 ] && return; sleep 1; done; }
for ctx in ${CTXS//,/ }; do
  gate
  /usr/bin/python3 $FIXED --host 127.0.0.1 --port 8000 --model glm53-flash-trellismx-p8-k45 --duration 20 --max-tokens 8192 \
    --token-targeting exact --display-mode plain --output $D/decode-$ctx-c1.json --contexts $ctx --concurrency 1 --skip-prefill \
    --cell-warmup-timeout-seconds 180 --temperature 0 < /dev/null > $D/decode-$ctx-c1.log 2>&1
  python3 -c "import json;r=json.load(open('$D/decode-$ctx-c1.json'))['results'][0];print('$ARM C1 $ctx', round(r['aggregate_tps'],1), 'tok/s  mtp_accept', round(r.get('server_spec_accept_rate') or 0,3))" 2>/dev/null || echo "$ARM C1 $ctx FAILED"
done
gate
python3 $BENCH --host 127.0.0.1 --port 8000 --model glm53-flash-trellismx-p8-k45 --display-mode plain --output $D/prefill-quick.json \
  --prefill-only --prefill-contexts 8k,32k,128k --prefill-duration 20 < /dev/null > $D/prefill-quick.log 2>&1
python3 -c "import json;p=json.load(open('$D/prefill-quick.json')).get('prefill',{});print('$ARM prefill', {k:v.get('tok_per_sec') for k,v in p.items() if isinstance(v,dict)})" 2>/dev/null || echo "$ARM prefill FAILED"
