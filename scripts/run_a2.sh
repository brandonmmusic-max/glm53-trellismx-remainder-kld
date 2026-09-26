#!/bin/bash
# Test A launcher: production image, scratch b12x over /opt/glm53-flash/b12x, no network,
# one GPU, production untouched. Compile caches off (they key on the spec, not the arm).
set -euo pipefail
HERE=<workspace>/p8-remainder-kvrot-20260923
GPU=${GPU:-0}
exec docker run --rm --name p8-dx2-a2-20260923 --gpus "\"device=${GPU}\"" --network none \
  -e HOME=/work/cache/home -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e B12X_COMPILE_MEMORY_CACHE=0 -e B12X_COMPILE_DISK_CACHE=0 \
  -e B12X_COMPILE_CACHE_DIR=/work/cache/b12x-compile -e B12X_CUTE_COMPILE_CACHE_DIR=/work/cache/b12x-cute \
  -e CUTE_DSL_CACHE_DIR=/work/cache/cute-dsl -e TORCH_EXTENSIONS_DIR=/work/cache/torch-ext \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor -e B12X_SOURCE_DIR=/opt/glm53-flash/b12x \
  -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
  -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ZR_LAYERS="${ZR_LAYERS:-8,3}" -e ZR_MS="${ZR_MS:-1,4,16}" -e ZR_BLOCKS="${ZR_BLOCKS:-40}" \
  -e ZR_REPLAYS="${ZR_REPLAYS:-200}" -e ZR_INPUT_SETS="${ZR_INPUT_SETS:-4}" \
  -e A2_MS="${A2_MS:-1,4,16,64,512,2048}" -e A2_LAYERS="${A2_LAYERS:-8,3}" -e A2_BLOCKS="${A2_BLOCKS:-60}" -e A2_REPLAYS="${A2_REPLAYS:-200}" -e A2_ARM="${A2_ARM:-1}" -e A2_OUT="${A2_OUT:-/work/results/a2_timing_raw.json}" -e ZR_OUT="${ZR_OUT:-/work/results/timing_raw.json}" \
  -v "$HERE/b12x-dx2:/opt/glm53-flash/b12x:ro" \
  -v <model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1:/checkpoint:ro \
  -v <data-volume>:<data-volume>:ro \
  -v "$HERE:/work" \
  --entrypoint /bin/bash \
  verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf \
  -c '/opt/venv/bin/python /work/scripts/dx2_timing.py; rc=$?; chown -R "$HOST_UID:$HOST_GID" /work/results /work/cache; exit $rc'
