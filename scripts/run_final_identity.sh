#!/bin/bash
# Pre-flight identity check for the RP2 release image. MODE=save: r27 image + measured b12x-dx2rp2 mount.
# MODE=compare: release image, its own b12x (no mount). One GPU, no network, production untouched.
set -euo pipefail
HERE=<workspace>/p8-remainder-kvrot-20260923
GPU=${GPU:-1}
MODE=${MODE:?save or compare}
R27=verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf
REL=${REL_IMAGE:-verdictai/trellismx:glm53-flash-p8-r27-rp2-20260926}
if [[ $MODE == save ]]; then IMAGE=$R27; MOUNT=(-v "$HERE/b12x-dx2rp2:/opt/glm53-flash/b12x:ro"); else IMAGE=$REL; MOUNT=(); fi
exec docker run --rm --name p8-final-identity-$MODE --gpus "\"device=${GPU}\"" --network none \
  -e HOME=/work/cache/home -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e B12X_COMPILE_MEMORY_CACHE=0 -e B12X_COMPILE_DISK_CACHE=0 \
  -e B12X_COMPILE_CACHE_DIR=/work/cache/b12x-compile -e B12X_CUTE_COMPILE_CACHE_DIR=/work/cache/b12x-cute \
  -e CUTE_DSL_CACHE_DIR=/work/cache/cute-dsl -e TORCH_EXTENSIONS_DIR=/work/cache/torch-ext \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor -e B12X_SOURCE_DIR=/opt/glm53-flash/b12x \
  -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
  -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FI_LAYERS="${FI_LAYERS:-3,8,23,43}" -e FI_RANKS="${FI_RANKS:-0,3}" \
  "${MOUNT[@]}" \
  -v <model-volume>/glm53-trellismx-native6/trellismx-r27-local-checkpoint-v1:/checkpoint:ro \
  -v <data-volume>:<data-volume>:ro \
  -v "$HERE:/work" \
  --entrypoint /bin/bash "$IMAGE" \
  -c "/opt/venv/bin/python /work/scripts/final_identity_check.py $MODE; rc=\$?; chown -R \"\$HOST_UID:\$HOST_GID\" /work/results /work/cache; exit \$rc"
