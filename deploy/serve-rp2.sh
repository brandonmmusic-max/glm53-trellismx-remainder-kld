#!/usr/bin/env bash
set -euo pipefail

# TrellisMX r27 + RP2 (GLM-5.3-Flash, DCP4). RP2 carries the MoE activations as two FP8 terms (hi + lo) at both
# P8 hops on the decode kernels (steps with <= 16 tokens). EPI-PAR parallelizes the M1 FC1 epilogue with
# bit-identical outputs. The final 128-window true-decode KLD for this configuration was measured with FP8 MLA KV,
# which is the default here. Set B12X_P8_DOWN_REMAINDER= (empty) and B12X_P8_EPI_PAR= (empty) to run the plain r27
# kernels from this image.
export VLLM_TRELLISMX_CHECKPOINT=${VLLM_TRELLISMX_CHECKPOINT:-/checkpoint}
carrier=${MODEL_ROOT:-/model}
port=${PORT:-8000}
if ! [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((port > 65535)); then
  echo 'Choose a valid PORT' >&2
  exit 2
fi
test -f "$carrier/config.json"
test -f "$VLLM_TRELLISMX_CHECKPOINT/trellismx-manifest.json"
if [[ -n ${GLM53_P8_NATIVE:-} || -n ${GLM53_P8_PSEUDOQUANT:-} ]]; then
  echo 'Remove legacy P8 sitecustomize activation variables' >&2
  exit 2
fi

tp=${TP:-4}
dcp=${DCP:-4}
if [[ "$tp" != 4 || "$dcp" != 4 ]]; then
  echo "This launcher requires TP=4 and DCP=4; got TP=$tp DCP=$dcp" >&2
  exit 2
fi

# RP2 needs the broadcast-A FC1 path and the serial FC1 epilogue (the b12x guards reject anything else).
if [[ ${GLM53_P8_FC1_WARP_QUANT:-0} != 0 || ${GLM53_P8_FC1_BROADCAST_A:-1} != 1 ]]; then
  echo 'RP2 requires GLM53_P8_FC1_WARP_QUANT=0 and GLM53_P8_FC1_BROADCAST_A=1' >&2
  exit 2
fi
export B12X_P8_DOWN_REMAINDER=${B12X_P8_DOWN_REMAINDER-rp2}
export B12X_P8_EPI_PAR=${B12X_P8_EPI_PAR-1}
case "$B12X_P8_DOWN_REMAINDER" in
  rp2|rp|"") ;;
  *) echo "Unsupported B12X_P8_DOWN_REMAINDER=$B12X_P8_DOWN_REMAINDER (use rp2, rp, or empty)" >&2; exit 2 ;;
esac
echo "TrellisMX RP2 launcher: B12X_P8_DOWN_REMAINDER='${B12X_P8_DOWN_REMAINDER}' B12X_P8_EPI_PAR='${B12X_P8_EPI_PAR}' KV_CACHE_DTYPE='${KV_CACHE_DTYPE:-fp8}'" >&2

export TP="$tp"
export DCP="$dcp"
export CACHE_MODE=${CACHE_MODE:-vram}
export SPECULATOR=${SPECULATOR:-mtp}
export NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-1000000}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-48}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
export CP_KV_CACHE_INTERLEAVE_SIZE=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
export DCP_CKV_GATHER=${DCP_CKV_GATHER:-auto}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-flash-trellismx-p8-rp2}
export PORT="$port"
export MODEL_ROOT="$carrier"

export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-131072}
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-86016}
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=${VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD:-4096}

exec /usr/local/bin/serve-glm53-flash.sh "$carrier" "$@"
