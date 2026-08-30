#!/bin/bash
# Launch the FLUX.1-dev image-generation server.
#
# All paths are provided via environment variables so nothing machine-specific
# is baked in. Override any of them on the command line, e.g.:
#
#   FLUX_MODEL=/models/FLUX.1-dev \
#   TURBO_LORA=/models/FLUX.1-Turbo-Alpha \
#   GPUS=0,1,2,3 PORT=8999 \
#   bash scripts/serve_gen_flux.sh
set -e

FLUX_MODEL="${FLUX_MODEL:?Set FLUX_MODEL to your local FLUX.1-dev path}"
TURBO_LORA="${TURBO_LORA:-}"          # optional: FLUX.1-Turbo-Alpha LoRA for 8-step fast inference
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8999}"

ADAPTER_ARG=""
[ -n "$TURBO_LORA" ] && ADAPTER_ARG="--adapter_path $TURBO_LORA"

python serving/serve_image_gen.py \
    --model_path "$FLUX_MODEL" \
    $ADAPTER_ARG \
    --gpus "$GPUS" \
    --port "$PORT"
