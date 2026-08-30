#!/bin/bash
# Launch the Qwen-Image image-generation server (alternative backend to FLUX).
#
#   QWEN_IMAGE=/models/Qwen-Image GPUS=0,1,2,3 PORT=8999 \
#   bash scripts/serve_gen_qwen.sh
set -e

QWEN_IMAGE="${QWEN_IMAGE:?Set QWEN_IMAGE to your local Qwen-Image path}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8999}"

python serving/serve_image_gen_qwen.py \
    --model_path "$QWEN_IMAGE" \
    --gpus "$GPUS" \
    --port "$PORT"
