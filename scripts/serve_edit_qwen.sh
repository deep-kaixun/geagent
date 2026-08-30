#!/bin/bash
# Launch the Qwen-Image-Edit-2509 image-editing server (required for GenAgent v2).
#
#   QWEN_EDIT=/models/Qwen-Image-Edit-2509 \
#   LIGHTNING_LORA=/models/Qwen-Image-Edit-Lightning.safetensors \
#   GPUS=0,1 PORT=8998 \
#   bash scripts/serve_edit_qwen.sh
set -e

QWEN_EDIT="${QWEN_EDIT:?Set QWEN_EDIT to your local Qwen-Image-Edit-2509 path}"
LIGHTNING_LORA="${LIGHTNING_LORA:-}"   # optional: Lightning LoRA for 8-step fast editing
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8998}"

ADAPTER_ARG=""
[ -n "$LIGHTNING_LORA" ] && ADAPTER_ARG="--adapter_path $LIGHTNING_LORA"

python serving/serve_image_edit.py \
    --model_path "$QWEN_EDIT" \
    $ADAPTER_ARG \
    --gpus "$GPUS" \
    --port "$PORT"
