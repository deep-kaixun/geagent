#!/bin/bash
# GenAgent VLM Server
# Launches a vLLM server with OpenAI-compatible API for the GenAgent VLM.
#
# Usage:
#   bash serve_vlm.sh <model_path> [--port PORT] [--tp TP_SIZE] [--gpus GPU_IDS]
#
# Examples:
#   bash serve_vlm.sh /path/to/genagent-v1
#   bash serve_vlm.sh /path/to/genagent-v2 --port 18901 --tp 2 --gpus "0,1"

set -e

MODEL_PATH=""
PORT=18901
TP_SIZE=2
GPUS=""
MODEL_NAME="genagent"
MAX_MODEL_LEN=16384

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"; shift 2 ;;
        --tp)
            TP_SIZE="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --model-name)
            MODEL_NAME="$2"; shift 2 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            fi
            shift ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is required."
    echo "Usage: bash serve_vlm.sh <model_path> [--port PORT] [--tp TP_SIZE] [--gpus GPU_IDS]"
    exit 1
fi

if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "Starting GenAgent VLM server..."
echo "  Model:     $MODEL_PATH"
echo "  Port:      $PORT"
echo "  TP Size:   $TP_SIZE"
echo "  Model Len: $MAX_MODEL_LEN"
echo ""

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization 0.8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --seed 42 \
    --tensor-parallel-size "$TP_SIZE" \
    --served-model-name "$MODEL_NAME" \
    --trust_remote_code
