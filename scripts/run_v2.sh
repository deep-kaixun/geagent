#!/bin/bash
# GenAgent v2 Inference Example
# Prerequisites: VLM server, image generation server, and image editing server must be running.
#
# 1. Start VLM server:
#    bash serving/serve_vlm.sh /path/to/genagent-v2-model --gpus "0,1"
#
# 2. Start image generation server:
#    python serving/serve_image_gen.py --model_path /path/to/FLUX.1-dev --gpus 2,3 --port 8999
#
# 3. Start image editing server:
#    python serving/serve_image_edit.py --model_path /path/to/Qwen-Image-Edit-2509 \
#        --adapter_path /path/to/Lightning-LoRA.safetensors --gpus 4,5 --port 8998
#
# 4. Run this script:
#    bash scripts/run_v2.sh

set -e

VLM_URL=${VLM_URL:-"http://localhost:18901/v1"}
GEN_URL=${GEN_URL:-"http://localhost:8999"}
EDIT_URL=${EDIT_URL:-"http://localhost:8998"}
INPUT=${INPUT:-"examples/prompts.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/v2_results"}
NUM_PROCESS=${NUM_PROCESS:-1}

echo "GenAgent v2 Inference"
echo "  VLM URL:    $VLM_URL"
echo "  Gen URL:    $GEN_URL"
echo "  Edit URL:   $EDIT_URL"
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT_DIR"
echo "  Processes:  $NUM_PROCESS"
echo ""

python core/genagent_v2.py \
    --vlm_url "$VLM_URL" \
    --gen_url "$GEN_URL" \
    --edit_url "$EDIT_URL" \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --num_process "$NUM_PROCESS"
