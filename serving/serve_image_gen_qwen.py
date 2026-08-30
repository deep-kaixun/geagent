"""
GenAgent Image Generation Server (Qwen-Image)

A FastAPI server that generates images using Qwen-Image model.
An alternative to the FLUX-based server, supporting multi-GPU inference.

Usage:
    python serve_image_gen_qwen.py --model_path /path/to/Qwen-Image --gpus 0,1,2,3 --port 8999

API:
    POST /generate/
    Body: {"prompt": ["a photo of a cat", "a sunset over mountains"]}
    Response: {"status": "success", "images": ["base64...", "base64..."]}
"""

import argparse
import asyncio
import base64
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

app = FastAPI(
    title="GenAgent Image Generation Server (Qwen-Image)",
    description="Qwen-Image based text-to-image generation server for GenAgent.",
    version="1.0.0",
)


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


class ImageRequest(BaseModel):
    prompt: list = Field(..., description="List of text prompts for image generation.")
    height: int = Field(1024, description="Image height.", gt=0)
    width: int = Field(1024, description="Image width.", gt=0)
    num_inference_steps: int = Field(50, description="Number of denoising steps.", gt=0)
    guidance_scale: float = Field(7.5, description="Guidance scale.", ge=0)
    num_images_per_prompt: int = Field(1, description="Number of images per prompt.", ge=1, le=4)


pipelines: list = []
gpu_task_count: List[int] = []
gpu_locks: list = []  # one lock per pipeline: diffusers scheduler is stateful & NOT thread-safe;
                      # concurrent pipe() calls on the same pipeline race on scheduler._step_index
                      # -> "IndexError: index 51 is out of bounds". Serialize per-GPU.
executor: ThreadPoolExecutor = None

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image")


def pil2base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _generate_on_gpu(gpu_id: int, input_args: dict):
    pipe = pipelines[gpu_id]
    # Hold the per-GPU lock across the whole diffusion loop: the pipeline's scheduler is
    # stateful (mutates _step_index / sigmas each step) and shared across threads, so two
    # concurrent pipe() calls on the same pipeline corrupt each other's step index.
    with gpu_locks[gpu_id]:
        images = pipe(
            prompt=input_args["prompt"],
            height=input_args.get("height", 1024),
            width=input_args.get("width", 1024),
            negative_prompt=" ",
            true_cfg_scale=4.0,
            num_inference_steps=input_args.get("num_inference_steps", 50),
            generator=torch.Generator("cpu").manual_seed(0),
            num_images_per_prompt=input_args.get("num_images_per_prompt", 1),
        ).images
    return [pil2base64(img) for img in images]


def pick_least_busy_gpu() -> int:
    min_tasks = min(gpu_task_count)
    candidates = [i for i, count in enumerate(gpu_task_count) if count == min_tasks]
    return candidates[0]


@app.on_event("startup")
async def load_models():
    global pipelines, gpu_task_count, executor, MODEL_PATH

    MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_PATH)

    gpu_list_str = os.environ.get("GPU_LIST", "0")
    gpu_list = [int(g.strip()) for g in gpu_list_str.split(",")]

    print(f"Loading Qwen-Image model on GPUs: {gpu_list}")
    print(f"Model: {MODEL_PATH}")

    for gpu_id in gpu_list:
        device = f"cuda:{gpu_id}"
        try:
            pipe = DiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device=device)
            pipelines.append(pipe)
            gpu_task_count.append(0)
            gpu_locks.append(threading.Lock())
            print(f"GPU {gpu_id} ready (VRAM: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB)")
        except Exception as e:
            print(f"Failed to load GPU {gpu_id}: {e}")
            raise RuntimeError(f"Failed to load model on GPU {gpu_id}") from e

    executor = ThreadPoolExecutor(max_workers=len(gpu_list), thread_name_prefix="GPU-Worker")
    print(f"All {len(pipelines)} GPUs initialized.")


@app.post("/generate/", response_model=dict)
async def generate_image(request: ImageRequest):
    loop = asyncio.get_event_loop()
    gpu_id = pick_least_busy_gpu()
    gpu_task_count[gpu_id] += 1

    try:
        result = await loop.run_in_executor(executor, _generate_on_gpu, gpu_id, request.dict())
        return {"status": "success", "images": result}
    except Exception as e:
        import traceback
        print(f"[GEN-ERROR gpu_idx={gpu_id}] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        gpu_task_count[gpu_id] -= 1


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "total_gpus": len(pipelines),
        "gpu_task_count": gpu_task_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GenAgent Image Generation Server (Qwen-Image)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Qwen-Image model")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--port", type=int, default=8999, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    os.environ["GPU_LIST"] = args.gpus

    uvicorn.run(app, host=args.host, port=args.port)
