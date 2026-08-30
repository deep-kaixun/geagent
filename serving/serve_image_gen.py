"""
GenAgent Image Generation Server (FLUX)

A FastAPI server that generates images using FLUX.1-dev with Turbo-Alpha LoRA.
Supports multi-GPU inference with per-GPU request queuing.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python serve_image_gen.py --model_path /path/to/FLUX.1-dev

    # Multi-GPU
    python serve_image_gen.py --model_path /path/to/FLUX.1-dev --gpus 0,1,2,3 --port 8999

    # With LoRA adapter
    python serve_image_gen.py --model_path /path/to/FLUX.1-dev --adapter_path /path/to/turbo-alpha

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
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

app = FastAPI(
    title="GenAgent Image Generation Server",
    description="FLUX-based text-to-image generation server for GenAgent.",
    version="1.0.0",
)

gpu_queues = {}
gpu_processing = {}


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
    num_inference_steps: int = Field(8, description="Number of denoising steps.", gt=0)
    guidance_scale: float = Field(3.5, description="Classifier-free guidance scale.", ge=0)
    num_images_per_prompt: int = Field(1, description="Number of images per prompt.", ge=1, le=4)


class GPUWorker:
    def __init__(self, gpu_id: int, pipeline: FluxPipeline):
        self.gpu_id = gpu_id
        self.pipeline = pipeline
        self.lock = threading.Lock()
        self.is_busy = False
        self.total_processed = 0
        self.total_time = 0.0
        self.last_request_time = 0

    def execute(self, input_args: dict) -> List[str]:
        with self.lock:
            self.is_busy = True
            start_time = time.time()
            try:
                with torch.cuda.device(self.gpu_id):
                    torch.cuda.empty_cache()
                    images = self.pipeline(
                        prompt=input_args["prompt"],
                        height=input_args.get("height", 1024),
                        width=input_args.get("width", 1024),
                        guidance_scale=input_args.get("guidance_scale", 3.5),
                        num_inference_steps=input_args.get("num_inference_steps", 8),
                        max_sequence_length=512,
                        generator=torch.Generator("cpu").manual_seed(0),
                        num_images_per_prompt=input_args.get("num_images_per_prompt", 1),
                    ).images
                    elapsed = time.time() - start_time
                    self.total_processed += 1
                    self.total_time += elapsed
                    return [pil2base64(img) for img in images]
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail=f"GPU {self.gpu_id} out of memory")
            except Exception as e:
                raise
            finally:
                self.is_busy = False
                self.last_request_time = time.time()

    def get_stats(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "is_busy": self.is_busy,
            "total_processed": self.total_processed,
            "avg_time_seconds": round(self.total_time / self.total_processed, 2) if self.total_processed > 0 else 0,
            "vram_allocated_gb": round(torch.cuda.memory_allocated(self.gpu_id) / 1e9, 2),
        }


gpu_workers: List[GPUWorker] = []
executor: ThreadPoolExecutor = None

MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-dev")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "alimama-creative/FLUX.1-Turbo-Alpha")


def pil2base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.on_event("startup")
async def load_models():
    global gpu_workers, executor, MODEL_PATH, ADAPTER_PATH

    MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_PATH)
    ADAPTER_PATH = os.environ.get("ADAPTER_PATH", ADAPTER_PATH)

    gpu_list_str = os.environ.get("GPU_LIST", "0")
    gpu_list = [int(g.strip()) for g in gpu_list_str.split(",")]

    print(f"Loading FLUX model on GPUs: {gpu_list}")
    print(f"Model: {MODEL_PATH}")
    print(f"Adapter: {ADAPTER_PATH}")

    for gpu_id in gpu_list:
        device = f"cuda:{gpu_id}"
        with torch.cuda.device(gpu_id):
            pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device)
            if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
                pipe.load_lora_weights(ADAPTER_PATH)
                pipe.fuse_lora()
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()

            # Warmup
            with torch.no_grad():
                _ = pipe(prompt=["warmup"], height=512, width=512, num_inference_steps=1, guidance_scale=3.5).images
            torch.cuda.empty_cache()

        worker = GPUWorker(gpu_id, pipe)
        gpu_workers.append(worker)
        print(f"GPU {gpu_id} ready (VRAM: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB)")

    executor = ThreadPoolExecutor(max_workers=len(gpu_list), thread_name_prefix="GPU-Worker")
    print(f"All {len(gpu_workers)} GPUs initialized.")


@app.on_event("startup")
async def init_queues():
    for worker in gpu_workers:
        gpu_queues[worker.gpu_id] = asyncio.Queue()
        gpu_processing[worker.gpu_id] = False

    for worker in gpu_workers:
        asyncio.create_task(_process_queue(worker))


async def _process_queue(worker):
    gpu_id = worker.gpu_id
    queue = gpu_queues[gpu_id]
    while True:
        task_data, future = await queue.get()
        try:
            gpu_processing[gpu_id] = True
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, worker.execute, task_data)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            gpu_processing[gpu_id] = False
            queue.task_done()


@app.post("/generate/", response_model=dict)
async def generate_image(request: ImageRequest):
    best_gpu_id = min(gpu_queues.keys(), key=lambda gid: gpu_queues[gid].qsize())
    queue = gpu_queues[best_gpu_id]

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await queue.put((request.dict(), future))

    try:
        result = await asyncio.wait_for(future, timeout=600.0)
        return {"status": "success", "images": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "total_gpus": len(gpu_workers),
        "busy_gpus": sum(1 for w in gpu_workers if w.is_busy),
        "gpu_stats": [w.get_stats() for w in gpu_workers],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GenAgent Image Generation Server")
    parser.add_argument("--model_path", type=str, default=None, help="Path to FLUX model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--port", type=int, default=8999, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.adapter_path:
        os.environ["ADAPTER_PATH"] = args.adapter_path
    os.environ["GPU_LIST"] = args.gpus

    uvicorn.run(app, host=args.host, port=args.port)
