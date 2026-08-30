"""
GenAgent Image Editing Server (Qwen-Image-Edit)

A FastAPI server for image editing using Qwen-Image-Edit-2509 with Lightning LoRA.
Supports multi-GPU inference with per-GPU request queuing.
Used by GenAgent v2 for the image_editing_tool.

Usage:
    python serve_image_edit.py --model_path /path/to/Qwen-Image-Edit-2509

    # With Lightning LoRA for faster inference
    python serve_image_edit.py \\
        --model_path /path/to/Qwen-Image-Edit-2509 \\
        --adapter_path /path/to/Lightning-LoRA.safetensors \\
        --gpus 0,1,2,3 --port 8998

API:
    POST /generate/
    Body: {"prompt": ["make the sky blue"], "image": ["base64_encoded_image"]}
    Response: {"status": "success", "images": ["base64..."]}
"""

import argparse
import asyncio
import base64
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

app = FastAPI(
    title="GenAgent Image Editing Server",
    description="Qwen-Image-Edit based image editing server for GenAgent v2.",
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
    prompt: list = Field(..., description="List of editing instructions.")
    image: list = Field(..., description="List of base64-encoded source images.")
    height: int = Field(1024, description="Output image height.", gt=0)
    width: int = Field(1024, description="Output image width.", gt=0)
    num_inference_steps: int = Field(8, description="Number of denoising steps.", gt=0)
    guidance_scale: float = Field(1.0, description="Guidance scale.", ge=0)
    num_images_per_prompt: int = Field(1, ge=1, le=4)


def pil2base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))


class GPUWorker:
    def __init__(self, gpu_id: int, pipeline):
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

                    if isinstance(input_args["image"], str):
                        input_args["image"] = base64_to_pil(input_args["image"])
                    elif isinstance(input_args["image"], list):
                        input_args["image"] = [
                            base64_to_pil(img) if isinstance(img, str) else img
                            for img in input_args["image"]
                        ]

                    if "negative_prompt" not in input_args:
                        input_args["negative_prompt"] = " "
                    if "true_cfg_scale" not in input_args:
                        input_args["true_cfg_scale"] = 1.0
                    if "num_inference_steps" not in input_args:
                        input_args["num_inference_steps"] = 8
                    if "callback_on_step_end" not in input_args:
                        input_args["callback_on_step_end"] = None
                    if "guidance_scale" not in input_args:
                        input_args["guidance_scale"] = 1.0
                    if "num_images_per_prompt" not in input_args:
                        input_args["num_images_per_prompt"] = 1

                    seed = input_args.pop("seed", 42)
                    input_args["generator"] = torch.manual_seed(seed)

                    images = self.pipeline(**input_args).images

                    elapsed = time.time() - start_time
                    self.total_processed += 1
                    self.total_time += elapsed
                    return [pil2base64(img) for img in images]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail=f"GPU {self.gpu_id} out of memory")
            except Exception:
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

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")

SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


@app.on_event("startup")
async def load_models():
    global gpu_workers, executor, MODEL_PATH, ADAPTER_PATH

    MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_PATH)
    ADAPTER_PATH = os.environ.get("ADAPTER_PATH", ADAPTER_PATH)

    gpu_list_str = os.environ.get("GPU_LIST", "0")
    gpu_list = [int(g.strip()) for g in gpu_list_str.split(",")]

    print(f"Loading Qwen-Image-Edit model on GPUs: {gpu_list}")
    print(f"Model: {MODEL_PATH}")
    if ADAPTER_PATH:
        print(f"Adapter: {ADAPTER_PATH}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)

    for gpu_id in gpu_list:
        device = f"cuda:{gpu_id}"
        with torch.cuda.device(gpu_id):
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                MODEL_PATH, scheduler=scheduler, torch_dtype=torch.bfloat16
            ).to(device=device)

            if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
                pipe.load_lora_weights(ADAPTER_PATH)
                pipe.fuse_lora()

            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
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
async def edit_image(request: ImageRequest):
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
        raise HTTPException(status_code=500, detail=f"Editing failed: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "total_gpus": len(gpu_workers),
        "busy_gpus": sum(1 for w in gpu_workers if w.is_busy),
        "gpu_stats": [w.get_stats() for w in gpu_workers],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GenAgent Image Editing Server")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Qwen-Image-Edit model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to Lightning LoRA adapter")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--port", type=int, default=8998, help="Server port")
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
