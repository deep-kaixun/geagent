"""
GenAgent v1 Inference

Agentic text-to-image generation using a VLM with image_generation_tool.
The VLM generates a caption, calls the image generation tool, judges the result,
and iterates up to 2 rounds (max 2 images per prompt).

Usage:
    python genagent_v1.py \
        --vlm_url http://localhost:18901/v1 \
        --gen_url http://localhost:8999 \
        --input examples/prompts.json \
        --output_dir outputs/v1_results \
        --num_process 1
"""

import argparse
import base64
import json
import logging
import multiprocessing
import os
import random
import time
from io import BytesIO
from multiprocessing import Pool

import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

multiprocessing.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = '''You are a helpful assistant. Your task is to carefully understand the user's input prompt and transform it into a more detailed, precise, and well-structured description, making it fully optimized for image generation models to produce outputs that strictly align with the user's requirements(e.g. objects, relationships, colors, shapes, aesthetics, styles, text, physical law, etc.).
# Tools
You may call **image_generation_tool** to assist with the user requirements.
Function signatures are provided within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_generation_tool","description":"Generate an image based on the provided prompt","parameters":{"type":"object","properties":{"caption":{"type":"string","description":"A descriptive caption for the image to be generated"}},"required":["caption"]}}}
</tools>

# How to call a tool
Return a JSON object with the function name and arguments inside <tool_call></tool_call> XML tags, strictly following the format:
<tool_call>
{"name": "image_generation_tool", "arguments": {"caption": "Your detailed description of the image to be generated"}}
</tool_call>

# Tips
- You can call the image_generation_tool multiple times (at least one) to evaluate the quality of the refine the prompt and help you refine the prompt. Formart strictly as: <think>Your detailed comparative analysis</think> <tool_call>...</tool_call>.
- For the generated results returned by the image generation tool, you must first judge whether they meet the user's requirements. Enclose your evaluation within <judge></judge>. Then, based on your judgment, you must strictly follow one of the two formats below:
    **Case 1: Image is consistent with the Prompt**
    <judge>
    [Your rationale goes here, explaining why the image meets the requirements of the user requirements.]
    </judge>
    <answer>done</answer>

    **Case 2: Image is inconsistent with the Prompt**
    <judge>
    [Your rationale goes here, analyzing the aspects where the image fails to meet the requirements of the user requirements.]
    </judge>
    <think>
    [Your suggestions for improvement go here, explaining what should be optimized to better achieve the original intent.]
    </think>
    <tool_call>
    {"name": "image_generation_tool", "arguments": {"caption": "[The new optimized prompt]"}}
    </tool_call>
'''

START_TOKEN = "<tool_call>"
END_TOKEN = "</tool_call>"


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_image_gen(prompt, gen_url, max_retries=10):
    for attempt in range(max_retries):
        try:
            payload = {"prompt": [prompt], "height": 512, "width": 512}
            response = requests.post(f"{gen_url}/generate/", json=payload, timeout=5000)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_bytes = base64.b64decode(data["images"][0])
                return Image.open(BytesIO(img_bytes))
        except Exception as e:
            logging.warning(f"Image gen attempt {attempt}: {e}")
            time.sleep(1)
    return None


# Globals set at module level after arg parsing
client = None
model_name = None
gen_url = None


def process_single(prompt_text, prompt_id, output_dir):
    status = "success"
    img = None
    caption = None

    for attempt in range(3):
        question = prompt_text
        chat_message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": f"Prompt: {question}"}]},
        ]

        response_message = ""
        try_count = 0
        img = None
        img_cnt = 0

        try:
            while "</answer>" not in response_message:
                if try_count > 8:
                    break

                params = {
                    "model": model_name,
                    "messages": chat_message,
                    "temperature": 0,
                    "stop": ["</tool_call>"],
                }
                response = client.chat.completions.create(**params)
                response_message = response.choices[0].message.content

                if START_TOKEN in response_message:
                    response_message = response_message.strip() + "</tool_call>"
                    action_str = response_message.split(START_TOKEN)[1].split(END_TOKEN)[0].strip()
                    action_data = json.loads(action_str)
                    caption = action_data["arguments"]["caption"]

                    img = call_image_gen(caption, gen_url)
                    if img is None:
                        raise RuntimeError("Image generation failed after retries")

                    img_b64 = encode_pil_image_to_base64(img)
                    tool_response = [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": "</tool_response>"},
                    ]

                    chat_message.append({"role": "assistant", "content": response_message})
                    chat_message.append({"role": "user", "content": tool_response})

                    img_cnt += 1
                    if img_cnt == 2:
                        break

                try_count += 1

            if img is not None:
                break
        except Exception as e:
            logging.error(f"Attempt {attempt} failed for {prompt_id}: {e}")
            status = "error"

    if img is not None:
        img.save(os.path.join(output_dir, f"{prompt_id}.png"))
    else:
        fallback_img = call_image_gen(prompt_text, gen_url)
        if fallback_img is not None:
            fallback_img.save(os.path.join(output_dir, f"{prompt_id}.png"))
        status = "fallback"

    return prompt_id


def _worker_init(vlm_url, gen_url_arg, model_name_arg):
    global client, model_name, gen_url
    client = OpenAI(api_key="EMPTY", base_url=vlm_url)
    model_name = model_name_arg
    gen_url = gen_url_arg


def _worker_fn(args):
    return process_single(*args)


def main():
    global client, model_name, gen_url

    parser = argparse.ArgumentParser(description="GenAgent v1 Inference")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:18901/v1", help="VLM server URL")
    parser.add_argument("--gen_url", type=str, default="http://localhost:8999", help="Image generation server URL")
    parser.add_argument("--input", type=str, required=True, help="Input JSON or JSONL file with prompts")
    parser.add_argument("--output_dir", type=str, default="outputs/v1", help="Output directory")
    parser.add_argument("--num_process", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--model_name", type=str, default=None, help="VLM model name (auto-detected if not set)")
    args = parser.parse_args()

    gen_url = args.gen_url

    client = OpenAI(api_key="EMPTY", base_url=args.vlm_url)
    if args.model_name:
        model_name = args.model_name
    else:
        resp = requests.get(f"{args.vlm_url}/models")
        model_name = resp.json()["data"][0]["id"]
    logging.info(f"Using VLM model: {model_name}")

    # Load input data
    input_path = args.input
    if input_path.endswith(".jsonl"):
        with open(input_path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip already processed
    existing = set(os.listdir(args.output_dir))
    tasks = []
    for item in data:
        pid = str(item.get("prompt_id", item.get("id", "")))
        if f"{pid}.png" not in existing:
            tasks.append(item)
    logging.info(f"Total: {len(data)}, Remaining: {len(tasks)}")

    if not tasks:
        logging.info("All prompts already processed.")
        return

    all_inputs = [
        (item.get("prompt", item.get("Prompt", "")), str(item.get("prompt_id", item.get("id", ""))), args.output_dir)
        for item in tasks
    ]

    with Pool(args.num_process, initializer=_worker_init, initargs=(args.vlm_url, args.gen_url, model_name)) as p:
        results = p.imap_unordered(_worker_fn, all_inputs)
        completed = 0
        for result in tqdm(results, total=len(tasks)):
            if result is not None:
                completed += 1

    logging.info(f"Done. Completed: {completed}/{len(tasks)}")


if __name__ == "__main__":
    main()
