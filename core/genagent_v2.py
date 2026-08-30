"""
GenAgent v2 Inference

Agentic text-to-image generation using a VLM with both image_generation_tool
and image_editing_tool. The VLM supports thinking (<think> tags) and can
generate, judge, and iteratively edit images up to 3 rounds.

Usage:
    python genagent_v2.py \
        --vlm_url http://localhost:18901/v1 \
        --gen_url http://localhost:8999 \
        --edit_url http://localhost:8998 \
        --input examples/prompts.json \
        --output_dir outputs/v2_results \
        --num_process 1
"""

import argparse
import ast
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

SYSTEM_PROMPT = '''You are a helpful and intelligent image generation assistant. Your task is to carefully understand the user's input prompt and transform it into a more detailed, precise, and well-structured description, making it fully optimized for image generation models to produce outputs that strictly align with the user's requirements (e.g., objects, relationships, colors, shapes, aesthetics, styles, text, physical laws, etc.).

# Tools
You have access to two specific tools to assist with user requirements:
1.  **image_generation_tool**: Creates a new image from scratch.
2.  **image_editing_tool**: Modifies or tweaks an existing image based on instructions.

Function signatures are provided within <tools></tools> XML tags:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "image_generation_tool",
      "description": "Generate a new image based on the provided prompt description.",
      "parameters": {
        "type": "object",
        "properties": {
          "caption": {
            "type": "string",
            "description": "A highly detailed, descriptive caption for the image generation model."
          }
        },
        "required": ["caption"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "image_editing_tool",
      "description": "Edit or modify the current image based on natural language instructions.",
      "parameters": {
        "type": "object",
        "properties": {
          "instruction": {
            "type": "string",
            "description": "Specific instructions on what to change in the image."
          }
        },
        "required": ["instruction"]
      }
    }
  }
]
</tools>

# How to call a tool
Return a JSON object with the function name and arguments inside <tool_call></tool_call> XML tags. You must strictly follow this format:
<tool_call>
{"name": "image_generation_tool", "arguments": {"caption": "Your detailed description..."}}
</tool_call>

# Workflow & Guidelines
1.  **Initial Generation**: You must always start by calling the `image_generation_tool`.
2.  **Evaluation**: Upon receiving a tool result (an image), you must evaluate whether it meets the user's requirements. Enclose your evaluation within <judge></judge>.
3.  **Decision Logic**: Based on your judgment, you must strictly follow one of the three cases below:

    **Case 1: Image is Perfect (Done)**
    *Criteria: The image strictly aligns with the prompt, style, and details.*
    <think>
    [ Your thinking process ]
    </think>
    <judge>
    [Explain why the image perfectly meets the user's requirements.]
    </judge>
    <answer>done</answer>

    **Case 2: Image is Mostly Correct but Needs Tweaking (Edit)**
    *Criteria: The composition and style are good, but there are local flaws (e.g., wrong object color, minor artifacts, unwanted elements).*
    <think>
    [ Your thinking process ]
    </think>
    <judge>
    [Analyze the image. Acknowledge what is good, but specifically point out the flaws that need fixing.]
    </judge>
    <tool_call>
    {"name": "image_editing_tool", "arguments": {"instruction": "[Your specific editing instruction]"}}
    </tool_call>

    **Case 3: Image is Poor or Incorrect (Re-generate)**
    *Criteria: The image fails fundamentally (e.g., wrong subject, wrong art style, bad composition, severe hallucinations).*
    <think>
    [ Your thinking process ]
    </think>
    <judge>
    [Analyze why the image failed to meet the core requirements.]
    </judge>
    <tool_call>
    {"name": "image_generation_tool", "arguments": {"caption": "[The new, optimized prompt]"}}
    </tool_call>

4.  **Iteration**: You can call tools multiple times until **Case 1** is achieved.
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
            payload = {"prompt": [prompt], "height": 1024, "width": 1024}
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


def call_image_edit(instruction, source_image, edit_url, max_retries=10):
    for attempt in range(max_retries):
        try:
            buffered = BytesIO()
            source_image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {"prompt": [instruction], "image": [img_b64]}
            response = requests.post(f"{edit_url}/generate/", json=payload, timeout=5000)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_bytes = base64.b64decode(data["images"][0])
                return Image.open(BytesIO(img_bytes))
        except Exception as e:
            logging.warning(f"Image edit attempt {attempt}: {e}")
            time.sleep(1)
    return None


# Globals set at module level after arg parsing
client_list = []
model_name_list = []
gen_url = None
edit_url = None
enable_thinking = False


def process_single(prompt_text, prompt_id, output_dir):
    status = "fail"
    final_img = None

    for attempt in range(3):
        try:
            client_idx = random.randint(0, len(client_list) - 1)
            client = client_list[client_idx]
            m_name = model_name_list[client_idx]

            chat_message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Prompt: {prompt_text}"},
            ]

            response_message = ""
            img_cnt = 0
            turn_idx = 0

            while turn_idx < 10:
                if "<answer>done</answer>" in response_message:
                    status = "success"
                    break

                params = {
                    "model": m_name,
                    "messages": chat_message,
                    "temperature": 0.0,
                    "stop": ["</tool_call>"],
                }

                try:
                    if enable_thinking:
                        response = client.chat.completions.create(**params, extra_body={"enable_thinking": True})
                    else:
                        response = client.chat.completions.create(**params)
                    response_message = response.choices[0].message.content or ""
                except Exception as api_e:
                    logging.error(f"API Error: {api_e}")
                    raise api_e

                if START_TOKEN in response_message:
                    full_text = response_message.strip()
                    if not full_text.endswith("</tool_call>"):
                        full_text += "</tool_call>"

                    tool_section = full_text.split(START_TOKEN)[1].split(END_TOKEN)[0].strip()
                    try:
                        action_data = json.loads(tool_section)
                    except json.JSONDecodeError:
                        action_data = ast.literal_eval(tool_section)

                    tool_name = action_data.get("name")
                    tool_args = action_data.get("arguments", {})

                    current_img = None
                    if tool_name == "image_generation_tool":
                        caption = tool_args.get("caption", "")
                        current_img = call_image_gen(caption, gen_url)
                    elif tool_name == "image_editing_tool":
                        instruction = tool_args.get("instruction", "")
                        if final_img is None:
                            raise ValueError("Edit called without existing image")
                        current_img = call_image_edit(instruction, final_img, edit_url)
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    if current_img is None:
                        raise RuntimeError(f"{tool_name} failed after retries")

                    final_img = current_img

                    img_b64 = encode_pil_image_to_base64(current_img)
                    tool_return = [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": "</tool_response>"},
                    ]

                    chat_message.append({"role": "assistant", "content": response_message})
                    chat_message.append({"role": "user", "content": tool_return})

                    img_cnt += 1
                    turn_idx += 1
                    if img_cnt == 3:
                        status = "success"
                        break
                else:
                    chat_message.append({"role": "assistant", "content": response_message})
                    turn_idx += 1

            if status == "success":
                break

            if final_img is not None:
                status = "success"
                break

        except Exception as e:
            logging.error(f"Attempt {attempt} failed for {prompt_id}: {e}")
            if "429" in str(e):
                wait_time = (2 ** (attempt + 1)) + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
            else:
                time.sleep(2)

    if final_img is not None:
        final_img.save(os.path.join(output_dir, f"{prompt_id}.png"))
    else:
        # Fallback: direct generation without agent
        fallback_img = call_image_gen(prompt_text, gen_url)
        if fallback_img is not None:
            fallback_img.save(os.path.join(output_dir, f"{prompt_id}.png"))
        status = "fallback"

    return prompt_id


def _worker_init(vlm_url, gen_url_arg, edit_url_arg, model_name_arg, thinking):
    global client_list, model_name_list, gen_url, edit_url, enable_thinking
    client_list.append(OpenAI(api_key="EMPTY", base_url=vlm_url))
    model_name_list.append(model_name_arg)
    gen_url = gen_url_arg
    edit_url = edit_url_arg
    enable_thinking = thinking


def _worker_fn(task):
    return process_single(task["prompt"], task["prompt_id"], task["output_dir"])


def main():
    global client_list, model_name_list, gen_url, edit_url, enable_thinking

    parser = argparse.ArgumentParser(description="GenAgent v2 Inference")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:18901/v1", help="VLM server URL")
    parser.add_argument("--gen_url", type=str, default="http://localhost:8999", help="Image generation server URL")
    parser.add_argument("--edit_url", type=str, default="http://localhost:8998", help="Image editing server URL")
    parser.add_argument("--input", type=str, required=True, help="Input JSON or JSONL file with prompts")
    parser.add_argument("--output_dir", type=str, default="outputs/v2", help="Output directory")
    parser.add_argument("--num_process", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--model_name", type=str, default=None, help="VLM model name (auto-detected if not set)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode (for models that support it)")
    args = parser.parse_args()

    gen_url = args.gen_url
    edit_url = args.edit_url
    enable_thinking = args.enable_thinking

    client = OpenAI(api_key="EMPTY", base_url=args.vlm_url)
    client_list.append(client)

    if args.model_name:
        model_name_list.append(args.model_name)
    else:
        resp = requests.get(f"{args.vlm_url}/models")
        model_name_list.append(resp.json()["data"][0]["id"])
    logging.info(f"Using VLM model: {model_name_list[0]}")

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
        prompt = item.get("prompt", item.get("Prompt", ""))
        if f"{pid}.png" not in existing:
            tasks.append({"prompt_id": pid, "prompt": prompt, "output_dir": args.output_dir})
    logging.info(f"Total: {len(data)}, Remaining: {len(tasks)}")

    if not tasks:
        logging.info("All prompts already processed.")
        return

    with Pool(args.num_process, initializer=_worker_init,
              initargs=(args.vlm_url, args.gen_url, args.edit_url, model_name_list[0], enable_thinking)) as p:
        results = p.imap_unordered(_worker_fn, tasks)
        completed = 0
        for result in tqdm(results, total=len(tasks)):
            if result is not None:
                completed += 1

    logging.info(f"Done. Completed: {completed}/{len(tasks)}")


if __name__ == "__main__":
    main()
