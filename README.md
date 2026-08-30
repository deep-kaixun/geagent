# GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning

### GenAgent v1 (generation-only)

```
   User Prompt
        │
        ▼
  ┌───────────────┐   ①  image_generation_tool    ┌──────────────┐
  │      VLM      │ ────────────────────────────▶ │ Gen backbone │
  │  (controller) │ ◀──────────────────────────── │              │
  └───────────────┘   ②  <tool_response> image     └──────────────┘
        │
        ▼
   ③ judge: good enough?
        ├─ no  ─▶ regenerate ─┐
        │                     │  (loop back to ①)
        │  ◀──────────────────┘
        └─ yes ─▶ final image
```

### GenAgent v2 (generation + editing)

```
   User Prompt
        │
        ▼
  ┌───────────────┐   image_generation_tool    ┌──────────────┐
  │      VLM      │ ─────────────────────────▶ │ Gen backbone │
  │  (controller) │                             └──────────────┘
  │               │   image_editing_tool       ┌──────────────┐
  │               │ ─────────────────────────▶ │  Qwen-Edit   │
  └───────────────┘ ◀───────────────────────── └──────────────┘
        │                  returns image
        ▼
   judge: done? / edit? / regenerate?
        ├─ edit       ─▶ refine details  ─┐
        ├─ regenerate ─▶ start over       │  (loop back)
        │  ◀──────────────────────────────┘
        └─ done       ─▶ final image
```

## Model Weights

Model weights are released on HuggingFace:

- [GenAgent v1](https://huggingface.co/fengyao123/genagent)
- [GenAgent v2](https://huggingface.co/fengyao123/genagent)

## Quick Start

### 1. Launch the servers

**VLM server** (tensor-parallel across 2 GPUs):
```bash
bash serving/serve_vlm.sh /path/to/GenAgent-v1 --gpus "0,1" --port 18901
```

**Image generation server** — Qwen-Image:
```bash
QWEN_IMAGE=/path/to/Qwen-Image GPUS=2,3 PORT=8999 bash scripts/serve_gen_qwen.sh
```

**Image editing server** (v2 only):
```bash
QWEN_EDIT=/path/to/Qwen-Image-Edit-2509 GPUS=4,5 PORT=8998 bash scripts/serve_edit_qwen.sh
```

### 2. Run inference

**v1:**
```bash
python core/genagent_v1.py \
    --vlm_url http://localhost:18901/v1 \
    --gen_url http://localhost:8999 \
    --input examples/prompts.json \
    --output_dir outputs/v1_results \
    --num_process 1
```

**v2:**
```bash
python core/genagent_v2.py \
    --vlm_url http://localhost:18901/v1 \
    --gen_url http://localhost:8999 \
    --edit_url http://localhost:8998 \
    --input examples/prompts.json \
    --output_dir outputs/v2_results \
    --num_process 1
```

## Citation

```bibtex
@article{jiang2026genagent,
  title={Genagent: Scaling text-to-image generation via agentic multimodal reasoning},
  author={Jiang, Kaixun and Wang, Yuzheng and Zhou, Junjie and Li, Pandeng and Liu, Zhihang and Xie, Chen-Wei and Chen, Zhaoyu and Zheng, Yun and Zhang, Wenqiang},
  journal={arXiv preprint arXiv:2601.18543},
  year={2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
