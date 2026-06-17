"""本地GPU读图脚本 — Qwen2-VL-2B-Instruct"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
print(f"[加载模型] → {DEVICE}...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
t1 = time.time()

if DEVICE == "cuda":
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[加载完成] {t1-t0:.1f}秒 | 显存: {mem:.1f}GB")
else:
    print(f"[加载完成] {t1-t0:.1f}秒 (CPU)")

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path or not os.path.exists(image_path):
    print("用法: python read_image.py <图片路径> [提示词]")
    sys.exit(1)

prompt = sys.argv[2] if len(sys.argv) > 2 else "请仔细识别图片中的文字，输出所有看到的数字和文字内容。"

image = Image.open(image_path).convert("RGB")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}" if os.name == 'nt' else f"file://{image_path}"},
        {"type": "text", "text": prompt}
    ]}
]

try:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception as e:
    # fallback: 直接用PIL image
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(DEVICE)

t2 = time.time()
print(f"[处理] {t2-t1:.1f}秒")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

t3 = time.time()
result = processor.decode(outputs[0], skip_special_tokens=True)
if "assistant" in result:
    result = result.split("assistant")[-1].strip()

print(f"\n{'='*60}")
print(result)
print(f"{'='*60}")
print(f"[推理: {t3-t2:.1f}秒 | 总计: {t3-t0:.1f}秒]")
