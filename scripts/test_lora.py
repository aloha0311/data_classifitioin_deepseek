#!/usr/bin/env python3
"""测试LoRA权重是否被正确加载"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.tokenizer_fix import get_chinese_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")

print("=" * 60)
print("LoRA权重加载测试")
print("=" * 60)

# 1. 加载tokenizer
tokenizer = get_chinese_tokenizer(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载基础模型
print("\n[1] 加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

# 3. 加载LoRA权重
print("[2] 加载LoRA权重...")
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
print(f"    LoRA模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# 4. 检查LoRA权重是否有变化
print("\n[3] 检查LoRA权重是否生效...")
# 获取原始基础模型和LoRA模型的部分参数
for name, param in list(model.named_parameters())[:5]:
    print(f"    {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

# 5. 简单测试
print("\n[4] 简单生成测试...")

test_prompt = "请输出数字1到5，用逗号分隔："

messages = [{"role": "user", "content": test_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

input_len = len(inputs['input_ids'][0])
result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"    输入: {test_prompt}")
print(f"    输出: {result.strip()}")

# 6. 实际任务测试
print("\n[5] 实际任务测试（无分类任务）...")
test_prompt2 = "行业：medical\n字段名：血压\n样本值：120, 118\n答案："

messages2 = [{"role": "user", "content": test_prompt2}]
text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
inputs2 = tokenizer(text2, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

input_len2 = len(inputs2['input_ids'][0])
result2 = tokenizer.decode(outputs2[0][input_len2:], skip_special_tokens=True)
print(f"    输入: {test_prompt2}")
print(f"    输出: {result2.strip()}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
