#!/usr/bin/env python3
"""
模型评估脚本：在验证集上评估微调后的模型效果
简化输出，只显示预测结果和评估指标
"""
import os
import json
import torch
import sys
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.tokenizer_fix import get_chinese_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")
VAL_FILE = os.path.join(BASE_DIR, "data/sft/val.jsonl")

def load_model(use_lora=False):
    tokenizer = get_chinese_tokenizer(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    
    lora_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if use_lora and os.path.exists(lora_path):
        print("使用LoRA微调模型...")
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        print("使用基础模型...")
        model = base_model
    
    model.eval()
    return model, tokenizer

def load_validation_data():
    samples = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def extract_label(response):
    """从模型输出中提取标签，只在"答案："之后的内容中查找"""
    response = response.strip()
    
    # 找到"答案："或"Assistant:"之后的内容
    answer_markers = ["答案：", "Assistant:", "答案是", "答案:"]
    content_after_answer = response
    for marker in answer_markers:
        if marker in response:
            content_after_answer = response.split(marker)[-1]
            break
    
    all_labels = [
        "ID类/主键ID", "结构类/分类代码", "结构类/产品代码", "结构类/企业代码", "结构类/标准代码",
        "属性类/名称标题", "属性类/类别标签", "属性类/描述文本", "属性类/技能标签", "属性类/地址位置",
        "度量类/计量数值", "度量类/计数统计", "度量类/比率比例", "度量类/时间度量", "度量类/序号排序",
        "身份类/人口统计", "身份类/联系方式", "身份类/教育背景", "身份类/职业信息",
        "状态类/二元标志", "状态类/状态枚举", "状态类/时间标记",
        "扩展类/扩展代码", "扩展类/其他字段",
        "第1级/公开", "第2级/内部", "第3级/敏感", "第4级/机密"
    ]
    
    # 只在答案部分查找标签
    for label in all_labels:
        if label in content_after_answer:
            # 确保标签是答案的开始部分（前面是换行或冒号）
            idx = content_after_answer.find(label)
            before = content_after_answer[:idx].strip()
            if not before or before[-1] in ['\n', '：', ':', ' ']:
                return label
    
    # 如果没找到，返回原始内容（取第一行作为答案）
    first_line = content_after_answer.split('\n')[0].strip()
    return first_line

def predict(model, tokenizer, sample, max_new_tokens=20):
    instruction = sample.get("instruction", "")
    messages = [{"role": "user", "content": instruction}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_len = len(encoded["input_ids"])
    input_ids = encoded["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return extract_label(response)

def main(use_lora=False):
    print("=" * 70)
    print("DeepSeek 模型评估结果")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, tokenizer = load_model(use_lora=use_lora)
    samples = load_validation_data()
    print(f"测试样本数: {len(samples)}\n")
    
    y_true = []
    y_pred = []
    results = []
    
    for i, sample in enumerate(samples):
        expected = sample.get("output", "")
        try:
            predicted = predict(model, tokenizer, sample)
            y_true.append(expected)
            y_pred.append(predicted)
            results.append({
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted
            })
        except Exception as e:
            print(f"样本 {i+1} 预测失败: {e}")
            y_true.append(expected)
            y_pred.append("预测失败")
            results.append({
                "expected": expected,
                "predicted": "预测失败",
                "correct": False
            })
    
    # 计算评估指标
    print("=" * 70)
    print("评估指标")
    print("=" * 70)
    
    # 整体准确率
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results)
    print(f"\n准确率 (Accuracy): {accuracy:.2%} ({correct}/{len(results)})")
    
    # 计算各类别指标
    all_labels = sorted(set(y_true))
    
    # Macro 平均
    precision_macro = precision_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    
    print(f"精确率 (Macro Precision): {precision_macro:.2%}")
    print(f"召回率 (Macro Recall):    {recall_macro:.2%}")
    print(f"F1分数 (Macro F1):        {f1_macro:.2%}")
    
    # Weighted 平均
    precision_weighted = precision_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    
    print(f"精确率 (Weighted): {precision_weighted:.2%}")
    print(f"召回率 (Weighted): {recall_weighted:.2%}")
    print(f"F1分数 (Weighted): {f1_weighted:.2%}")
    
    # 详细预测结果
    print("\n" + "=" * 70)
    print("预测结果详情")
    print("=" * 70)
    
    print(f"\n{'序号':>4}  {'预期分类':<20} {'预测分类':<20} {'结果'}")
    print("-" * 70)
    
    for i, r in enumerate(results):
        status = "✓" if r["correct"] else "✗"
        print(f"{i+1:>4}  {r['expected']:<20} {r['predicted']:<20} {status}")
    
    # 错误统计
    print("\n" + "=" * 70)
    print("错误分析")
    print("=" * 70)
    
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误数: {len(errors)}/{len(results)}")
        error_pairs = Counter([(e['expected'], e['predicted']) for e in errors])
        print("\n错误类型统计:")
        for (expected, predicted), count in error_pairs.most_common():
            print(f"  预期: {expected} -> 预测: {predicted}  ({count}次)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="使用LoRA微调模型")
    args = parser.parse_args()
    main(use_lora=args.lora)
