#!/usr/bin/env python3
"""
模型对比评估脚本
对比基础模型和微调模型的性能差异，生成详细的对比报告
"""
import os
import json
import torch
import sys
import time
from transformers import AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter, defaultdict
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.tokenizer_fix import get_chinese_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_model(use_lora=False):
    """加载模型"""
    print("\n" + "=" * 60)
    print(f"加载{'微调' if use_lora else '基础'}模型...")
    print("=" * 60)
    
    tokenizer = get_chinese_tokenizer(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        device_map="auto", 
        low_cpu_mem_usage=True,
    )
    
    lora_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if use_lora and os.path.exists(lora_path):
        print("✓ 检测到LoRA权重，正在加载...")
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    elif use_lora:
        print("⚠ 未找到LoRA权重，使用基础模型")
        model = base_model
    else:
        print("✓ 使用基础模型")
        model = base_model
    
    model.eval()
    return model, tokenizer


def load_test_data():
    """加载测试数据"""
    test_file = os.path.join(BASE_DIR, "data/sft/test.jsonl")
    val_file = os.path.join(BASE_DIR, "data/validation/balanced_val.jsonl")
    
    filepath = test_file if os.path.exists(test_file) else val_file
    print(f"\n加载测试数据: {filepath}")
    
    samples = []
    industries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                # 提取行业信息
                instruction = sample.get('instruction', '')
                if '行业：' in instruction:
                    try:
                        industry = instruction.split('行业：')[1].split('\n')[0].strip()
                    except:
                        industry = 'unknown'
                else:
                    industry = 'unknown'
                
                sample['industry'] = industry
                samples.append(sample)
                industries.append(industry)
    
    print(f"✓ 加载了 {len(samples)} 条测试样本")
    
    # 行业分布
    industry_counts = Counter(industries)
    print("\n行业分布:")
    for ind, count in industry_counts.most_common():
        print(f"  {ind}: {count}")
    
    return samples


def extract_label(response):
    """从模型输出中提取标签"""
    response = response.strip()
    
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
        "扩展类/扩展代码", "扩展类/其他字段"
    ]
    
    for label in all_labels:
        if label in content_after_answer:
            idx = content_after_answer.find(label)
            before = content_after_answer[:idx].strip()
            if not before or before[-1] in ['\n', '：', ':', ' ']:
                return label
    
    first_line = content_after_answer.split('\n')[0].strip()
    return first_line if first_line else "未识别"


def predict(model, tokenizer, sample, max_new_tokens=20):
    """单条预测"""
    instruction = sample.get("instruction", "")
    messages = [{"role": "user", "content": instruction}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_len = len(encoded["input_ids"])
    input_ids = encoded["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return extract_label(response)


def evaluate_model(model, tokenizer, samples, model_name):
    """评估模型性能"""
    print(f"\n{'=' * 60}")
    print(f"评估 {model_name}...")
    print('=' * 60)
    
    start_time = time.time()
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
                "correct": expected == predicted,
                "industry": sample.get("industry", "unknown")
            })
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(samples)}")
        except Exception as e:
            print(f"  样本 {i+1} 预测失败: {e}")
            y_true.append(expected)
            y_pred.append("预测失败")
            results.append({
                "expected": expected,
                "predicted": "预测失败",
                "correct": False,
                "industry": sample.get("industry", "unknown")
            })
    
    elapsed_time = time.time() - start_time
    
    # 计算评估指标
    all_labels = sorted(set(y_true))
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    
    metrics = {
        "model_name": model_name,
        "num_samples": len(samples),
        "elapsed_time": elapsed_time,
        "time_per_sample": elapsed_time / len(samples),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }
    
    print(f"\n评估完成! 耗时: {elapsed_time:.2f}秒")
    print(f"\n核心指标:")
    print(f"  准确率 (Accuracy):   {accuracy:.2%}")
    print(f"  F1分数 (Macro):       {f1_macro:.2%}")
    print(f"  F1分数 (Weighted):    {f1_weighted:.2%}")
    
    return metrics, results


def analyze_by_industry(results):
    """按行业分析准确率"""
    industry_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for r in results:
        ind = r["industry"]
        industry_results[ind]["total"] += 1
        if r["correct"]:
            industry_results[ind]["correct"] += 1
    
    analysis = {}
    for ind, stats in industry_results.items():
        analysis[ind] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        }
    
    return analysis


def analyze_errors(results):
    """分析错误类型"""
    errors = [r for r in results if not r["correct"]]
    
    error_pairs = Counter([(e['expected'], e['predicted']) for e in errors])
    error_by_industry = defaultdict(list)
    for e in errors:
        error_by_industry[e['industry']].append(e)
    
    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(results) if results else 0,
        "top_errors": error_pairs.most_common(10),
        "errors_by_industry": {ind: len(errs) for ind, errs in error_by_industry.items()}
    }


def generate_comparison_report(base_metrics, finetuned_metrics, base_results, finetuned_results):
    """生成对比报告"""
    print("\n" + "=" * 60)
    print("模型对比报告")
    print("=" * 60)
    
    # 核心指标对比
    print("\n【核心指标对比】")
    print(f"{'指标':<20} {'基础模型':>12} {'微调模型':>12} {'提升':>12}")
    print("-" * 60)
    
    metrics_names = [
        ("accuracy", "准确率"),
        ("precision_macro", "精确率(Macro)"),
        ("recall_macro", "召回率(Macro)"),
        ("f1_macro", "F1(Macro)"),
        ("precision_weighted", "精确率(Weighted)"),
        ("recall_weighted", "召回率(Weighted)"),
        ("f1_weighted", "F1(Weighted)")
    ]
    
    improvements = {}
    for key, name in metrics_names:
        base_val = base_metrics[key]
        ft_val = finetuned_metrics[key]
        improvement = ft_val - base_val
        improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
        improvements[key] = {
            "base": base_val,
            "finetuned": ft_val,
            "improvement": improvement,
            "improvement_pct": improvement_pct
        }
        print(f"{name:<16} {base_val:>11.2%} {ft_val:>11.2%} {improvement_pct:>+10.1f}%")
    
    # 行业性能对比
    print("\n【行业性能对比】")
    base_industry = analyze_by_industry(base_results)
    ft_industry = analyze_by_industry(finetuned_results)
    
    all_industries = set(base_industry.keys()) | set(ft_industry.keys())
    print(f"{'行业':<15} {'基础准确率':>12} {'微调准确率':>12} {'提升':>10}")
    print("-" * 55)
    
    for ind in sorted(all_industries):
        base_acc = base_industry.get(ind, {}).get("accuracy", 0)
        ft_acc = ft_industry.get(ind, {}).get("accuracy", 0)
        improvement = ft_acc - base_acc
        print(f"{ind:<15} {base_acc:>11.1%} {ft_acc:>11.1%} {improvement:>+9.1%}")
    
    # 错误分析对比
    print("\n【错误分析对比】")
    base_errors = analyze_errors(base_results)
    ft_errors = analyze_errors(finetuned_results)
    
    print(f"基础模型错误数: {base_errors['total_errors']} ({base_errors['error_rate']:.1%})")
    print(f"微调模型错误数: {ft_errors['total_errors']} ({ft_errors['error_rate']:.1%})")
    
    # Top错误类型
    print("\n微调模型Top错误类型:")
    for (expected, predicted), count in ft_errors['top_errors'][:5]:
        print(f"  {expected} → {predicted}: {count}次")
    
    # 效率对比
    print("\n【效率对比】")
    print(f"{'指标':<20} {'基础模型':>15} {'微调模型':>15}")
    print("-" * 55)
    print(f"{'总耗时(秒)':<20} {base_metrics['elapsed_time']:>15.2f} {finetuned_metrics['elapsed_time']:>15.2f}")
    print(f"{'单样本耗时(秒)':<20} {base_metrics['time_per_sample']:>15.3f} {finetuned_metrics['time_per_sample']:>15.3f}")
    
    # 保存报告
    report = {
        "evaluation_date": "2026-03-25",
        "test_samples": len(base_results),
        "base_model_metrics": base_metrics,
        "finetuned_model_metrics": finetuned_metrics,
        "improvements": improvements,
        "industry_comparison": {
            "base": base_industry,
            "finetuned": ft_industry
        },
        "error_analysis": {
            "base": base_errors,
            "finetuned": ft_errors
        }
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_file = os.path.join(RESULTS_DIR, "model_comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存: {report_file}")
    
    return report


def main():
    print("=" * 60)
    print("DeepSeek 模型对比评估")
    print("=" * 60)
    print("本脚本将评估基础模型和微调模型的性能差异")
    
    # 加载测试数据
    samples = load_test_data()
    
    # 评估基础模型
    base_model, base_tokenizer = load_model(use_lora=False)
    base_metrics, base_results = evaluate_model(base_model, base_tokenizer, samples, "DeepSeek-7B-Base")
    
    # 清理GPU内存
    del base_model
    torch.cuda.empty_cache()
    
    # 评估微调模型
    finetuned_model, finetuned_tokenizer = load_model(use_lora=True)
    finetuned_metrics, finetuned_results = evaluate_model(finetuned_model, finetuned_tokenizer, samples, "DeepSeek-7B-LoRA")
    
    # 生成对比报告
    report = generate_comparison_report(base_metrics, finetuned_metrics, base_results, finetuned_results)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
