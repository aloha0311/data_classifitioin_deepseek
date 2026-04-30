#!/usr/bin/env python3
"""
训练数据准备脚本：将标注数据转换为模型训练格式

这个脚本会：
1. 读取标注数据
2. 生成分类任务和分级任务的训练样本
3. 划分训练集和验证集
4. 保存到 data/sft/ 目录
"""
import os
import json
import random
from pathlib import Path
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/sft")

print("=" * 60)
print("训练数据准备脚本")
print("=" * 60)

# 分类标签（从schema文件加载）
CLASSIFICATION_LABELS = [
    "ID类/主键ID",
    "结构类/分类代码",
    "结构类/产品代码",
    "结构类/企业代码",
    "结构类/标准代码",
    "属性类/名称标题",
    "属性类/类别标签",
    "属性类/描述文本",
    "属性类/技能标签",
    "属性类/地址位置",
    "度量类/计量数值",
    "度量类/计数统计",
    "度量类/比率比例",
    "度量类/时间度量",
    "度量类/序号排序",
    "身份类/人口统计",
    "身份类/联系方式",
    "身份类/教育背景",
    "身份类/职业信息",
    "状态类/二元标志",
    "状态类/状态枚举",
    "状态类/时间标记",
    "扩展类/扩展代码",
    "扩展类/其他字段"
]

# 分级标签
GRADING_LABELS = [
    "第1级/公开",
    "第2级/内部",
    "第3级/敏感",
    "第4级/机密"
]

def build_classification_prompt(item):
    """构建分类任务的提示"""
    column_name = item.get("column_name", "")
    sample_values = item.get("sample_values", [])
    industry = item.get("industry", "")
    
    # 限制样本值数量
    sample_preview = sample_values[:5]
    sample_str = ", ".join(str(v) for v in sample_preview)
    if len(sample_values) > 5:
        sample_str += f" ... (共{len(sample_values)}个)"
    
    # 构建提示模板
    prompt = (
        "你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段属于哪一类。\n\n"
        f"行业：{industry}\n"
        f"字段名：{column_name}\n"
        f"样本值示例：{sample_str}\n\n"
        "请从以下分类标签中选择最合适的一个（只输出标签路径，不要其他内容）：\n"
    )
    
    # 添加所有分类标签
    for label in CLASSIFICATION_LABELS:
        prompt += f"- {label}\n"
    
    prompt += "\n答案："
    
    return prompt

def build_grading_prompt(item):
    """构建分级任务的提示"""
    column_name = item.get("column_name", "")
    sample_values = item.get("sample_values", [])
    industry = item.get("industry", "")
    classification = item.get("classification_label", "")
    
    # 限制样本值数量
    sample_preview = sample_values[:5]
    sample_str = ", ".join(str(v) for v in sample_preview)
    if len(sample_values) > 5:
        sample_str += f" ... (共{len(sample_values)}个)"
    
    # 构建提示模板
    prompt = (
        "你是一个数据分类分级助手。请根据字段名、行业、样本值和分类结果判断字段的安全分级。\n\n"
        f"行业：{industry}\n"
        f"字段名：{column_name}\n"
        f"样本值示例：{sample_str}\n"
        f"分类结果：{classification}\n\n"
        "请从以下分级标签中选择最合适的一个（只输出标签路径，不要其他内容）：\n"
    )
    
    # 添加所有分级标签
    for label in GRADING_LABELS:
        prompt += f"- {label}\n"
    
    prompt += "\n答案："
    
    return prompt

def load_processed_data():
    """加载标注数据"""
    data_file = os.path.join(PROCESSED_DIR, "all_labeled_data.jsonl")
    
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data

def create_classification_samples(data):
    """创建分类任务样本"""
    samples = []
    
    for item in data:
        classification_label = item.get("classification_label", "")
        if not classification_label:
            continue
        
        prompt = build_classification_prompt(item)
        
        samples.append({
            "instruction": prompt,
            "input": "",
            "output": classification_label,
            "task": "classification",
            "industry": item.get("industry", ""),
            "column_name": item.get("column_name", "")
        })
    
    return samples

def create_grading_samples(data):
    """创建分级任务样本"""
    samples = []
    
    for item in data:
        grading_label = item.get("grading_label", "")
        if not grading_label:
            continue
        
        prompt = build_grading_prompt(item)
        
        samples.append({
            "instruction": prompt,
            "input": "",
            "output": grading_label,
            "task": "grading",
            "industry": item.get("industry", ""),
            "column_name": item.get("column_name", "")
        })
    
    return samples

def split_dataset(samples, train_ratio=0.8, seed=42):
    """划分训练集和验证集"""
    random.seed(seed)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    return train_samples, val_samples

def save_jsonl(samples, filepath):
    """保存为JSONL格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in samples:
            # 移除额外字段，只保留训练需要的字段
            train_item = {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"]
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + "\n")

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载标注数据
    print("\n加载标注数据...")
    data = load_processed_data()
    print(f"  共 {len(data)} 条标注数据")
    
    # 统计按行业分布
    industry_dist = Counter(d.get("industry", "") for d in data)
    print("\n按行业分布:")
    for industry, count in industry_dist.most_common():
        print(f"  {industry}: {count}")
    
    # 创建分类样本
    print("\n创建分类任务样本...")
    classification_samples = create_classification_samples(data)
    print(f"  共 {len(classification_samples)} 个分类样本")
    
    # 创建分级样本
    print("\n创建分级任务样本...")
    grading_samples = create_grading_samples(data)
    print(f"  共 {len(grading_samples)} 个分级样本")
    
    # 合并所有样本
    all_samples = classification_samples + grading_samples
    print(f"\n总共 {len(all_samples)} 个训练样本")
    
    # 划分数据集
    print("\n划分训练集和验证集...")
    train_samples, val_samples = split_dataset(all_samples)
    print(f"  训练集: {len(train_samples)} 条")
    print(f"  验证集: {len(val_samples)} 条")
    
    # 保存数据
    print("\n保存数据文件...")
    train_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_file = os.path.join(OUTPUT_DIR, "val.jsonl")
    
    save_jsonl(train_samples, train_file)
    save_jsonl(val_samples, val_file)
    
    print(f"  训练集已保存: {train_file}")
    print(f"  验证集已保存: {val_file}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    
    # 统计分类标签分布
    train_class_dist = Counter(s["output"] for s in train_samples)
    print("\n训练集分类标签分布:")
    for label, count in train_class_dist.most_common():
        print(f"  {label}: {count}")
    
    print("\n下一步：运行训练脚本")
    print(f"  python scripts/train_model.py")

if __name__ == "__main__":
    main()
