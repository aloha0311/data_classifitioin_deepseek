#!/usr/bin/env python3
"""
批量预测新数据的分类和分级
支持用户选择文件，简洁输出
"""
import os
import sys
import json
import glob
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.tokenizer_fix import get_chinese_tokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from scripts.grading_rules import (
    get_classification_labels,
    get_grading_labels,
    infer_grading,
    load_grading_rules
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")
DATA_DIR = os.path.join(BASE_DIR, "data/new")
RESULTS_DIR = os.path.join(BASE_DIR, "results/csv_predict")

os.makedirs(RESULTS_DIR, exist_ok=True)

# 从配置加载标签
CLASSIFICATION_LABELS = get_classification_labels()
GRADING_LABELS = get_grading_labels()

INDUSTRY_KEYWORDS = {
    'medical': ['patient', 'diagnos', 'disease', 'hospital', 'clinic', 'breast', 'cancer', 'diabetes'],
    'education': ['student', 'school', 'course', 'grade', 'score', 'exam', 'university', 'learn', '课程', '学生', '考试'],
    'business': ['company', 'job', 'position', 'salary', 'employee', 'recruit', '岗位', '薪资', '公司', '招聘', '天猫', '美妆'],
    'financial': ['bank', 'credit', 'loan', 'finance', 'customer', 'account', 'transaction', '银行', '信用', '贷款', '客户'],
    'industrial': ['machine', 'device', 'product', 'batch', 'serial', 'equipment', 'sensor', '设备', '工业', '测试']
}


def list_files():
    """列出可选的数据文件"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    xls_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx")) + glob.glob(os.path.join(DATA_DIR, "*.xls"))
    return sorted(csv_files + xls_files)


def select_files():
    """让用户选择要处理的文件"""
    files = list_files()
    
    if not files:
        print(f"未找到数据文件: {DATA_DIR}")
        return []
    
    print("=" * 60)
    print("可用文件列表：")
    print("=" * 60)
    for i, f in enumerate(files, 1):
        filename = os.path.basename(f)
        print(f"  {i}. {filename}")
    print()
    
    while True:
        print("请输入要处理的文件编号（支持多选，用逗号分隔，如: 1,3,5）：")
        print("输入 'a' 处理所有文件，输入 'q' 退出：")
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            return []
        elif choice.lower() == 'a':
            return files
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = [files[i-1] for i in indices if 1 <= i <= len(files)]
                if selected:
                    return selected
                print("输入无效，请重试")
            except ValueError:
                print("输入无效，请重试")


def infer_industry(filename, columns):
    """根据文件名和列名推断行业"""
    filename_lower = filename.lower()
    columns_str = ' '.join(columns).lower()
    
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in filename_lower or kw in columns_str:
                return industry
    return 'business'


def get_column_samples(series, n=5):
    """获取列的样本值"""
    valid_values = series.dropna().astype(str)
    valid_values = valid_values[valid_values.str.len() < 100]
    unique_vals = valid_values.unique()[:n]
    return ', '.join(unique_vals) if len(unique_vals) > 0 else 'N/A'


def load_model():
    """加载模型"""
    print("\n加载模型...")
    tokenizer = get_chinese_tokenizer(MODEL_DIR)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        print("加载LoRA权重...")
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        print("警告: 未找到LoRA权重，使用基础模型")
        model = base_model
    
    model.eval()
    return model, tokenizer


def extract_label(text, labels):
    """从模型输出中提取标签"""
    text = text.strip()
    for label in labels:
        if label in text:
            return label
    return text.split('\n')[0].strip() if text else "未识别"


def predict_classification(model, tokenizer, industry, column_name, samples):
    """预测分类"""
    labels_text = '\n'.join([f"- {label}" for label in CLASSIFICATION_LABELS])
    prompt = f"""行业：{industry}
字段名：{column_name}
样本值示例：{samples}

请从以下分类标签中选择最合适的一个（只输出标签）：
{labels_text}

答案："""
    
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return extract_label(response, CLASSIFICATION_LABELS)


def predict_grading(model, tokenizer, industry, column_name, samples, classification):
    """预测分级 - 基于分类映射，不依赖模型直接预测"""
    # 使用 grading_rules 模块的 infer_grading 函数
    grading = infer_grading(classification, column_name, samples)
    return grading


def process_file(filepath, model, tokenizer):
    """处理单个文件"""
    filename = os.path.basename(filepath)
    print(f"\n处理: {filename}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        print(f"  不支持的文件格式")
        return None
    
    industry = infer_industry(filename, df.columns.tolist())
    print(f"  行业: {industry}, 字段数: {len(df.columns)}")
    
    results = []
    for col in df.columns:
        samples = get_column_samples(df[col])
        
        classification = predict_classification(model, tokenizer, industry, col, samples)
        grading = predict_grading(model, tokenizer, industry, col, samples, classification)
        
        results.append({
            "字段名": col,
            "分类": classification,
            "分级": grading
        })
        
        print(f"  {col}: {classification} / {grading}")
    
    return results


def save_results(filename, results):
    """保存结果到CSV"""
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  -> 已保存: {output_path}")


def main():
    selected_files = select_files()
    
    if not selected_files:
        print("未选择文件，退出")
        return
    
    model, tokenizer = load_model()
    
    for filepath in selected_files:
        results = process_file(filepath, model, tokenizer)
        if results:
            filename = os.path.basename(filepath)
            output_name = filename.replace('.csv', '_预测结果.csv').replace('.xlsx', '_预测结果.csv').replace('.xls', '_预测结果.csv')
            save_results(output_name, results)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
