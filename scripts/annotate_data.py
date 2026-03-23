#!/usr/bin/env python3
"""
数据标注脚本：将原始CSV文件转换为标注数据

这个脚本会：
1. 读取原始CSV文件
2. 提取每个字段的元信息（字段名、样本值、统计特征）
3. 根据分类分级标准自动生成标注（可人工调整）
4. 保存到 processed/ 目录
"""
import os
import json
import csv
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
LABEL_DIR = os.path.join(BASE_DIR, "data/labels")

print("=" * 60)
print("数据标注脚本")
print("=" * 60)

def load_label_schemas():
    """加载分类和分级标签体系"""
    with open(os.path.join(LABEL_DIR, "classification_schema.json"), 'r', encoding='utf-8') as f:
        classification_schema = json.load(f)
    
    with open(os.path.join(LABEL_DIR, "grading_schema.json"), 'r', encoding='utf-8') as f:
        grading_schema = json.load(f)
    
    return classification_schema, grading_schema

def analyze_field(column_name, sample_values):
    """分析字段特征"""
    if not sample_values:
        return {}
    
    # 统计样本值
    total_values = len(sample_values)
    unique_values = len(set(sample_values))
    non_empty = len([v for v in sample_values if v and str(v).strip()])
    
    # 长度分析
    lengths = [len(str(v)) for v in sample_values if v]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    
    # 字符类型分析
    digit_count = sum(1 for v in sample_values if v and str(v).replace('.', '').replace('-', '').isdigit())
    letter_count = sum(1 for v in sample_values if v and any(c.isalpha() for c in str(v)))
    mix_count = sum(1 for v in sample_values if v and any(c.isalpha() for c in str(v)) and any(c.isdigit() for c in str(v)))
    
    # 常见模式识别
    patterns = []
    if digit_count / total_values > 0.8:
        patterns.append("纯数字")
    if letter_count / total_values > 0.8:
        patterns.append("纯字母")
    if mix_count / total_values > 0.8:
        patterns.append("字母数字混合")
    
    # 检查是否为ID
    is_id = column_name.endswith('id') or '编号' in column_name or '代码' in column_name
    
    # 检查是否为代码类
    is_code = '代码' in column_name or '编号' in column_name or 'id' in column_name.lower()
    
    # 检查是否为名称
    is_name = '名称' in column_name or '标题' in column_name
    
    # 检查是否为描述
    is_description = '描述' in column_name or '说明' in column_name or '要求' in column_name
    
    # 检查是否为金额
    is_amount = '薪' in column_name or '价格' in column_name or '金额' in column_name or '费用' in column_name
    
    # 检查是否为数量
    is_count = '人数' in column_name or '次数' in column_name or '浏览量' in column_name or '申请数' in column_name
    
    # 检查是否为比率
    is_rate = '率' in column_name or '%' in str(sample_values[0]) if sample_values else False
    
    # 检查是否为日期时间
    is_datetime = '日期' in column_name or '时间' in column_name or '月份' in column_name
    
    # 检查二元标志
    is_binary = len(set(sample_values)) <= 2 and set(str(v).lower() for v in sample_values).issubset({'yes', 'no', '0', '1', '是', '否', 'true', 'false'})
    
    # 检查是否枚举
    unique_ratio = unique_values / non_empty if non_empty > 0 else 0
    is_enum = unique_ratio < 0.1 and not is_binary
    
    return {
        "total_values": total_values,
        "unique_values": unique_values,
        "unique_ratio": round(unique_ratio, 3),
        "avg_length": round(avg_length, 2),
        "min_length": min_length,
        "max_length": max_length,
        "digit_ratio": round(digit_count / total_values, 3) if total_values > 0 else 0,
        "letter_ratio": round(letter_count / total_values, 3) if total_values > 0 else 0,
        "mix_ratio": round(mix_count / total_values, 3) if total_values > 0 else 0,
        "patterns": patterns,
        "is_id": is_id,
        "is_code": is_code,
        "is_name": is_name,
        "is_description": is_description,
        "is_amount": is_amount,
        "is_count": is_count,
        "is_rate": is_rate,
        "is_datetime": is_datetime,
        "is_binary": is_binary,
        "is_enum": is_enum,
        "sample_values": sample_values[:5]
    }

def auto_classify(column_name, features, industry=""):
    """根据规则自动分类"""
    # ID类
    if features.get('is_id') or 'id' in column_name.lower():
        if '主键' in column_name.lower() or column_name.endswith('id'):
            return "ID类/主键ID"
    
    # 结构类
    if features.get('is_code'):
        if '分类' in column_name:
            return "结构类/分类代码"
        if '产品' in column_name or '零件' in column_name:
            return "结构类/产品代码"
        if '企业' in column_name or '公司' in column_name:
            return "结构类/企业代码"
        if '批次' in column_name or '标准' in column_name:
            return "结构类/标准代码"
        # 默认代码类
        if '产品' in str(features.get('sample_values', [''])[0]) if features.get('sample_values') else False:
            return "结构类/产品代码"
        if column_name == '分类代码':
            return "结构类/分类代码"
    
    # 属性类
    if features.get('is_name'):
        return "属性类/名称标题"
    
    if features.get('is_description'):
        if '技能' in column_name or '要求' in column_name:
            return "属性类/技能标签"
        return "属性类/描述文本"
    
    if '城市' in column_name or '地址' in column_name or '位置' in column_name:
        return "属性类/地址位置"
    
    if '类别' in column_name:
        return "属性类/类别标签"
    
    # 度量类
    if features.get('is_amount') or '薪' in column_name or '价格' in column_name:
        return "度量类/计量数值"
    
    if features.get('is_count'):
        return "度量类/计数统计"
    
    if features.get('is_rate'):
        return "度量类/比率比例"
    
    if '时长' in column_name or '时间' in column_name or '间隔' in column_name:
        return "度量类/时间度量"
    
    if '序列' in column_name or '序号' in column_name:
        return "度量类/序号排序"
    
    # 检查数字类
    if features.get('digit_ratio', 0) > 0.8 and features.get('avg_length', 0) < 20:
        # 可能是ID或代码
        if features.get('is_enum') and features.get('unique_ratio', 1) < 0.05:
            return "度量类/序号排序"
    
    # 身份类
    if '年龄' in column_name:
        return "身份类/人口统计"
    
    if '婚姻' in column_name:
        return "身份类/人口统计"
    
    if '联系' in column_name and ('电话' in column_name or '手机' in column_name or '邮箱' in column_name or '邮件' in column_name):
        return "身份类/联系方式"
    
    if '教育' in column_name or '学历' in column_name:
        return "身份类/教育背景"
    
    if '职业' in column_name or '工作' in column_name:
        return "身份类/职业信息"
    
    # 状态类
    if features.get('is_binary'):
        if '是否' in column_name or '有没有' in column_name:
            return "状态类/二元标志"
    
    if '联系' in column_name and ('方式' in column_name or '类型' in column_name):
        return "状态类/状态枚举"
    
    if '结果' in column_name or '营销' in column_name:
        return "状态类/状态枚举"
    
    if features.get('is_datetime'):
        if '发布' in column_name or '日期' in column_name:
            return "状态类/时间标记"
        return "状态类/时间标记"
    
    # 扩展类
    if '扩展' in column_name:
        return "扩展类/扩展代码"
    
    # 默认：根据avg_length和模式判断
    if features.get('mix_ratio', 0) > 0.5:
        return "结构类/产品代码"
    
    return "扩展类/其他字段"

def auto_grade(classification_label, column_name, features):
    """根据分类自动确定分级"""
    mapping = {
        "ID类/主键ID": "第1级/公开",
        "结构类/分类代码": "第1级/公开",
        "结构类/产品代码": "第1级/公开",
        "结构类/企业代码": "第1级/公开",
        "结构类/标准代码": "第1级/公开",
        "属性类/名称标题": "第1级/公开",
        "属性类/类别标签": "第1级/公开",
        "属性类/描述文本": "第2级/内部",
        "属性类/技能标签": "第2级/内部",
        "属性类/地址位置": "第2级/内部",
        "度量类/计量数值": "第2级/内部",
        "度量类/计数统计": "第2级/内部",
        "度量类/比率比例": "第2级/内部",
        "度量类/时间度量": "第2级/内部",
        "度量类/序号排序": "第1级/公开",
        "身份类/人口统计": "第3级/敏感",
        "身份类/联系方式": "第4级/机密",
        "身份类/教育背景": "第3级/敏感",
        "身份类/职业信息": "第3级/敏感",
        "状态类/二元标志": "第2级/内部",
        "状态类/状态枚举": "第2级/内部",
        "状态类/时间标记": "第2级/内部",
        "扩展类/扩展代码": "第2级/内部",
        "扩展类/其他字段": "第1级/公开"
    }
    
    return mapping.get(classification_label, "第2级/内部")

def process_csv_file(csv_path, industry):
    """处理单个CSV文件"""
    fields = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        
        if not columns:
            return fields
        
        rows = list(reader)
        
        for column_name in columns:
            if column_name == 'industry':
                continue
            
            sample_values = []
            for row in rows:
                value = row.get(column_name, "")
                if value and str(value).strip():
                    sample_values.append(str(value).strip())
            
            if not sample_values:
                continue
            
            # 分析特征
            features = analyze_field(column_name, sample_values)
            
            # 自动分类
            classification_label = auto_classify(column_name, features, industry)
            
            # 自动分级
            grading_label = auto_grade(classification_label, column_name, features)
            
            field_data = {
                "industry": industry,
                "column_name": column_name,
                "sample_values": sample_values,
                "classification_label": classification_label,
                "grading_label": grading_label,
                "features": features
            }
            
            fields.append(field_data)
    
    return fields

def main():
    parser = argparse.ArgumentParser(description="数据标注脚本")
    parser.add_argument("--industry", type=str, help="指定行业名称")
    parser.add_argument("--all", action="store_true", help="处理所有CSV文件")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 加载标签体系
    classification_schema, grading_schema = load_label_schemas()
    print(f"\n已加载分类标签：{len(classification_schema['flattened_labels'])} 个")
    print(f"已加载分级标签：{len(grading_schema['flattened_grades'])} 个")
    
    all_data = []
    
    if args.all:
        # 处理所有CSV文件
        csv_files = list(Path(RAW_DIR).glob("*.csv"))
        print(f"\n找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            industry = csv_file.stem.replace('_train', '').replace('_test', '')
            print(f"\n处理: {csv_file.name} (行业: {industry})")
            fields = process_csv_file(csv_file, industry)
            all_data.extend(fields)
            print(f"  提取了 {len(fields)} 个字段")
    
    elif args.industry:
        # 处理指定行业
        csv_files = list(Path(RAW_DIR).glob(f"{args.industry}*.csv"))
        for csv_file in csv_files:
            industry = args.industry
            print(f"\n处理: {csv_file.name} (行业: {industry})")
            fields = process_csv_file(csv_file, industry)
            all_data.extend(fields)
            print(f"  提取了 {len(fields)} 个字段")
    else:
        print("请指定 --industry <行业名> 或 --all")
        return
    
    # 统计分类分布
    classification_dist = Counter(d['classification_label'] for d in all_data)
    grading_dist = Counter(d['grading_label'] for d in all_data)
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总共提取字段数: {len(all_data)}")
    
    print("\n分类分布:")
    for label, count in classification_dist.most_common():
        print(f"  {label}: {count}")
    
    print("\n分级分布:")
    for label, count in grading_dist.most_common():
        print(f"  {label}: {count}")
    
    # 保存到文件
    output_file = os.path.join(PROCESSED_DIR, "all_labeled_data.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n标注数据已保存到: {output_file}")

if __name__ == "__main__":
    main()
