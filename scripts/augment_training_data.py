#!/usr/bin/env python3
"""
数据增强脚本：平衡分级数据比例，增加第2、3、4级样本
"""
import json
import random
import os
from pathlib import Path
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SFT_DIR = os.path.join(DATA_DIR, "sft")
NEWTEST_DIR = os.path.join(DATA_DIR, "newtest")

# 分类标签
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
    "扩展类/其他字段",
]

# 分级标签
GRADING_LABELS = [
    "第1级/公开",
    "第2级/内部",
    "第3级/敏感",
    "第4级/机密",
]


def load_all_annotated_data():
    """加载所有标注数据"""
    all_data = []
    for json_file in Path(NEWTEST_DIR).glob("*_annotated.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations = data.get('annotations', [])
            industry = data.get('industry', '其他')
            for ann in annotations:
                ann['industry'] = industry
                ann['source_file'] = json_file.stem
            all_data.extend(annotations)
    return all_data


def build_combined_prompt(industry, field_name, samples_str, classification, grading):
    """构建分类分级同时输出的prompt"""
    return f"""你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段的分类和分级。

【分类标签】
- ID类：主键ID
- 属性类：名称标题、地址位置、描述文本、类别标签、技能标签
- 度量类：计量数值、计数统计、时间度量、比率比例、序号排序
- 状态类：时间标记、状态枚举、二元标志
- 身份类：人口统计、教育背景、职业信息、联系方式
- 结构类：产品代码、分类代码、企业代码、标准代码
- 扩展类：扩展代码、其他字段

【分级标准】
- 第1级/公开：一般性公开数据
- 第2级/内部：内部使用数据
- 第3级/敏感：敏感数据，需要保护
- 第4级/机密：高度敏感数据，严格保密

行业：{industry}
字段名：{field_name}
样本值示例：{samples_str}

请分析字段含义，输出分类和分级（格式：分类，分级）。

答案："""


def create_samples(data_list, target_dist=None):
    """创建训练样本，支持数据增强
    
    Args:
        data_list: 标注数据列表
        target_dist: 目标分级分布，如 {"第1级/公开": 0.35, "第2级/内部": 0.25, ...}
    """
    samples = []
    
    # 统计当前分布
    current_dist = Counter(item['grading'] for item in data_list)
    print("当前分级分布:")
    for g, c in sorted(current_dist.items()):
        print(f"  {g}: {c}")
    
    # 计算需要增强的样本
    if target_dist:
        total = len(data_list)
        target_counts = {g: int(total * p) for g, p in target_dist.items()}
        print(f"\n目标分布 (基于 {total} 条):")
        for g, c in sorted(target_counts.items()):
            print(f"  {g}: {c}")
    
    # 首先添加所有原始样本
    for item in data_list:
        field_name = item.get('field', '')
        industry = item.get('industry', '其他')
        classification = item.get('classification', '')
        grading = item.get('grading', '')
        samples_list = item.get('samples', [])
        
        if not classification or not grading:
            continue
            
        samples_str = ', '.join(str(s) for s in samples_list[:5])
        if len(samples_list) > 5:
            samples_str += f' ... (共{len(samples_list)}个)'
        
        prompt = build_combined_prompt(industry, field_name, samples_str, classification, grading)
        
        samples.append({
            "instruction": prompt,
            "input": "",
            "output": f"{classification}，{grading}",
            "classification": classification,
            "grading": grading,
            "industry": industry,
            "field": field_name,
        })
    
    # 数据增强：对第2、3、4级样本进行增强
    enhanced_samples = []
    
    # 收集各分级样本
    grade_samples = {g: [] for g in GRADING_LABELS}
    for s in samples:
        grade_samples[s['grading']].append(s)
    
    # 目标分布
    if target_dist:
        total_target = sum(target_counts.values())
        
        for grade, target_count in target_counts.items():
            current_count = len(grade_samples[grade])
            
            # 需要增强的数量
            if target_count > current_count:
                need = target_count - current_count
                print(f"\n{grade}: 当前 {current_count} 条，需要增强 {need} 条")
                
                # 通过同义词替换字段名来增强
                base_samples = grade_samples[grade]
                if base_samples:
                    synonyms = get_field_synonyms()
                    
                    for i in range(need):
                        base = base_samples[i % len(base_samples)]
                        
                        # 生成变体
                        variant = create_variant(base, synonyms, grade)
                        if variant:
                            enhanced_samples.append(variant)
    
    samples.extend(enhanced_samples)
    
    return samples


def get_field_synonyms():
    """获取字段名同义词映射，用于数据增强"""
    return {
        # ID类
        "id": ["编号", "编码", "标识", "标识符", "序列号"],
        "name": ["名称", "姓名", "名", "称谓"],
        "address": ["地址", "位置", "住址", "所在地"],
        "phone": ["电话", "手机", "联系方式", "号码"],
        "email": ["邮箱", "电子邮件", "邮件"],
        "date": ["日期", "时间", "时间戳", "时刻"],
        "price": ["价格", "价钱", "金额", "费用", "成本"],
        "count": ["数量", "数目", "个数", "计数"],
        "age": ["年龄", "年岁"],
        "gender": ["性别", "男女"],
        "status": ["状态", "状态码", "状态标识"],
        "type": ["类型", "类别", "种类"],
        "description": ["描述", "说明", "详情", "简介"],
        "code": ["代码", "编码", "代号"],
    }


def create_variant(sample, synonyms, target_grade=None):
    """创建样本变体"""
    field = sample['field'].lower()
    classification = sample['classification']
    grading = target_grade if target_grade else sample['grading']
    
    # 尝试找到匹配的同义词
    variant_field = None
    for key, syns in synonyms.items():
        if key in field:
            # 随机选择一个同义词
            new_suffix = random.choice(syns)
            # 保留原始字段名的前缀部分
            for prefix in ['user_', 'cust_', 'order_', 'product_', 'info_', '']:
                if prefix + key in field or field.endswith(key):
                    base_name = field.replace(key, '').replace(prefix, '')
                    if base_name:
                        variant_field = prefix + base_name + '_' + new_suffix
                    else:
                        variant_field = new_suffix
                    break
            if variant_field:
                break
    
    if not variant_field:
        # 使用原始字段名添加序号
        variant_field = f"{sample['field']}_v{random.randint(1, 100)}"
    
    # 构建新的样本
    prompt = build_combined_prompt(
        sample['industry'],
        variant_field,
        sample.get('instruction', '').split('样本值示例：')[1].split('\n')[0] if '样本值示例：' in sample.get('instruction', '') else '无样本',
        classification,
        grading
    )
    
    return {
        "instruction": prompt,
        "input": "",
        "output": f"{classification}，{grading}",
        "classification": classification,
        "grading": grading,
        "industry": sample['industry'],
        "field": variant_field,
        "is_variant": True,
    }


def split_and_save(samples, train_ratio=0.85, seed=42):
    """划分数据集并保存"""
    random.seed(seed)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # 保存
    train_file = os.path.join(SFT_DIR, "train.jsonl")
    val_file = os.path.join(SFT_DIR, "val.jsonl")
    
    def save_jsonl(data_list, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                # 只保留训练需要的字段
                train_item = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"],
                }
                f.write(json.dumps(train_item, ensure_ascii=False) + '\n')
    
    save_jsonl(train_samples, train_file)
    save_jsonl(val_samples, val_file)
    
    return len(train_samples), len(val_samples)


def main():
    print("=" * 60)
    print("数据增强脚本：平衡分级数据比例")
    print("=" * 60)
    
    # 加载所有标注数据
    print("\n步骤1：加载标注数据...")
    all_data = load_all_annotated_data()
    print(f"  共加载 {len(all_data)} 条标注数据")
    
    # 统计分布
    grade_dist = Counter(item.get('grading', '') for item in all_data)
    print("\n原始分级分布:")
    for g, c in sorted(grade_dist.items()):
        print(f"  {g}: {c} ({c/len(all_data)*100:.1f}%)")
    
    # 目标分布（更平衡）
    target_distribution = {
        "第1级/公开": 0.40,   # 40%
        "第2级/内部": 0.25,   # 25%
        "第3级/敏感": 0.20,   # 20%
        "第4级/机密": 0.15,   # 15%
    }
    
    # 创建增强样本
    print("\n步骤2：创建增强样本...")
    all_samples = create_samples(all_data, target_distribution)
    print(f"  增强后共 {len(all_samples)} 条样本")
    
    # 统计增强后的分布
    final_dist = Counter(s['grading'] for s in all_samples)
    print("\n增强后分级分布:")
    for g, c in sorted(final_dist.items()):
        print(f"  {g}: {c} ({c/len(all_samples)*100:.1f}%)")
    
    # 划分并保存
    print("\n步骤3：划分数据集并保存...")
    train_count, val_count = split_and_save(all_samples)
    print(f"  训练集: {train_count} 条")
    print(f"  验证集: {val_count} 条")
    
    print("\n" + "=" * 60)
    print("数据增强完成！")
    print("下一步：运行训练脚本")
    print("  python scripts/train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
