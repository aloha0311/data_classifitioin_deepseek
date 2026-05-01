#!/usr/bin/env python3
"""
将人工标注的数据转换为微调训练格式
"""
import json
import os
from pathlib import Path

# 参考示例 - 用于构建prompt
REFERENCE_EXAMPLES = """
  字段：条目id → ID类/主键ID
  字段：公司名称 → 属性类/名称标题
  字段：所属城市 → 属性类/地址位置
  字段：要求技能 → 属性类/技能标签
  字段：岗位要求 → 属性类/描述文本
  字段：公司规模 → 属性类/类别标签
  字段：序列号 → 度量类/序号排序
  字段：课程时长 → 度量类/时间度量
  字段：完课率 → 度量类/比率比例
  字段：报名人数 → 度量类/计数统计
  字段：最低薪资 → 度量类/计量数值
  字段：扩展数据代码 → 扩展类/扩展代码
  字段：是否有房贷 → 状态类/二元标志
  字段：上一次联系的月份 → 状态类/时间标记
  字段：之前营销活动的结果 → 状态类/状态枚举
  字段：产品代码 → 结构类/产品代码
  字段：分类代码 → 结构类/分类代码
  字段：批次号 → 结构类/标准代码
  字段：年龄 → 身份类/人口统计
  字段：教育情况 → 身份类/教育背景
  字段：要求经历 → 身份类/职业信息
  字段：联系方式 → 身份类/联系方式"""


def create_instruction(industry, field_name, samples):
    """创建分类指令"""
    return f"""你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段属于哪一类。

行业：{industry}
字段名：{field_name}
样本值示例：{samples}

参考示例：
{REFERENCE_EXAMPLES}

请从以上分类标签中选择最合适的一个（只输出标签路径，不要其他内容）。

答案："""


def create_grading_instruction(field_name, classification, samples):
    """创建分级指令"""
    return f"""你是一个数据分类分级助手。请根据字段名、分类和样本值判断字段属于哪一级别。

字段名：{field_name}
分类：{classification}
样本值示例：{samples}

分级标准：
- 第1级/公开：一般性公开数据，可自由传播
- 第2级/内部：内部使用数据，不宜对外公开
- 第3级/敏感：敏感数据，需要保护处理
- 第4级/机密：高度敏感数据，严格保密

请根据字段的分类和样本值，判断合适的分级（只输出分级标签，不要其他内容）。

答案："""


def convert_to_jsonl(annotation_file, output_file, task_type="classification"):
    """将标注数据转换为jsonl格式
    
    Args:
        annotation_file: 标注JSON文件路径
        output_file: 输出jsonl文件路径
        task_type: "classification" 或 "grading"
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    industry = data.get('industry', '其他')
    
    results = []
    
    for item in annotations:
        field_name = item['field']
        classification = item['classification']
        grading = item['grading']
        samples = item.get('samples', [])
        samples_str = ', '.join(str(s) for s in samples[:5])
        if len(samples) > 5:
            samples_str += f' ... (共{len(samples)}个)'
        
        if task_type == "classification":
            instruction = create_instruction(industry, field_name, samples_str)
            output = classification
        else:
            instruction = create_grading_instruction(field_name, classification, samples_str)
            output = grading
        
        results.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(results)


def main():
    # 设置路径
    script_dir = Path(__file__).parent
    newtest_dir = script_dir.parent / 'data' / 'newtest'
    output_dir = script_dir.parent / 'data' / 'sft'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    total_classification = 0
    total_grading = 0
    
    # 处理每个标注文件
    for json_file in newtest_dir.glob('*_annotated.json'):
        print(f"\n处理文件: {json_file.name}")
        
        # 分类任务
        cls_output = output_dir / f"train_{json_file.stem}_cls.jsonl"
        count = convert_to_jsonl(json_file, cls_output, task_type="classification")
        print(f"  - 分类任务: {count} 条")
        total_classification += count
        
        # 分级任务
        grade_output = output_dir / f"train_{json_file.stem}_grade.jsonl"
        count = convert_to_jsonl(json_file, grade_output, task_type="grading")
        print(f"  - 分级任务: {count} 条")
        total_grading += count
    
    print(f"\n总计:")
    print(f"  - 分类任务: {total_classification} 条")
    print(f"  - 分级任务: {total_grading} 条")
    print(f"  - 输出目录: {output_dir}")


if __name__ == '__main__':
    main()
