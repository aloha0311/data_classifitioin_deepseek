#!/usr/bin/env python3
"""
数据分类分级规则加载模块
所有脚本通过此模块加载分类分级规则配置
"""
import os
import json
import re
from typing import Dict, List, Optional

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_DIR = os.path.join(SCRIPT_DIR, "data", "labels")
RULES_FILE = os.path.join(RULES_DIR, "grading_rules.json")
CLASSIFICATION_FILE = os.path.join(RULES_DIR, "classification_schema.json")

# 全局缓存
_rules_cache = None
_classification_cache = None


def load_grading_rules() -> dict:
    """加载分级规则配置"""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    
    if not os.path.exists(RULES_FILE):
        raise FileNotFoundError(f"规则文件不存在: {RULES_FILE}")
    
    with open(RULES_FILE, 'r', encoding='utf-8') as f:
        _rules_cache = json.load(f)
    
    return _rules_cache


def get_classification_labels() -> List[str]:
    """获取所有分类标签"""
    global _classification_cache
    if _classification_cache is not None:
        return _classification_cache
    
    if not os.path.exists(CLASSIFICATION_FILE):
        raise FileNotFoundError(f"分类配置文件不存在: {CLASSIFICATION_FILE}")
    
    with open(CLASSIFICATION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        _classification_cache = data.get("flattened_labels", [])
    
    return _classification_cache


def get_grading_labels() -> List[str]:
    """获取所有分级标签"""
    rules = load_grading_rules()
    return rules.get("grading_levels", ["第1级/公开", "第2级/内部", "第3级/敏感", "第4级/机密"])


def infer_grading(classification: str, field_name: str, samples: str = "") -> str:
    """根据分类推断分级，并结合字段名和样本值进行精细调整
    
    Args:
        classification: 分类结果
        field_name: 字段名
        samples: 样本值（可选）
    
    Returns:
        分级标签
    """
    rules = load_grading_rules()
    field_lower = field_name.lower()
    samples_lower = samples.lower() if samples else ""
    
    # 1. 获取基础分级
    mapping = rules.get("classification_to_grading", {})
    base_grading = mapping.get(classification, mapping.get("_default", "第2级/内部"))
    
    # 2. 检查机密关键字
    critical_kw = rules.get("grading_rules", {}).get("critical_keywords", {}).get("keywords", [])
    for keyword in critical_kw:
        if keyword in field_lower or keyword in samples_lower:
            return "第4级/机密"
    
    # 3. 检查敏感关键字
    sensitive_kw = rules.get("grading_rules", {}).get("sensitive_keywords", {}).get("keywords", [])
    for keyword in sensitive_kw:
        if keyword in field_lower or keyword in samples_lower:
            return "第3级/敏感"
    
    # 4. 检查样本值模式
    sample_rules = rules.get("grading_rules", {}).get("sample_patterns", {}).get("rules", [])
    for rule in sample_rules:
        pattern = rule.get("pattern", "")
        min_len = rule.get("min_length", 0)
        grade = rule.get("grade", base_grading)
        
        if pattern and re.search(pattern, samples) and len(samples) >= min_len:
            return grade
    
    return base_grading


# 便捷别名
def infer_grading_from_classification(classification: str, field_name: str, samples: str = "") -> str:
    """infer_grading 的别名，保持向后兼容"""
    return infer_grading(classification, field_name, samples)


if __name__ == "__main__":
    # 测试代码
    print("分类标签:", get_classification_labels())
    print("分级标签:", get_grading_labels())
    
    # 测试推断
    test_cases = [
        ("ID类/主键ID", "user_id", "12345"),
        ("度量类/计量数值", "salary", "15000"),
        ("属性类/名称标题", "customer_name", "张三"),
        ("身份类/联系方式", "phone", "13812345678"),
    ]
    
    print("\n分级推断测试:")
    for cls, field, sample in test_cases:
        grade = infer_grading(cls, field, sample)
        print(f"  {cls} + {field} -> {grade}")
