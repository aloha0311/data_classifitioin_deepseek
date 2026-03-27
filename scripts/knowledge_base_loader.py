#!/usr/bin/env python3
"""
知识库规则加载模块
从 data/knowledge_base/ 目录加载用户定义的规则
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(SCRIPT_DIR, "data", "knowledge_base")
GENERAL_RULES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "general_rules.json")
INDUSTRY_RULES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "industry_rules.json")

# 全局缓存
_general_rules_cache = None
_industry_rules_cache = None


def load_general_rules() -> List[dict]:
    """加载通用规则"""
    global _general_rules_cache
    if _general_rules_cache is not None:
        return _general_rules_cache

    if not os.path.exists(GENERAL_RULES_FILE):
        _general_rules_cache = []
        return _general_rules_cache

    try:
        with open(GENERAL_RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)
            _general_rules_cache = rules if isinstance(rules, list) else []
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载通用规则失败: {e}")
        _general_rules_cache = []

    return _general_rules_cache


def load_industry_rules() -> Dict[str, List[dict]]:
    """加载行业规则"""
    global _industry_rules_cache
    if _industry_rules_cache is not None:
        return _industry_rules_cache

    if not os.path.exists(INDUSTRY_RULES_FILE):
        _industry_rules_cache = {}
        return _industry_rules_cache

    try:
        with open(INDUSTRY_RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)
            _industry_rules_cache = rules if isinstance(rules, dict) else {}
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载行业规则失败: {e}")
        _industry_rules_cache = {}

    return _industry_rules_cache


def reload_knowledge_base():
    """重新加载知识库（清除缓存）"""
    global _general_rules_cache, _industry_rules_cache
    _general_rules_cache = None
    _industry_rules_cache = None


def match_field_with_rules(field_name: str, industry: str = None) -> List[Tuple[dict, float]]:
    """
    使用规则匹配字段

    Args:
        field_name: 字段名
        industry: 行业（可选）

    Returns:
        匹配的规则列表，按权重降序排列
    """
    field_name_lower = field_name.lower()
    matches = []

    # 1. 先匹配行业规则（优先级更高）
    if industry:
        industry_rules = load_industry_rules()
        industry_specific = industry_rules.get(industry, [])
        for rule in industry_specific:
            rule_field = rule.get('field', '').lower()
            if rule_field and (rule_field == field_name_lower or rule_field in field_name_lower):
                weight = rule.get('weight', 1.0)
                matches.append((rule, weight + 0.5))  # 行业规则权重加成

    # 2. 匹配通用规则
    general_rules = load_general_rules()
    for rule in general_rules:
        patterns = rule.get('patterns', [])
        if not patterns:
            continue

        for pattern in patterns:
            # 支持正则表达式或简单字符串匹配
            try:
                if re.search(pattern, field_name, re.IGNORECASE):
                    weight = rule.get('weight', 0.9)
                    matches.append((rule, weight))
                    break
            except re.error:
                # 如果不是有效的正则表达式，当作普通字符串匹配
                if pattern.lower() in field_name_lower:
                    weight = rule.get('weight', 0.9)
                    matches.append((rule, weight))
                    break

    # 按权重排序
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def get_classification_from_rules(field_name: str, industry: str = None) -> Optional[str]:
    """
    从规则中获取分类

    Args:
        field_name: 字段名
        industry: 行业

    Returns:
        匹配的分类，如果没有匹配则返回 None
    """
    matches = match_field_with_rules(field_name, industry)
    if matches:
        return matches[0][0].get('category')
    return None


def get_grading_from_rules(field_name: str, industry: str = None) -> Optional[str]:
    """
    从规则中获取分级

    Args:
        field_name: 字段名
        industry: 行业

    Returns:
        匹配的分级，如果没有匹配则返回 None
    """
    matches = match_field_with_rules(field_name, industry)
    if matches:
        return matches[0][0].get('grading')
    return None


def save_general_rules(rules: List[dict]) -> bool:
    """保存通用规则到文件"""
    global _general_rules_cache
    try:
        with open(GENERAL_RULES_FILE, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        _general_rules_cache = rules
        return True
    except IOError as e:
        print(f"保存通用规则失败: {e}")
        return False


def save_industry_rules(rules: Dict[str, List[dict]]) -> bool:
    """保存行业规则到文件"""
    global _industry_rules_cache
    try:
        with open(INDUSTRY_RULES_FILE, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        _industry_rules_cache = rules
        return True
    except IOError as e:
        print(f"保存行业规则失败: {e}")
        return False


def get_knowledge_base_stats() -> dict:
    """获取知识库统计信息"""
    general = load_general_rules()
    industry = load_industry_rules()

    total_industry = sum(len(rules) for rules in industry.values())

    return {
        "totalRules": len(general),
        "totalIndustryRules": total_industry,
        "cachedFields": len(general) + total_industry,
        "industries": list(industry.keys())
    }


if __name__ == "__main__":
    # 测试代码
    print("通用规则:", len(load_general_rules()))
    print("行业规则:", len(load_industry_rules()))
    print("统计:", get_knowledge_base_stats())

    # 测试匹配
    test_fields = ["customer_id", "balance", "password", "email"]
    for field in test_fields:
        matches = match_field_with_rules(field, "金融")
        if matches:
            print(f"{field}: 匹配到 {matches[0][0].get('category')}, {matches[0][0].get('grading')}")
        else:
            print(f"{field}: 未匹配")
