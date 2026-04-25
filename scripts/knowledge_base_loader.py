#!/usr/bin/env python3
"""
知识库规则加载模块
从 data/knowledge_base/ 目录加载用户定义的规则
支持基于向量相似度的智能匹配
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(SCRIPT_DIR, "data", "knowledge_base")
GENERAL_RULES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "general_rules.json")
INDUSTRY_RULES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "industry_rules.json")

# 全局缓存
_general_rules_cache = None
_industry_rules_cache = None
_similarity_cache = {}  # 相似度计算缓存


@dataclass
class KBMatchResult:
    """知识库匹配结果"""
    field: str
    matched_field: str
    similarity: float
    category: str
    grading: str
    source: str  # "general" 或 "industry:xxx"
    is_conflict: bool = False
    conflict_with: Optional[dict] = None


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
    global _general_rules_cache, _industry_rules_cache, _similarity_cache
    _general_rules_cache = None
    _industry_rules_cache = None
    _similarity_cache = {}


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


# ============= 向量相似度匹配功能 =============

def _get_similarity_calculator():
    """懒加载相似度计算器"""
    global _similarity_cache
    if 'calculator' not in _similarity_cache:
        from scripts.similarity_calculator import FieldSimilarityCalculator
        _similarity_cache['calculator'] = FieldSimilarityCalculator()
    return _similarity_cache['calculator']


def compute_field_similarity(field1: str, field2: str) -> float:
    """
    计算两个字段名的相似度

    Args:
        field1: 字段名1
        field2: 字段名2

    Returns:
        相似度得分 (0-1)
    """
    calculator = _get_similarity_calculator()
    result = calculator.compute_similarity(field1, field2)
    return result.combined_score


def find_similar_fields_in_kb(target_field: str,
                              industry: str = None,
                              top_k: int = 5,
                              threshold: float = 0.5) -> List[KBMatchResult]:
    """
    在知识库中查找与目标字段相似的字段

    Args:
        target_field: 目标字段名
        industry: 行业（可选）
        top_k: 返回前k个最相似的字段
        threshold: 相似度阈值

    Returns:
        匹配的字段列表
    """
    calculator = _get_similarity_calculator()
    results = []

    # 1. 搜索通用规则
    general_rules = load_general_rules()
    for rule in general_rules:
        patterns = rule.get('patterns', [])
        for pattern in patterns:
            sim = calculator.compute_similarity(target_field, pattern).combined_score
            if sim >= threshold:
                results.append(KBMatchResult(
                    field=target_field,
                    matched_field=pattern,
                    similarity=sim,
                    category=rule.get('category', ''),
                    grading=rule.get('grading', ''),
                    source='general'
                ))

    # 2. 搜索行业规则（如果指定了行业）
    if industry:
        industry_rules = load_industry_rules()
        industry_specific = industry_rules.get(industry, [])
        for rule in industry_specific:
            field = rule.get('field', '')
            sim = calculator.compute_similarity(target_field, field).combined_score
            if sim >= threshold:
                results.append(KBMatchResult(
                    field=target_field,
                    matched_field=field,
                    similarity=sim,
                    category=rule.get('category', ''),
                    grading=rule.get('grading', ''),
                    source=f'industry:{industry}'
                ))

    # 按相似度排序
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:top_k]


def detect_conflicts_with_similarity(target_field: str,
                                    predicted_category: str,
                                    predicted_grading: str,
                                    industry: str = None) -> List[dict]:
    """
    检测新字段与知识库中现有规则的冲突

    Args:
        target_field: 目标字段名
        predicted_category: 预测的分类
        predicted_grading: 预测的分级
        industry: 行业

    Returns:
        冲突列表
    """
    similar = find_similar_fields_in_kb(
        target_field,
        industry=industry,
        top_k=10,
        threshold=0.4  # 相似度超过0.4就可能存在冲突
    )

    conflicts = []
    for match in similar:
        # 检查分类或分级是否冲突
        if (match.category and match.category != predicted_category) or \
           (match.grading and match.grading != predicted_grading):
            match.is_conflict = True
            match.conflict_with = {
                'field': match.matched_field,
                'category': match.category,
                'grading': match.grading,
                'similarity': match.similarity
            }
            conflicts.append({
                'type': 'classification_conflict' if match.category != predicted_category else 'grading_conflict',
                'similar_field': match.matched_field,
                'kb_category': match.category,
                'kb_grading': match.grading,
                'predicted_category': predicted_category,
                'predicted_grading': predicted_grading,
                'similarity': match.similarity,
                'suggestion': f"字段 '{target_field}' 与知识库中的 '{match.matched_field}' 相似度较高({match.similarity:.2f})，但分类/分级结果不一致，请确认"
            })

    return conflicts


def add_rule_with_similarity_check(field: str,
                                   category: str,
                                   grading: str,
                                   industry: str = None,
                                   auto_confirm: bool = False) -> dict:
    """
    添加规则前先进行相似度检查

    Args:
        field: 字段名
        category: 分类
        grading: 分级
        industry: 行业
        auto_confirm: 是否自动确认（当没有冲突时）

    Returns:
        结果字典，包含添加状态和冲突信息
    """
    # 先检查是否有相似字段
    similar = find_similar_fields_in_kb(field, industry, top_k=5, threshold=0.5)

    result = {
        'field': field,
        'category': category,
        'grading': grading,
        'added': False,
        'conflicts': [],
        'warnings': []
    }

    # 检查冲突
    for match in similar:
        if match.category == category and match.grading == grading:
            # 完全一致，只是重复
            result['warnings'].append(f"字段 '{field}' 与知识库中的 '{match.matched_field}' 完全一致，无需重复添加")
        elif match.similarity >= 0.7:
            # 高度相似但结果不同，可能是冲突
            result['conflicts'].append({
                'similar_field': match.matched_field,
                'kb_category': match.category,
                'kb_grading': match.grading,
                'similarity': match.similarity,
                'severity': 'high' if match.similarity >= 0.85 else 'medium'
            })

    return result


def batch_similarity_search(fields: List[str],
                          industry: str = None,
                          threshold: float = 0.5) -> Dict[str, List[KBMatchResult]]:
    """
    批量字段相似度搜索

    Args:
        fields: 字段列表
        industry: 行业
        threshold: 相似度阈值

    Returns:
        {field_name: [matched_results]}
    """
    results = {}
    for field in fields:
        results[field] = find_similar_fields_in_kb(
            field,
            industry=industry,
            top_k=5,
            threshold=threshold
        )
    return results


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
