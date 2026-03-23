#!/usr/bin/env python3
"""
动态知识库模块
支持规则动态更新和增量学习
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DynamicKnowledgeBase:
    """动态知识库"""
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            kb_path: 知识库文件路径
        """
        self.kb_path = kb_path or os.path.join(BASE_DIR, "data/knowledge_base.json")
        self.rules = []  # 分类规则
        self.industry_rules = defaultdict(list)  # 行业特定规则
        self.field_cache = {}  # 字段缓存（加速查询）
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库"""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.rules = data.get('rules', [])
                self.industry_rules = defaultdict(list, data.get('industry_rules', {}))
                self.field_cache = data.get('field_cache', {})
        else:
            self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        # 通用分类规则
        self.rules = [
            # ID类
            {
                "id": "rule_001",
                "patterns": [r'^id$', r'_id$', r'^no$', r'编号', r'编码'],
                "category": "ID类/主键ID",
                "grading": "第1级/公开",
                "weight": 1.0,
                "description": "唯一标识符"
            },
            # 名称类
            {
                "id": "rule_002",
                "patterns": [r'name', r'名称', r'标题', r'title', r'username'],
                "category": "属性类/名称标题",
                "grading": "第1级/公开",
                "weight": 0.9,
                "description": "名称或标题字段"
            },
            # 价格类
            {
                "id": "rule_003",
                "patterns": [r'price', r'金额', r'价格', r'money', r'cost', r'fee'],
                "category": "度量类/计量数值",
                "grading": "第2级/内部",
                "weight": 0.95,
                "description": "价格或金额"
            },
            # 数量类
            {
                "id": "rule_004",
                "patterns": [r'count', r'数量', r'quantity', r'num', r'次数'],
                "category": "度量类/计数统计",
                "grading": "第2级/内部",
                "weight": 0.9,
                "description": "计数统计"
            },
            # 时间类
            {
                "id": "rule_005",
                "patterns": [r'date', r'time', r'时间', r'日期', r'created', r'updated'],
                "category": "状态类/时间标记",
                "grading": "第2级/内部",
                "weight": 0.9,
                "description": "时间标记字段"
            },
            # 地址类
            {
                "id": "rule_006",
                "patterns": [r'address', r'地址', r'city', r'城市', r'location', r'国家'],
                "category": "属性类/地址位置",
                "grading": "第1级/公开",
                "weight": 0.9,
                "description": "地址位置"
            },
            # 状态类
            {
                "id": "rule_007",
                "patterns": [r'status', r'状态', r'type', r'类型', r'flag', r'标志'],
                "category": "状态类/状态枚举",
                "grading": "第2级/内部",
                "weight": 0.85,
                "description": "状态枚举"
            },
            # 年龄类
            {
                "id": "rule_008",
                "patterns": [r'age', r'年龄'],
                "category": "身份类/人口统计",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "年龄信息"
            },
            # 性别类
            {
                "id": "rule_009",
                "patterns": [r'gender', r'性别', r'sex'],
                "category": "身份类/人口统计",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "性别信息"
            },
            # 收入类
            {
                "id": "rule_010",
                "patterns": [r'income', r'收入', r'salary', r'工资', r'薪酬'],
                "category": "度量类/计量数值",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "收入相关"
            },
            # 联系方式
            {
                "id": "rule_011",
                "patterns": [r'phone', r'电话', r'mobile', r'手机', r'email', r'邮箱'],
                "category": "身份类/联系方式",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "联系方式"
            },
            # 教育类
            {
                "id": "rule_012",
                "patterns": [r'education', r'教育', r'degree', r'学历'],
                "category": "身份类/教育背景",
                "grading": "第3级/敏感",
                "weight": 0.95,
                "description": "教育背景"
            },
        ]
        
        # 行业特定规则
        self.industry_rules = {
            "金融": [
                {"field": "credit_score", "category": "度量类/计量数值", "grading": "第3级/敏感"},
                {"field": "balance", "category": "度量类/计量数值", "grading": "第3级/敏感"},
                {"field": "limit", "category": "度量类/计量数值", "grading": "第3级/敏感"},
            ],
            "医疗": [
                {"field": "diagnosis", "category": "属性类/描述文本", "grading": "第3级/敏感"},
                {"field": "blood_pressure", "category": "度量类/计量数值", "grading": "第3级/敏感"},
            ],
            "教育": [
                {"field": "score", "category": "度量类/计量数值", "grading": "第2级/内部"},
                {"field": "grade", "category": "属性类/类别标签", "grading": "第1级/公开"},
            ]
        }
    
    def save(self):
        """保存知识库"""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        data = {
            "rules": self.rules,
            "industry_rules": dict(self.industry_rules),
            "field_cache": self.field_cache,
            "updated_at": datetime.now().isoformat()
        }
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"知识库已保存至: {self.kb_path}")
    
    def match(self, field_name: str, industry: Optional[str] = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        匹配字段获取分类和分级
        
        Args:
            field_name: 字段名
            industry: 行业（可选）
            
        Returns:
            (category, grading, confidence)
        """
        field_lower = field_name.lower()
        
        # 1. 先检查缓存
        if field_lower in self.field_cache:
            cached = self.field_cache[field_lower]
            return cached['category'], cached['grading'], cached['confidence']
        
        # 2. 行业特定规则优先
        if industry and industry in self.industry_rules:
            for rule in self.industry_rules[industry]:
                if rule['field'].lower() in field_lower:
                    return rule['category'], rule['grading'], 1.0
        
        # 3. 通用规则匹配
        best_match = None
        best_weight = 0
        
        for rule in self.rules:
            for pattern in rule['patterns']:
                if re.search(pattern, field_lower, re.IGNORECASE):
                    if rule['weight'] > best_weight:
                        best_weight = rule['weight']
                        best_match = rule
                    break
        
        if best_match:
            # 更新缓存
            self.field_cache[field_lower] = {
                'category': best_match['category'],
                'grading': best_match['grading'],
                'confidence': best_weight
            }
            return best_match['category'], best_match['grading'], best_weight
        
        return None, None, 0.0
    
    def add_rule(self, field_name: str, category: str, grading: str, 
                  description: str = "", industry: Optional[str] = None):
        """
        添加新规则
        
        Args:
            field_name: 字段名
            category: 分类标签
            grading: 分级标签
            description: 规则描述
            industry: 行业（可选）
        """
        if industry:
            # 添加行业规则
            rule = {
                "field": field_name,
                "category": category,
                "grading": grading,
                "description": description
            }
            self.industry_rules[industry].append(rule)
            print(f"已添加行业规则: {industry} -> {field_name}")
        else:
            # 添加通用规则
            rule = {
                "id": f"rule_{len(self.rules) + 1:03d}",
                "patterns": [field_name.lower(), field_name],
                "category": category,
                "grading": grading,
                "weight": 0.95,
                "description": description
            }
            self.rules.append(rule)
            print(f"已添加通用规则: {field_name}")
        
        # 更新缓存
        self.field_cache[field_name.lower()] = {
            'category': category,
            'grading': grading,
            'confidence': 1.0
        }
    
    def learn_from_prediction(self, field_name: str, category: str, grading: str):
        """
        从预测结果中学习（增量更新）
        
        Args:
            field_name: 字段名
            category: 预测的分类
            grading: 预测的分级
        """
        # 检查是否已存在
        field_lower = field_name.lower()
        
        # 更新缓存
        self.field_cache[field_lower] = {
            'category': category,
            'grading': grading,
            'confidence': 0.95,
            'learned_from': 'prediction',
            'learned_at': datetime.now().isoformat()
        }
        
        print(f"已从预测结果学习: {field_name} -> {category}")
    
    def merge_similar_rules(self):
        """合并相似规则，减少冲突"""
        merged_count = 0
        
        # 简化实现：合并相同分类的规则
        for rule in self.rules:
            if 'patterns' in rule and len(rule['patterns']) > 1:
                # 检查是否有其他规则的patterns与当前规则相同分类
                for other_rule in self.rules:
                    if rule['id'] != other_rule['id'] and rule['category'] == other_rule['category']:
                        # 合并patterns
                        combined = list(set(rule['patterns'] + other_rule['patterns']))
                        if len(combined) > len(rule['patterns']):
                            rule['patterns'] = combined
                            merged_count += 1
        
        if merged_count > 0:
            print(f"已合并 {merged_count} 个相似规则")
    
    def detect_conflicts(self) -> List[Dict]:
        """
        检测规则冲突
        
        Returns:
            冲突列表
        """
        conflicts = []
        
        # 检查同一字段名在不同规则中的冲突
        field_rules = defaultdict(list)
        for rule in self.rules:
            for pattern in rule['patterns']:
                field_rules[pattern].append({
                    'category': rule['category'],
                    'grading': rule['grading'],
                    'rule_id': rule['id']
                })
        
        for field, rules in field_rules.items():
            if len(rules) > 1:
                categories = set(r['category'] for r in rules)
                gradings = set(r['grading'] for r in rules)
                if len(categories) > 1 or len(gradings) > 1:
                    conflicts.append({
                        'field': field,
                        'rules': rules,
                        'type': 'category_conflict' if len(categories) > 1 else 'grading_conflict'
                    })
        
        return conflicts
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_rules": len(self.rules),
            "total_industry_rules": sum(len(r) for r in self.industry_rules.values()),
            "cached_fields": len(self.field_cache),
            "industries": list(self.industry_rules.keys()),
            "categories": list(set(r['category'] for r in self.rules))
        }


def demo():
    """演示动态知识库"""
    kb = DynamicKnowledgeBase()
    
    print("=" * 60)
    print("动态知识库演示")
    print("=" * 60)
    
    # 测试匹配
    test_fields = [
        ("customer_id", "金融"),
        ("price", "商业"),
        ("age", "医疗"),
        ("created_at", None),
        ("my_custom_field", None),
    ]
    
    print("\n字段匹配测试:")
    print("-" * 60)
    for field, industry in test_fields:
        category, grading, confidence = kb.match(field, industry)
        print(f"{field:20} -> {category} | {grading} | 置信度: {confidence:.2f}")
    
    # 添加新规则
    print("\n添加新规则:")
    kb.add_rule("customer_level", "属性类/类别标签", "第2级/内部", "客户等级", "金融")
    
    # 测试新规则
    print("\n测试新规则:")
    category, grading, confidence = kb.match("customer_level", "金融")
    print(f"customer_level -> {category} | {grading} | 置信度: {confidence:.2f}")
    
    # 获取统计
    print("\n知识库统计:")
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 检测冲突
    print("\n规则冲突检测:")
    conflicts = kb.detect_conflicts()
    if conflicts:
        for conflict in conflicts[:3]:
            print(f"  冲突: {conflict}")
    else:
        print("  未检测到冲突")


def create_knowledge_base_from_predictions():
    """从预测结果创建知识库"""
    import glob
    
    kb = DynamicKnowledgeBase()
    
    # 读取预测结果
    pred_files = glob.glob(os.path.join(BASE_DIR, "results/*.json"))
    
    new_rules_added = 0
    for pred_file in pred_files:
        if "evaluation" in pred_file:
            continue
        
        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for filename, file_data in data.items():
                industry = file_data.get('行业', '通用')
                for field_info in file_data.get('字段', []):
                    field_name = field_info.get('字段', '')
                    category = field_info.get('分类', '')
                    grading = field_info.get('分级', '')
                    
                    if field_name and category and grading:
                        # 学习预测结果
                        kb.learn_from_prediction(field_name, category, grading)
                        new_rules_added += 1
        except Exception as e:
            print(f"处理文件失败: {pred_file}, 错误: {e}")
    
    # 保存更新后的知识库
    kb.save()
    print(f"\n已从预测结果学习 {new_rules_added} 条规则")
    
    return kb


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--learn":
            # 从预测结果学习
            create_knowledge_base_from_predictions()
        elif sys.argv[1] == "--save":
            # 保存当前知识库
            kb = DynamicKnowledgeBase()
            kb.save()
        else:
            print("用法: python dynamic_knowledge_base.py [--learn|--save]")
    else:
        demo()
