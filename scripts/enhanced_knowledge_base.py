#!/usr/bin/env python3
"""
增强的动态知识库模块
支持向量相似度匹配、语义向量存储、增量学习和模式冲突检测
"""
import os
import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VectorStore:
    """向量存储模块（简化版，无需额外依赖）"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = {}  # field_name -> vector
        self.metadata = {}  # field_name -> metadata
    
    def add_vector(self, field_name: str, vector: List[float], metadata: Dict = None):
        """添加向量"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]  # L2归一化
        self.vectors[field_name] = vector
        self.metadata[field_name] = metadata or {}
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        搜索最相似的向量
        
        Returns:
            List of (field_name, similarity, metadata)
        """
        if not self.vectors or not query_vector:
            return []
        
        # 归一化查询向量
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query = [v / norm for v in query_vector]
        else:
            query = query_vector
        
        # 计算余弦相似度
        similarities = []
        for field_name, vec in self.vectors.items():
            sim = sum(q * v for q, v in zip(query, vec))
            similarities.append((field_name, sim, self.metadata[field_name]))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def save(self, filepath: str):
        """保存向量库"""
        data = {
            "dimension": self.dimension,
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载向量库"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.dimension = data.get("dimension", 384)
            self.vectors = data.get("vectors", {})
            self.metadata = data.get("metadata", {})


class SemanticHasher:
    """语义哈希生成器 - 将字段名转换为语义向量"""
    
    def __init__(self):
        # 语义关键词映射
        self.semantic_map = {
            # ID类
            "id": [1, 0, 0, 0, 0, 0, 0, 0],
            "主键": [1, 0, 0, 0, 0, 0, 0, 0],
            "编号": [0.9, 0, 0, 0, 0, 0, 0, 0],
            "编码": [0.8, 0, 0, 0, 0, 0, 0, 0],
            # 名称类
            "name": [0, 1, 0, 0, 0, 0, 0, 0],
            "名称": [0, 1, 0, 0, 0, 0, 0, 0],
            "标题": [0, 0.9, 0, 0, 0, 0, 0, 0],
            # 数值类
            "price": [0, 0, 1, 0, 0, 0, 0, 0],
            "金额": [0, 0, 1, 0, 0, 0, 0, 0],
            "数量": [0, 0, 0.9, 0, 0, 0, 0, 0],
            "count": [0, 0, 0.9, 0, 0, 0, 0, 0],
            "比率": [0, 0, 0, 1, 0, 0, 0, 0],
            "rate": [0, 0, 0, 1, 0, 0, 0, 0],
            # 时间类
            "date": [0, 0, 0, 0, 1, 0, 0, 0],
            "time": [0, 0, 0, 0, 1, 0, 0, 0],
            "时间": [0, 0, 0, 0, 0.9, 0, 0, 0],
            "created": [0, 0, 0, 0, 1, 0, 0, 0],
            # 状态类
            "status": [0, 0, 0, 0, 0, 1, 0, 0],
            "状态": [0, 0, 0, 0, 0, 1, 0, 0],
            "type": [0, 0, 0, 0, 0, 0.8, 0, 0],
            # 身份类
            "phone": [0, 0, 0, 0, 0, 0, 1, 0],
            "电话": [0, 0, 0, 0, 0, 0, 1, 0],
            "email": [0, 0, 0, 0, 0, 0, 1, 0],
            "age": [0, 0, 0, 0, 0, 0, 0, 1],
            "年龄": [0, 0, 0, 0, 0, 0, 0, 1],
            "教育": [0, 0, 0, 0, 0, 0, 0, 0.9],
            "职业": [0, 0, 0, 0, 0, 0, 0, 0.8],
        }
    
    def hash_field(self, field_name: str) -> List[float]:
        """将字段名转换为语义向量"""
        field_lower = field_name.lower()
        vector = [0.0] * 8
        
        for keyword, vec in self.semantic_map.items():
            if keyword in field_lower:
                for i, v in enumerate(vec):
                    vector[i] = max(vector[i], v)
        
        # 如果没有匹配，返回空向量
        if sum(vector) == 0:
            return [0.0] * 8
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector


class EnhancedKnowledgeBase:
    """增强的动态知识库"""
    
    def __init__(self, kb_path: Optional[str] = None, vector_store_path: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            kb_path: 知识库文件路径
            vector_store_path: 向量存储文件路径
        """
        self.kb_path = kb_path or os.path.join(BASE_DIR, "data/knowledge_base.json")
        self.vector_store_path = vector_store_path or os.path.join(BASE_DIR, "data/vector_store.json")
        
        self.rules = []
        self.industry_rules = defaultdict(list)
        self.field_cache = {}
        
        self.vector_store = VectorStore()
        self.semantic_hasher = SemanticHasher()
        
        self._load_knowledge_base()
        self._load_vector_store()
    
    def _load_knowledge_base(self):
        """加载知识库"""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.rules = data.get('rules', [])
                self.industry_rules = defaultdict(list, data.get('industry_rules', {}))
                self.field_cache = data.get('field_cache', {})
    
    def _load_vector_store(self):
        """加载向量存储"""
        self.vector_store.load(self.vector_store_path)
    
    def _init_default_rules(self):
        """初始化默认规则"""
        self.rules = [
            {
                "id": "rule_001",
                "patterns": [r'^id$', r'_id$', r'^no$', r'编号', r'编码'],
                "category": "ID类/主键ID",
                "grading": "第1级/公开",
                "weight": 1.0,
                "description": "唯一标识符"
            },
            {
                "id": "rule_002",
                "patterns": [r'name', r'名称', r'标题', r'title', r'username'],
                "category": "属性类/名称标题",
                "grading": "第1级/公开",
                "weight": 0.9,
                "description": "名称或标题字段"
            },
            {
                "id": "rule_003",
                "patterns": [r'price', r'金额', r'价格', r'money', r'cost', r'fee'],
                "category": "度量类/计量数值",
                "grading": "第2级/内部",
                "weight": 0.95,
                "description": "价格或金额"
            },
            {
                "id": "rule_004",
                "patterns": [r'count', r'数量', r'quantity', r'num', r'次数'],
                "category": "度量类/计数统计",
                "grading": "第2级/内部",
                "weight": 0.9,
                "description": "计数统计"
            },
            {
                "id": "rule_005",
                "patterns": [r'date', r'time', r'时间', r'日期', r'created', r'updated'],
                "category": "状态类/时间标记",
                "grading": "第2级/内部",
                "weight": 0.9,
                "description": "时间标记字段"
            },
            {
                "id": "rule_006",
                "patterns": [r'address', r'地址', r'city', r'城市', r'location', r'国家'],
                "category": "属性类/地址位置",
                "grading": "第1级/公开",
                "weight": 0.9,
                "description": "地址位置"
            },
            {
                "id": "rule_007",
                "patterns": [r'status', r'状态', r'type', r'类型', r'flag', r'标志'],
                "category": "状态类/状态枚举",
                "grading": "第2级/内部",
                "weight": 0.85,
                "description": "状态枚举"
            },
            {
                "id": "rule_008",
                "patterns": [r'age', r'年龄'],
                "category": "身份类/人口统计",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "年龄信息"
            },
            {
                "id": "rule_009",
                "patterns": [r'gender', r'性别', r'sex'],
                "category": "身份类/人口统计",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "性别信息"
            },
            {
                "id": "rule_010",
                "patterns": [r'income', r'收入', r'salary', r'工资', r'薪酬'],
                "category": "度量类/计量数值",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "收入相关"
            },
            {
                "id": "rule_011",
                "patterns": [r'phone', r'电话', r'mobile', r'手机', r'email', r'邮箱'],
                "category": "身份类/联系方式",
                "grading": "第3级/敏感",
                "weight": 1.0,
                "description": "联系方式"
            },
            {
                "id": "rule_012",
                "patterns": [r'education', r'教育', r'degree', r'学历'],
                "category": "身份类/教育背景",
                "grading": "第3级/敏感",
                "weight": 0.95,
                "description": "教育背景"
            },
        ]
        
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
        
        # 保存向量存储
        self.vector_store.save(self.vector_store_path)
        print(f"向量存储已保存至: {self.vector_store_path}")
    
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
    
    def match_with_vector(self, field_name: str, industry: Optional[str] = None) -> Dict:
        """
        使用向量相似度匹配字段
        
        Returns:
            包含多种匹配结果的字典
        """
        # 规则匹配
        category, grading, confidence = self.match(field_name, industry)
        
        # 向量相似度搜索
        query_vector = self.semantic_hasher.hash_field(field_name)
        vector_results = self.vector_store.search(query_vector, top_k=5)
        
        return {
            "field_name": field_name,
            "industry": industry,
            "rule_match": {
                "category": category,
                "grading": grading,
                "confidence": confidence
            },
            "vector_matches": [
                {
                    "field": name,
                    "similarity": sim,
                    "metadata": meta
                }
                for name, sim, meta in vector_results
            ]
        }
    
    def add_rule(self, field_name: str, category: str, grading: str, 
                  description: str = "", industry: Optional[str] = None):
        """添加新规则"""
        if industry:
            rule = {
                "field": field_name,
                "category": category,
                "grading": grading,
                "description": description
            }
            self.industry_rules[industry].append(rule)
            print(f"已添加行业规则: {industry} -> {field_name}")
        else:
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
        
        # 添加到向量存储
        vector = self.semantic_hasher.hash_field(field_name)
        if sum(vector) > 0:
            self.vector_store.add_vector(
                field_name,
                vector,
                {"category": category, "grading": grading}
            )
    
    def learn_from_prediction(self, field_name: str, category: str, grading: str, 
                           correct: bool = True):
        """从预测结果中学习"""
        field_lower = field_name.lower()
        
        # 更新缓存
        self.field_cache[field_lower] = {
            'category': category,
            'grading': grading,
            'confidence': 0.95 if correct else 0.5,
            'learned_from': 'prediction',
            'learned_at': datetime.now().isoformat()
        }
        
        # 添加到向量存储
        vector = self.semantic_hasher.hash_field(field_name)
        if sum(vector) > 0:
            self.vector_store.add_vector(
                field_lower,
                vector,
                {"category": category, "grading": grading, "learned": True}
            )
    
    def detect_conflicts(self) -> List[Dict]:
        """检测规则冲突"""
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
    
    def detect_new_field_conflicts(self, field_name: str, category: str, grading: str) -> List[Dict]:
        """检测新字段与现有模式的冲突"""
        # 使用向量相似度检测
        query_vector = self.semantic_hasher.hash_field(field_name)
        similar_fields = self.vector_store.search(query_vector, top_k=3)
        
        conflicts = []
        for similar_name, similarity, metadata in similar_fields:
            if similarity > 0.7:  # 相似度阈值
                similar_category = metadata.get('category', '')
                similar_grading = metadata.get('grading', '')
                
                if similar_category != category:
                    conflicts.append({
                        'field': field_name,
                        'similar_to': similar_name,
                        'similarity': similarity,
                        'expected': category,
                        'actual': similar_category,
                        'reason': 'category_conflict'
                    })
                
                if similar_grading != grading:
                    conflicts.append({
                        'field': field_name,
                        'similar_to': similar_name,
                        'similarity': similarity,
                        'expected_grading': grading,
                        'actual_grading': similar_grading,
                        'reason': 'grading_conflict'
                    })
        
        return conflicts
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_rules": len(self.rules),
            "total_industry_rules": sum(len(r) for r in self.industry_rules.values()),
            "cached_fields": len(self.field_cache),
            "vector_store_size": len(self.vector_store.vectors),
            "industries": list(self.industry_rules.keys()),
            "categories": list(set(r['category'] for r in self.rules))
        }


def demo():
    """演示增强知识库"""
    kb = EnhancedKnowledgeBase()
    
    print("=" * 60)
    print("增强动态知识库演示")
    print("=" * 60)
    
    # 测试向量匹配
    test_fields = [
        "customer_id",
        "price",
        "user_age",
        "phone_number",
        "email_address",
        "create_time",
        "status",
        "education_level"
    ]
    
    print("\n【向量相似度匹配测试】")
    print("-" * 60)
    for field in test_fields:
        result = kb.match_with_vector(field, "金融")
        print(f"\n字段: {field}")
        print(f"  规则匹配: {result['rule_match']['category']} / {result['rule_match']['grading']}")
        if result['vector_matches']:
            print(f"  最相似字段: {result['vector_matches'][0]['field']} (相似度: {result['vector_matches'][0]['similarity']:.3f})")
    
    # 添加新规则
    print("\n\n【添加新规则测试】")
    print("-" * 60)
    kb.add_rule("customer_level", "属性类/类别标签", "第2级/内部", "客户等级", "金融")
    
    # 测试冲突检测
    print("\n\n【冲突检测测试】")
    print("-" * 60)
    conflicts = kb.detect_new_field_conflicts("customer_rank", "度量类/计量数值", "第1级/公开")
    if conflicts:
        print("检测到潜在冲突:")
        for c in conflicts:
            print(f"  {c}")
    else:
        print("未检测到冲突")
    
    # 获取统计
    print("\n\n【知识库统计】")
    print("-" * 60)
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            kb = EnhancedKnowledgeBase()
            kb.save()
        else:
            print("用法: python enhanced_knowledge_base.py [--save]")
    else:
        demo()
