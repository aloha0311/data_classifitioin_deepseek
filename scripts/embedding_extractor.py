#!/usr/bin/env python3
"""
字段语义建模模块
对字段进行多维度语义特征提取与表示，支持结构特征、语义特征和数据特征的联合建模
"""
import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import math

# 项目路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


@dataclass
class FieldStatistics:
    """字段统计特征"""
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    avg_length: float = 0.0
    max_length: int = 0
    min_length: int = 0
    std_length: float = 0.0


@dataclass
class DataPattern:
    """数据模式"""
    pattern_type: str  # numeric, text, date, boolean, mixed
    pattern_regex: str
    match_ratio: float
    examples: List[str] = field(default_factory=list)


@dataclass
class FieldSemanticFeatures:
    """字段语义特征"""
    # 基础信息
    field_name: str
    normalized_name: str
    
    # 结构特征
    structure_features: Dict[str, float] = field(default_factory=dict)
    
    # 语义特征
    semantic_features: Dict[str, float] = field(default_factory=dict)
    
    # 数据特征
    data_features: Dict[str, Any] = field(default_factory=dict)
    
    # 综合表示向量
    embedding_vector: np.ndarray = None
    
    # 分类推断
    inferred_category: str = ""
    inferred_confidence: float = 0.0


class FieldSemanticModeler:
    """
    字段语义建模器
    
    实现多模态语义建模，提取三类特征：
    1. 结构特征：字段命名规范、后缀模式
    2. 语义特征：基于关键词的语义推断
    3. 数据特征：样本值分布、格式、类型
    
    最终融合为联合特征表示向量。
    """
    
    # 字段分类关键词映射
    CATEGORY_KEYWORDS = {
        'ID类/主键ID': ['id', 'uuid', 'guid', 'key', 'pk', 'fk', '主键', '标识'],
        '结构类/分类代码': ['code', 'no', 'num', '分类', '类型', 'code', '编码'],
        '属性类/名称标题': ['name', 'title', 'label', '标题', '名称', '姓名'],
        '属性类/描述文本': ['desc', 'description', 'remark', 'note', '描述', '说明', '备注'],
        '属性类/地址位置': ['address', 'location', 'addr', '位置', '地址', 'city', 'province'],
        '属性类/技能标签': ['skill', 'ability', 'skill', '专长', '技能'],
        '度量类/计量数值': ['amount', 'price', 'cost', 'fee', 'salary', '金额', '价格', '工资'],
        '度量类/计数统计': ['count', 'num', 'total', 'quantity', '数量', '次数', '人数'],
        '度量类/比率比例': ['rate', 'ratio', 'percent', '率', '比', '占比'],
        '度量类/时间度量': ['date', 'time', 'duration', '时间', '时长', '间隔'],
        '度量类/序号排序': ['seq', 'index', 'rank', 'order', '序号', '排名', '排序'],
        '身份类/人口统计': ['age', 'gender', 'sex', 'birth', '年龄', '性别', '生日'],
        '身份类/联系方式': ['phone', 'mobile', 'email', 'tel', '电话', '手机', '邮箱'],
        '身份类/教育背景': ['education', 'school', 'degree', '学历', '学校', '学位'],
        '身份类/职业信息': ['job', 'position', 'title', '职业', '职位', '岗位'],
        '状态类/二元标志': ['flag', 'is_', 'has_', '是否', '状态', '标识'],
        '状态类/状态枚举': ['status', 'state', 'stage', '状态', '阶段', '结果'],
        '状态类/时间标记': ['created', 'updated', 'at', '时间', '日期', '创建', '更新'],
        '扩展类/扩展代码': ['ext', 'extra', 'custom', '扩展', '自定义'],
        '扩展类/其他字段': ['other', 'misc', 'reserved', '其他', '备用'],
    }
    
    # 后缀模式及其权重
    SUFFIX_PATTERNS = {
        'id': ('ID类/主键ID', 0.95),
        '_id': ('ID类/主键ID', 0.90),
        'code': ('结构类/分类代码', 0.85),
        '_no': ('结构类/分类代码', 0.80),
        'name': ('属性类/名称标题', 0.90),
        '_name': ('属性类/名称标题', 0.90),
        'title': ('属性类/名称标题', 0.85),
        'desc': ('属性类/描述文本', 0.85),
        'description': ('属性类/描述文本', 0.90),
        'address': ('属性类/地址位置', 0.95),
        'amount': ('度量类/计量数值', 0.90),
        'price': ('度量类/计量数值', 0.90),
        'cost': ('度量类/计量数值', 0.85),
        'fee': ('度量类/计量数值', 0.85),
        'count': ('度量类/计数统计', 0.90),
        'num': ('度量类/计数统计', 0.85),
        'total': ('度量类/计数统计', 0.85),
        'rate': ('度量类/比率比例', 0.90),
        'ratio': ('度量类/比率比例', 0.90),
        'date': ('度量类/时间度量', 0.95),
        'time': ('度量类/时间度量', 0.90),
        'created': ('状态类/时间标记', 0.95),
        'updated': ('状态类/时间标记', 0.95),
        'phone': ('身份类/联系方式', 0.95),
        'email': ('身份类/联系方式', 0.95),
        'age': ('身份类/人口统计', 0.90),
        'gender': ('身份类/人口统计', 0.95),
        'status': ('状态类/状态枚举', 0.90),
        'type': ('状态类/状态枚举', 0.85),
        'flag': ('状态类/二元标志', 0.90),
    }
    
    # 数值特征关键词
    NUMERIC_KEYWORDS = ['amount', 'price', 'cost', 'fee', 'count', 'num', 'total',
                        'rate', 'ratio', 'age', 'salary', 'balance', 'score']
    
    # 日期特征关键词
    DATE_KEYWORDS = ['date', 'time', 'created', 'updated', 'at', 'birth', 'expire']
    
    # 敏感字段关键词
    SENSITIVE_KEYWORDS = ['password', 'secret', 'key', 'token', 'auth', 'credential',
                          'ssn', 'id_card', 'bank', 'card', 'credit']
    
    def __init__(self, embedding_dim: int = 128):
        """
        初始化语义建模器
        
        Args:
            embedding_dim: 嵌入向量维度
        """
        self.embedding_dim = embedding_dim
        self._build_feature_weights()
    
    def _build_feature_weights(self):
        """构建特征权重表"""
        # 结构特征权重
        self.structure_weights = {
            'has_underscore': 0.5,
            'has_number': 0.3,
            'has_prefix': 0.4,
            'suffix_match': 1.0,
            'length_norm': 0.2,
        }
        
        # 语义特征权重
        self.semantic_weights = {
            'keyword_match': 1.0,
            'category_confidence': 1.5,
            'sensitive_match': 2.0,
        }
    
    def normalize_field_name(self, field_name: str) -> str:
        """标准化字段名"""
        # 转小写
        normalized = field_name.lower()
        # 替换分隔符为空格
        for sep in ['_', '-', '.', '/', '\\']:
            normalized = normalized.replace(sep, ' ')
        # 移除多余空格
        normalized = ' '.join(normalized.split())
        return normalized
    
    def extract_structure_features(self, field_name: str) -> Dict[str, float]:
        """
        提取结构特征
        
        包括：
        - 下划线数量（snake_case命名）
        - 驼峰拆分后的词数
        - 后缀匹配得分
        - 字段长度
        """
        features = {}
        normalized = self.normalize_field_name(field_name)
        
        # 基础特征
        features['has_underscore'] = 1.0 if '_' in field_name else 0.0
        features['has_number'] = 1.0 if any(c.isdigit() for c in field_name) else 0.0
        features['word_count'] = len(normalized.split())
        features['length'] = len(field_name) / 50.0  # 归一化
        features['is_snake_case'] = 1.0 if '_' in normalized else 0.0
        features['is_camel_case'] = 1.0 if (any(c.isupper() for c in field_name[1:]) and '_' not in field_name) else 0.0
        
        # 后缀匹配
        suffix_score = 0.0
        matched_suffix = None
        for suffix, (category, score) in self.SUFFIX_PATTERNS.items():
            if field_name.lower().endswith(suffix):
                if matched_suffix is None or len(suffix) > len(matched_suffix):
                    suffix_score = score
                    matched_suffix = suffix
        
        features['suffix_match'] = suffix_score
        features['suffix_category'] = self.SUFFIX_PATTERNS.get(matched_suffix, (None, 0.0))[0] if matched_suffix else None
        
        return features
    
    def extract_semantic_features(self, field_name: str) -> Dict[str, float]:
        """
        提取语义特征
        
        基于关键词匹配推断字段语义类别
        """
        features = {}
        normalized = self.normalize_field_name(field_name)
        words = set(normalized.split())
        
        # 计算每个类别的匹配得分
        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0.0
            matched = 0
            for kw in keywords:
                if kw in normalized:
                    score += 1.0
                    matched += 1
            if matched > 0:
                # 考虑匹配比例
                category_scores[category] = score / len(keywords)
        
        features['category_scores'] = category_scores
        
        # 推断最可能的类别
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            features['inferred_category'] = best_category[0]
            features['category_confidence'] = best_category[1]
        else:
            features['inferred_category'] = '扩展类/其他字段'
            features['category_confidence'] = 0.0
        
        # 敏感字段检测
        is_sensitive = any(kw in normalized for kw in self.SENSITIVE_KEYWORDS)
        features['is_sensitive'] = 1.0 if is_sensitive else 0.0
        
        # 数值字段检测
        is_numeric = any(kw in normalized for kw in self.NUMERIC_KEYWORDS)
        features['is_numeric'] = 1.0 if is_numeric else 0.0
        
        # 日期字段检测
        is_date = any(kw in normalized for kw in self.DATE_KEYWORDS)
        features['is_date'] = 1.0 if is_date else 0.0
        
        return features
    
    def analyze_data_pattern(self, samples: List[str]) -> DataPattern:
        """
        分析样本数据模式
        
        识别字段值的数据类型：数值、文本、日期、布尔值等
        """
        if not samples:
            return DataPattern(
                pattern_type='unknown',
                pattern_regex='',
                match_ratio=0.0
            )
        
        samples = [str(s).strip() for s in samples if s]
        if not samples:
            return DataPattern(
                pattern_type='unknown',
                pattern_regex='',
                match_ratio=0.0
            )
        
        # 模式定义
        patterns = {
            'integer': (r'^-?\d+$', '整数'),
            'float': (r'^-?\d+\.?\d*$', '浮点数'),
            'email': (r'.+@.+\..+', '邮箱'),
            'phone': (r'^1[3-9]\d{9}$', '手机号'),
            'id_card': (r'^\d{15}|\d{18}$', '身份证'),
            'date_iso': (r'^\d{4}-\d{2}-\d{2}', '日期(ISO)'),
            'date_cn': (r'^\d{4}年\d{1,2}月\d{1,2}日', '日期(中文)'),
            'datetime': (r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', '日期时间'),
            'boolean': (r'^(true|false|是|否|0|1)$', '布尔值'),
            'url': (r'^https?://', 'URL'),
            'ipv4': (r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', 'IP地址'),
        }
        
        # 统计各模式匹配数量
        pattern_matches = Counter()
        for sample in samples:
            sample_lower = sample.lower()
            for ptype, (regex, _) in patterns.items():
                if re.match(regex, sample, re.IGNORECASE):
                    pattern_matches[ptype] += 1
        
        # 确定主模式
        if pattern_matches:
            main_pattern = pattern_matches.most_common(1)[0]
            pattern_type = main_pattern[0]
            match_ratio = main_pattern[1] / len(samples)
        else:
            # 默认为文本
            pattern_type = 'text'
            match_ratio = 1.0
        
        # 数值类型细分
        if pattern_type in ['integer', 'float']:
            # 检查是否为百分比
            if any('%' in s for s in samples):
                pattern_type = 'percentage'
            # 检查是否为金额
            elif any(any(c in s for c in ['¥', '$', '€', '元', '美元']) for s in samples):
                pattern_type = 'currency'
            # 归一化为已知类型
            else:
                pattern_type = 'float'
        
        # 确保 pattern_type 在预定义列表中
        known_types = ['integer', 'float', 'text', 'email', 'phone', 'id_card',
                      'date_iso', 'date_cn', 'datetime', 'boolean', 'url', 'ipv4', 'percentage', 'currency']
        if pattern_type not in known_types:
            pattern_type = 'text'
        
        return DataPattern(
            pattern_type=pattern_type,
            pattern_regex=patterns.get(pattern_type, ('', ''))[0],
            match_ratio=match_ratio,
            examples=samples[:5]
        )
    
    def compute_statistics(self, samples: List[str]) -> FieldStatistics:
        """
        计算字段统计特征
        """
        stats = FieldStatistics()
        
        if not samples:
            return stats
        
        # 过滤空值
        valid_samples = [str(s) for s in samples if s is not None and str(s).strip()]
        total = len(samples)
        
        if not valid_samples:
            return stats
        
        # 空值统计
        stats.null_count = total - len(valid_samples)
        stats.null_ratio = stats.null_count / total if total > 0 else 0.0
        
        # 唯一值统计
        stats.unique_count = len(set(valid_samples))
        stats.unique_ratio = stats.unique_count / len(valid_samples)
        
        # 长度统计
        lengths = [len(s) for s in valid_samples]
        stats.avg_length = np.mean(lengths) if lengths else 0.0
        stats.max_length = max(lengths) if lengths else 0
        stats.min_length = min(lengths) if lengths else 0
        stats.std_length = np.std(lengths) if len(lengths) > 1 else 0.0
        
        return stats
    
    def build_embedding_vector(self, 
                              structure_features: Dict[str, float],
                              semantic_features: Dict[str, float],
                              data_features: Dict[str, Any]) -> np.ndarray:
        """
        构建联合特征嵌入向量
        
        融合结构特征、语义特征和数据特征为固定维度的向量表示
        """
        embedding = np.zeros(self.embedding_dim)
        
        # 结构特征编码 (dim 0-31)
        struct_idx = 0
        for key, value in structure_features.items():
            if struct_idx >= 32:
                break
            # 跳过非数值类型的特征
            if isinstance(value, (int, float)):
                embedding[struct_idx] = float(value)
                struct_idx += 1
            elif key == 'word_count':
                embedding[struct_idx] = float(value)
                struct_idx += 1
        
        # 语义特征编码 (dim 32-95)
        # 类别one-hot编码
        category_scores = semantic_features.get('category_scores', {})
        for i, (cat, score) in enumerate(sorted(category_scores.items())):
            if i < 24:  # 24个类别
                embedding[32 + i] = float(score)
        
        # 其他语义特征
        embedding[56] = float(semantic_features.get('is_sensitive', 0))
        embedding[57] = float(semantic_features.get('is_numeric', 0))
        embedding[58] = float(semantic_features.get('is_date', 0))
        embedding[59] = float(semantic_features.get('category_confidence', 0))
        
        # 数据特征编码 (dim 60-95)
        if 'pattern_type' in data_features:
            pattern_types = ['unknown', 'integer', 'float', 'text', 'email', 
                          'phone', 'id_card', 'date_iso', 'date_cn', 'datetime',
                          'boolean', 'url', 'ipv4', 'percentage', 'currency']
            ptype = data_features.get('pattern_type', 'unknown')
            pattern_idx = pattern_types.index(ptype) if ptype in pattern_types else 0
            if pattern_idx < len(pattern_types):
                embedding[60 + pattern_idx % 20] = 1.0
        
        # 统计特征
        stats = data_features.get('stats')
        if stats:
            embedding[80] = float(stats.unique_ratio)
            embedding[81] = float(stats.null_ratio)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def model_field(self, field_name: str, samples: List[str] = None) -> FieldSemanticFeatures:
        """
        对字段进行完整语义建模
        
        Args:
            field_name: 字段名
            samples: 字段样本值列表
        
        Returns:
            FieldSemanticFeatures: 字段语义特征
        """
        # 标准化字段名
        normalized = self.normalize_field_name(field_name)
        
        # 提取结构特征
        structure_features = self.extract_structure_features(field_name)
        
        # 提取语义特征
        semantic_features = self.extract_semantic_features(field_name)
        
        # 数据特征
        data_features = {}
        if samples:
            data_features['pattern'] = self.analyze_data_pattern(samples)
            data_features['stats'] = self.compute_statistics(samples)
            data_features['pattern_type'] = data_features['pattern'].pattern_type
        
        # 构建嵌入向量
        embedding_vector = self.build_embedding_vector(
            structure_features, 
            semantic_features, 
            data_features
        )
        
        return FieldSemanticFeatures(
            field_name=field_name,
            normalized_name=normalized,
            structure_features=structure_features,
            semantic_features=semantic_features,
            data_features=data_features,
            embedding_vector=embedding_vector,
            inferred_category=semantic_features.get('inferred_category', '扩展类/其他字段'),
            inferred_confidence=semantic_features.get('category_confidence', 0.0)
        )
    
    def batch_model_fields(self, 
                          field_data: List[Tuple[str, List[str]]]) -> List[FieldSemanticFeatures]:
        """
        批量建模字段
        
        Args:
            field_data: [(field_name, samples), ...]
        
        Returns:
            字段语义特征列表
        """
        return [self.model_field(name, samples) for name, samples in field_data]
    
    def compute_field_distance(self, 
                              features1: FieldSemanticFeatures,
                              features2: FieldSemanticFeatures) -> float:
        """
        计算两个字段的距离（基于嵌入向量）
        """
        if features1.embedding_vector is not None and features2.embedding_vector is not None:
            # 余弦距离
            dot = np.dot(features1.embedding_vector, features2.embedding_vector)
            return 1 - dot  # 余弦距离 = 1 - 余弦相似度
        return 1.0
    
    def find_similar_fields_by_semantic(self,
                                       target: FieldSemanticFeatures,
                                       candidates: List[FieldSemanticFeatures],
                                       top_k: int = 5) -> List[Tuple[FieldSemanticFeatures, float]]:
        """
        基于语义特征查找相似字段
        
        Args:
            target: 目标字段
            candidates: 候选字段列表
            top_k: 返回前k个
        
        Returns:
            [(字段特征, 相似度), ...]
        """
        similarities = []
        for candidate in candidates:
            if candidate.field_name == target.field_name:
                continue
            dist = self.compute_field_distance(target, candidate)
            sim = 1 - dist
            similarities.append((candidate, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def demo():
    """演示函数"""
    modeler = FieldSemanticModeler()
    
    test_fields = [
        ("customer_id", ["1001", "1002", "1003"]),
        ("user_email", ["test@example.com", "user@test.cn"]),
        ("order_amount", ["150.50", "299.00", "88.88"]),
        ("created_at", ["2024-01-15", "2024-02-20", "2024-03-10"]),
        ("phone_number", ["13812345678", "13998765432"]),
        ("customer_name", ["张三", "李四", "王五"]),
        ("is_active", ["true", "false", "true"]),
    ]
    
    print("=" * 70)
    print("字段语义建模测试")
    print("=" * 70)
    
    for field_name, samples in test_fields:
        features = modeler.model_field(field_name, samples)
        
        print(f"\n字段: {field_name}")
        print(f"  样本: {samples[:3]}")
        print(f"  标准化: {features.normalized_name}")
        print(f"  结构特征:")
        for k, v in list(features.structure_features.items())[:4]:
            print(f"    - {k}: {v}")
        print(f"  语义推断:")
        print(f"    - 类别: {features.inferred_category}")
        print(f"    - 置信度: {features.category_confidence:.2f}")
        print(f"  数据特征:")
        if features.data_features:
            pattern = features.data_features.get('pattern')
            if pattern:
                print(f"    - 模式类型: {pattern.pattern_type}")
                print(f"    - 匹配率: {pattern.match_ratio:.2f}")


if __name__ == "__main__":
    demo()
