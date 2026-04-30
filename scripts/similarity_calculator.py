#!/usr/bin/env python3
"""
向量相似度计算模块
用于计算字段名之间的语义相似度，支持知识库匹配和冲突检测
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 项目路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EMBEDDINGS_DIR = os.path.join(PROJECT_DIR, "data", "embeddings")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的编辑距离 (Levenshtein Distance)
    使用动态规划实现
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


@dataclass
class SimilarityResult:
    """相似度计算结果"""
    field1: str
    field2: str
    char_similarity: float  # 字符级TF-IDF相似度
    semantic_similarity: float  # 语义相似度（基于编辑距离）
    combined_score: float  # 综合得分
    match_level: str  # high/medium/low/none


class FieldSimilarityCalculator:
    """
    字段相似度计算器
    
    使用多种相似度算法融合：
    1. TF-IDF字符级n-gram相似度（捕捉命名模式）
    2. 编辑距离归一化相似度（捕捉编辑差异）
    3. 关键词匹配度（捕捉语义关键词）
    """
    
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (2, 4),
                 use_cache: bool = True):
        """
        初始化相似度计算器
        
        Args:
            ngram_range: 字符n-gram范围
            use_cache: 是否使用缓存
        """
        self.ngram_range = ngram_range
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}
        
        # TF-IDF向量化器（字符级）
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # 字符级，考虑词边界
            ngram_range=ngram_range,
            lowercase=True,
            max_features=5000
        )
        
        # 编辑距离计算器
        self.levenshtein = levenshtein_distance
        
        # 关键词权重表（用于领域特定匹配）
        self.keyword_weights = {
            # ID类
            'id': 1.0, 'uuid': 1.0, 'guid': 1.0, 'key': 0.9, 'pk': 1.0, 'fk': 1.0,
            # 名称类
            'name': 0.9, 'title': 0.8, 'label': 0.7, 'desc': 0.8, 'description': 0.8,
            # 联系方式类
            'phone': 1.0, 'mobile': 1.0, 'email': 1.0, 'tel': 0.9, 'address': 0.8,
            # 金额类
            'amount': 0.9, 'price': 0.9, 'cost': 0.8, 'fee': 0.8, 'salary': 1.0, 'balance': 0.9,
            # 标识类
            'code': 0.8, 'no': 0.7, 'num': 0.7, 'status': 0.8, 'type': 0.7,
            # 时间类
            'date': 0.9, 'time': 0.9, 'created': 0.8, 'updated': 0.8, 'at': 0.6,
            # 统计类
            'count': 0.8, 'total': 0.8, 'sum': 0.8, 'num': 0.7, 'quantity': 0.8,
            # 身份类
            'age': 0.9, 'gender': 1.0, 'sex': 1.0, 'birth': 0.9, 'job': 0.8, 'company': 0.8,
        }
        
        # 创建停用词列表
        self.stopwords = {'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'and', 'or', 'is', 'are'}
    
    def preprocess_field(self, field: str) -> str:
        """预处理字段名"""
        # 转小写
        field = field.lower()
        # 替换常见分隔符为空格
        for sep in ['_', '-', '.', '/']:
            field = field.replace(sep, ' ')
        # 移除多余空格
        field = ' '.join(field.split())
        return field
    
    def compute_char_tfidf_similarity(self, field1: str, field2: str) -> float:
        """
        计算字符级TF-IDF相似度
        
        通过n-gram特征捕捉字段命名模式，如：
        - user_id 与 customer_id 高度相似
        - created_at 与 updated_at 高度相似
        """
        try:
            # 预处理
            f1 = self.preprocess_field(field1)
            f2 = self.preprocess_field(field2)
            
            # 检查缓存
            cache_key = f"{f1}|{f2}"
            if self.use_cache and cache_key in self._cache:
                return self._cache[cache_key]
            
            # 拟向量
            vectors = self.vectorizer.fit_transform([f1, f2])
            
            # 计算余弦相似度
            sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # 缓存结果
            if self.use_cache:
                self._cache[cache_key] = float(sim)
            
            return float(sim)
        except Exception:
            return 0.0
    
    def compute_edit_similarity(self, field1: str, field2: str) -> float:
        """
        计算编辑距离归一化相似度
        
        基于Levenshtein距离计算两个字符串的相似程度
        """
        try:
            f1 = self.preprocess_field(field1)
            f2 = self.preprocess_field(field2)
            
            # 计算编辑距离
            distance = self.levenshtein.distance(f1, f2)
            
            # 归一化到[0, 1]范围
            max_len = max(len(f1), len(f2))
            if max_len == 0:
                return 1.0
            
            similarity = 1 - (distance / max_len)
            return float(similarity)
        except Exception:
            return 0.0
    
    def compute_keyword_similarity(self, field1: str, field2: str) -> float:
        """
        计算关键词匹配相似度
        
        提取两个字段中的关键词及其权重，计算加权匹配度
        """
        try:
            f1 = self.preprocess_field(field1)
            f2 = self.preprocess_field(field2)
            
            # 提取关键词
            words1 = set(f1.split()) - self.stopwords
            words2 = set(f2.split()) - self.stopwords
            
            if not words1 or not words2:
                return 0.0
            
            # 计算权重匹配
            weight_sum = 0.0
            total_weight = 0.0
            
            for word in words1 | words2:
                weight = self.keyword_weights.get(word, 0.5)
                total_weight += weight
                if word in words1 and word in words2:
                    weight_sum += weight * 2  # 两边都有的词权重翻倍
                elif word in words1:
                    weight_sum += weight
                elif word in words2:
                    weight_sum += weight
            
            # 归一化
            if total_weight == 0:
                return 0.0
            
            return float(weight_sum / total_weight)
        except Exception:
            return 0.0
    
    def compute_suffix_similarity(self, field1: str, field2: str) -> float:
        """
        计算后缀相似度
        
        字段名后缀通常表示相同的数据类型：
        - xxx_id, xxx_name, xxx_date
        """
        try:
            f1 = self.preprocess_field(field1)
            f2 = self.preprocess_field(field2)
            
            # 提取后缀（最后两个词）
            words1 = f1.split()
            words2 = f2.split()
            
            suffix1 = ' '.join(words1[-2:]) if len(words1) >= 2 else f1
            suffix2 = ' '.join(words2[-2:]) if len(words2) >= 2 else f2
            
            # 计算后缀相似度
            if suffix1 == suffix2:
                return 1.0
            
            # 使用编辑距离
            distance = self.levenshtein.distance(suffix1, suffix2)
            max_len = max(len(suffix1), len(suffix2))
            
            return 1 - (distance / max_len) if max_len > 0 else 0.0
        except Exception:
            return 0.0
    
    def compute_similarity(self, field1: str, field2: str, 
                         weights: Dict[str, float] = None) -> SimilarityResult:
        """
        计算综合相似度
        
        Args:
            field1: 字段名1
            field2: 字段名2
            weights: 各相似度权重，默认 {'tfidf': 0.3, 'edit': 0.2, 'keyword': 0.3, 'suffix': 0.2}
        
        Returns:
            SimilarityResult: 相似度计算结果
        """
        if weights is None:
            weights = {
                'tfidf': 0.3,
                'edit': 0.2,
                'keyword': 0.3,
                'suffix': 0.2
            }
        
        # 计算各项相似度
        char_sim = self.compute_char_tfidf_similarity(field1, field2)
        edit_sim = self.compute_edit_similarity(field1, field2)
        keyword_sim = self.compute_keyword_similarity(field1, field2)
        suffix_sim = self.compute_suffix_similarity(field1, field2)
        
        # 综合得分
        combined_score = (
            char_sim * weights.get('tfidf', 0.3) +
            edit_sim * weights.get('edit', 0.2) +
            keyword_sim * weights.get('keyword', 0.3) +
            suffix_sim * weights.get('suffix', 0.2)
        )
        
        # 确定匹配级别
        if combined_score >= 0.8:
            match_level = 'high'
        elif combined_score >= 0.6:
            match_level = 'medium'
        elif combined_score >= 0.4:
            match_level = 'low'
        else:
            match_level = 'none'
        
        return SimilarityResult(
            field1=field1,
            field2=field2,
            char_similarity=round(char_sim, 4),
            semantic_similarity=round(keyword_sim, 4),
            combined_score=round(combined_score, 4),
            match_level=match_level
        )
    
    def find_similar_fields(self, 
                           target_field: str, 
                           candidate_fields: List[str],
                           top_k: int = 5,
                           threshold: float = 0.3) -> List[SimilarityResult]:
        """
        在候选字段中查找与目标字段最相似的字段
        
        Args:
            target_field: 目标字段
            candidate_fields: 候选字段列表
            top_k: 返回前k个最相似的字段
            threshold: 相似度阈值，低于此值的将被过滤
        
        Returns:
            按相似度降序排列的结果列表
        """
        results = []
        
        for field in candidate_fields:
            if field == target_field:
                continue
            
            result = self.compute_similarity(target_field, field)
            if result.combined_score >= threshold:
                results.append(result)
        
        # 按综合得分降序排序
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_k]
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def save_embeddings(self, filepath: str):
        """
        保存嵌入向量到文件
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'cache': {k: v.tolist() for k, v in self._cache.items()},
            'config': {
                'ngram_range': self.ngram_range
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_embeddings(self, filepath: str):
        """
        从文件加载嵌入向量
        
        Args:
            filepath: 加载路径
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._cache = {k: np.array(v) for k, v in data.get('cache', {}).items()}


def batch_compute_similarities(field_pairs: List[Tuple[str, str]]) -> List[SimilarityResult]:
    """
    批量计算字段对相似度
    
    Args:
        field_pairs: [(field1, field2), ...]
    
    Returns:
        相似度结果列表
    """
    calculator = FieldSimilarityCalculator()
    return [calculator.compute_similarity(f1, f2) for f1, f2 in field_pairs]


def find_similar_in_knowledge_base(target_field: str, 
                                   knowledge_base: List[dict],
                                   threshold: float = 0.5) -> List[dict]:
    """
    在知识库中查找相似字段
    
    Args:
        target_field: 目标字段名
        knowledge_base: 知识库规则列表
        threshold: 相似度阈值
    
    Returns:
        匹配的规则列表，按相似度降序
    """
    calculator = FieldSimilarityCalculator()
    
    # 提取知识库中的所有字段名
    kb_fields = []
    for rule in knowledge_base:
        patterns = rule.get('patterns', [])
        kb_fields.extend(patterns)
    
    # 计算相似度
    similar = calculator.find_similar_fields(
        target_field, 
        kb_fields, 
        top_k=10,
        threshold=threshold
    )
    
    # 构建结果
    matched_rules = []
    for result in similar:
        for rule in knowledge_base:
            if result.field2 in rule.get('patterns', []):
                matched_rules.append({
                    'rule': rule,
                    'similarity': result.combined_score,
                    'match_level': result.match_level
                })
                break
    
    return matched_rules


if __name__ == "__main__":
    # 测试代码
    calculator = FieldSimilarityCalculator()
    
    test_pairs = [
        ("customer_id", "user_id"),
        ("customer_id", "order_id"),
        ("customer_name", "customer_address"),
        ("created_at", "updated_at"),
        ("phone_number", "mobile_phone"),
        ("salary", "bonus"),
        ("user_name", "username"),
        ("password", "passwd"),
        ("email_address", "email"),
        ("balance_amount", "transaction_amount"),
    ]
    
    print("=" * 70)
    print("字段相似度计算测试")
    print("=" * 70)
    print(f"{'字段1':<25} {'字段2':<25} {'TF-IDF':<8} {'编辑':<8} {'关键词':<8} {'综合':<8} {'级别'}")
    print("-" * 100)
    
    for f1, f2 in test_pairs:
        result = calculator.compute_similarity(f1, f2)
        print(f"{result.field1:<25} {result.field2:<25} "
              f"{result.char_similarity:<8.3f} {result.semantic_similarity:<8.3f} "
              f"{result.combined_score:<8.3f} {result.match_level}")
    
    print("\n" + "=" * 70)
    print("批量查找相似字段测试")
    print("=" * 70)
    
    candidates = [
        "user_id", "customer_id", "order_id", "product_id",
        "username", "customer_name", "product_name", "order_name",
        "email", "phone", "mobile", "address",
        "created_at", "updated_at", "delete_time",
        "price", "amount", "quantity", "total"
    ]
    
    target = "user_email"
    print(f"\n目标字段: {target}")
    print(f"候选字段: {candidates}")
    print("-" * 50)
    
    similar = calculator.find_similar_fields(target, candidates, top_k=5)
    for r in similar:
        print(f"  {r.field2:<20} 相似度: {r.combined_score:.4f} ({r.match_level})")
