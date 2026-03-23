#!/usr/bin/env python3
"""
数据统计特征提取模块
为每个数据字段提取多维统计特征，支持多模态数据表示
"""
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class StatisticalFeatureExtractor:
    """数据字段统计特征提取器"""
    
    def __init__(self):
        self.numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        self.date_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # 2024-01-01
            re.compile(r'^\d{4}/\d{2}/\d{2}$'),   # 2024/01/01
            re.compile(r'^\d{2}/\d{2}/\d{4}$'),   # 01/01/2024
        ]
        self.email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        self.url_pattern = re.compile(r'^https?://')
        self.phone_pattern = re.compile(r'^1[3-9]\d{9}$|^\d{3,4}-\d{7,8}$')
    
    def is_numeric(self, value: str) -> bool:
        """判断是否为数值"""
        try:
            str_val = str(value).strip().replace(',', '')
            float(str_val)
            return True
        except:
            return False
    
    def is_date(self, value: str) -> bool:
        """判断是否为日期"""
        value = str(value).strip()
        for pattern in self.date_patterns:
            if pattern.match(value):
                return True
        return False
    
    def infer_data_type(self, values: List[str]) -> str:
        """推断数据类型"""
        non_null = [v for v in values if str(v).strip() and str(v).strip() != 'N/A' and str(v).strip() != '?']
        if not non_null:
            return "unknown"
        
        numeric_count = sum(1 for v in non_null if self.is_numeric(v))
        date_count = sum(1 for v in non_null if self.is_date(v))
        
        ratio = max(numeric_count, date_count) / len(non_null)
        
        if ratio > 0.8:
            if date_count > numeric_count:
                return "date"
            return "numeric"
        
        # 检查文本类型
        text_sample = non_null[:10]
        avg_len = np.mean([len(str(v)) for v in text_sample])
        
        if avg_len < 50:
            return "short_text"
        return "long_text"
    
    def extract_char_features(self, values: List[str]) -> Dict[str, float]:
        """提取字符类型特征"""
        non_null = [str(v).strip() for v in values if str(v).strip() and str(v) != 'N/A' and str(v) != '?']
        if not non_null:
            return {"digit_ratio": 0, "letter_ratio": 0, "chinese_ratio": 0, "special_ratio": 0}
        
        total_chars = sum(len(s) for s in non_null)
        if total_chars == 0:
            return {"digit_ratio": 0, "letter_ratio": 0, "chinese_ratio": 0, "special_ratio": 0}
        
        digit_count = sum(sum(c.isdigit() for c in s) for s in non_null)
        letter_count = sum(sum(c.isalpha() and not '\u4e00' <= c <= '\u9fff' for c in s) for s in non_null)
        chinese_count = sum(sum('\u4e00' <= c <= '\u9fff' for c in s) for s in non_null)
        special_count = total_chars - digit_count - letter_count - chinese_count
        
        return {
            "digit_ratio": round(digit_count / total_chars, 4),
            "letter_ratio": round(letter_count / total_chars, 4),
            "chinese_ratio": round(chinese_count / total_chars, 4),
            "special_ratio": round(special_count / total_chars, 4)
        }
    
    def extract_value_features(self, values: List[str]) -> Dict[str, Any]:
        """提取数值类特征"""
        non_null = [v for v in values if str(v).strip() and str(v).strip() not in ['N/A', '?', 'nan']]
        if not non_null:
            return {"min": None, "max": None, "mean": None, "std": None, "median": None}
        
        numeric_values = []
        for v in non_null:
            try:
                numeric_values.append(float(str(v).replace(',', '').strip()))
            except:
                continue
        
        if not numeric_values:
            return {"min": None, "max": None, "mean": None, "std": None, "median": None}
        
        return {
            "min": round(min(numeric_values), 4),
            "max": round(max(numeric_values), 4),
            "mean": round(np.mean(numeric_values), 4),
            "std": round(np.std(numeric_values), 4) if len(numeric_values) > 1 else 0,
            "median": round(np.median(numeric_values), 4)
        }
    
    def extract_distribution_features(self, values: List[str]) -> Dict[str, Any]:
        """提取分布特征"""
        non_null = [str(v).strip() for v in values if str(v).strip() and str(v) not in ['N/A', '?', 'nan']]
        if not non_null:
            return {"unique_count": 0, "unique_ratio": 0, "total_count": 0}
        
        total = len(non_null)
        unique = len(set(non_null))
        
        return {
            "unique_count": unique,
            "unique_ratio": round(unique / total, 4),
            "total_count": total
        }
    
    def extract_semantic_hints(self, column_name: str) -> List[str]:
        """提取语义提示词"""
        hints = []
        name_lower = column_name.lower()
        
        # ID类
        if any(k in name_lower for k in ['id', '编号', '编码', '序列号', '主键']):
            hints.append("唯一标识符")
        
        # 名称类
        if any(k in name_lower for k in ['name', '名称', '标题', 'title', '姓名']):
            hints.append("名称类字段")
        
        # 数值类
        if any(k in name_lower for k in ['count', '数量', '金额', '价格', '分数', 'score', 'price', 'amount']):
            hints.append("数值度量字段")
        
        # 时间类
        if any(k in name_lower for k in ['time', '时间', '日期', 'date', 'year', 'month', 'day']):
            hints.append("时间标记字段")
        
        # 地址类
        if any(k in name_lower for k in ['addr', '地址', '位置', 'location', '城市', 'city', 'country']):
            hints.append("地址位置字段")
        
        # 状态类
        if any(k in name_lower for k in ['status', '状态', 'flag', '标志', 'type', '类型']):
            hints.append("状态枚举字段")
        
        # 描述类
        if any(k in name_lower for k in ['desc', '描述', 'text', '文本', 'content']):
            hints.append("描述文本字段")
        
        return hints
    
    def extract_all_features(self, df: pd.DataFrame, column: str, n_samples: int = 5) -> Dict[str, Any]:
        """提取单个字段的所有特征"""
        try:
            series = df[column].astype(str).replace(['nan', 'None', ''], np.nan)
            values = series.dropna().tolist()
            
            # 样本值
            valid_values = [v for v in values if str(v).strip() and str(v) not in ['N/A', '?', 'nan']]
            sample_values = valid_values[:n_samples] if len(valid_values) >= n_samples else valid_values
            
            features = {
                "column_name": column,
                "data_type": self.infer_data_type(values),
                "samples": sample_values,
                "char_features": self.extract_char_features(values),
                "value_features": self.extract_value_features(values),
                "distribution_features": self.extract_distribution_features(values),
                "semantic_hints": self.extract_semantic_hints(column)
            }
            
            return features
        except Exception as e:
            return {
                "column_name": column,
                "error": str(e)
            }
    
    def extract_dataframe_features(self, df: pd.DataFrame, n_samples: int = 5) -> Dict[str, Dict]:
        """提取整个DataFrame的所有字段特征"""
        features = {}
        for col in df.columns:
            try:
                features[col] = self.extract_all_features(df, col, n_samples)
            except Exception as e:
                features[col] = {"error": str(e)}
        return features


def demo():
    """演示统计特征提取"""
    extractor = StatisticalFeatureExtractor()
    
    # 测试数据
    test_data = {
        'id': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'price': ['99.5', '199.0', '299.5', '150.0', '89.9'],
        'create_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'city': ['北京', '上海', '深圳', '广州', '杭州'],
        'description': ['这是一段很长的描述文本用于测试', '第二段描述文本', '第三段描述文本', '第四段', '第五段']
    }
    
    df = pd.DataFrame(test_data)
    features = extractor.extract_dataframe_features(df)
    
    print("=" * 60)
    print("数据统计特征提取演示")
    print("=" * 60)
    
    for col, feat in features.items():
        print(f"\n字段: {col}")
        print(f"  数据类型: {feat.get('data_type')}")
        print(f"  字符特征: {feat.get('char_features')}")
        print(f"  数值特征: {feat.get('value_features')}")
        print(f"  分布特征: {feat.get('distribution_features')}")
        print(f"  语义提示: {feat.get('semantic_hints')}")


if __name__ == "__main__":
    demo()
