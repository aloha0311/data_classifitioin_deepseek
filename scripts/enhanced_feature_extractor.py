#!/usr/bin/env python3
"""
增强的语义特征提取模块
将统计特征整合到推理流程中，为模型提供更丰富的输入信息
"""
import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FieldFeatures:
    """字段特征数据类"""
    column_name: str
    data_type: str  # numeric, date, text, boolean, enum
    is_unique: bool
    is_nullable: bool
    null_ratio: float
    unique_ratio: float
    unique_count: int
    total_count: int
    
    # 数值特征
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    median_value: Optional[float] = None
    
    # 文本特征
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    has_chinese: bool = False
    has_digits: bool = False
    has_letters: bool = False
    
    # 语义提示
    semantic_hints: List[str] = None
    
    # 样本值
    samples: List[str] = None
    
    def __post_init__(self):
        if self.semantic_hints is None:
            self.semantic_hints = []
        if self.samples is None:
            self.samples = []
    
    def to_dict(self) -> Dict:
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "is_unique": self.is_unique,
            "is_nullable": self.is_nullable,
            "null_ratio": round(self.null_ratio, 4),
            "unique_ratio": round(self.unique_ratio, 4),
            "unique_count": self.unique_count,
            "total_count": self.total_count,
            "numeric_stats": {
                "min": self.min_value,
                "max": self.max_value,
                "mean": self.mean_value,
                "std": self.std_value,
                "median": self.median_value
            } if self.data_type == "numeric" else None,
            "text_stats": {
                "avg_length": self.avg_length,
                "max_length": self.max_length,
                "min_length": self.min_length,
                "has_chinese": self.has_chinese,
                "has_digits": self.has_digits,
                "has_letters": self.has_letters
            } if self.data_type == "text" else None,
            "semantic_hints": self.semantic_hints,
            "samples": self.samples
        }


class EnhancedFeatureExtractor:
    """增强的特征提取器"""
    
    def __init__(self):
        # 语义模式库
        self.semantic_patterns = {
            "ID类/主键ID": [
                r'^id$', r'_id$', r'^no$', r'编号', r'编码', r'^uuid',
                r'^guid$', r'^pk$', r'主键', r'标识', r'序列'
            ],
            "属性类/名称标题": [
                r'name', r'名称', r'标题', r'title', r'username', r'标题',
                r'公司名', r'产品名', r'姓名', r'店名', r'品牌'
            ],
            "属性类/类别标签": [
                r'category', r'tag', r'label', r'type', r'分类', r'类别',
                r'标签', r'等级', r'level', r'grade', r'品类'
            ],
            "属性类/描述文本": [
                r'desc', r'description', r'desc', r'text', r'文本',
                r'描述', r'说明', r'备注', r'remark', r'note'
            ],
            "属性类/技能标签": [
                r'skill', r'ability', r'skills', r'技能', r'专长',
                r'expertise', r'competency', r'能力'
            ],
            "属性类/地址位置": [
                r'address', r'addr', r'location', r'地址', r'位置',
                r'城市', r'city', r'country', r'国家', r'区域', r'省', r'县'
            ],
            "度量类/计量数值": [
                r'price', r'amount', r'money', r'金额', r'价格', r'费用',
                r'salary', r'工资', r'age', r'年龄', r'长度', r'宽度', r'高度'
            ],
            "度量类/计数统计": [
                r'count', r'数量', r'quantity', r'num', r'次数',
                r'人数', r'次数', r'total', r'sum'
            ],
            "度量类/比率比例": [
                r'rate', r'ratio', r'percent', r'百分比', r'比率',
                r'比例', r'占比', r'probability'
            ],
            "度量类/时间度量": [
                r'duration', r'interval', r'hours', r'minutes', r'seconds',
                r'时长', r'间隔', r'持续', r'工时'
            ],
            "度量类/序号排序": [
                r'seq', r'index', r'rank', r'order', r'序号',
                r'排序', r'顺位', r'排名', r'position.*\d+'
            ],
            "身份类/人口统计": [
                r'age', r'年龄', r'gender', r'性别', r'sex',
                r'生日', r'birthday', r'出生', r'民族', r'婚姻'
            ],
            "身份类/联系方式": [
                r'phone', r'tel', r'mobile', r'电话', r'手机',
                r'email', r'mail', r'邮箱', r'地址.*联系', r'contact'
            ],
            "身份类/教育背景": [
                r'education', r'教育', r'degree', r'学历', r'school',
                r'学校', r'university', r'major', r'专业'
            ],
            "身份类/职业信息": [
                r'job', r'occupation', r'position', r'职业', r'岗位',
                r'department', r'部门', r'company', r'公司', r'雇主'
            ],
            "状态类/二元标志": [
                r'^is_', r'^has_', r'flag', r'标志', r'是否',
                r'有.*无', r'_bool$', r'激活', r'启用'
            ],
            "状态类/状态枚举": [
                r'status', r'状态', r'state', r'stage', r'phase',
                r'阶段', r'结果', r'result', r'_type$'
            ],
            "状态类/时间标记": [
                r'_at$', r'_time$', r'_date$', r'时间', r'日期',
                r'created', r'updated', r'datetime', r'timestamp'
            ],
            "结构类/产品代码": [
                r'product', r'sku', r'item_code', r'产品', r'商品',
                r'货号', r'型号'
            ],
            "结构类/企业代码": [
                r'company', r'org', r'dept', r'企业', r'公司',
                r'部门', r'组织', r'institution'
            ],
            "结构类/标准代码": [
                r'country', r'currency', r'iso', r'国家', r'货币',
                r'标准', r'region.*code'
            ]
        }
    
    def is_numeric(self, value: str) -> bool:
        """判断是否为数值"""
        if pd.isna(value) or str(value).strip() == '':
            return False
        try:
            str_val = str(value).strip().replace(',', '')
            float(str_val)
            return True
        except:
            return False
    
    def infer_data_type(self, series: pd.Series) -> str:
        """推断数据类型"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        # 数值型检测
        numeric_count = sum(1 for v in non_null if self.is_numeric(v))
        if numeric_count / len(non_null) > 0.8:
            return "numeric"
        
        # 日期型检测
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}', r'^\d{4}/\d{2}/\d{2}',
            r'^\d{2}/\d{2}/\d{4}', r'^\d{10,}$'
        ]
        date_count = sum(1 for v in non_null if any(re.match(p, str(v)) for p in date_patterns))
        if date_count / len(non_null) > 0.8:
            return "date"
        
        # 布尔型检测
        unique_vals = set(str(v).strip().lower() for v in non_null)
        bool_values = {'true', 'false', '是', '否', '1', '0', 'yes', 'no', 'y', 'n'}
        if unique_vals.issubset(bool_values) and len(unique_vals) <= 2:
            return "boolean"
        
        # 枚举型检测（唯一值较少）
        unique_ratio = len(unique_vals) / len(non_null)
        if unique_ratio < 0.1 and len(unique_vals) < 20:
            return "enum"
        
        # 文本型
        return "text"
    
    def extract_semantic_hints(self, column_name: str) -> List[str]:
        """提取语义提示"""
        hints = []
        name_lower = column_name.lower().replace('_', '').replace('-', '')
        
        for category, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower, re.IGNORECASE):
                    hints.append(category)
                    break
        
        return hints
    
    def extract_features(self, df: pd.DataFrame, column: str, n_samples: int = 5) -> FieldFeatures:
        """提取单个字段的特征"""
        series = df[column]
        col_name = column
        
        # 基本统计
        total_count = len(series)
        null_count = series.isna().sum()
        null_ratio = null_count / total_count if total_count > 0 else 0
        
        non_null = series.dropna()
        non_null_str = non_null.astype(str)
        non_null_str = non_null_str[non_null_str.str.strip() != '']
        
        unique_vals = set(non_null_str)
        unique_count = len(unique_vals)
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0
        
        # 数据类型
        data_type = self.infer_data_type(series)
        
        # 样本值
        samples = list(non_null_str.unique()[:n_samples])
        
        # 数值特征
        min_val, max_val, mean_val, std_val, median_val = None, None, None, None, None
        if data_type == "numeric":
            numeric_vals = []
            for v in non_null:
                try:
                    numeric_vals.append(float(str(v).replace(',', '').strip()))
                except:
                    continue
            if numeric_vals:
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                mean_val = np.mean(numeric_vals)
                std_val = np.std(numeric_vals) if len(numeric_vals) > 1 else 0
                median_val = np.median(numeric_vals)
        
        # 文本特征
        avg_len, max_len, min_len, has_cn, has_digit, has_letter = None, None, None, False, False, False
        if data_type in ["text", "enum", "boolean"]:
            lengths = [len(str(v)) for v in non_null]
            if lengths:
                avg_len = np.mean(lengths)
                max_len = max(lengths)
                min_len = min(lengths)
            
            # 字符组成
            all_chars = ''.join(str(v) for v in non_null[:100])
            has_cn = bool(re.search(r'[\u4e00-\u9fff]', all_chars))
            has_digit = bool(re.search(r'\d', all_chars))
            has_letter = bool(re.search(r'[a-zA-Z]', all_chars))
        
        # 语义提示
        semantic_hints = self.extract_semantic_hints(col_name)
        
        return FieldFeatures(
            column_name=col_name,
            data_type=data_type,
            is_unique=unique_ratio == 1.0,
            is_nullable=null_ratio > 0,
            null_ratio=null_ratio,
            unique_ratio=unique_ratio,
            unique_count=unique_count,
            total_count=total_count,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            median_value=median_val,
            avg_length=avg_len,
            max_length=max_len,
            min_length=min_len,
            has_chinese=has_cn,
            has_digits=has_digit,
            has_letters=has_letter,
            semantic_hints=semantic_hints,
            samples=samples
        )
    
    def extract_all_features(self, df: pd.DataFrame, n_samples: int = 5) -> Dict[str, FieldFeatures]:
        """提取所有字段的特征"""
        features = {}
        for col in df.columns:
            try:
                features[col] = self.extract_features(df, col, n_samples)
            except Exception as e:
                print(f"  提取字段 '{col}' 特征失败: {e}")
        return features


class EnhancedPromptBuilder:
    """增强的Prompt构建器"""
    
    def __init__(self, extractor: EnhancedFeatureExtractor):
        self.extractor = extractor
    
    def build_classification_prompt(
        self,
        industry: str,
        column_name: str,
        features: FieldFeatures,
        labels: List[str],
        include_examples: bool = True
    ) -> str:
        """构建分类Prompt"""
        # 基础信息
        prompt_parts = [
            f"你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段属于哪一类。",
            f"",
            f"行业：{industry}",
            f"字段名：{column_name}",
        ]
        
        # 数据类型
        prompt_parts.append(f"数据类型：{features.data_type}")
        
        # 样本值
        if features.samples:
            samples_str = ', '.join(str(s) for s in features.samples[:5])
            prompt_parts.append(f"样本值示例：{samples_str}")
        
        # 统计特征
        if features.data_type == "numeric" and features.mean_value is not None:
            stats_parts = []
            if features.min_value is not None:
                stats_parts.append(f"最小值: {features.min_value:.2f}")
            if features.max_value is not None:
                stats_parts.append(f"最大值: {features.max_value:.2f}")
            if features.mean_value is not None:
                stats_parts.append(f"平均值: {features.mean_value:.2f}")
            if stats_parts:
                prompt_parts.append(f"数值统计：{', '.join(stats_parts)}")
        
        # 语义提示
        if features.semantic_hints:
            prompt_parts.append(f"语义提示：{', '.join(features.semantic_hints)}")
        
        # 唯一性
        if features.unique_ratio < 0.1:
            prompt_parts.append(f"枚举值数量：{features.unique_count}个")
        
        # 标签列表
        labels_text = '\n'.join([f"- {label}" for label in labels])
        prompt_parts.append(f"\n请从以下分类标签中选择最合适的一个（只输出标签路径，不要其他内容）：")
        prompt_parts.append(labels_text)
        prompt_parts.append(f"\n答案：")
        
        return '\n'.join(prompt_parts)
    
    def build_grading_prompt(
        self,
        industry: str,
        column_name: str,
        features: FieldFeatures,
        classification: str,
        grades: List[str]
    ) -> str:
        """构建分级Prompt"""
        prompt_parts = [
            f"你是一个数据分类分级助手。请根据字段名、行业、样本值和分类结果判断字段的安全分级。",
            f"",
            f"行业：{industry}",
            f"字段名：{column_name}",
            f"分类结果：{classification}",
        ]
        
        # 样本值
        if features.samples:
            samples_str = ', '.join(str(s) for s in features.samples[:5])
            prompt_parts.append(f"样本值示例：{samples_str}")
        
        # 分级标准
        prompt_parts.append(f"\n分级标准参考：")
        prompt_parts.append(f"- 第1级/公开：可向公众公开的数据")
        prompt_parts.append(f"- 第2级/内部：仅限内部人员访问")
        prompt_parts.append(f"- 第3级/敏感：涉及个人/敏感信息")
        prompt_parts.append(f"- 第4级/机密：高度敏感的个人信息")
        
        # 分级列表
        grades_text = '\n'.join([f"- {grade}" for grade in grades])
        prompt_parts.append(f"\n请从以下分级标签中选择最合适的一个（只输出标签路径）：")
        prompt_parts.append(grades_text)
        prompt_parts.append(f"\n答案：")
        
        return '\n'.join(prompt_parts)


def demo():
    """演示增强特征提取"""
    extractor = EnhancedFeatureExtractor()
    prompt_builder = EnhancedPromptBuilder(extractor)
    
    # 测试数据
    test_data = {
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'price': ['99.5', '199.0', '299.5', '150.0', '89.9'],
        'create_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'city': ['北京', '上海', '深圳', '广州', '杭州'],
        'description': ['这是一段很长的描述文本用于测试', '第二段描述文本', '第三段描述文本', '第四段', '第五段'],
        'is_active': ['是', '否', '是', '是', '否'],
        'status': ['pending', 'approved', 'rejected', 'pending', 'approved'],
        'age': [25, 30, 35, 28, 42],
        'phone': ['13800138001', '13900139002', '13700137003', '13600136004', '13500135005'],
        'education': ['本科', '硕士', '博士', '本科', '本科'],
    }
    
    df = pd.DataFrame(test_data)
    features_dict = extractor.extract_all_features(df)
    
    CLASSIFICATION_LABELS = [
        "ID类/主键ID", "结构类/分类代码", "结构类/产品代码", "结构类/企业代码", "结构类/标准代码",
        "属性类/名称标题", "属性类/类别标签", "属性类/描述文本", "属性类/技能标签", "属性类/地址位置",
        "度量类/计量数值", "度量类/计数统计", "度量类/比率比例", "度量类/时间度量", "度量类/序号排序",
        "身份类/人口统计", "身份类/联系方式", "身份类/教育背景", "身份类/职业信息",
        "状态类/二元标志", "状态类/状态枚举", "状态类/时间标记",
        "扩展类/扩展代码", "扩展类/其他字段"
    ]
    
    print("=" * 60)
    print("增强特征提取演示")
    print("=" * 60)
    
    for col, features in features_dict.items():
        print(f"\n字段: {col}")
        print(f"  数据类型: {features.data_type}")
        print(f"  唯一值比例: {features.unique_ratio:.2%}")
        print(f"  语义提示: {features.semantic_hints}")
        
        # 构建增强Prompt
        prompt = prompt_builder.build_classification_prompt(
            industry="金融",
            column_name=col,
            features=features,
            labels=CLASSIFICATION_LABELS
        )
        print(f"\n  增强Prompt预览:")
        print(f"  {'-' * 40}")
        for line in prompt.split('\n')[:10]:
            print(f"  {line}")
        if len(prompt.split('\n')) > 10:
            print(f"  ...")


def process_csv_with_features(csv_path: str, output_path: str = None):
    """处理CSV文件并提取特征"""
    extractor = EnhancedFeatureExtractor()
    prompt_builder = EnhancedPromptBuilder(extractor)
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 提取特征
    features_dict = extractor.extract_all_features(df)
    
    # 保存特征
    features_output = {col: features.to_dict() for col, features in features_dict.items()}
    
    if output_path is None:
        output_path = csv_path.replace('.csv', '_features.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features_output, f, ensure_ascii=False, indent=2)
    
    print(f"特征已保存: {output_path}")
    return features_dict


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 处理指定CSV文件
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        process_csv_with_features(csv_path, output_path)
    else:
        demo()
