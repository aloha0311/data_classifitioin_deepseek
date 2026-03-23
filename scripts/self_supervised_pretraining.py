#!/usr/bin/env python3
"""
自监督预训练脚本
基于字段语义的自监督学习任务
"""
import os
import json
import random
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 字段语义知识库（用于自监督学习）
FIELD_SEMANTICS = {
    # ID类
    "id": {"type": "numeric", "description": "唯一标识符，用于唯一标识记录"},
    "customer_id": {"type": "numeric", "description": "客户唯一标识"},
    "user_id": {"type": "numeric", "description": "用户唯一标识"},
    "order_id": {"type": "numeric", "description": "订单唯一标识"},
    "product_id": {"type": "numeric", "description": "产品唯一标识"},
    
    # 名称类
    "name": {"type": "text", "description": "名称或标题"},
    "title": {"type": "text", "description": "标题或名称"},
    "username": {"type": "text", "description": "用户名"},
    "company_name": {"type": "text", "description": "公司名称"},
    "address": {"type": "text", "description": "地址位置"},
    
    # 数值类
    "price": {"type": "numeric", "description": "价格或金额"},
    "amount": {"type": "numeric", "description": "数量或金额"},
    "quantity": {"type": "numeric", "description": "数量"},
    "score": {"type": "numeric", "description": "分数或评分"},
    "count": {"type": "numeric", "description": "计数或次数"},
    
    # 时间类
    "date": {"type": "date", "description": "日期"},
    "time": {"type": "date", "description": "时间"},
    "created_at": {"type": "date", "description": "创建时间"},
    "updated_at": {"type": "date", "description": "更新时间"},
    "birthday": {"type": "date", "description": "出生日期"},
    
    # 状态类
    "status": {"type": "enum", "description": "状态或类型"},
    "type": {"type": "enum", "description": "类型或分类"},
    "gender": {"type": "enum", "description": "性别"},
    "is_active": {"type": "boolean", "description": "是否激活"},
    
    # 人口统计类
    "age": {"type": "numeric", "description": "年龄"},
    "income": {"type": "numeric", "description": "收入"},
    "education": {"type": "enum", "description": "教育程度"},
    "occupation": {"type": "text", "description": "职业"},
}

# 字段命名模式
FIELD_PATTERNS = {
    "id|id_no|id_number|no|编号|编码": "ID类/主键ID",
    "name|title|名称|标题|姓名": "属性类/名称标题",
    "price|amount|money|金额|价格": "度量类/计量数值",
    "count|num|quantity|数量|次数": "度量类/计数统计",
    "date|time|datetime|时间|日期|created|updated": "状态类/时间标记",
    "age|生日": "身份类/人口统计",
    "phone|tel|mobile|电话|手机|email|邮箱|contact": "身份类/联系方式",
    "address|location|city|country|region|地址|城市|国家": "属性类/地址位置",
    "status|state|type|flag|状态|类型": "状态类/状态枚举",
    "category|class|type|分类|类别": "属性类/类别标签",
    "description|desc|text|content|描述|说明": "属性类/描述文本",
    "percent|ratio|rate|百分比|比率|比例": "度量类/比率比例",
    "year|month|day|hour|年|月|日": "度量类/时间度量",
    "code|编号|序号|seq": "结构类/标准代码",
}


class SelfSupervisedDataGenerator:
    """自监督学习数据生成器"""
    
    def __init__(self):
        self.semantics = FIELD_SEMANTICS
        self.patterns = FIELD_PATTERNS
        
    def generate_semantic_completion_task(self, field_name: str) -> Dict:
        """
        生成语义补全任务
        输入: 字段名 -> 输出: 语义描述
        """
        field_lower = field_name.lower().replace("_", "").replace("-", "")
        
        # 匹配语义
        matched_semantic = None
        for pattern, semantics in self.semantics.items():
            if pattern in field_lower:
                matched_semantic = semantics
                break
        
        if matched_semantic:
            prompt = f"根据字段名推断其业务含义。\n\n字段名：{field_name}\n\n该字段可能的语义描述是："
            response = matched_semantic["description"]
        else:
            prompt = f"根据字段名推断其业务含义。\n\n字段名：{field_name}\n\n该字段可能的语义描述是："
            response = "需要根据具体业务场景判断"
        
        return {
            "instruction": prompt,
            "input": "",
            "output": response
        }
    
    def generate_type_inference_task(self, field_name: str) -> Dict:
        """
        生成数据类型推断任务
        输入: 字段名 -> 输出: 数据类型
        """
        field_lower = field_name.lower().replace("_", "").replace("-", "")
        
        # 匹配类型
        data_type = "未知类型"
        for pattern, semantics in self.semantics.items():
            if pattern in field_lower:
                data_type = semantics["type"]
                break
        
        # 类型映射
        type_mapping = {
            "numeric": "数值型（整数或浮点数）",
            "text": "文本型（字符串）",
            "date": "日期时间型",
            "enum": "枚举型（有限离散值）",
            "boolean": "布尔型（是/否）"
        }
        
        prompt = f"根据字段名判断其数据类型。\n\n字段名：{field_name}\n\n数据类型是："
        output = type_mapping.get(data_type, "未知类型")
        
        return {
            "instruction": prompt,
            "input": "",
            "output": output
        }
    
    def generate_pattern_matching_task(self, field_name: str) -> Dict:
        """
        生成模式匹配任务（分类）
        输入: 字段名 -> 输出: 分类标签
        """
        field_lower = field_name.lower().replace("_", "").replace("-", "")
        
        matched_label = "扩展类/其他字段"
        for pattern, label in self.patterns.items():
            if pattern in field_lower:
                matched_label = label
                break
        
        prompt = f"""根据字段名判断其所属类别。

字段名：{field_name}

请从以下类别中选择最合适的：
- ID类/主键ID
- 属性类/名称标题
- 度量类/计量数值
- 度量类/计数统计
- 状态类/时间标记
- 身份类/人口统计
- 属性类/地址位置
- 状态类/状态枚举
- 属性类/类别标签
- 扩展类/其他字段

分类结果："""
        
        return {
            "instruction": prompt,
            "input": "",
            "output": matched_label
        }
    
    def generate_all_tasks(self, field_name: str) -> List[Dict]:
        """为一个字段生成所有自监督学习任务"""
        return [
            self.generate_semantic_completion_task(field_name),
            self.generate_type_inference_task(field_name),
            self.generate_pattern_matching_task(field_name),
        ]
    
    def generate_dataset(self, num_samples: int = 100) -> List[Dict]:
        """生成自监督学习数据集"""
        # 从知识库中采样字段
        fields = list(self.semantics.keys()) * (num_samples // len(self.semantics) + 1)
        random.shuffle(fields)
        fields = fields[:num_samples]
        
        tasks = []
        for field in fields:
            tasks.extend(self.generate_all_tasks(field))
        
        random.shuffle(tasks)
        return tasks


def create_pretraining_data():
    """创建预训练数据"""
    generator = SelfSupervisedDataGenerator()
    
    # 生成自监督样本
    pretrain_samples = generator.generate_dataset(num_samples=200)
    
    # 生成有监督样本（从现有数据）
    sup_data_path = os.path.join(BASE_DIR, "data/sft/train.jsonl")
    supervised_samples = []
    
    if os.path.exists(sup_data_path):
        with open(sup_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    supervised_samples.append(json.loads(line))
    
    # 合并数据
    all_samples = pretrain_samples + supervised_samples
    random.shuffle(all_samples)
    
    # 保存
    output_path = os.path.join(BASE_DIR, "data/sft/pretrain_mixed.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"生成 {len(pretrain_samples)} 条自监督样本")
    print(f"合并 {len(supervised_samples)} 条有监督样本")
    print(f"总计 {len(all_samples)} 条样本")
    print(f"保存至: {output_path}")
    
    return all_samples


def demo():
    """演示自监督学习数据生成"""
    generator = SelfSupervisedDataGenerator()
    
    test_fields = ["customer_id", "price", "created_at", "status", "address"]
    
    print("=" * 60)
    print("自监督学习任务演示")
    print("=" * 60)
    
    for field in test_fields:
        print(f"\n字段: {field}")
        tasks = generator.generate_all_tasks(field)
        for i, task in enumerate(tasks):
            print(f"  任务{i+1}: {task['output']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        create_pretraining_data()
    else:
        demo()
