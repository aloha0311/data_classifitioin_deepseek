#!/usr/bin/env python3
"""
生成高质量的分类分级微调数据集

这个脚本生成的数据集可以直接用于微调 DeepSeek-7B 模型，
使其能够准确地识别字段的分类和分级。
"""
import json
import os
from pathlib import Path

# 输出目录
output_dir = Path("/home/zsq/deepseek_project/data/finetune")
output_dir.mkdir(parents=True, exist_ok=True)

# ============ 微调数据：高质量的 (字段名, 行业, 样本, 分类, 分级) 样本 ============
# 这些样本覆盖了各种场景，可以用于创建高质量的训练数据

TRAINING_SAMPLES = [
    # ========== 金融行业 ==========
    {"industry": "金融", "field": "customer_id", "samples": "C100001, C100002, C100003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "金融", "field": "account_no", "samples": "6222021234567890, 6222029876543210", "classification": "ID类/主键ID", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "customer_name", "samples": "张三, 李四, 王五", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "金融", "field": "id_card_no", "samples": "110101199001011234, 310101198505052345", "classification": "身份类/人口统计", "grading": "第4级/机密"},
    {"industry": "金融", "field": "phone_no", "samples": "13812345678, 13998765432", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "email", "samples": "user@example.com, test@email.com", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "bank_card_no", "samples": "6217000012345678901, 6217999998765432", "classification": "ID类/主键ID", "grading": "第4级/机密"},
    {"industry": "金融", "field": "account_balance", "samples": "12345.67, 98765.43, 5000.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "credit_score", "samples": "680, 720, 850, 650", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "loan_amount", "samples": "50000.00, 100000.00, 200000.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "monthly_income", "samples": "15000.00, 20000.00, 8000.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "transaction_amount", "samples": "100.50, 2000.00, 500.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "金融", "field": "department", "samples": "技术部, 市场部, 财务部", "classification": "身份类/职业信息", "grading": "第2级/内部"},
    {"industry": "金融", "field": "employee_level", "samples": "1, 2, 3, 4, 5", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "金融", "field": "is_active", "samples": "是, 否, Y, N", "classification": "状态类/二元标志", "grading": "第1级/公开"},
    {"industry": "金融", "field": "risk_level", "samples": "低风险, 中风险, 高风险", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "金融", "field": "login_password", "samples": "******, ******", "classification": "ID类/主键ID", "grading": "第4级/机密"},
    {"industry": "金融", "field": "last_login_ip", "samples": "192.168.1.1, 10.0.0.100", "classification": "度量类/时间度量", "grading": "第2级/内部"},

    # ========== 医疗行业 ==========
    {"industry": "医疗", "field": "patient_id", "samples": "P20240001, P20240002, P20240003", "classification": "ID类/主键ID", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "patient_name", "samples": "张三, 李四, 王五", "classification": "属性类/名称标题", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "gender", "samples": "男, 女, 其他", "classification": "身份类/人口统计", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "birth_date", "samples": "1990-01-01, 1985-05-15, 2000-12-20", "classification": "度量类/时间度量", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "age", "samples": "25, 30, 45, 60", "classification": "身份类/人口统计", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "phone_no", "samples": "13812345678, 13998765432", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "医疗", "field": "id_card_no", "samples": "110101199001011234, 310101198505052345", "classification": "身份类/人口统计", "grading": "第4级/机密"},
    {"industry": "医疗", "field": "home_address", "samples": "北京市朝阳区XX路XX号, 上海市浦东新区XX路XX号", "classification": "属性类/地址位置", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "blood_type", "samples": "A, B, AB, O", "classification": "身份类/人口统计", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "allergy_info", "samples": "青霉素过敏, 海鲜过敏, 无", "classification": "身份类/人口统计", "grading": "第3级/敏感"},
    {"industry": "医疗", "field": "medical_history", "samples": "高血压2年, 糖尿病1年, 无重大病史", "classification": "身份类/人口统计", "grading": "第3级/敏感"},
    {"industry": "医疗", "field": "diagnosis", "samples": "高血压, 糖尿病, 肺炎, 胃炎", "classification": "属性类/类别标签", "grading": "第3级/敏感"},
    {"industry": "医疗", "field": "doctor_id", "samples": "D0001, D0002, D0003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "医疗", "field": "department", "samples": "内科, 外科, 儿科, 妇产科", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "医疗", "field": "total_cost", "samples": "5000.00, 12000.00, 25000.00", "classification": "度量类/计量数值", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "insurance_no", "samples": "INS12345678, INS98765432", "classification": "ID类/主键ID", "grading": "第2级/内部"},
    {"industry": "医疗", "field": "blood_pressure", "samples": "120/80, 140/90, 130/85", "classification": "度量类/计量数值", "grading": "第2级/内部"},

    # ========== 教育行业 ==========
    {"industry": "教育", "field": "student_id", "samples": "S20240001, S20240002, S20240003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "教育", "field": "student_name", "samples": "张三, 李四, 王五", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "教育", "field": "gender", "samples": "男, 女", "classification": "身份类/人口统计", "grading": "第1级/公开"},
    {"industry": "教育", "field": "birth_date", "samples": "2010-01-01, 2009-05-15, 2011-12-20", "classification": "度量类/时间度量", "grading": "第1级/公开"},
    {"industry": "教育", "field": "class_id", "samples": "C001, C002, C003", "classification": "结构类/分类代码", "grading": "第1级/公开"},
    {"industry": "教育", "field": "chinese_score", "samples": "85, 92, 78, 95, 88", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "教育", "field": "math_score", "samples": "90, 85, 95, 80, 88", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "教育", "field": "english_score", "samples": "88, 92, 78, 90, 85", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "教育", "field": "total_score", "samples": "520, 550, 480, 600, 540", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "教育", "field": "avg_score", "samples": "86.7, 92.0, 80.0, 95.5", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "教育", "field": "rank", "samples": "1, 5, 10, 20, 50", "classification": "度量类/序号排序", "grading": "第1级/公开"},
    {"industry": "教育", "field": "attendance_rate", "samples": "95.5%, 98.0%, 92.3%, 100%", "classification": "度量类/比率比例", "grading": "第1级/公开"},
    {"industry": "教育", "field": "parent_name", "samples": "张华, 李明, 王强", "classification": "属性类/名称标题", "grading": "第2级/内部"},
    {"industry": "教育", "field": "parent_phone", "samples": "13812345678, 13998765432", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "教育", "field": "scholarship", "samples": "一等奖学金, 二等奖学金, 无", "classification": "属性类/类别标签", "grading": "第2级/内部"},

    # ========== 工业行业 ==========
    {"industry": "工业", "field": "device_id", "samples": "DEV001, DEV002, DEV003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "工业", "field": "device_name", "samples": "数控机床1号, CNC加工中心, 注塑机A", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "工业", "field": "device_type", "samples": "数控机床, CNC, 注塑机, 焊接机器人", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "工业", "field": "serial_no", "samples": "SN20240001, SN20240002", "classification": "结构类/产品代码", "grading": "第1级/公开"},
    {"industry": "工业", "field": "temperature", "samples": "25.5, 30.0, 35.2, 40.1", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "humidity", "samples": "45.5%, 50.0%, 55.2%", "classification": "度量类/比率比例", "grading": "第1级/公开"},
    {"industry": "工业", "field": "pressure", "samples": "1.01, 1.05, 0.98, 1.10", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "vibration", "samples": "0.5, 1.2, 0.8, 2.0", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "rotation_speed", "samples": "1500, 2000, 3000, 5000", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "power_consumption", "samples": "15.5, 20.0, 25.5, 30.0", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "output_count", "samples": "1000, 2000, 5000, 10000", "classification": "度量类/计数统计", "grading": "第1级/公开"},
    {"industry": "工业", "field": "defect_count", "samples": "5, 10, 20, 50", "classification": "度量类/计数统计", "grading": "第1级/公开"},
    {"industry": "工业", "field": "uptime_hours", "samples": "5000, 8000, 10000, 15000", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "工业", "field": "error_code", "samples": "正常, E001, E002, E003", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "工业", "field": "status", "samples": "运行中, 待机, 故障, 维护中", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "工业", "field": "operator_id", "samples": "OP001, OP002, OP003", "classification": "ID类/主键ID", "grading": "第2级/内部"},
    {"industry": "工业", "field": "location", "samples": "车间1-线1, 车间2-线5, 车间3-线10", "classification": "属性类/地址位置", "grading": "第1级/公开"},
    {"industry": "工业", "field": "ip_address", "samples": "192.168.1.100, 10.0.0.50", "classification": "结构类/产品代码", "grading": "第2级/内部"},

    # ========== 电商行业 ==========
    {"industry": "电商", "field": "order_id", "samples": "ORD2024010100001, ORD2024010200002", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "电商", "field": "customer_id", "samples": "C100001, C100002, C100003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "电商", "field": "customer_name", "samples": "张三, 李四, 王五", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "电商", "field": "customer_phone", "samples": "13812345678, 13998765432", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "电商", "field": "product_id", "samples": "P00001, P00002, P00003", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "电商", "field": "product_name", "samples": "iPhone15, MacBook Pro, AirPods", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "电商", "field": "category", "samples": "电子产品, 服装, 食品, 家居", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "电商", "field": "quantity", "samples": "1, 2, 3, 5, 10", "classification": "度量类/计数统计", "grading": "第1级/公开"},
    {"industry": "电商", "field": "unit_price", "samples": "99.00, 199.50, 2999.00, 5999.00", "classification": "度量类/计量数值", "grading": "第1级/公开"},
    {"industry": "电商", "field": "total_amount", "samples": "199.00, 5998.00, 299.50", "classification": "度量类/计量数值", "grading": "第2级/内部"},
    {"industry": "电商", "field": "order_status", "samples": "待支付, 已支付, 已发货, 已完成", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "电商", "field": "payment_method", "samples": "微信支付, 支付宝, 银行卡", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "电商", "field": "shipping_address", "samples": "北京市朝阳区XX路XX号, 上海市浦东新区XX路XX号", "classification": "属性类/地址位置", "grading": "第3级/敏感"},
    {"industry": "电商", "field": "order_time", "samples": "2024-01-01 10:30:00, 2024-01-02 15:45:00", "classification": "度量类/时间度量", "grading": "第1级/公开"},
    {"industry": "电商", "field": "coupon_discount", "samples": "10.00, 20.00, 50.00, 0.00", "classification": "度量类/计量数值", "grading": "第2级/内部"},

    # ========== 政务行业 ==========
    {"industry": "政务", "field": "case_id", "samples": "CASE20240100001, CASE20240100002", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "政务", "field": "citizen_name", "samples": "张三, 李四, 王五", "classification": "属性类/名称标题", "grading": "第2级/内部"},
    {"industry": "政务", "field": "citizen_id", "samples": "110101199001011234, 310101198505052345", "classification": "身份类/人口统计", "grading": "第4级/机密"},
    {"industry": "政务", "field": "phone", "samples": "13812345678, 13998765432", "classification": "身份类/联系方式", "grading": "第3级/敏感"},
    {"industry": "政务", "field": "address", "samples": "XX市XX区XX街道XX号", "classification": "属性类/地址位置", "grading": "第2级/内部"},
    {"industry": "政务", "field": "case_type", "samples": "身份证办理, 户籍迁移, 社保查询", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "政务", "field": "case_status", "samples": "受理中, 审核中, 已完成, 已驳回", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "政务", "field": "department", "samples": "公安局, 民政局, 人社局, 税务局", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "政务", "field": "handler_name", "samples": "办理员1, 办理员2, 办理员3", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "政务", "field": "tax_id", "samples": "T1234567890, T9876543210", "classification": "结构类/分类代码", "grading": "第3级/敏感"},
    {"industry": "政务", "field": "social_security_no", "samples": "SS1234567890, SS9876543210", "classification": "ID类/主键ID", "grading": "第4级/机密"},
    {"industry": "政务", "field": "annual_income", "samples": "50000.00, 100000.00, 200000.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "政务", "field": "tax_amount", "samples": "5000.00, 10000.00, 20000.00", "classification": "度量类/计量数值", "grading": "第3级/敏感"},
    {"industry": "政务", "field": "satisfaction", "samples": "非常满意, 满意, 一般, 不满意", "classification": "状态类/状态枚举", "grading": "第1级/公开"},

    # ========== 通用字段 ==========
    {"industry": "通用", "field": "id", "samples": "1, 2, 3, 4, 5", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "通用", "field": "uuid", "samples": "550e8400-e29b-41d4-a716-446655440000", "classification": "ID类/主键ID", "grading": "第1级/公开"},
    {"industry": "通用", "field": "created_at", "samples": "2024-01-01 10:30:00, 2024-01-02 15:45:00", "classification": "度量类/时间度量", "grading": "第1级/公开"},
    {"industry": "通用", "field": "updated_at", "samples": "2024-01-01 10:30:00, 2024-01-02 15:45:00", "classification": "度量类/时间度量", "grading": "第1级/公开"},
    {"industry": "通用", "field": "created_by", "samples": "admin, user1, user2", "classification": "属性类/名称标题", "grading": "第2级/内部"},
    {"industry": "通用", "field": "updated_by", "samples": "admin, user1, user2", "classification": "属性类/名称标题", "grading": "第2级/内部"},
    {"industry": "通用", "field": "remark", "samples": "备注信息, 测试数据, 特殊情况", "classification": "属性类/描述文本", "grading": "第1级/公开"},
    {"industry": "通用", "field": "description", "samples": "这是描述文本, 产品说明, 服务条款", "classification": "属性类/描述文本", "grading": "第1级/公开"},
    {"industry": "通用", "field": "status", "samples": "正常, 禁用, 删除, 待审核", "classification": "状态类/状态枚举", "grading": "第1级/公开"},
    {"industry": "通用", "field": "is_deleted", "samples": "是, 否, Y, N, 0, 1", "classification": "状态类/二元标志", "grading": "第1级/公开"},
    {"industry": "通用", "field": "is_active", "samples": "是, 否, Y, N, true, false", "classification": "状态类/二元标志", "grading": "第1级/公开"},
    {"industry": "通用", "field": "version", "samples": "1.0, 2.0, 3.0, 1.2.3", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "通用", "field": "sort_order", "samples": "1, 2, 3, 10, 100", "classification": "度量类/序号排序", "grading": "第1级/公开"},
    {"industry": "通用", "field": "enable_flag", "samples": "启用, 禁用, Y, N", "classification": "状态类/二元标志", "grading": "第1级/公开"},
    {"industry": "通用", "field": "type", "samples": "类型A, 类型B, 类型C", "classification": "属性类/类别标签", "grading": "第1级/公开"},
    {"industry": "通用", "field": "code", "samples": "CODE001, CODE002, CODE003", "classification": "结构类/分类代码", "grading": "第1级/公开"},
    {"industry": "通用", "field": "name", "samples": "名称1, 名称2, 名称3", "classification": "属性类/名称标题", "grading": "第1级/公开"},
    {"industry": "通用", "field": "title", "samples": "标题1, 标题2, 标题3", "classification": "属性类/名称标题", "grading": "第1级/公开"},
]


def generate_alpaca_format():
    """生成 Alpaca 格式的微调数据"""
    outputs = []

    for sample in TRAINING_SAMPLES:
        # 构建 instruction
        instruction = f"""请根据以下信息判断字段的分类和分级。

字段名：{sample['field']}
行业：{sample['industry']}
样本值：{sample['samples']}

请输出该字段的分类和分级。"""

        # 构建 input
        input_text = f"字段名：{sample['field']}\n行业：{sample['industry']}\n样本值：{sample['samples']}"

        # 构建 output
        output_text = f"分类：{sample['classification']}\n分级：{sample['grading']}"

        outputs.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    return outputs


def generate_conversation_format():
    """生成对话格式的微调数据"""
    outputs = []

    for sample in TRAINING_SAMPLES:
        conversation = [
            {
                "role": "user",
                "content": f"""请判断以下字段的分类和分级。

字段名：{sample['field']}
行业：{sample['industry']}
样本值：{sample['samples']}"""
            },
            {
                "role": "assistant",
                "content": f"分类：{sample['classification']}\n分级：{sample['grading']}"
            }
        ]
        outputs.append({"conversations": conversation})

    return outputs


def generate_qa_format():
    """生成问答格式的微调数据"""
    outputs = []

    for sample in TRAINING_SAMPLES:
        # Q1: 分类
        q1 = f"字段 '{sample['field']}' 在 {sample['industry']} 行业中应该属于哪个分类？"
        a1 = sample['classification']

        # Q2: 分级
        q2 = f"字段 '{sample['field']}' 的数据敏感度等级是什么？"
        a2 = sample['grading']

        # Q3: 综合
        q3 = f"请分析字段 '{sample['field']}'（行业：{sample['industry']}，样本：{sample['samples']}）的分类和分级。"
        a3 = f"分类：{sample['classification']}，分级：{sample['grading']}"

        outputs.append({
            "question_1": q1,
            "answer_1": a1,
            "question_2": q2,
            "answer_2": a2,
            "question_3": q3,
            "answer_3": a3
        })

    return outputs


def main():
    print("=" * 60)
    print("生成分类分级微调数据集")
    print("=" * 60)

    # 1. Alpaca 格式
    alpaca_data = generate_alpaca_format()
    alpaca_path = output_dir / "train_alpaca.json"
    with open(alpaca_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 生成 Alpaca 格式: {alpaca_path}")
    print(f"   样本数: {len(alpaca_data)}")

    # 2. 对话格式
    conv_data = generate_conversation_format()
    conv_path = output_dir / "train_conversation.json"
    with open(conv_path, 'w', encoding='utf-8') as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 生成对话格式: {conv_path}")
    print(f"   样本数: {len(conv_data)}")

    # 3. 问答格式
    qa_data = generate_qa_format()
    qa_path = output_dir / "train_qa.json"
    with open(qa_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 生成问答格式: {qa_path}")
    print(f"   样本数: {len(qa_data)}")

    # 4. 生成纯文本格式（用于简单微调）
    text_path = output_dir / "train_text.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        for sample in TRAINING_SAMPLES:
            f.write(f"字段: {sample['field']}\n")
            f.write(f"行业: {sample['industry']}\n")
            f.write(f"样本: {sample['samples']}\n")
            f.write(f"分类: {sample['classification']}\n")
            f.write(f"分级: {sample['grading']}\n")
            f.write("-" * 40 + "\n")
    print(f"\n✅ 生成纯文本格式: {text_path}")

    # 5. 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    # 按行业统计
    by_industry = {}
    for sample in TRAINING_SAMPLES:
        industry = sample['industry']
        by_industry[industry] = by_industry.get(industry, 0) + 1

    print("\n按行业分布：")
    for industry, count in sorted(by_industry.items()):
        print(f"  {industry}: {count} 条")

    # 按分类统计
    by_class = {}
    for sample in TRAINING_SAMPLES:
        cls = sample['classification']
        by_class[cls] = by_class.get(cls, 0) + 1

    print("\n按分类分布：")
    for cls, count in sorted(by_class.items()):
        print(f"  {cls}: {count} 条")

    # 按分级统计
    by_grade = {}
    for sample in TRAINING_SAMPLES:
        grade = sample['grading']
        by_grade[grade] = by_grade.get(grade, 0) + 1

    print("\n按分级分布：")
    for grade, count in sorted(by_grade.items()):
        print(f"  {grade}: {count} 条")

    print("\n" + "=" * 60)
    print(f"所有数据集已保存到: {output_dir}")
    print("=" * 60)

    print("\n使用方法：")
    print("1. Alpaca 格式: 适用于 LLaMA-Factory、Axolotl 等微调框架")
    print("2. 对话格式: 适用于对话模型微调")
    print("3. 问答格式: 适用于问答模型微调")
    print("4. 纯文本格式: 适用于简单的文本生成微调")


if __name__ == "__main__":
    main()
