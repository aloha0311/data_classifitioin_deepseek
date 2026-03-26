#!/usr/bin/env python3
"""生成测试数据集"""
import csv
import random
import os
from datetime import datetime, timedelta

output_dir = "/home/zsq/deepseek_project/data/new"
os.makedirs(output_dir, exist_ok=True)

# ============= 数据集1: 金融行业用户数据 (约20字段, 25行) =============
finance_data = {
    "user_id": [f"U{str(i).zfill(6)}" for i in range(1, 26)],
    "username": [f"user_{i}" for i in range(1, 26)],
    "real_name": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
                  "郑十一", "王十二", "刘十三", "陈十四", "杨十五", "黄十六", "林十七",
                  "何十八", "高十九", "马二十", "朱二十一", "胡二十二", "郭二十三", "何二十四", "罗二十五", None, None],
    "id_card": [f"{random.randint(110000, 659000)}{random.randint(1950, 2005)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}{random.randint(1000, 9999)}" if i < 23 else None for i in range(25)],
    "phone": [f"138{random.randint(10000000, 99999999)}" for _ in range(25)],
    "email": [f"user{i}@example.com" for i in range(1, 26)],
    "bank_card": [f"{random.randint(400000, 499999)}{random.randint(100000000000, 999999999999)}" if i < 20 else None for i in range(25)],
    "account_balance": [round(random.uniform(0, 100000), 2) for _ in range(25)],
    "credit_score": [random.randint(300, 850) for _ in range(25)],
    "loan_amount": [random.choice([None, None, None, round(random.uniform(10000, 500000), 2)]) for _ in range(25)],
    "monthly_income": [round(random.uniform(3000, 50000), 2) for _ in range(25)],
    "department": random.choice(["技术部", "市场部", "财务部", "人事部", "运营部"]),
    "employee_level": [random.randint(1, 10) for _ in range(25)],
    "hire_date": [(datetime.now() - timedelta(days=random.randint(1, 3650))).strftime("%Y-%m-%d") for _ in range(25)],
    "is_active": random.choice(["是", "否"]),
    "risk_level": random.choice(["低风险", "中风险", "高风险"]),
    "last_login_ip": [f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(25)],
    "password_hash": ["*"] * 25,
    "transaction_count": [random.randint(0, 1000) for _ in range(25)],
    "avg_transaction_amount": [round(random.uniform(10, 5000), 2) for _ in range(25)],
}

# ============= 数据集2: 医疗患者数据 (约35字段, 20行) =============
medical_data = {
    "patient_id": [f"P{str(i).zfill(8)}" for i in range(1, 21)],
    "patient_name": ["患者" + str(i) for i in range(1, 21)],
    "gender": random.choices(["男", "女", "其他"], k=20),
    "birth_date": [(datetime.now() - timedelta(days=random.randint(1, 365*80))).strftime("%Y-%m-%d") for _ in range(20)],
    "age": [random.randint(1, 90) for _ in range(20)],
    "phone": [f"139{random.randint(10000000, 99999999)}" for _ in range(20)],
    "emergency_contact": [f"138{random.randint(10000000, 99999999)}" for _ in range(20)],
    "emergency_name": ["紧急联系人" + str(i) for i in range(1, 21)],
    "id_card": [f"{random.randint(110000, 659000)}{random.randint(1950, 2010)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}{random.randint(1000, 9999)}" for _ in range(20)],
    "home_address": ["地址" + str(i) + "号" for i in range(1, 21)],
    "blood_type": random.choices(["A", "B", "AB", "O"], k=20),
    "allergy_info": random.choices(["无", "青霉素", "海鲜过敏", "花粉过敏", None], k=20, weights=[60, 10, 10, 10, 10]),
    "medical_history": ["高血压2年, 糖尿病1年", "无重大病史", "心脏病史", "过敏性鼻炎", "胆结石手术史"],
    "diagnosis": ["高血压", "糖尿病", "肺炎", "胃炎", "腰椎间盘突出"],
    "doctor_id": [f"D{str(random.randint(1, 50)).zfill(4)}" for _ in range(20)],
    "doctor_name": ["医生" + str(i) for i in range(1, 21)],
    "department": random.choices(["内科", "外科", "儿科", "妇产科", "骨科"], k=20),
    "admission_date": [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d") for _ in range(20)],
    "discharge_date": [(datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d") if random.random() > 0.3 else None for _ in range(20)],
    "treatment_days": [random.randint(1, 30) for _ in range(20)],
    "total_cost": [round(random.uniform(500, 50000), 2) for _ in range(20)],
    "insurance_type": random.choices(["城镇职工医保", "城乡居民医保", "商业保险", "自费"], k=20),
    "insurance_number": [f"INS{random.randint(10000000, 99999999)}" if random.random() > 0.2 else None for _ in range(20)],
    "prescription_count": [random.randint(0, 20) for _ in range(20)],
    "lab_test_count": [random.randint(0, 50) for _ in range(20)],
    "blood_pressure_high": [random.randint(90, 180) for _ in range(20)],
    "blood_pressure_low": [random.randint(60, 110) for _ in range(20)],
    "heart_rate": [random.randint(60, 120) for _ in range(20)],
    "body_temperature": [round(random.uniform(36.0, 38.5), 1) for _ in range(20)],
    "weight_kg": [round(random.uniform(40, 100), 1) for _ in range(20)],
    "height_cm": [random.randint(150, 190) for _ in range(20)],
    "surgery_history": random.choices(["无", "阑尾切除术(2018)", "剖腹产(2020)", "骨折手术(2019)", None], k=20),
    "family_history": random.choices(["无", "父亲有高血压", "母亲有糖尿病", "家族有癌症史", None], k=20),
    "smoking_status": random.choices(["不吸烟", "偶尔吸烟", "经常吸烟", "已戒烟"], k=20),
    "drinking_status": random.choices(["不饮酒", "偶尔饮酒", "经常饮酒"], k=20),
}

# ============= 数据集3: 电商订单数据 (约15字段, 30行) =============
ecommerce_data = {
    "order_id": [f"ORD{str(i).zfill(10)}" for i in range(1, 31)],
    "customer_id": [f"C{str(random.randint(1, 1000)).zfill(6)}" for _ in range(30)],
    "customer_name": ["客户" + str(i) for i in range(1, 31)],
    "customer_phone": [f"15{random.randint(10000000, 99999999)}" for _ in range(30)],
    "product_id": [f"P{str(random.randint(1000, 9999)).zfill(5)}" for _ in range(30)],
    "product_name": ["商品" + str(i) for i in range(1, 31)],
    "category": random.choices(["电子产品", "服装", "食品", "家居", "美妆", "图书"], k=30),
    "quantity": [random.randint(1, 10) for _ in range(30)],
    "unit_price": [round(random.uniform(9.9, 999.9), 2) for _ in range(30)],
    "total_amount": [],  # 计算得出
    "order_status": random.choices(["待支付", "已支付", "已发货", "已完成", "已取消", "退款中"], k=30, weights=[10, 20, 30, 25, 10, 5]),
    "payment_method": random.choices(["微信支付", "支付宝", "银行卡", "货到付款"], k=30),
    "shipping_address": ["收货地址" + str(i) for i in range(1, 31)],
    "order_time": [(datetime.now() - timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(30)],
    "shipping_time": [(datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S") if random.random() > 0.3 else None for _ in range(30)],
    "coupon_discount": [round(random.uniform(0, 50), 2) if random.random() > 0.5 else 0 for _ in range(30)],
    "points_earned": [random.randint(0, 500) for _ in range(30)],
}

# 计算总价
ecommerce_data["total_amount"] = [
    round(row["quantity"] * row["unit_price"] - row["coupon_discount"], 2)
    for row in [dict(zip(ecommerce_data.keys(), values)) for values in zip(*ecommerce_data.values())]
]

# ============= 数据集4: 教育学生成绩数据 (约25字段, 18行) =============
education_data = {
    "student_id": [f"S{str(i).zfill(8)}" for i in range(1, 19)],
    "student_name": ["学生" + str(i) for i in range(1, 19)],
    "gender": random.choices(["男", "女"], k=18),
    "birth_date": [(datetime(2010, 1, 1) + timedelta(days=random.randint(0, 2000))).strftime("%Y-%m-%d") for _ in range(18)],
    "class_id": [f"C{str(random.randint(1, 20)).zfill(3)}" for _ in range(18)],
    "class_name": ["班级" + str(random.randint(1, 6)) + "班" for _ in range(18)],
    "teacher_id": [f"T{str(random.randint(1, 50)).zfill(4)}" for _ in range(18)],
    "chinese_score": [random.randint(60, 100) for _ in range(18)],
    "math_score": [random.randint(50, 100) for _ in range(18)],
    "english_score": [random.randint(55, 100) for _ in range(18)],
    "physics_score": [random.randint(60, 100) if random.random() > 0.3 else None for _ in range(18)],
    "chemistry_score": [random.randint(60, 100) if random.random() > 0.3 else None for _ in range(18)],
    "history_score": [random.randint(60, 100) for _ in range(18)],
    "geography_score": [random.randint(60, 100) for _ in range(18)],
    "political_score": [random.randint(60, 100) for _ in range(18)],
    "avg_score": [],  # 计算
    "total_score": [],  # 计算
    "rank": [i for i in range(1, 19)],
    "attendance_rate": [round(random.uniform(85, 100), 1) for _ in range(18)],
    "homework_completion": [round(random.uniform(80, 100), 1) for _ in range(18)],
    "parent_name": ["家长" + str(i) for i in range(1, 19)],
    "parent_phone": [f"13{random.randint(10000000, 99999999)}" for _ in range(18)],
    "parent_email": [f"parent{i}@email.com" for i in range(1, 19)],
    "scholarship": random.choices(["无", "一等奖学金", "二等奖学金", "三等奖学金"], k=18, weights=[70, 10, 12, 8]),
    "discipline_score": [random.randint(60, 100) for _ in range(18)],
}

# 计算平均分和总分
for i in range(18):
    scores = [education_data["chinese_score"][i], education_data["math_score"][i],
              education_data["english_score"][i], education_data["history_score"][i],
              education_data["geography_score"][i], education_data["political_score"][i]]
    education_data["total_score"].append(sum(scores))
    education_data["avg_score"].append(round(sum(scores) / len(scores), 1))

# ============= 数据集5: 工业设备传感器数据 (约40字段, 15行) =============
industrial_data = {
    "device_id": [f"DEV{str(i).zfill(6)}" for i in range(1, 16)],
    "device_name": ["设备" + str(i) for i in range(1, 16)],
    "device_type": random.choices(["数控机床", "CNC", "注塑机", "焊接机器人", "传送带"], k=15),
    "manufacturer": ["厂商" + str(random.randint(1, 5)) for _ in range(15)],
    "model": ["型号" + str(random.randint(100, 999)) for _ in range(15)],
    "serial_number": [f"SN{str(random.randint(100000, 999999))}" for _ in range(15)],
    "install_date": [(datetime.now() - timedelta(days=random.randint(365, 1825))).strftime("%Y-%m-%d") for _ in range(15)],
    "location": ["车间" + str(random.randint(1, 5)) + "-线" + str(random.randint(1, 10)) for _ in range(15)],
    "temperature_1": [round(random.uniform(20, 80), 2) for _ in range(15)],
    "temperature_2": [round(random.uniform(20, 80), 2) for _ in range(15)],
    "temperature_3": [round(random.uniform(20, 80), 2) for _ in range(15)],
    "humidity": [round(random.uniform(30, 80), 1) for _ in range(15)],
    "pressure_1": [round(random.uniform(0.5, 1.5), 3) for _ in range(15)],
    "pressure_2": [round(random.uniform(0.5, 1.5), 3) for _ in range(15)],
    "vibration_x": [round(random.uniform(0, 10), 3) for _ in range(15)],
    "vibration_y": [round(random.uniform(0, 10), 3) for _ in range(15)],
    "vibration_z": [round(random.uniform(0, 10), 3) for _ in range(15)],
    "rotation_speed": [random.randint(1000, 5000) for _ in range(15)],
    "power_consumption": [round(random.uniform(5, 50), 2) for _ in range(15)],
    "voltage": [round(random.uniform(360, 420), 1) for _ in range(15)],
    "current": [round(random.uniform(10, 100), 2) for _ in range(15)],
    "frequency": [round(random.uniform(49.5, 50.5), 2) for _ in range(15)],
    "output_count": [random.randint(0, 10000) for _ in range(15)],
    "defect_count": [random.randint(0, 50) for _ in range(15)],
    "uptime_hours": [random.randint(0, 20000) for _ in range(15)],
    "downtime_hours": [random.randint(0, 1000) for _ in range(15)],
    "maintenance_count": [random.randint(0, 20) for _ in range(15)],
    "last_maintenance": [(datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d") for _ in range(15)],
    "next_maintenance": [(datetime.now() + timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(15)],
    "error_code": random.choices(["正常", "E001", "E002", "E003", None], k=15, weights=[70, 10, 8, 7, 5]),
    "status": random.choices(["运行中", "待机", "故障", "维护中"], k=15, weights=[60, 20, 10, 10]),
    "operator_id": [f"O{str(random.randint(1, 100)).zfill(4)}" for _ in range(15)],
    "operator_name": ["操作员" + str(i) for i in range(1, 16)],
    "shift": random.choices(["早班", "中班", "晚班"], k=15),
    "production_rate": [round(random.uniform(80, 99), 1) for _ in range(15)],
    "quality_rate": [round(random.uniform(95, 99.9), 2) for _ in range(15)],
    "oee": [round(random.uniform(60, 90), 1) for _ in range(15)],  # 设备综合效率
    "firmware_version": [f"V{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 99)}" for _ in range(15)],
    "ip_address": [f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}" for _ in range(15)],
    "mac_address": [":".join([f"{random.randint(0, 255):02x}" for _ in range(6)]) for _ in range(15)],
}

# ============= 数据集6: 政务数据 (约30字段, 22行) =============
government_data = {
    "case_id": [f"CASE{str(i).zfill(10)}" for i in range(1, 23)],
    "citizen_name": ["市民" + str(i) for i in range(1, 23)],
    "citizen_id": [f"{random.randint(110000, 659000)}{random.randint(1950, 2005)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}{random.randint(1000, 9999)}" for _ in range(23)],
    "phone": [f"13{random.randint(10000000, 99999999)}" for _ in range(23)],
    "email": [f"citizen{i}@mail.gov.cn" for i in range(1, 23)],
    "address": ["XX市XX区XX街道XX号" for _ in range(23)],
    "case_type": random.choices(["身份证办理", "户籍迁移", "社保查询", "公积金提取", "营业执照办理", "税务申报"], k=23),
    "case_status": random.choices(["受理中", "审核中", "已完成", "已驳回"], k=23, weights=[20, 30, 40, 10]),
    "submit_time": [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(23)],
    "complete_time": [(datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d %H:%M:%S") if random.random() > 0.3 else None for _ in range(23)],
    "department": random.choices(["公安局", "民政局", "人社局", "税务局", "工商局", "公积金中心"], k=23),
    "handler_id": [f"H{str(random.randint(1, 200)).zfill(5)}" for _ in range(23)],
    "handler_name": ["办理员" + str(i) for i in range(1, 24)],
    "id_card_number": [f"{random.randint(110000, 659000)}{random.randint(1950, 2005)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}{random.randint(1000, 9999)}" if random.random() > 0.2 else None for _ in range(23)],
    "household_type": random.choices(["城镇", "农村"], k=23),
    "income_level": random.choices(["低收入", "中等收入", "中高收入", "高收入"], k=23, weights=[20, 40, 25, 15]),
    "tax_id": [f"T{str(random.randint(100000000, 999999999)).zfill(10)}" if random.random() > 0.3 else None for _ in range(23)],
    "social_security_number": [f"SS{str(random.randint(100000000, 999999999)).zfill(10)}" for _ in range(23)],
    "housing_fund_number": [f"HF{str(random.randint(100000000, 999999999)).zfill(10)}" if random.random() > 0.4 else None for _ in range(23)],
    "business_license_number": [f"BL{str(random.randint(100000000, 999999999)).zfill(10)}" if random.random() > 0.6 else None for _ in range(23)],
    "company_name": ["公司" + str(random.randint(1, 100)) + "有限公司" if random.random() > 0.5 else None for _ in range(23)],
    "company_address": ["公司地址" + str(i) if random.random() > 0.5 else None for i in range(23)],
    "annual_income": [round(random.uniform(30000, 500000), 2) if random.random() > 0.3 else None for _ in range(23)],
    "tax_amount": [round(random.uniform(1000, 100000), 2) if random.random() > 0.4 else None for _ in range(23)],
    "approval_score": [round(random.uniform(60, 100), 1) for _ in range(23)],
    "satisfaction": random.choices(["非常满意", "满意", "一般", "不满意"], k=23, weights=[30, 40, 20, 10]),
    "remark": ["备注信息" + str(i) if random.random() > 0.5 else None for i in range(23)],
    "attachment_count": [random.randint(0, 10) for _ in range(23)],
    "process_days": [random.randint(1, 30) for _ in range(23)],
    "priority": random.choices(["普通", "加急", "VIP"], k=23, weights=[70, 20, 10]),
}

# 写入CSV文件
datasets = [
    ("金融用户数据_20字段_25行.csv", finance_data),
    ("医疗患者数据_35字段_20行.csv", medical_data),
    ("电商订单数据_15字段_30行.csv", ecommerce_data),
    ("学生成绩数据_25字段_18行.csv", education_data),
    ("工业传感器数据_40字段_15行.csv", industrial_data),
    ("政务办事数据_30字段_23行.csv", government_data),
]

for filename, data in datasets:
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()

        # 获取行数
        num_rows = len(next(iter(data.values())))

        # 写入每一行
        for i in range(num_rows):
            row = {k: v[i] if i < len(v) else None for k, v in data.items()}
            writer.writerow(row)

    print(f"生成: {filename} ({len(data)}字段, {num_rows}行)")

print(f"\n所有数据集已保存到: {output_dir}")
