#!/usr/bin/env python3
"""
批量处理所有CSV数据集，生成训练和验证JSONL文件
"""
import os
import json
import pandas as pd
import glob
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/sft")

CLASSIFICATION_LABELS = [
    "ID类/主键ID", "结构类/分类代码", "结构类/产品代码", "结构类/企业代码", "结构类/标准代码",
    "属性类/名称标题", "属性类/类别标签", "属性类/描述文本", "属性类/技能标签", "属性类/地址位置",
    "度量类/计量数值", "度量类/计数统计", "度量类/比率比例", "度量类/时间度量", "度量类/序号排序",
    "身份类/人口统计", "身份类/联系方式", "身份类/教育背景", "身份类/职业信息",
    "状态类/二元标志", "状态类/状态枚举", "状态类/时间标记",
    "扩展类/扩展代码", "扩展类/其他字段"
]

# 完整标签映射 - 覆盖所有数据集
ALL_FIELD_LABELS = {
    # === 原有数据集 ===
    # 乳腺癌数据集 (medical)
    "乳腺癌": {
        "id": ("ID类/主键ID", "第1级/公开"),
        "Clump Thickness": ("度量类/计量数值", "第1级/公开"),
        "Uniformity of Cell Size": ("度量类/计量数值", "第1级/公开"),
        "Uniformity of Cell Shape": ("度量类/计量数值", "第1级/公开"),
        "Marginal Adhesion": ("度量类/计量数值", "第1级/公开"),
        "Single Epithelial Cell Size": ("度量类/计量数值", "第1级/公开"),
        "Bare Nuclei": ("度量类/计量数值", "第1级/公开"),
        "Bland Chromatin": ("度量类/计量数值", "第1级/公开"),
        "Normal Nucleoli": ("度量类/计量数值", "第1级/公开"),
        "Mitoses": ("度量类/计量数值", "第1级/公开"),
        "Class": ("状态类/二元标志", "第2级/内部"),
    },
    # 天猫双十一美妆数据 (business)
    "天猫": {
        "update_time": ("状态类/时间标记", "第2级/内部"),
        "id": ("ID类/主键ID", "第1级/公开"),
        "title": ("属性类/名称标题", "第1级/公开"),
        "price": ("度量类/计量数值", "第1级/公开"),
        "sale_count": ("度量类/计数统计", "第2级/内部"),
        "comment_count": ("度量类/计数统计", "第2级/内部"),
        "店名": ("属性类/名称标题", "第1级/公开"),
        "sub_type": ("属性类/类别标签", "第1级/公开"),
        "main_type": ("属性类/类别标签", "第1级/公开"),
        "是否为男士专用": ("状态类/二元标志", "第1级/公开"),
        "销售额": ("度量类/计量数值", "第2级/内部"),
        "day": ("度量类/时间度量", "第1级/公开"),
    },
    # 学生考试表现 (education)
    "学生考试": {
        "gender": ("身份类/人口统计", "第3级/敏感"),
        "race/ethnicity": ("身份类/人口统计", "第3级/敏感"),
        "parental level of education": ("身份类/教育背景", "第3级/敏感"),
        "lunch": ("属性类/类别标签", "第1级/公开"),
        "test preparation course": ("属性类/技能标签", "第1级/公开"),
        "math score": ("度量类/计量数值", "第2级/内部"),
        "reading score": ("度量类/计量数值", "第2级/内部"),
        "writing score": ("度量类/计量数值", "第2级/内部"),
    },
    # 工业设备测试数据 (industrial)
    "工业": {
        "采集编号": ("ID类/主键ID", "第1级/公开"),
        "工单": ("结构类/分类代码", "第2级/内部"),
        "产品编号": ("结构类/产品代码", "第2级/内部"),
        "烧程编号": ("度量类/时间度量", "第1级/公开"),
        "产品序列号": ("度量类/序号排序", "第2级/内部"),
        "工序": ("度量类/计数统计", "第1级/公开"),
        "设备编码": ("结构类/企业代码", "第2级/内部"),
        "采集参数编码": ("结构类/标准代码", "第1级/公开"),
        "采集数据列表": ("扩展类/扩展代码", "第1级/公开"),
        "采集数据结果值": ("度量类/计量数值", "第1级/公开"),
        "采集数据结果": ("状态类/二元标志", "第1级/公开"),
        "参数版本号": ("结构类/标准代码", "第1级/公开"),
        "标签一": ("扩展类/其他字段", "第1级/公开"),
        "标签二": ("扩展类/其他字段", "第1级/公开"),
        "标签三": ("扩展类/其他字段", "第1级/公开"),
        "标签四": ("扩展类/其他字段", "第1级/公开"),
        "标签五": ("扩展类/其他字段", "第1级/公开"),
        "域": ("度量类/计量数值", "第1级/公开"),
        "进入使用": ("状态类/二元标志", "第1级/公开"),
        "使用日期": ("状态类/时间标记", "第2级/内部"),
        "状态": ("状态类/状态枚举", "第1级/公开"),
        "标识": ("度量类/计量数值", "第1级/公开"),
        "标签六": ("扩展类/其他字段", "第1级/公开"),
        "标签七": ("扩展类/其他字段", "第1级/公开"),
        "标签八": ("扩展类/其他字段", "第1级/公开"),
        "标签九": ("状态类/二元标志", "第1级/公开"),
        "麦斯数据库零零一": ("扩展类/其他字段", "第1级/公开"),
        "麦斯数据库零零二": ("扩展类/其他字段", "第1级/公开"),
        "麦斯数据库零零三": ("扩展类/其他字段", "第1级/公开"),
        "麦斯数据库零零四": ("扩展类/其他字段", "第1级/公开"),
        "标签十": ("扩展类/其他字段", "第1级/公开"),
        "开始采集时间": ("状态类/时间标记", "第2级/内部"),
        "结束采集时间": ("状态类/时间标记", "第2级/内部"),
        "日期三": ("状态类/时间标记", "第2级/内部"),
        "日期四": ("状态类/时间标记", "第2级/内部"),
        "日期五": ("状态类/时间标记", "第2级/内部"),
        "麦斯数据库五": ("扩展类/其他字段", "第1级/公开"),
    },

    # === 新增数据集 ===
    # 金融/银行数据 (financial)
    "信用卡客户违约": {
        "ID": ("ID类/主键ID", "第1级/公开"),
        "LIMIT_BAL": ("度量类/计量数值", "第3级/敏感"),
        "SEX": ("身份类/人口统计", "第3级/敏感"),
        "EDUCATION": ("身份类/教育背景", "第3级/敏感"),
        "MARRIAGE": ("身份类/人口统计", "第3级/敏感"),
        "AGE": ("身份类/人口统计", "第3级/敏感"),
        "PAY_1": ("状态类/状态枚举", "第2级/内部"),
        "PAY_2": ("状态类/状态枚举", "第2级/内部"),
        "PAY_3": ("状态类/状态枚举", "第2级/内部"),
        "PAY_4": ("状态类/状态枚举", "第2级/内部"),
        "PAY_5": ("状态类/状态枚举", "第2级/内部"),
        "PAY_6": ("状态类/状态枚举", "第2级/内部"),
        "BILL_AMT1": ("度量类/计量数值", "第2级/内部"),
        "BILL_AMT2": ("度量类/计量数值", "第2级/内部"),
        "BILL_AMT3": ("度量类/计量数值", "第2级/内部"),
        "BILL_AMT4": ("度量类/计量数值", "第2级/内部"),
        "BILL_AMT5": ("度量类/计量数值", "第2级/内部"),
        "BILL_AMT6": ("度量类/计量数值", "第2级/内部"),
        "PAY_AMT1": ("度量类/计量数值", "第2级/内部"),
        "PAY_AMT2": ("度量类/计量数值", "第2级/内部"),
    },
    "健康保险交叉销售预测": {
        "id": ("ID类/主键ID", "第1级/公开"),
        "Gender": ("身份类/人口统计", "第3级/敏感"),
        "Age": ("身份类/人口统计", "第3级/敏感"),
        "Driving_License": ("状态类/二元标志", "第1级/公开"),
        "Region_Code": ("结构类/标准代码", "第1级/公开"),
        "Previously_Insured": ("状态类/二元标志", "第2级/内部"),
        "Vehicle_Age": ("度量类/计量数值", "第1级/公开"),
        "Vehicle_Damage": ("状态类/二元标志", "第2级/内部"),
        "Annual_Premium": ("度量类/计量数值", "第2级/内部"),
        "Policy_Sales_Channel": ("结构类/标准代码", "第1级/公开"),
        "Vintage": ("度量类/时间度量", "第2级/内部"),
        "Response": ("状态类/二元标志", "第2级/内部"),
    },
    "贷款审批预测": {
        "loan_id": ("ID类/主键ID", "第1级/公开"),
        "no_of_dependents": ("度量类/计数统计", "第3级/敏感"),
        "education": ("身份类/教育背景", "第3级/敏感"),
        "self_employed": ("状态类/二元标志", "第1级/公开"),
        "income_annum": ("度量类/计量数值", "第3级/敏感"),
        "loan_amount": ("度量类/计量数值", "第2级/内部"),
        "loan_term": ("度量类/时间度量", "第1级/公开"),
        "cibil_score": ("度量类/计量数值", "第3级/敏感"),
        "residential_assets_value": ("度量类/计量数值", "第3级/敏感"),
        "commercial_assets_value": ("度量类/计量数值", "第3级/敏感"),
        "luxury_assets_value": ("度量类/计量数值", "第3级/敏感"),
        "bank_asset_value": ("度量类/计量数值", "第3级/敏感"),
        "loan_status": ("状态类/二元标志", "第2级/内部"),
    },
    "银行客户流失": {
        "RowNumber": ("度量类/序号排序", "第1级/公开"),
        "CustomerId": ("ID类/主键ID", "第2级/内部"),
        "Surname": ("属性类/名称标题", "第2级/内部"),
        "CreditScore": ("度量类/计量数值", "第3级/敏感"),
        "Geography": ("属性类/地址位置", "第2级/内部"),
        "Gender": ("身份类/人口统计", "第3级/敏感"),
        "Age": ("身份类/人口统计", "第3级/敏感"),
        "Tenure": ("度量类/时间度量", "第2级/内部"),
        "Balance": ("度量类/计量数值", "第3级/敏感"),
        "NumOfProducts": ("度量类/计数统计", "第1级/公开"),
        "HasCrCard": ("状态类/二元标志", "第1级/公开"),
        "IsActiveMember": ("状态类/二元标志", "第2级/内部"),
        "EstimatedSalary": ("度量类/计量数值", "第3级/敏感"),
        "Exited": ("状态类/二元标志", "第2级/内部"),
    },
    "工科毕业生薪酬预测": {
        "ID": ("ID类/主键ID", "第1级/公开"),
        "Gender": ("身份类/人口统计", "第3级/敏感"),
        "DOB": ("状态类/时间标记", "第3级/敏感"),
        "10percentage": ("度量类/计量数值", "第2级/内部"),
        "10board": ("属性类/类别标签", "第1级/公开"),
        "12graduation": ("状态类/时间标记", "第2级/内部"),
        "12percentage": ("度量类/计量数值", "第2级/内部"),
        "12board": ("属性类/类别标签", "第1级/公开"),
        "CollegeID": ("结构类/标准代码", "第1级/公开"),
        "CollegeTier": ("属性类/类别标签", "第1级/公开"),
        "Degree": ("属性类/类别标签", "第1级/公开"),
        "Specialization": ("属性类/类别标签", "第1级/公开"),
        "collegeGPA": ("度量类/计量数值", "第2级/内部"),
        "CollegeCityID": ("结构类/标准代码", "第1级/公开"),
        "CollegeCityTier": ("属性类/类别标签", "第1级/公开"),
        "CollegeState": ("属性类/地址位置", "第1级/公开"),
        "GraduationYear": ("状态类/时间标记", "第2级/内部"),
        "English": ("度量类/计量数值", "第1级/公开"),
        "Logical": ("度量类/计量数值", "第1级/公开"),
        "Quant": ("度量类/计量数值", "第1级/公开"),
    },

    # 医疗健康数据 (medical)
    "心脏病和中风预防": {
        "Year": ("状态类/时间标记", "第1级/公开"),
        "LocationAbbr": ("属性类/地址位置", "第1级/公开"),
        "LocationDesc": ("属性类/地址位置", "第1级/公开"),
        "Datasource": ("属性类/类别标签", "第1级/公开"),
        "PriorityArea1": ("属性类/类别标签", "第1级/公开"),
        "Category": ("属性类/类别标签", "第1级/公开"),
        "Topic": ("属性类/类别标签", "第1级/公开"),
        "Indicator": ("属性类/名称标题", "第1级/公开"),
        "Data_Value_Type": ("属性类/类别标签", "第1级/公开"),
        "Data_Value_Unit": ("属性类/类别标签", "第1级/公开"),
        "Data_Value": ("度量类/计量数值", "第1级/公开"),
        "Confidence_Limit_Low": ("度量类/计量数值", "第1级/公开"),
        "Confidence_Limit_High": ("度量类/计量数值", "第1级/公开"),
        "Break_Out_Category": ("属性类/类别标签", "第1级/公开"),
    },
    "世界卫生组织自杀统计": {
        "country": ("属性类/地址位置", "第1级/公开"),
        "year": ("状态类/时间标记", "第1级/公开"),
        "sex": ("身份类/人口统计", "第3级/敏感"),
        "age": ("身份类/人口统计", "第3级/敏感"),
        "suicides_no": ("度量类/计数统计", "第2级/内部"),
        "population": ("度量类/计数统计", "第1级/公开"),
    },
    "医疗费用个人数据集": {
        "age": ("身份类/人口统计", "第3级/敏感"),
        "sex": ("身份类/人口统计", "第3级/敏感"),
        "bmi": ("度量类/计量数值", "第2级/内部"),
        "children": ("度量类/计数统计", "第3级/敏感"),
        "smoker": ("状态类/二元标志", "第3级/敏感"),
        "region": ("属性类/地址位置", "第1级/公开"),
        "charges": ("度量类/计量数值", "第3级/敏感"),
    },
    "养鱼场数据监测": {
        "deviceId": ("ID类/主键ID", "第1级/公开"),
        "water_temp": ("度量类/计量数值", "第1级/公开"),
        "air_temp": ("度量类/计量数值", "第1级/公开"),
        "ph": ("度量类/计量数值", "第1级/公开"),
        "timestamp": ("状态类/时间标记", "第2级/内部"),
        "month": ("状态类/时间标记", "第1级/公开"),
        "day": ("度量类/时间度量", "第1级/公开"),
        "hours": ("度量类/时间度量", "第1级/公开"),
    },
    "印度的每日发电量": {
        "Date": ("状态类/时间标记", "第1级/公开"),
        "Region": ("属性类/地址位置", "第1级/公开"),
        "Thermal Generation Actual (in MU)": ("度量类/计量数值", "第1级/公开"),
        "Thermal Generation Estimated (in MU)": ("度量类/计量数值", "第1级/公开"),
        "Nuclear Generation Actual (in MU)": ("度量类/计量数值", "第1级/公开"),
        "Nuclear Generation Estimated (in MU)": ("度量类/计量数值", "第1级/公开"),
        "Hydro Generation Actual (in MU)": ("度量类/计量数值", "第1级/公开"),
        "Hydro Generation Estimated (in MU)": ("度量类/计量数值", "第1级/公开"),
    },
    "各国霍乱病例数据集": {
        "Country": ("属性类/地址位置", "第1级/公开"),
        "Year": ("状态类/时间标记", "第1级/公开"),
        "Number of reported cases of cholera": ("度量类/计数统计", "第2级/内部"),
        "Number of reported deaths from cholera": ("度量类/计数统计", "第2级/内部"),
        "Cholera case fatality rate": ("度量类/比率比例", "第1级/公开"),
        "WHO Region": ("属性类/地址位置", "第1级/公开"),
    },

    # 商业/零售数据 (business)
    "女性电子商务服装评论": {
        "Unnamed: 0": ("度量类/序号排序", "第1级/公开"),
        "Clothing ID": ("ID类/主键ID", "第1级/公开"),
        "Age": ("身份类/人口统计", "第3级/敏感"),
        "Title": ("属性类/名称标题", "第1级/公开"),
        "Review Text": ("属性类/描述文本", "第2级/内部"),
        "Rating": ("度量类/计量数值", "第1级/公开"),
        "Recommended IND": ("状态类/二元标志", "第1级/公开"),
        "Positive Feedback Count": ("度量类/计数统计", "第1级/公开"),
        "Division Name": ("属性类/类别标签", "第1级/公开"),
        "Department Name": ("属性类/类别标签", "第1级/公开"),
        "Class Name": ("属性类/类别标签", "第1级/公开"),
    },
    "餐厅销售": {
        "DATE": ("状态类/时间标记", "第1级/公开"),
        "SALES": ("度量类/计量数值", "第2级/内部"),
        "IS_WEEKEND": ("状态类/二元标志", "第1级/公开"),
        "IS_HOLIDAY": ("状态类/二元标志", "第1级/公开"),
        "IS_FESTIVE_DATE": ("状态类/二元标志", "第1级/公开"),
        "IS_PRE_FESTIVE_DATE": ("状态类/二元标志", "第1级/公开"),
        "IS_AFTER_FESTIVE_DATE": ("状态类/二元标志", "第1级/公开"),
        "IS_PEOPLE_WEEK_PAYMENT": ("状态类/二元标志", "第1级/公开"),
        "IS_LOW_SEASON": ("状态类/二元标志", "第1级/公开"),
        "AMOUNT_OTHER_OPENED_RESTAURANTS": ("度量类/计数统计", "第1级/公开"),
        "WEATHER_PRECIPITATION": ("度量类/计量数值", "第1级/公开"),
        "WEATHER_TEMPERATURE": ("度量类/计量数值", "第1级/公开"),
        "WEATHER_HUMIDITY": ("度量类/计量数值", "第1级/公开"),
    },
    "上海车牌拍卖价格": {
        "Date": ("状态类/时间标记", "第1级/公开"),
        "Total number of license issued": ("度量类/计数统计", "第1级/公开"),
        "lowest price": ("度量类/计量数值", "第1级/公开"),
        "avg price": ("度量类/计量数值", "第1级/公开"),
        "Total number of applicants": ("度量类/计数统计", "第1级/公开"),
    },
    "巧克力棒评级": {
        "Company": ("属性类/名称标题", "第1级/公开"),
        "Company Location": ("属性类/地址位置", "第1级/公开"),
        "Review Date": ("状态类/时间标记", "第1级/公开"),
        "Country of Bean Origin": ("属性类/地址位置", "第1级/公开"),
        "Country of Production": ("属性类/地址位置", "第1级/公开"),
        "Cocoa Percent": ("度量类/比率比例", "第1级/公开"),
        "Rating": ("度量类/计量数值", "第1级/公开"),
    },

    # 教育数据 (education)
    "ted演讲数据集": {
        "comments": ("度量类/计数统计", "第1级/公开"),
        "description": ("属性类/描述文本", "第1级/公开"),
        "duration": ("度量类/时间度量", "第1级/公开"),
        "event": ("属性类/类别标签", "第1级/公开"),
        "film_date": ("状态类/时间标记", "第1级/公开"),
        "languages": ("度量类/计数统计", "第1级/公开"),
        "main_speaker": ("身份类/职业信息", "第2级/内部"),
        "name": ("属性类/名称标题", "第1级/公开"),
        "num_speaker": ("度量类/计数统计", "第1级/公开"),
        "published_date": ("状态类/时间标记", "第1级/公开"),
        "ratings": ("扩展类/扩展代码", "第1级/公开"),
        "related_talks": ("扩展类/扩展代码", "第1级/公开"),
        "speaker_occupation": ("身份类/职业信息", "第2级/内部"),
        "tags": ("属性类/技能标签", "第1级/公开"),
        "title": ("属性类/名称标题", "第1级/公开"),
        "url": ("属性类/描述文本", "第1级/公开"),
        "views": ("度量类/计数统计", "第1级/公开"),
    },
    "校园招聘": {
        "sl_no": ("度量类/序号排序", "第1级/公开"),
        "gender": ("身份类/人口统计", "第3级/敏感"),
        "ssc_p": ("度量类/计量数值", "第2级/内部"),
        "ssc_b": ("属性类/类别标签", "第1级/公开"),
        "hsc_p": ("度量类/计量数值", "第2级/内部"),
        "hsc_b": ("属性类/类别标签", "第1级/公开"),
        "hsc_s": ("属性类/类别标签", "第1级/公开"),
        "degree_p": ("度量类/计量数值", "第2级/内部"),
        "degree_t": ("属性类/类别标签", "第1级/公开"),
        "workex": ("状态类/二元标志", "第2级/内部"),
        "etest_p": ("度量类/计量数值", "第1级/公开"),
        "specialisation": ("属性类/类别标签", "第1级/公开"),
        "mba_p": ("度量类/计量数值", "第2级/内部"),
        "status": ("状态类/状态枚举", "第2级/内部"),
        "salary": ("度量类/计量数值", "第3级/敏感"),
    },

    # 交通数据 (transport)
    "共享单车数据集": {
        "Unnamed: 0": ("度量类/序号排序", "第1级/公开"),
        "orderid": ("ID类/主键ID", "第1级/公开"),
        "bikeid": ("结构类/产品代码", "第1级/公开"),
        "userid": ("ID类/主键ID", "第2级/内部"),
        "start_time": ("状态类/时间标记", "第2级/内部"),
        "start_location_x": ("属性类/地址位置", "第1级/公开"),
        "start_location_y": ("属性类/地址位置", "第1级/公开"),
        "end_time": ("状态类/时间标记", "第2级/内部"),
        "end_location_x": ("属性类/地址位置", "第1级/公开"),
        "end_location_y": ("属性类/地址位置", "第1级/公开"),
        "track": ("扩展类/扩展代码", "第1级/公开"),
        "start_year": ("状态类/时间标记", "第1级/公开"),
        "start_month": ("状态类/时间标记", "第1级/公开"),
        "start_day": ("度量类/时间度量", "第1级/公开"),
        "start_weekday": ("属性类/类别标签", "第1级/公开"),
        "start_hour": ("度量类/时间度量", "第1级/公开"),
        "weekend": ("状态类/二元标志", "第1级/公开"),
        "起始点距离": ("度量类/计量数值", "第1级/公开"),
    },
    "美国枪击案数据集": {
        "id": ("ID类/主键ID", "第1级/公开"),
        "name": ("属性类/名称标题", "第2级/内部"),
        "date": ("状态类/时间标记", "第2级/内部"),
        "manner_of_death": ("属性类/类别标签", "第2级/内部"),
        "armed": ("属性类/类别标签", "第1级/公开"),
        "age": ("身份类/人口统计", "第3级/敏感"),
        "gender": ("身份类/人口统计", "第3级/敏感"),
        "race": ("身份类/人口统计", "第3级/敏感"),
        "city": ("属性类/地址位置", "第1级/公开"),
        "state": ("属性类/地址位置", "第1级/公开"),
        "signs_of_mental_illness": ("状态类/二元标志", "第2级/内部"),
        "threat_level": ("属性类/类别标签", "第1级/公开"),
        "flee": ("属性类/类别标签", "第1级/公开"),
        "body_camera": ("状态类/二元标志", "第1级/公开"),
    },
    "航班分配问题数据集": {
        "Origin": ("属性类/地址位置", "第1级/公开"),
        "Destination": ("属性类/地址位置", "第1级/公开"),
        "Flight1": ("结构类/产品代码", "第1级/公开"),
        "Fligh2": ("结构类/产品代码", "第1级/公开"),
        "Flight3": ("结构类/产品代码", "第1级/公开"),
        "Class": ("属性类/类别标签", "第1级/公开"),
        "Fare": ("度量类/计量数值", "第1级/公开"),
        "TRD": ("度量类/时间度量", "第1级/公开"),
        "deptime": ("状态类/时间标记", "第1级/公开"),
        "arrtime": ("状态类/时间标记", "第1级/公开"),
        "delta_time": ("度量类/时间度量", "第1级/公开"),
        "Transfer times": ("度量类/计数统计", "第1级/公开"),
    },

    # 经济数据 (economic)
    "经济自由指数": {
        "CountryID": ("ID类/主键ID", "第1级/公开"),
        "Country Name": ("属性类/名称标题", "第1级/公开"),
        "WEBNAME": ("属性类/名称标题", "第1级/公开"),
        "Region": ("属性类/地址位置", "第1级/公开"),
        "World Rank": ("度量类/序号排序", "第1级/公开"),
        "Region Rank": ("度量类/序号排序", "第1级/公开"),
        "2019 Score": ("度量类/计量数值", "第1级/公开"),
        "Property Rights": ("度量类/计量数值", "第1级/公开"),
        "Judical Effectiveness": ("度量类/计量数值", "第1级/公开"),
        "Government Integrity": ("度量类/计量数值", "第1级/公开"),
        "Tax Burden": ("度量类/计量数值", "第1级/公开"),
        "Gov't Spending": ("度量类/计量数值", "第1级/公开"),
        "Fiscal Health": ("度量类/计量数值", "第1级/公开"),
        "Business Freedom": ("度量类/计量数值", "第1级/公开"),
        "Labor Freedom": ("度量类/计量数值", "第1级/公开"),
        "Monetary Freedom": ("度量类/计量数值", "第1级/公开"),
        "Trade Freedom": ("度量类/计量数值", "第1级/公开"),
        "Investment Freedom": ("度量类/计量数值", "第1级/公开"),
        "Financial Freedom": ("度量类/计量数值", "第1级/公开"),
        "Tariff Rate (%)": ("度量类/比率比例", "第1级/公开"),
    },
    "就业信息": {
        "年份": ("状态类/时间标记", "第1级/公开"),
        "城市名称": ("属性类/地址位置", "第1级/公开"),
        "unemploymentRate": ("度量类/比率比例", "第1级/公开"),
    },

    # 房地产数据 (real estate)
    "美国King County的房屋销售数据": {
        "id": ("ID类/主键ID", "第1级/公开"),
        "date": ("状态类/时间标记", "第1级/公开"),
        "price": ("度量类/计量数值", "第3级/敏感"),
        "bedrooms": ("度量类/计数统计", "第1级/公开"),
        "bathrooms": ("度量类/计数统计", "第1级/公开"),
        "sqft_living": ("度量类/计量数值", "第1级/公开"),
        "sqft_lot": ("度量类/计量数值", "第1级/公开"),
        "floors": ("度量类/计量数值", "第1级/公开"),
        "waterfront": ("状态类/二元标志", "第2级/内部"),
        "view": ("度量类/计量数值", "第1级/公开"),
        "condition": ("属性类/类别标签", "第1级/公开"),
        "grade": ("属性类/类别标签", "第1级/公开"),
        "sqft_above": ("度量类/计量数值", "第1级/公开"),
        "sqft_basement": ("度量类/计量数值", "第1级/公开"),
        "yr_built": ("度量类/时间度量", "第1级/公开"),
        "yr_renovated": ("度量类/时间度量", "第2级/内部"),
        "zipcode": ("结构类/标准代码", "第1级/公开"),
        "lat": ("属性类/地址位置", "第1级/公开"),
        "long": ("属性类/地址位置", "第1级/公开"),
        "sqft_living15": ("度量类/计量数值", "第1级/公开"),
    },

    # 新闻/假新闻数据 (news)
    "真实的假新闻": {
        "uuid": ("ID类/主键ID", "第1级/公开"),
        "ord_in_thread": ("度量类/序号排序", "第1级/公开"),
        "author": ("身份类/职业信息", "第2级/内部"),
        "published": ("状态类/时间标记", "第1级/公开"),
        "title": ("属性类/名称标题", "第1级/公开"),
        "text": ("属性类/描述文本", "第1级/公开"),
        "language": ("属性类/类别标签", "第1级/公开"),
        "crawled": ("状态类/时间标记", "第1级/公开"),
        "site_url": ("属性类/地址位置", "第1级/公开"),
        "country": ("属性类/地址位置", "第1级/公开"),
        "domain_rank": ("度量类/序号排序", "第1级/公开"),
        "thread_title": ("属性类/名称标题", "第1级/公开"),
        "spam_score": ("度量类/计量数值", "第1级/公开"),
        "main_img_url": ("属性类/描述文本", "第1级/公开"),
        "replies_count": ("度量类/计数统计", "第1级/公开"),
        "participants_count": ("度量类/计数统计", "第1级/公开"),
        "likes": ("度量类/计数统计", "第1级/公开"),
        "comments": ("度量类/计数统计", "第1级/公开"),
        "shares": ("度量类/计数统计", "第1级/公开"),
        "type": ("属性类/类别标签", "第1级/公开"),
    },

    # 葡萄酒评论数据 (wine)
    "葡萄酒评论": {
        "Unnamed: 0": ("度量类/序号排序", "第1级/公开"),
        "country": ("属性类/地址位置", "第1级/公开"),
        "description": ("属性类/描述文本", "第1级/公开"),
        "designation": ("属性类/名称标题", "第1级/公开"),
        "points": ("度量类/计量数值", "第1级/公开"),
        "price": ("度量类/计量数值", "第1级/公开"),
        "province": ("属性类/地址位置", "第1级/公开"),
        "region_1": ("属性类/地址位置", "第1级/公开"),
        "region_2": ("属性类/地址位置", "第1级/公开"),
        "variety": ("属性类/类别标签", "第1级/公开"),
        "winery": ("属性类/名称标题", "第1级/公开"),
    },
    
    # 学生考试数据 (education)
    "学生表现": {
        "gender": ("身份类/人口统计", "第3级/敏感"),
        "race/ethnicity": ("身份类/人口统计", "第3级/敏感"),
        "parental level of education": ("身份类/教育背景", "第3级/敏感"),
        "lunch": ("属性类/类别标签", "第1级/公开"),
        "test preparation course": ("属性类/技能标签", "第1级/公开"),
        "math score": ("度量类/计量数值", "第2级/内部"),
        "reading score": ("度量类/计量数值", "第2级/内部"),
        "writing score": ("度量类/计量数值", "第2级/内部"),
    },
    
    # 业务数据 business
    "业务数据": {
        "岗位id": ("ID类/主键ID", "第1级/公开"),
        "岗位名称": ("属性类/名称标题", "第1级/公开"),
        "岗位类别": ("属性类/类别标签", "第1级/公开"),
        "公司名称": ("属性类/名称标题", "第2级/内部"),
        "公司规模": ("属性类/类别标签", "第1级/公开"),
        "公司类型": ("属性类/类别标签", "第1级/公开"),
        "所属城市": ("属性类/地址位置", "第1级/公开"),
        "要求学历": ("身份类/教育背景", "第2级/内部"),
        "要求经历": ("身份类/职业信息", "第2级/内部"),
        "最低薪资": ("度量类/计量数值", "第3级/敏感"),
        "最高薪资": ("度量类/计量数值", "第3级/敏感"),
        "平均薪资": ("度量类/计量数值", "第3级/敏感"),
        "要求技能": ("属性类/技能标签", "第1级/公开"),
        "岗位描述": ("属性类/描述文本", "第1级/公开"),
        "岗位要求": ("属性类/描述文本", "第1级/公开"),
        "发布日期": ("状态类/时间标记", "第1级/公开"),
        "浏览量": ("度量类/计数统计", "第1级/公开"),
        "申请数": ("度量类/计数统计", "第2级/内部"),
    },
    
    # 课程数据 education
    "课程数据": {
        "课程id": ("ID类/主键ID", "第1级/公开"),
        "课程类别": ("属性类/类别标签", "第1级/公开"),
        "课程时长": ("度量类/时间度量", "第1级/公开"),
        "课程章节数": ("度量类/计数统计", "第1级/公开"),
        "报名人数": ("度量类/计数统计", "第2级/内部"),
        "完课率": ("度量类/比率比例", "第2级/内部"),
        "课程发布平台数量": ("度量类/计数统计", "第1级/公开"),
        "课程价格": ("度量类/计量数值", "第1级/公开"),
        "课程评价": ("属性类/描述文本", "第1级/公开"),
        "考试平均得分": ("度量类/计量数值", "第2级/内部"),
    },
    
    # 金融数据 financial
    "金融数据": {
        "条目id": ("ID类/主键ID", "第1级/公开"),
        "客户年龄": ("身份类/人口统计", "第3级/敏感"),
        "职业": ("身份类/职业信息", "第3级/敏感"),
        "婚姻情况": ("身份类/人口统计", "第3级/敏感"),
        "教育情况": ("身份类/教育背景", "第3级/敏感"),
        "信用卡是否违约": ("状态类/二元标志", "第3级/敏感"),
        "是否有房贷": ("状态类/二元标志", "第3级/敏感"),
        "联系方式": ("身份类/联系方式", "第3级/敏感"),
        "上一次联系的月份": ("状态类/时间标记", "第2级/内部"),
        "上一次联系的星期几": ("属性类/类别标签", "第1级/公开"),
        "上一次联系的时长（秒）": ("度量类/时间度量", "第2级/内部"),
        "活动期间联系客户的次数": ("度量类/计数统计", "第2级/内部"),
        "上一次与客户联系后的间隔天数": ("度量类/时间度量", "第2级/内部"),
        "在本次营销活动前，与客户联系的次数": ("度量类/计数统计", "第2级/内部"),
        "之前营销活动的结果": ("状态类/状态枚举", "第2级/内部"),
        "就业变动率": ("度量类/比率比例", "第1级/公开"),
        "消费者价格指数": ("度量类/计量数值", "第1级/公开"),
        "消费者信心指数": ("度量类/计量数值", "第1级/公开"),
        "银行同业拆借率 3个月利率": ("度量类/比率比例", "第1级/公开"),
        "雇员人数": ("度量类/计数统计", "第2级/内部"),
    },
    
    # 工业数据 industrial
    "工业设备": {
        "分类代码": ("结构类/分类代码", "第2级/内部"),
        "产品代码": ("结构类/产品代码", "第2级/内部"),
        "批次号": ("结构类/标准代码", "第1级/公开"),
        "序列号": ("度量类/序号排序", "第2级/内部"),
        "扩展数据代码": ("扩展类/扩展代码", "第1级/公开"),
    },
    
    # 医疗数据 medical
    "血糖医疗": {
        "怀孕次数": ("度量类/计数统计", "第3级/敏感"),
        "血糖水平": ("度量类/计量数值", "第2级/内部"),
        "血压": ("度量类/计量数值", "第2级/内部"),
        "皮肤厚度": ("度量类/计量数值", "第1级/公开"),
        "胰岛素": ("度量类/计量数值", "第2级/内部"),
        "BMI": ("度量类/计量数值", "第2级/内部"),
        "糖尿病遗传函数": ("度量类/计量数值", "第3级/敏感"),
        "年龄": ("身份类/人口统计", "第3级/敏感"),
        "是否有糖尿病": ("状态类/二元标志", "第3级/敏感"),
    },
}

def match_dataset(filename):
    """根据文件名匹配数据集标签映射"""
    # 精确匹配
    for key in ALL_FIELD_LABELS:
        if key in filename:
            return ALL_FIELD_LABELS[key]
    
    # 处理 _train.csv 格式
    if "_train.csv" in filename or "business_train" in filename:
        # 业务数据
        if "business" in filename.lower() or "岗位" in filename:
            return ALL_FIELD_LABELS.get("业务数据", {})
        # 课程数据
        elif "education" in filename.lower() or "课程" in filename:
            return ALL_FIELD_LABELS.get("课程数据", {})
        # 金融数据
        elif "financial" in filename.lower() or "金融" in filename:
            return ALL_FIELD_LABELS.get("金融数据", {})
        # 工业数据
        elif "industrial" in filename.lower() or "工业" in filename:
            return ALL_FIELD_LABELS.get("工业设备", {})
        # 医疗数据
        elif "medical" in filename.lower() or "医疗" in filename:
            return ALL_FIELD_LABELS.get("血糖医疗", {})
    
    # 处理 _extra_test.csv 后缀
    for key in ALL_FIELD_LABELS:
        # 学生表现
        if "学生" in key and ("学生" in filename or "Students" in filename or "Performance" in filename):
            if "考试" in key:
                return ALL_FIELD_LABELS[key]
        # 巧克力棒
        if "巧克力" in key and ("巧克力" in filename or "chocolate" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 贷款
        if "贷款" in key and ("贷款" in filename or "loan" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 餐厅
        if "餐厅" in key and ("餐厅" in filename or "restaurant" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 心脏病
        if "心脏病" in key and ("心脏病" in filename or "heart" in filename.lower() or "stroke" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 经济自由
        if "经济自由" in key and ("经济自由" in filename or "freedom" in filename.lower() or "economic" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 业务 business
        if "天猫" in key and ("天猫" in filename or "business" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 课程 education
        if "课程" in key and ("education" in filename.lower() or "课程" in filename):
            return ALL_FIELD_LABELS[key]
        # 金融 financial  
        if ("金融" in key or "信用卡" in key) and ("financial" in filename.lower() or "金融" in filename or "credit" in filename.lower()):
            return ALL_FIELD_LABELS[key]
        # 工业 industrial
        if "工业" in key and ("industrial" in filename.lower() or "工业" in filename):
            return ALL_FIELD_LABELS[key]
        # 医疗 medical
        if ("乳腺癌" in key or "医疗" in key) and ("medical" in filename.lower() or "医疗" in filename or "cancer" in filename.lower()):
            return ALL_FIELD_LABELS[key]
    return None

def get_industry(filename):
    """根据文件名推断行业"""
    filename_lower = filename.lower()
    if any(k in filename_lower for k in ["银行", "信用卡", "保险", "贷款", "薪酬", "financial", "credit", "insurance", "loan", "salary", "bank"]):
        return "金融"
    elif any(k in filename_lower for k in ["医疗", "健康", "心脏病", "自杀", "medical", "health", "cancer", "hospital"]):
        return "医疗"
    elif any(k in filename_lower for k in ["教育", "学生", "ted", "education", "学生表现"]):
        return "教育"
    elif any(k in filename_lower for k in ["工业", "设备", "发电", "industrial"]):
        return "工业"
    elif any(k in filename_lower for k in ["单车", "航班", "枪击", "bike", "flight", "shooting"]):
        return "交通"
    elif any(k in filename_lower for k in ["房屋", "房价", "房产", "house", "real estate"]):
        return "房地产"
    elif any(k in filename_lower for k in ["经济", "就业", "economic", "employment", "freedom"]):
        return "经济"
    elif any(k in filename_lower for k in ["新闻", "fake", "news"]):
        return "新闻"
    elif any(k in filename_lower for k in ["葡萄酒", "酒", "wine"]):
        return "食品饮料"
    elif any(k in filename_lower for k in ["巧克力", "chocolate"]):
        return "食品饮料"
    elif any(k in filename_lower for k in ["美妆", "服装", "餐厅", "business", "ecommerce", "fashion", "restaurant"]):
        return "商业"
    elif any(k in filename_lower for k in ["工业", "霍乱"]):
        return "公共卫生"
    return "商业"

def get_samples(series, n=5):
    """获取列的样本值"""
    valid = series.dropna().astype(str)
    valid = valid[valid.str.len() < 80]
    unique = valid.unique()[:n]
    count = len(valid.unique())
    if len(unique) == 0:
        return "(无样本)"
    samples_str = ', '.join(unique)
    return f"{samples_str} ... (共{count}个)"

def build_classification_prompt(industry, col, samples):
    """构建分类提示"""
    labels_text = '\n'.join([f"- {l}" for l in CLASSIFICATION_LABELS])
    return f"""你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段属于哪一类。

行业：{industry}
字段名：{col}
样本值示例：{samples}

请从以下分类标签中选择最合适的一个（只输出标签路径，不要其他内容）：
{labels_text}

答案："""

def process_csv_to_samples(filepath, val_ratio=0.15):
    """处理单个CSV文件，返回样本列表"""
    filename = os.path.basename(filepath)
    label_map = match_dataset(filename)
    industry = get_industry(filename)
    
    if label_map is None:
        print(f"  警告: 未找到标签映射: {filename}")
        return []
    
    try:
        # 尝试不同编码
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except:
                continue
        
        samples = []
        matched = 0
        unmatched = 0
        
        for col in df.columns:
            samples_text = get_samples(df[col])
            if col in label_map:
                classification, grading = label_map[col]
                sample = {
                    "instruction": build_classification_prompt(industry, col, samples_text),
                    "input": "",
                    "output": classification
                }
                samples.append(sample)
                matched += 1
            else:
                unmatched += 1
        
        print(f"  ✓ 匹配 {matched} 个字段, 未匹配 {unmatched} 个")
        return samples
    except Exception as e:
        print(f"  错误: {e}")
        return []

def main():
    print("=" * 60)
    print("批量处理CSV数据集")
    print("=" * 60)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    print(f"\n找到 {len(csv_files)} 个CSV文件\n")
    
    all_samples = []
    
    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        print(f"处理: {filename}")
        samples = process_csv_to_samples(filepath)
        all_samples.extend(samples)
    
    print(f"\n总计生成 {len(all_samples)} 条样本")
    
    # 打乱并划分训练集和验证集
    random.seed(42)
    random.shuffle(all_samples)
    
    val_count = max(30, int(len(all_samples) * 0.15))  # 至少30条验证
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]
    
    # 保存文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_file = os.path.join(OUTPUT_DIR, "val.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n训练集: {len(train_samples)} 条 -> {train_file}")
    print(f"验证集: {len(val_samples)} 条 -> {val_file}")
    
    # 统计分类分布
    from collections import Counter
    classifications = [s["output"] for s in all_samples]
    print("\n分类分布 (Top 15):")
    for label, count in Counter(classifications).most_common(15):
        print(f"  {label}: {count}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
