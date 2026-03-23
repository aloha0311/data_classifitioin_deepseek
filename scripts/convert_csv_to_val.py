#!/usr/bin/env python3
"""
将4个CSV测试集转换为带标签的val.jsonl格式
"""
import os
import json
import pandas as pd
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/new")
OUTPUT_FILE = os.path.join(BASE_DIR, "data/sft/val.jsonl")

CLASSIFICATION_LABELS = [
    "ID类/主键ID", "结构类/分类代码", "结构类/产品代码", "结构类/企业代码", "结构类/标准代码",
    "属性类/名称标题", "属性类/类别标签", "属性类/描述文本", "属性类/技能标签", "属性类/地址位置",
    "度量类/计量数值", "度量类/计数统计", "度量类/比率比例", "度量类/时间度量", "度量类/序号排序",
    "身份类/人口统计", "身份类/联系方式", "身份类/教育背景", "身份类/职业信息",
    "状态类/二元标志", "状态类/状态枚举", "状态类/时间标记",
    "扩展类/扩展代码", "扩展类/其他字段"
]

GRADING_LABELS = ["第1级/公开", "第2级/内部", "第3级/敏感", "第4级/机密"]

# 字段分类分级映射
FIELD_LABELS = {
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
        "是否为男士专用": ("属性类/技能标签", "第1级/公开"),
        "销售额": ("度量类/计量数值", "第2级/内部"),
        "day": ("度量类/时间度量", "第1级/公开"),
    },
    # 学生考试表现 (education)
    "学生": {
        "gender": ("身份类/人口统计", "第3级/敏感"),
        "race/ethnicity": ("身份类/人口统计", "第3级/敏感"),
        "parental level of education": ("身份类/教育背景", "第3级/敏感"),
        "lunch": ("属性类/描述文本", "第1级/公开"),
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
        "标识": ("状态类/二元标志", "第1级/公开"),
        "标签六": ("扩展类/其他字段", "第1级/公开"),
        "标签七": ("扩展类/其他字段", "第1级/公开"),
        "标签八": ("扩展类/其他字段", "第1级/公开"),
        "标签九": ("扩展类/其他字段", "第1级/公开"),
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
    }
}

def get_industry(filename):
    """根据文件名推断行业"""
    if "乳腺癌" in filename:
        return "medical"
    elif "天猫" in filename or "美妆" in filename:
        return "business"
    elif "学生" in filename:
        return "education"
    elif "工业" in filename:
        return "industrial"
    return "business"

def get_label_mapping(filename):
    """获取文件名对应的标签映射"""
    if "乳腺癌" in filename:
        return FIELD_LABELS["乳腺癌"]
    elif "天猫" in filename or "美妆" in filename:
        return FIELD_LABELS["天猫"]
    elif "学生" in filename:
        return FIELD_LABELS["学生"]
    elif "工业" in filename:
        return FIELD_LABELS["工业"]
    return {}

def get_samples(series, n=5):
    """获取列的样本值"""
    valid = series.dropna().astype(str)
    valid = valid[valid.str.len() < 80]
    unique = valid.unique()[:n]
    count = len(valid.unique())
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

def build_classification_sample(industry, col, samples, output):
    """构建分类样本"""
    prompt = build_classification_prompt(industry, col, samples)
    return {
        "instruction": prompt,
        "input": "",
        "output": output
    }

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_samples = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"处理: {filename}")
        
        df = pd.read_csv(filepath, encoding='utf-8')
        industry = get_industry(filename)
        label_map = get_label_mapping(filename)
        
        for col in df.columns:
            samples = get_samples(df[col])
            if col in label_map:
                classification, grading = label_map[col]
                sample = build_classification_sample(industry, col, samples, classification)
                all_samples.append(sample)
                print(f"  ✓ {col}: {classification}")
            else:
                print(f"  ✗ {col}: 未找到标签映射")
    
    # 保存为JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n已生成 {len(all_samples)} 条样本: {OUTPUT_FILE}")
    
    # 统计分类分布
    from collections import Counter
    classifications = [s["output"] for s in all_samples]
    print("\n分类分布:")
    for label, count in Counter(classifications).most_common():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
