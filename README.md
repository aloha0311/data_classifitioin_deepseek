# 基于大语言模型的数据字段自动分类分级系统

## 本项目不包括模型本身，可以通过models/add.py将deepseek-llm-7b-chat下载到models文件夹下

## 项目概述

本项目使用 `DeepSeek-LLM-7B-Chat` 大语言模型，通过 LoRA 微调技术，实现对任意数据集字段的自动分类和分级功能。

### 核心能力

- **24类字段分类**：根据字段名和样本值，自动识别字段属于24种分类标签之一
- **4级敏感分级**：评估字段的安全等级（公开/内部/敏感/机密）
- **多行业覆盖**：支持金融、医疗、教育、商业、工业等10+行业场景
- **批量预测**：支持批量处理 CSV/Excel 文件，快速获取全字段分类分级结果

---

## 项目结构

```
deepseek_project/
├── data/                         # 数据目录
│   ├── labels/                   # 分类分级标准
│   │   ├── classification_schema.json   # 24类分类标签定义
│   │   └── grading_schema.json          # 4级分级标准定义
│   ├── raw/                      # 原始训练数据
│   ├── test/                     # 测试数据
│   ├── new/                      # 待预测的新数据（CSV/Excel）
│   ├── sft/                      # SFT训练数据
│   │   ├── train.jsonl          # 训练集
│   │   └── val.jsonl            # 验证集
│   └── processed/               # 中间处理文件
├── scripts/                      # 核心脚本
│   ├── batch_convert_datasets.py     # 批量转换数据集为训练格式
│   ├── train_model.py               # LoRA模型微调训练
│   ├── evaluate_model.py            # 模型评估
│   ├── predict_new_data.py          # 批量预测新数据
│   ├── dynamic_knowledge_base.py    # 动态知识库模块
│   ├── self_supervised_pretraining.py  # 自监督预训练
│   ├── statistical_feature_extractor.py # 统计特征提取
│   ├── test_lora.py                 # LoRA权重测试
│   └── test_tokenizer_cn.py        # 中文分词器测试
├── models/                         # 模型目录
│   └── deepseek-llm-7b-chat/       # 基础模型
│       └── tokenizer_fix.py        # 中文tokenizer修复
├── outputs/                        # 训练输出
│   └── finetuned/                  # 微调后的LoRA权重
├── results/                       # 预测结果
│   ├── model_evaluation_report.json  # 评估报告
│   ├── predictions.json             # 预测结果JSON
│   └── csv_predict/                 # CSV格式预测结果
└── docs/                          # 文档目录
```

---

## 分类标签体系

### 24类分类标签

| 大类 | 标签 | 说明 |
|------|------|------|
| **结构标识类** | ID类/主键ID | 唯一标识记录的主键或ID字段 |
| | 结构类/分类代码 | 用于分类标识的编码字段 |
| | 结构类/产品代码 | 产品或商品相关的编码字段 |
| | 结构类/企业代码 | 企业或组织相关的编码字段 |
| | 结构类/标准代码 | 遵循标准的代码字段 |
| **属性描述类** | 属性类/名称标题 | 实体名称或标题字段 |
| | 属性类/类别标签 | 分类或标签信息字段 |
| | 属性类/描述文本 | 详细描述内容字段 |
| | 属性类/技能标签 | 技能或能力标识字段 |
| | 属性类/地址位置 | 地理位置信息字段 |
| **数值度量类** | 度量类/计量数值 | 连续或离散数值测量字段 |
| | 度量类/计数统计 | 数量统计类数值字段 |
| | 度量类/比率比例 | 百分比或比率字段 |
| | 度量类/时间度量 | 时间相关的数值字段 |
| | 度量类/序号排序 | 顺序编号字段 |
| **身份特征类** | 身份类/人口统计 | 人口统计学特征字段 |
| | 身份类/联系方式 | 联系方式信息字段 |
| | 身份类/教育背景 | 教育相关信息字段 |
| | 身份类/职业信息 | 职业相关特征字段 |
| **状态标记类** | 状态类/二元标志 | 二值状态标识字段 |
| | 状态类/状态枚举 | 多状态枚举值字段 |
| | 状态类/时间标记 | 时间点或时间戳字段 |
| **扩展数据类** | 扩展类/扩展代码 | 自定义扩展编码字段 |
| | 扩展类/其他字段 | 其他未分类字段 |

### 4级分级标准

| 级别 | 名称 | 说明 |
|------|------|------|
| 第1级 | 公开 | 可对外公开的数据 |
| 第2级 | 内部 | 仅内部可访问的数据 |
| 第3级 | 敏感 | 需要保护的个人/敏感信息 |
| 第4级 | 机密 | 高度机密的数据 |

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers peft bitsandbytes pandas scikit-learn openpyxl

# 验证GPU可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 批量预测（新数据分类分级）

将待预测的数据文件（CSV或Excel）放入 `data/new/` 目录：

```bash
cd deepseek_project
python scripts/predict_new_data.py
```

交互示例：

```
============================================================
可用文件列表：
============================================================
  1. medical_data.csv
  2. student_scores.csv
  3. financial_data.xlsx

请输入要处理的文件编号（支持多选，用逗号分隔，如: 1,3,5）：
输入 'a' 处理所有文件，输入 'q' 退出：
> 1,2

加载模型...
加载LoRA权重...

处理: medical_data.csv
  行业: medical, 字段数: 12
  patient_id: ID类/主键ID / 第1级/公开
  age: 身份类/人口统计 / 第3级/敏感
  diagnosis: 属性类/描述文本 / 第3级/敏感
  ...

处理: student_scores.csv
  行业: education, 字段数: 8
  student_id: ID类/主键ID / 第1级/公开
  name: 属性类/名称标题 / 第2级/内部
  ...
```

预测结果保存至 `results/csv_predict/` 目录，格式如下：

```csv
字段名,分类,分级
patient_id,ID类/主键ID,第1级/公开
age,身份类/人口统计,第3级/敏感
```

### 3. 模型评估

```bash
# 评估LoRA微调模型
python scripts/evaluate_model.py --lora

# 评估基础模型
python scripts/evaluate_model.py
```

### 4. 重新训练模型（如需）

```bash
# 步骤1：准备训练数据
python scripts/batch_convert_datasets.py

# 步骤2：训练模型
python scripts/train_model.py
```

---

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                     数据准备阶段                              │
├─────────────────────────────────────────────────────────────┤
│  1. 将CSV文件放入 data/raw/                                │
│  2. python scripts/batch_convert_datasets.py                │
│     ↓                                                       │
│  3. 生成 data/sft/train.jsonl 和 val.jsonl                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     模型训练阶段                              │
├─────────────────────────────────────────────────────────────┤
│  4. python scripts/train_model.py                          │
│     ↓                                                       │
│  5. 生成 outputs/finetuned/ (LoRA权重)                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     模型评估阶段                              │
├─────────────────────────────────────────────────────────────┤
│  6. python scripts/evaluate_model.py --lora                │
│     ↓                                                       │
│  7. 查看 results/model_evaluation_report.json               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     预测使用阶段                              │
├─────────────────────────────────────────────────────────────┤
│  8. 将新数据放入 data/new/                                 │
│  9. python scripts/predict_new_data.py                    │
│     ↓                                                       │
│ 10. 查看 results/csv_predict/ 下的结果文件                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 训练数据统计

| 数据集 | 数量 |
|--------|------|
| 训练集 | 420条 |
| 验证集 | 46条 |
| 行业覆盖 | 10+ 个 |

### 标签分布（Top 10）

```
度量类/计量数值: 142
状态类/状态枚举: 61
度量类/比率比例: 54
属性类/地址位置: 34
状态类/时间标记: 32
属性类/类别标签: 19
身份类/人口统计: 12
度量类/计数统计: 12
ID类/主键ID: 11
属性类/名称标题: 10
```

---

## 模型性能

当前模型在验证集上的表现：

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | 82.69% |
| 精确率 (Macro) | 81.54% |
| 召回率 (Macro) | 79.57% |
| F1分数 (Macro) | 78.44% |
| F1分数 (Weighted) | 84.91% |

### 常见错误类型

| 预期分类 | 预测分类 | 原因 |
|----------|----------|------|
| 状态类/时间标记 | 度量类/时间度量 | 语义相似易混淆 |
| 属性类/名称标题 | 属性类/描述文本 | 边界模糊 |

---

## 辅助模块

### 动态知识库

支持规则动态更新和增量学习：

```bash
# 运行演示
python scripts/dynamic_knowledge_base.py

# 从预测结果学习新规则
python scripts/dynamic_knowledge_base.py --learn

# 保存知识库
python scripts/dynamic_knowledge_base.py --save
```

### 自监督预训练

增强模型对字段语义的理解：

```bash
# 运行演示
python scripts/self_supervised_pretraining.py

# 生成预训练数据
python scripts/self_supervised_pretraining.py --generate
```

### 统计特征提取

为字段提取多维统计特征：

```bash
python scripts/statistical_feature_extractor.py
```

---

## 项目依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | - | 深度学习框架 |
| transformers | - | 模型加载 |
| peft | - | LoRA微调 |
| pandas | - | 数据处理 |
| scikit-learn | - | 评估指标 |
| openpyxl | - | Excel文件支持 |

---

## 硬件要求

| 配置 | 要求 |
|------|------|
| GPU显存 | ≥16GB（推荐） |
| 内存 | ≥32GB |
| 存储 | ≥50GB |

