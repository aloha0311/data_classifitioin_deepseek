# DeepSeek 数据分类分级系统

基于 DeepSeek-7B 大语言模型的数据自动化分类分级系统，支持字段级分类分级推理。

## 项目结构

```
deepseek_project/
├── docs/                    # 项目文档
│   ├── 项目过程文档.md       # 开发过程文档
│   ├── 算法文档.md          # 算法原理说明
│   └── 标注规范文档.md       # 数据标注规范
├── scripts/                 # 核心脚本
│   ├── train_model.py       # 模型训练脚本
│   ├── evaluate_model.py    # 模型评估脚本
│   ├── predict_new_data.py  # 新数据预测脚本
│   ├── compare_models.py    # 模型对比评估脚本
│   ├── enhanced_feature_extractor.py  # 增强特征提取
│   ├── enhanced_knowledge_base.py     # 增强知识库
│   ├── api_server.py        # FastAPI推理服务
│   └── split_dataset.py     # 数据集划分脚本
├── models/                  # 模型文件
│   ├── deepseek-llm-7b-chat/
│   └── tokenizer_fix.py     # 分词器修复
├── data/                    # 数据目录
│   ├── sft/                 # SFT训练数据
│   ├── raw/                 # 原始数据
│   ├── labels/              # 标签定义
│   └── knowledge_base.json  # 知识库
├── outputs/                 # 输出目录
│   └── finetuned/           # 微调模型输出
├── results/                 # 评估结果
├── frontend/                # 前端项目
│   ├── src/
│   │   ├── views/           # 页面组件
│   │   ├── router/          # 路由配置
│   │   └── api/             # API接口
│   └── package.json
└── docker-compose.yml       # Docker部署配置
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
# 安装Python依赖
pip install -r requirements.txt

# 安装Node.js依赖 (前端)
cd frontend && npm install
```

### 2. 模型准备

下载 DeepSeek-7B-Chat 模型到 `models/deepseek-llm-7b-chat` 目录。

### 3. 启动API服务

```bash
# 方式一：直接运行
python scripts/api_server.py --port 8001

# 方式二：使用Docker
docker-compose up api
```

### 4. 启动前端

```bash
cd frontend
npm run dev
```

访问 http://localhost:3000 即可使用系统。

## 核心功能

### 模型对比评估

```bash
python scripts/compare_models.py
```

该脚本将评估基础模型和微调模型的性能差异，生成详细的对比报告。

### 增强特征提取

```bash
python scripts/enhanced_feature_extractor.py data/raw/your_file.csv
```

提取字段的统计特征和语义特征，为模型提供更丰富的输入信息。

### 知识库管理

```bash
# 初始化知识库
python scripts/enhanced_knowledge_base.py --save

# 运行演示
python scripts/enhanced_knowledge_base.py
```

## API接口

### 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

### 接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/labels` | 获取分类分级标签 |
| POST | `/classify` | 单字段分类分级 |
| POST | `/classify/batch` | 批量分类分级 |
| POST | `/classify/file` | 上传文件分类分级 |

### 调用示例

```bash
# 单字段分类
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "customer_age",
    "industry": "金融",
    "samples": ["25", "30", "35", "28", "42"]
  }'

# 文件分类
curl -X POST http://localhost:8000/classify/file \
  -F "file=@data.csv" \
  -F "industry=金融"
```

## 分类分级体系

### 分类标签（24类）

| 一级分类 | 二级分类 |
|----------|----------|
| ID类 | 主键ID |
| 结构类 | 分类代码、产品代码、企业代码、标准代码 |
| 属性类 | 名称标题、类别标签、描述文本、技能标签、地址位置 |
| 度量类 | 计量数值、计数统计、比率比例、时间度量、序号排序 |
| 身份类 | 人口统计、联系方式、教育背景、职业信息 |
| 状态类 | 二元标志、状态枚举、时间标记 |
| 扩展类 | 扩展代码、其他字段 |

### 分级标签（4级）

| 级别 | 名称 | 说明 |
|------|------|------|
| 第1级 | 公开 | 可向公众公开的数据 |
| 第2级 | 内部 | 仅限内部人员访问 |
| 第3级 | 敏感 | 涉及个人/敏感信息 |
| 第4级 | 机密 | 高度敏感的个人信息 |

## 技术栈

### 后端
- **模型**: DeepSeek-LLM-7B-Chat
- **微调**: LoRA (PEFT)
- **框架**: FastAPI + Uvicorn
- **依赖**: transformers, peft, torch

### 前端
- **框架**: Vue 3 + Composition API
- **UI**: Element Plus
- **图表**: ECharts
- **构建**: Vite

## 开发指南

### 添加新行业规则

```python
from scripts.enhanced_knowledge_base import EnhancedKnowledgeBase

kb = EnhancedKnowledgeBase()
kb.add_rule(
    field_name="field_name",
    category="分类标签",
    grading="分级标签",
    description="规则描述",
    industry="行业名称"
)
kb.save()
```

### 自定义特征提取

```python
from scripts.enhanced_feature_extractor import EnhancedFeatureExtractor
import pandas as pd

extractor = EnhancedFeatureExtractor()
df = pd.read_csv("your_data.csv")
features = extractor.extract_all_features(df)

for col, feat in features.items():
    print(f"{col}: {feat.data_type}, {feat.semantic_hints}")
```

## 部署说明

### Docker部署

```bash
# 启动所有服务
docker-compose up -d

# 仅启动API服务
docker-compose up -d api

# 仅启动前端
docker-compose up -d frontend
```

### GPU支持

确保Docker配置了NVIDIA Container Toolkit：
```bash
docker run --gpus all -p 8000:8000 deepseek-api
```

## 性能指标

基于当前数据集（420条训练样本，46条验证样本，50条测试样本）：

- **准确率**: 82.69%
- **F1分数(Macro)**: ~0.78
- **F1分数(Weighted)**: ~0.82
- **训练时间**: <2小时
- **可训练参数**: ~0.1%（LoRA）


## 参考标准

- GB/T 43697-2024 《数据安全技术 数据分类分级规则》
