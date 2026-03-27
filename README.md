# DeepSeek 数据分类分级系统

基于 DeepSeek-7B 大语言模型的数据自动化分类分级系统，支持字段级分类分级推理。

## 项目结构

```
deepseek_project/
├── README.md                   # 项目说明
├── requirements.txt            # Python 依赖
├── scripts/                    # 核心脚本
│   ├── api_server.py          # FastAPI 推理服务 (主入口)
│   ├── grading_rules.py       # 分类→分级规则映射
│   ├── knowledge_base_loader.py # 知识库规则加载
│   ├── batch_convert_datasets.py # 训练数据格式转换
│   ├── train_model.py         # LoRA 模型微调训练
│   ├── evaluate_model.py      # 模型评估
│   └── compare_models.py      # 基础模型 vs 微调模型对比
├── models/                    # 模型文件
│   ├── deepseek-llm-7b-chat/  # DeepSeek-7B-Chat 基座模型
│   └── tokenizer_fix.py       # 分词器修复 (支持中文)
├── outputs/finetuned/         # 微调后的 LoRA 权重
│   └── adapter_model.safetensors
├── data/                      # 数据目录
│   ├── knowledge_base/        # 知识库规则
│   │   ├── general_rules.json     # 通用分类规则
│   │   └── industry_rules.json    # 行业特定规则
│   ├── labels/                # 标签定义
│   │   ├── classification_schema.json  # 24类分类标签
│   │   └── grading_rules.json        # 分级规则
│   ├── sft/                   # SFT 训练数据
│   │   ├── train.jsonl        # 训练集
│   │   └── val.jsonl          # 验证集
│   ├── raw/                   # 原始训练数据 (CSV)
│   └── new/                   # 生成的测试数据集
├── results/                    # 评估与预测结果
│   ├── csv_predict/           # CSV 文件预测结果
│   └── *.json                 # 评估报告
└── frontend/                  # Vue.js 前端
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.js
        ├── App.vue
        ├── router/index.js    # 路由配置
        ├── api/               # API 客户端
        │   ├── index.js
        │   └── knowledge.js
        ├── views/             # 页面组件
        │   ├── DashboardView.vue       # 仪表盘
        │   ├── ClassificationView.vue  # 分类分级
        │   ├── KnowledgeBaseView.vue   # 知识库管理
        │   └── VisualizationView.vue    # 数据可视化
        └── assets/styles/main.scss
```

---

## 分类标签体系

### 24类分类标签

| 大类 | 标签 | 说明 |
|------|------|------|
| **结构标识类** | ID类/主键ID | 唯一标识记录的主键或ID字段 |
|  | 结构类/分类代码 | 用于分类标识的编码字段 |
|  | 结构类/产品代码 | 产品或商品相关的编码字段 |
|  | 结构类/企业代码 | 企业或组织相关的编码字段 |
|  | 结构类/标准代码 | 遵循标准的代码字段 |
| **属性描述类** | 属性类/名称标题 | 实体名称或标题字段 |
|  | 属性类/类别标签 | 分类或标签信息字段 |
|  | 属性类/描述文本 | 详细描述内容字段 |
|  | 属性类/技能标签 | 技能或能力标识字段 |
|  | 属性类/地址位置 | 地理位置信息字段 |
| **数值度量类** | 度量类/计量数值 | 连续或离散数值测量字段 |
|  | 度量类/计数统计 | 数量统计类数值字段 |
|  | 度量类/比率比例 | 百分比或比率字段 |
|  | 度量类/时间度量 | 时间相关的数值字段 |
|  | 度量类/序号排序 | 顺序编号字段 |
| **身份特征类** | 身份类/人口统计 | 人口统计学特征字段 |
|  | 身份类/联系方式 | 联系方式信息字段 |
|  | 身份类/教育背景 | 教育相关信息字段 |
|  | 身份类/职业信息 | 职业相关特征字段 |
| **状态标记类** | 状态类/二元标志 | 二值状态标识字段 |
|  | 状态类/状态枚举 | 多状态枚举值字段 |
|  | 状态类/时间标记 | 时间点或时间戳字段 |
| **扩展数据类** | 扩展类/扩展代码 | 自定义扩展编码字段 |
|  | 扩展类/其他字段 | 其他未分类字段 |

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
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Node.js 依赖 (前端)
cd frontend && npm install
```

### 2. 模型准备

从 HuggingFace 下载 DeepSeek-7B-Chat 模型到 `models/deepseek-llm-7b-chat` 目录。

```bash
# 或者使用 huggingface-cli
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat
```

### 3. 启动 API 服务

```bash
python scripts/api_server.py --port 8001
```

服务启动后会自动在后台加载模型，首次加载需要等待 1-3 分钟。

### 4. 启动前端

```bash
cd frontend
npm run dev
```

访问 http://localhost:3000 即可使用系统。

---

## 核心功能

### 字段分类分级

- **单字段分类**: 对单个字段进行分类和分级
- **批量分类**: 批量处理多个字段
- **文件上传**: 上传 CSV/Excel 文件，自动分析所有字段
- **流式返回**: 支持实时进度展示

### 知识库管理

- **通用规则**: 跨行业的通用字段分类规则
- **行业规则**: 特定行业的专业规则
- **规则冲突检测**: 自动检测规则冲突

### 模型评估

```bash
# 评估微调模型
python scripts/evaluate_model.py --lora

# 对比基础模型 vs 微调模型
python scripts/compare_models.py
```

---

## API 接口

### 基础信息

- **Base URL**: `http://localhost:8001`
- **Content-Type**: `application/json`

### 接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| GET | `/labels` | 获取分类分级标签 |
| GET | `/industries` | 获取支持的行业列表 |
| GET | `/knowledge/stats` | 知识库统计 |
| GET | `/knowledge/rules` | 获取知识库规则 |
| POST | `/knowledge/save` | 保存规则 |
| POST | `/knowledge/reload` | 重新加载知识库 |
| POST | `/knowledge/conflicts` | 检测规则冲突 |
| POST | `/classify` | 单字段分类分级 |
| POST | `/classify/batch` | 批量分类分级 |
| POST | `/classify/file` | 上传文件分类分级 |
| POST | `/classify/file/stream` | 流式文件分类分级 |

### 调用示例

```bash
# 单字段分类
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{
    "field_name": "customer_age",
    "industry": "金融",
    "samples": ["25", "30", "35", "28", "42"]
  }'

# 文件分类
curl -X POST http://localhost:8001/classify/file \
  -F "file=@data.csv" \
  -F "industry=金融"

# 流式文件分类 (SSE)
curl -N -X POST http://localhost:8001/classify/file/stream \
  -F "file=@data.csv" \
  -F "industry=金融"
```

### 响应示例

```json
{
  "success": true,
  "message": "分类分级完成",
  "data": {
    "field_name": "customer_age",
    "classification": "身份类/人口统计",
    "grading": "第3级/敏感",
    "confidence": 1.0,
    "warnings": []
  }
}
```

---

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `api_server.py` | FastAPI 推理服务主入口 |
| `grading_rules.py` | 分类→分级映射规则 |
| `knowledge_base_loader.py` | 知识库规则加载/保存 |
| `batch_convert_datasets.py` | 批量 CSV 转 SFT JSONL 格式 |
| `train_model.py` | LoRA 微调训练 |
| `evaluate_model.py` | 模型效果评估 |
| `compare_models.py` | 基础模型 vs 微调模型对比 |

---

## 技术栈

### 后端
- **模型**: DeepSeek-LLM-7B-Chat
- **微调**: LoRA (PEFT)
- **框架**: FastAPI + Uvicorn
- **依赖**: transformers, peft, torch, pandas

### 前端
- **框架**: Vue 3 + Composition API
- **UI**: Element Plus
- **图表**: ECharts
- **构建**: Vite
- **状态管理**: Pinia

---

## 性能指标

基于当前数据集的评估结果：

| 指标 | 数值 |
|------|------|
| 准确率 | ~82% |
| F1分数 (Macro) | ~0.78 |
| F1分数 (Weighted) | ~0.82 |
| 训练时间 | <2小时 (GPU) |
| 可训练参数 | ~0.1% (LoRA) |

---

## 部署说明

### 开发模式

```bash
# 终端1: 启动后端
python scripts/api_server.py --port 8001

# 终端2: 启动前端
cd frontend && npm run dev
```

### 生产部署

```bash
# 构建前端
cd frontend && npm run build

# 使用 nginx 或其他 Web 服务器托管 dist 目录
```

---

## 参考标准

- GB/T 43697-2024 《数据安全技术 数据分类分级规则》
