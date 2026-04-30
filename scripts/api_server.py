#!/usr/bin/env python3
"""
FastAPI推理服务
提供RESTful API接口，支持字段分类分级推理
"""
import os
import sys
import json
import torch
import pandas as pd
from io import BytesIO
from typing import List, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import asyncio
import threading

import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer_fix import get_chinese_tokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from scripts.grading_rules import (
    get_classification_labels,
    get_grading_labels,
    infer_grading,
    load_grading_rules
)
from scripts.knowledge_base_loader import (
    get_classification_from_rules,
    get_grading_from_rules,
    get_knowledge_base_stats,
    reload_knowledge_base,
    load_general_rules,
    load_industry_rules,
    save_general_rules,
    save_industry_rules
)
from scripts.similarity_calculator import FieldSimilarityCalculator, find_similar_in_knowledge_base
from scripts.embedding_extractor import FieldSemanticModeler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")

# 从配置加载标签
CLASSIFICATION_LABELS = get_classification_labels()
GRADING_LABELS = get_grading_labels()

# 加载分级规则
_grade_rules = load_grading_rules()
CLASSIFICATION_TO_GRADING = _grade_rules.get("classification_to_grading", {})

# 初始化相似度计算器和语义建模器
similarity_calculator = FieldSimilarityCalculator()
semantic_modeler = FieldSemanticModeler()

# ============= 分类规则映射 =============
# 用于根据字段名模式快速推断分类
CLASSIFICATION_RULES = [
    # ID类
    (r'^(id|uuid|guid)$', 'ID类/主键ID'),
    (r'_id$|_id\b', 'ID类/主键ID'),
    (r'^pk_|^fk_', 'ID类/主键ID'),

    # 结构类/代码
    (r'_code$|^code$', '结构类/分类代码'),
    (r'_no$|_no\b|no_', '结构类/分类代码'),
    (r'_seq$|^seq', '结构类/分类代码'),

    # 属性类/名称
    (r'_name$|^name$', '属性类/名称标题'),
    (r'_title$|^title$', '属性类/名称标题'),
    (r'_label$|^label$', '属性类/类别标签'),

    # 属性类/描述
    (r'_desc$|_desc\b', '属性类/描述文本'),
    (r'_description$', '属性类/描述文本'),
    (r'_remark$|_note$', '属性类/描述文本'),
    (r'_comment$', '属性类/描述文本'),

    # 属性类/地址
    (r'_address$|^address', '属性类/地址位置'),
    (r'_location$', '属性类/地址位置'),

    # 度量类/计量数值
    (r'_amount$|_amt$', '度量类/计量数值'),
    (r'_price$|^price', '度量类/计量数值'),
    (r'_cost$', '度量类/计量数值'),
    (r'_fee$', '度量类/计量数值'),
    (r'_salary$|^salary', '度量类/计量数值'),
    (r'_balance$', '度量类/计量数值'),
    (r'_income$', '度量类/计量数值'),
    (r'_tax$', '度量类/计量数值'),
    (r'_credit_score$', '度量类/计量数值'),

    # 度量类/时间
    (r'_date$|^date$', '度量类/时间度量'),
    (r'_time$|^time$', '度量类/时间度量'),
    (r'_at$|_at\b', '度量类/时间度量'),
    (r'_datetime$', '度量类/时间度量'),
    (r'_created$|_updated$', '度量类/时间度量'),

    # 度量类/计数
    (r'_count$', '度量类/计数统计'),
    (r'_num$|_num\b', '度量类/计数统计'),
    (r'_quantity$|_qty$', '度量类/计数统计'),
    (r'_total$', '度量类/计数统计'),

    # 身份类/联系方式
    (r'_phone$|^phone', '身份类/联系方式'),
    (r'_mobile$|^mobile', '身份类/联系方式'),
    (r'_tel$', '身份类/联系方式'),
    (r'_email$|^email', '身份类/联系方式'),
    (r'_fax$', '身份类/联系方式'),

    # 身份类/人口统计
    (r'_age$|^age$', '身份类/人口统计'),
    (r'_gender$|^gender', '身份类/人口统计'),
    (r'_sex$', '身份类/人口统计'),
    (r'_birth', '身份类/人口统计'),
    (r'_nation$|_nationality$', '身份类/人口统计'),

    # 身份类/职业
    (r'_job$|_position$', '身份类/职业信息'),
    (r'_company$|_employer$', '身份类/职业信息'),
    (r'_department$', '身份类/职业信息'),

    # 状态类
    (r'^is_|^has_|_flag$', '状态类/二元标志'),
    (r'_status$|^status', '状态类/状态枚举'),
    (r'_type$|^type', '状态类/状态枚举'),
]

def infer_classification_from_field_name(field_name: str) -> str:
    """根据字段名模式推断分类

    这是最可靠的兜底方案：当模型无法正确识别分类时使用。
    基于大量常见字段命名规范。

    Args:
        field_name: 字段名

    Returns:
        分类标签
    """
    field_lower = field_name.lower()

    for pattern, classification in CLASSIFICATION_RULES:
        import re
        if re.search(pattern, field_lower):
            return classification

    # 默认返回扩展类
    return "扩展类/其他字段"


model = None
tokenizer = None
use_lora = False
model_loading = False
model_loading_lock = threading.Lock()
_model_loading_thread = None

def load_model_async():
    """后台线程中加载模型"""
    global model, tokenizer, use_lora, model_loading
    
    with model_loading_lock:
        if model is not None or model_loading:
            return
        model_loading = True
    
    try:
        print("=" * 50)
        print("正在后台加载模型，请稍候...")
        print("=" * 50)
        
        tokenizer = get_chinese_tokenizer(MODEL_DIR)
        print("分词器加载完成")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("基础模型加载完成")
        
        lora_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
        if os.path.exists(lora_path):
            print("加载LoRA权重...")
            try:
                model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
                use_lora = True
            except Exception as e:
                print(f"LoRA加载失败: {e}, 使用基础模型")
                model = base_model
                use_lora = False
        else:
            print("使用基础模型...")
            model = base_model
            use_lora = False
        
        model.eval()
        print("=" * 50)
        print(f"模型加载完成! (LoRA: {use_lora})")
        print("=" * 50)
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with model_loading_lock:
            model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan事件处理 - 快速启动，不阻塞"""
    global model, tokenizer, use_lora, _model_loading_thread

    # 只启动一次后台加载，使用 daemon=True 避免阻塞
    if model is None and not model_loading:
        _model_loading_thread = threading.Thread(target=load_model_async, daemon=True)
        _model_loading_thread.start()

    print("=" * 50)
    print("服务已启动！")
    print(f"API地址: http://localhost:8001")
    if model is None:
        print("注意: 模型正在后台加载中，请等待完成...")
    print("=" * 50)

    yield

    # 清理
    model = None
    tokenizer = None
    use_lora = False


app = FastAPI(
    title="数据分类分级推理服务",
    description="基于DeepSeek-7B大语言模型的字段分类分级推理API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FieldClassificationRequest(BaseModel):
    field_name: str
    industry: str = "其他"
    samples: List[str] = []


class BatchClassificationRequest(BaseModel):
    fields: List[FieldClassificationRequest]


class ClassificationResult(BaseModel):
    field_name: str
    classification: str
    grading: str
    confidence: float = 1.0
    warnings: List[str] = []


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any = None
    error: str = None


def get_model():
    """获取模型，如果模型未加载则等待"""
    global model, model_loading
    
    if model is None and not model_loading:
        # 触发加载
        load_model_async()
    
    # 等待模型加载
    import time
    max_wait = 300  # 最多等待5分钟
    waited = 0
    while model is None and waited < max_wait:
        time.sleep(2)
        waited += 2
        print(f"等待模型加载... ({waited}s)")
    
    if model is None:
        raise RuntimeError("模型加载超时，请检查GPU和显存是否充足")
    
    return model, tokenizer


def extract_label(text: str, labels: List[str], label_type: str = "classification") -> str:
    """从LLM输出中提取标签

    Args:
        text: LLM的原始输出
        labels: 标签列表
        label_type: "classification" 或 "grading"，用于区分处理逻辑
    """
    text = text.strip()
    import re

    # 如果输出完全匹配某个标签
    for label in labels:
        if label == text:
            return label

    # 如果输出包含某个标签
    for label in labels:
        if label in text:
            return label

    # 如果是分级标签，特殊处理
    if label_type == "grading":
        # 1. 匹配 "第X级" 模式
        match = re.search(r'第[一二三四五六七八九十百千\d]+级[/／]?[^\s，,。\n]*', text)
        if match:
            matched_text = match.group()
            # 精确匹配分级标签
            for label in labels:
                if label == matched_text or matched_text.startswith(label):
                    return label
            # 解析数字
            grade_num_match = re.search(r'第([一二三四五六七八九十百千]+|\d+)级', matched_text)
            if grade_num_match:
                grade_num = grade_num_match.group(1)
                level_map = {
                    '一': '1', '二': '2', '三': '3', '四': '4',
                    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
                    '1': '1', '2': '2', '3': '3', '4': '4',
                    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10'
                }
                if grade_num in level_map:
                    level = level_map[grade_num]
                    for label in labels:
                        if label.startswith(f"第{level}级"):
                            return label

        # 2. 如果 LLM 输出单个数字（1, 2, 3, 4），直接映射
        digit_match = re.search(r'\b([1-4])\b', text.strip())
        if digit_match:
            level = digit_match.group(1)
            for label in labels:
                if label.startswith(f"第{level}级"):
                    return label

        # 3. 尝试从文本推断敏感度关键字
        text_lower = text.lower()
        if re.search(r'\b(公开|public|低敏感|无敏感|一般)\b', text_lower):
            return "第1级/公开"
        elif re.search(r'\b(内部|受限|internal|中敏感)\b', text_lower):
            return "第2级/内部"
        elif re.search(r'\b(敏感|隐私|个人|secret|高敏感)\b', text_lower):
            return "第3级/敏感"
        elif re.search(r'\b(机密|高度机密|confidential|绝密)\b', text_lower):
            return "第4级/机密"

        # 4. 默认返回第1级/公开
        return "第1级/公开"

    # 分类标签的处理
    # 如果输出只包含分类名称（不带前缀）
    for label in labels:
        label_short = label.split('/')[-1]
        if label_short in text:
            return label

    # 如果是完整的分类路径（如"属性类/名称标题"），尝试匹配
    for label in labels:
        parts = text.split('/')
        if len(parts) >= 2:
            label_parts = label.split('/')
            if parts[0].strip() in label_parts[0] or label_parts[0] in parts[0]:
                return label

    # 默认返回第一个匹配的分类（如果完全没有匹配）
    return labels[0] if labels else "扩展类/其他字段"


def predict_meaning(field_name: str, industry: str, samples: str, classification: str) -> str:
    """预测字段的业务含义"""
    global model, tokenizer

    m, t = get_model()

    prompt = f"""你是一个数据分类分级助手。请简要描述以下字段的业务含义和用途。

字段名：{field_name}
行业：{industry}
样本值：{samples}
分类：{classification}

请用一句话（不超过50字）描述该字段的业务含义和用途：

含义："""

    messages = [{"role": "user", "content": prompt}]
    encoded = t.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(m.device)

    with torch.no_grad():
        outputs = m.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=t.pad_token_id,
            eos_token_id=t.eos_token_id,
        )

    response = t.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    meaning = response.strip()

    # 清理输出
    if meaning.startswith('含义：'):
        meaning = meaning[3:].strip()
    elif meaning.startswith('该字段'):
        pass  # 保持原样
    else:
        # 截取第一句
        for sep in ['。', '，', '！', '？', '\n']:
            if sep in meaning:
                meaning = meaning.split(sep)[0] + sep
                break

    return meaning if meaning else "模型未提供含义解释"


def get_samples_from_series(series: pd.Series, n: int = 10) -> List[str]:
    """从数据列中获取样本值"""
    valid_values = series.dropna().astype(str)
    valid_values = valid_values[valid_values.str.len() < 100]
    unique_vals = valid_values.unique()[:n]
    return unique_vals.tolist() if len(unique_vals) > 0 else []


def predict_classification(field_name: str, industry: str, samples: str) -> str:
    """预测分类，优先使用知识库规则，模型作为兜底"""
    global model, tokenizer

    # 1. 优先使用知识库规则
    kb_classification = get_classification_from_rules(field_name, industry)
    if kb_classification:
        print(f"字段[{field_name}]分类结果: {kb_classification} (知识库规则)")
        return kb_classification

    # 2. 尝试用模型预测
    try:
        m, t = get_model()

        labels_text = '\n'.join([f"{i+1}. {label}" for i, label in enumerate(CLASSIFICATION_LABELS)])
        prompt = f"""你是一个数据分类分级助手。请根据字段名、行业和样本值判断字段属于哪一类。

行业：{industry}
字段名：{field_name}
样本值示例：{samples if samples else '无样本'}

请从以下{len(CLASSIFICATION_LABELS)}个分类标签中选择最合适的一个。只输出标签路径，不要其他内容。

{labels_text}

答案："""

        messages = [{"role": "user", "content": prompt}]
        encoded = t.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(m.device)

        with torch.no_grad():
            outputs = m.generate(
                input_ids,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=t.pad_token_id,
                eos_token_id=t.eos_token_id,
            )

        response = t.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        result = extract_label(response, CLASSIFICATION_LABELS, label_type="classification")

        # 3. 验证结果是否是有效的分类标签
        if result in CLASSIFICATION_LABELS:
            print(f"字段[{field_name}]分类结果: {result} (模型预测)")
            return result

        # 4. 模型输出无效，使用规则推断
        print(f"字段[{field_name}]模型输出无效: '{response[:30]}...', 使用规则推断")
        inferred = infer_classification_from_field_name(field_name)
        print(f"字段[{field_name}]推断分类: {inferred}")
        return inferred

    except Exception as e:
        # 5. 模型出错，使用规则推断
        print(f"字段[{field_name}]模型预测失败: {e}, 使用规则推断")
        inferred = infer_classification_from_field_name(field_name)
        print(f"字段[{field_name}]推断分类: {inferred}")
        return inferred


def predict_grading(field_name: str, industry: str, samples: str, classification: str) -> str:
    """预测分级，优先使用知识库规则，基于分类映射作为兜底"""
    # 1. 优先使用知识库规则
    kb_grading = get_grading_from_rules(field_name, industry)
    if kb_grading:
        print(f"字段[{field_name}]分级结果: {kb_grading} (知识库规则)")
        return kb_grading

    # 2. 基于分类推断分级
    grading = infer_grading(classification, field_name, samples)
    print(f"字段[{field_name}]分级结果: {grading} (基于分类映射: {classification})")
    return grading


@app.get("/")
async def root():
    return {
        "service": "数据分类分级推理服务",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "lora_enabled": use_lora,
        "message": "模型已就绪" if model is not None else "模型加载中，请稍候..."
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "message": "服务正常，模型已就绪" if model is not None else "服务正常，模型加载中..."
    }


@app.post("/classify", response_model=ApiResponse)
async def classify_field(request: FieldClassificationRequest):
    try:
        samples_str = ', '.join(request.samples[:5]) if request.samples else 'N/A'
        
        if model is None:
            return ApiResponse(
                success=False,
                message="模型正在加载中",
                error="模型尚未加载完成，请等待几秒后重试"
            )
        
        classification = predict_classification(request.field_name, request.industry, samples_str)
        grading = predict_grading(request.field_name, request.industry, samples_str, classification)
        
        result = ClassificationResult(
            field_name=request.field_name,
            classification=classification,
            grading=grading,
            confidence=1.0
        )
        
        return ApiResponse(success=True, message="分类分级完成", data=result.model_dump())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(success=False, message="分类分级失败", error=str(e))


@app.post("/classify/batch", response_model=ApiResponse)
async def classify_batch(request: BatchClassificationRequest):
    try:
        if model is None:
            return ApiResponse(
                success=False,
                message="模型正在加载中",
                error="模型尚未加载完成，请等待几秒后重试"
            )
        
        results = []
        for i, field_req in enumerate(request.fields):
            samples_str = ', '.join(field_req.samples[:5]) if field_req.samples else 'N/A'
            print(f"处理字段 {i+1}/{len(request.fields)}: {field_req.field_name}")
            
            classification = predict_classification(field_req.field_name, field_req.industry, samples_str)
            grading = predict_grading(field_req.field_name, field_req.industry, samples_str, classification)
            
            results.append({
                "field_name": field_req.field_name,
                "classification": classification,
                "grading": grading
            })
        
        return ApiResponse(
            success=True,
            message=f"批量分类分级完成，共{len(results)}个字段",
            data={"results": results}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(success=False, message="批量分类分级失败", error=str(e))


@app.post("/classify/file/stream")
async def classify_file_stream(file: UploadFile = File(...), industry: str = Form("其他")):
    """流式分类接口 - 每个字段分析完成后立即返回"""
    async def event_generator():
        try:
            if model is None:
                yield f"event: error\ndata: {json.dumps({'error': '模型尚未加载完成'})}\n\n"
                return

            content = await file.read()
            print(f"[Stream] 处理文件: {file.filename}, 大小: {len(content)} bytes")

            if file.filename.endswith('.csv'):
                df = pd.read_csv(BytesIO(content))
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(BytesIO(content))
            else:
                yield f"event: error\ndata: {json.dumps({'error': '不支持的文件格式'})}\n\n"
                return

            total_fields = len(df.columns)
            print(f"[Stream] 文件包含 {total_fields} 个字段")

            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({'filename': file.filename, 'total_fields': total_fields, 'industry': industry})}\n\n"

            for i, col in enumerate(df.columns):
                print(f"[Stream] 分析字段 {i+1}/{total_fields}: {col}")

                samples = get_samples_from_series(df[col])
                samples_str = ', '.join(samples[:5]) if samples else 'N/A'

                try:
                    classification = predict_classification(col, industry, samples_str)
                    grading = predict_grading(col, industry, samples_str, classification)
                    meaning = predict_meaning(col, industry, samples_str, classification)

                    field_result = {
                        "index": i,
                        "field_name": col,
                        "classification": classification,
                        "grading": grading,
                        "data_type": str(df[col].dtype),
                        "meaning": meaning,
                        "samples": samples[:10]
                    }

                    # 发送字段完成事件
                    yield f"event: field_done\ndata: {json.dumps(field_result)}\n\n"

                    # 发送进度事件
                    progress = int((i + 1) / total_fields * 100)
                    yield f"event: progress\ndata: {json.dumps({'current': i + 1, 'total': total_fields, 'percentage': progress, 'field_name': col})}\n\n"

                except Exception as field_error:
                    print(f"[Stream] 字段 {col} 分析失败: {field_error}")
                    yield f"event: field_error\ndata: {json.dumps({'index': i, 'field_name': col, 'error': str(field_error)})}\n\n"

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'filename': file.filename, 'total_fields': total_fields, 'industry': industry})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/classify/file", response_model=ApiResponse)
async def classify_file(file: UploadFile = File(...), industry: str = Form("其他")):
    try:
        if model is None:
            return ApiResponse(
                success=False,
                message="模型正在加载中",
                error="模型尚未加载完成，请等待几秒后重试。如果长时间未就绪，请检查GPU显存是否充足。"
            )

        content = await file.read()
        print(f"处理文件: {file.filename}, 大小: {len(content)} bytes")

        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(content))
        else:
            raise ValueError("不支持的文件格式，请上传CSV或Excel文件")

        print(f"文件包含 {len(df.columns)} 个字段")

        results = []
        for i, col in enumerate(df.columns):
            print(f"分析字段 {i+1}/{len(df.columns)}: {col}")
            samples = get_samples_from_series(df[col])
            samples_str = ', '.join(samples[:5]) if samples else 'N/A'

            classification = predict_classification(col, industry, samples_str)
            grading = predict_grading(col, industry, samples_str, classification)
            meaning = predict_meaning(col, industry, samples_str, classification)

            results.append({
                "field_name": col,
                "classification": classification,
                "grading": grading,
                "data_type": str(df[col].dtype),
                "meaning": meaning,
                "samples": samples[:10]  # 返回最多10个样本
            })

        return ApiResponse(
            success=True,
            message=f"文件处理完成，共{len(results)}个字段",
            data={
                "filename": file.filename,
                "total_fields": len(results),
                "industry": industry,
                "results": results
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(success=False, message="文件处理失败", error=str(e))


@app.get("/labels")
async def get_labels():
    return {
        "classification_labels": CLASSIFICATION_LABELS,
        "grading_labels": GRADING_LABELS,
        "classification_count": len(CLASSIFICATION_LABELS),
        "grading_count": len(GRADING_LABELS)
    }


@app.get("/industries")
async def get_industries():
    return {
        "industries": [
            {"code": "financial", "name": "金融"},
            {"code": "medical", "name": "医疗"},
            {"code": "education", "name": "教育"},
            {"code": "industrial", "name": "工业"},
            {"code": "commercial", "name": "商业"},
            {"code": "government", "name": "政务"},
            {"code": "other", "name": "其他"}
        ]
    }


@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    stats = get_knowledge_base_stats()
    return {
        "totalRules": stats.get("totalRules", 0),
        "totalIndustryRules": stats.get("totalIndustryRules", 0),
        "cachedFields": stats.get("cachedFields", 0),
        "industries": stats.get("industries", [])
    }


@app.post("/knowledge/reload")
async def reload_knowledge():
    """重新加载知识库"""
    reload_knowledge_base()
    return {"success": True, "message": "知识库已重新加载"}


@app.post("/knowledge/save")
async def save_knowledge_rules(request: dict):
    """保存规则到后端目录"""
    try:
        general_rules = request.get("general_rules", [])
        industry_rules = request.get("industry_rules", {})

        if save_general_rules(general_rules):
            pass
        else:
            return {"success": False, "message": "保存通用规则失败"}

        if save_industry_rules(industry_rules):
            pass
        else:
            return {"success": False, "message": "保存行业规则失败"}

        return {"success": True, "message": "规则已保存到后端目录"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"保存失败: {str(e)}"}


@app.get("/knowledge/rules")
async def get_knowledge_rules():
    """获取知识库规则"""
    general = load_general_rules()
    industry = load_industry_rules()
    return {
        "general_rules": general,
        "industry_rules": industry
    }


@app.get("/knowledge/conflicts")
async def detect_conflicts():
    """检测规则冲突"""
    general_rules = load_general_rules()
    industry_rules = load_industry_rules()

    conflicts = []

    # 检测同一字段不同分类
    field_map = {}

    for rule in general_rules:
        patterns = rule.get("patterns", [])
        for pattern in patterns:
            if pattern not in field_map:
                field_map[pattern] = []
            field_map[pattern].append({
                "type": "通用规则",
                "category": rule.get("category"),
                "grading": rule.get("grading")
            })

    for industry, rules in industry_rules.items():
        for rule in rules:
            field = rule.get("field", "")
            if field not in field_map:
                field_map[field] = []
            field_map[field].append({
                "type": f"行业规则({industry})",
                "category": rule.get("category"),
                "grading": rule.get("grading")
            })

    # 找出冲突
    for field, entries in field_map.items():
        if len(entries) > 1:
            categories = set(e["category"] for e in entries)
            gradings = set(e["grading"] for e in entries)
            if len(categories) > 1 or len(gradings) > 1:
                conflicts.append({
                    "field": field,
                    "entries": entries,
                    "issue": "同一字段存在多个不同的分类或分级"
                })

    return {
        "conflicts": conflicts,
        "total_conflicts": len(conflicts)
    }


@app.post("/knowledge/similarity-search")
async def kb_similarity_search(request: dict):
    """基于向量相似度在知识库中搜索相似字段"""
    try:
        from scripts.knowledge_base_loader import find_similar_fields_in_kb

        target_field = request.get("field", "")
        industry = request.get("industry")
        top_k = request.get("top_k", 5)
        threshold = request.get("threshold", 0.5)

        results = find_similar_fields_in_kb(
            target_field,
            industry=industry,
            top_k=top_k,
            threshold=threshold
        )

        return {
            "success": True,
            "data": {
                "target_field": target_field,
                "matched_fields": [
                    {
                        "field": r.matched_field,
                        "similarity": round(r.similarity, 4),
                        "category": r.category,
                        "grading": r.grading,
                        "source": r.source
                    }
                    for r in results
                ]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/knowledge/check-conflicts")
async def check_conflicts(request: dict):
    """检测新字段与知识库的冲突"""
    try:
        from scripts.knowledge_base_loader import detect_conflicts_with_similarity

        target_field = request.get("field", "")
        predicted_category = request.get("category", "")
        predicted_grading = request.get("grading", "")
        industry = request.get("industry")

        conflicts = detect_conflicts_with_similarity(
            target_field,
            predicted_category,
            predicted_grading,
            industry=industry
        )

        return {
            "success": True,
            "data": {
                "has_conflicts": len(conflicts) > 0,
                "conflicts": conflicts
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= 新增接口：向量相似度与语义建模 =============

class SimilarityRequest(BaseModel):
    field1: str
    field2: str


class BatchSimilarityRequest(BaseModel):
    target_field: str
    candidate_fields: List[str]
    top_k: int = 5
    threshold: float = 0.3


class SemanticModelingRequest(BaseModel):
    field_name: str
    samples: List[str] = []


@app.post("/similarity/calculate")
async def calculate_similarity(request: SimilarityRequest):
    """计算两个字段之间的相似度"""
    try:
        result = similarity_calculator.compute_similarity(request.field1, request.field2)
        return {
            "success": True,
            "data": {
                "field1": result.field1,
                "field2": result.field2,
                "char_similarity": result.char_similarity,
                "semantic_similarity": result.semantic_similarity,
                "combined_score": result.combined_score,
                "match_level": result.match_level
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/similarity/find")
async def find_similar_fields(request: BatchSimilarityRequest):
    """在候选字段中查找与目标字段最相似的字段"""
    try:
        similar = similarity_calculator.find_similar_fields(
            request.target_field,
            request.candidate_fields,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return {
            "success": True,
            "data": {
                "target_field": request.target_field,
                "similar_fields": [
                    {
                        "field": r.field2,
                        "char_similarity": r.char_similarity,
                        "semantic_similarity": r.semantic_similarity,
                        "combined_score": r.combined_score,
                        "match_level": r.match_level
                    }
                    for r in similar
                ]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/similarity/knowledge-base")
async def similarity_in_knowledge_base(request: SimilarityRequest):
    """在知识库中查找与目标字段相似的已分类字段"""
    try:
        general_rules = load_general_rules()
        matched = find_similar_in_knowledge_base(request.field1, general_rules, threshold=0.3)
        return {
            "success": True,
            "data": {
                "target_field": request.field1,
                "matched_rules": matched[:5]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/semantic/model")
async def model_field_semantic(request: SemanticModelingRequest):
    """对字段进行语义建模，提取多模态特征"""
    try:
        features = semantic_modeler.model_field(request.field_name, request.samples)
        return {
            "success": True,
            "data": {
                "field_name": features.field_name,
                "normalized_name": features.normalized_name,
                "structure_features": {
                    "suffix_match": features.structure_features.get("suffix_match", 0),
                    "inferred_category": features.structure_features.get("suffix_category"),
                    "word_count": features.structure_features.get("word_count", 0)
                },
                "semantic_features": {
                    "inferred_category": features.inferred_category,
                    "confidence": features.inferred_confidence,
                    "is_sensitive": features.semantic_features.get("is_sensitive", 0) == 1.0,
                    "is_numeric": features.semantic_features.get("is_numeric", 0) == 1.0,
                    "is_date": features.semantic_features.get("is_date", 0) == 1.0
                },
                "data_features": {
                    "pattern_type": features.data_features.get("pattern_type", "unknown") if features.data_features else "unknown",
                    "unique_ratio": features.data_features.get("stats", {}).unique_ratio if features.data_features else 0,
                    "null_ratio": features.data_features.get("stats", {}).null_ratio if features.data_features else 0
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/semantic/categories")
async def get_semantic_categories():
    """获取语义分类关键词映射"""
    return {
        "success": True,
        "data": {
            "categories": FieldSemanticModeler.CATEGORY_KEYWORDS,
            "suffix_patterns": FieldSemanticModeler.SUFFIX_PATTERNS
        }
    }


@app.get("/model/info")
async def get_model_info():
    """获取模型压缩相关信息"""
    from scripts.model_compression import ModelCompressor
    
    try:
        compressor = ModelCompressor(MODEL_DIR)
        original_size = compressor.get_model_size(MODEL_DIR)
        
        return {
            "success": True,
            "data": {
                "model_path": MODEL_DIR,
                "original_size_gb": round(original_size, 2),
                "compression_methods": ["int8", "qlora", "fp16", "pruned"],
                "estimated_compressed_sizes": {
                    "int8": round(original_size / 2, 2),
                    "qlora_4bit": round(original_size / 4, 2),
                    "fp16": round(original_size / 2, 2)
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_server(host: str = "0.0.0.0", port: int = 8001):
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据分类分级推理服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    args = parser.parse_args()
    
    print("=" * 60)
    print("启动数据分类分级推理服务...")
    print("=" * 60)
    run_server(args.host, args.port)
