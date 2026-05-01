#!/usr/bin/env python3
"""
模型微调脚本：使用LoRA技术微调DeepSeek模型
"""
import os
import json
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.tokenizer_fix import get_chinese_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models/deepseek-llm-7b-chat")
TRAIN_FILE = os.path.join(BASE_DIR, "data/sft/train.jsonl")
VAL_FILE = os.path.join(BASE_DIR, "data/sft/val.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/finetuned")

print("=" * 60)
print("DeepSeek 模型微调脚本")
print("=" * 60)

def preprocess_function(examples, tokenizer, max_length=512):
    """预处理训练数据 - 只训练输出部分
    
    格式: [SYS_PROMPT] + instruction + answer
    labels: -100 for prompt, actual tokens for answer
    """
    texts = []
    prompt_end_positions = []  # 记录每个样本中 prompt 结束的位置（token 级别）
    
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i] if examples["input"] else ""
        output = examples["output"][i].strip()
        
        # 构建 prompt（使用 chat template，带 generation prompt）
        if input_text:
            prompt = instruction + input_text
        else:
            prompt = instruction
        
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True  # 会自动加上 "Assistant:" 等
        )
        
        # 完整文本 = prompt + assistant 回复
        full_text = prompt_text + output
        texts.append(full_text)
        
        # 记录 prompt 部分的长度（字符级别），用于后续计算 token 位置
        prompt_end_positions.append(len(prompt_text))
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    
    # 构建 labels
    labels = []
    for idx, input_ids in enumerate(model_inputs["input_ids"]):
        label = [-100] * len(input_ids)  # 默认全部 mask
        
        # 找到 prompt 结束位置（需要用 tokenize 来确定）
        prompt_text = texts[idx]
        prompt_end_char = prompt_end_positions[idx]
        
        # 计算 prompt 对应的 token 数量
        prompt_tokens = tokenizer.encode(prompt_text[:prompt_end_char], add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # labels 只在 output 部分有值
        for j in range(prompt_len, len(input_ids)):
            if input_ids[j] != tokenizer.pad_token_id:
                label[j] = input_ids[j]
        
        labels.append(label)
    
    model_inputs["labels"] = labels
    return model_inputs

def load_and_prepare_dataset(train_file, val_file, tokenizer):
    """加载并准备数据集"""
    # 加载 JSONL 数据
    def load_jsonl(filepath):
        data = {"instruction": [], "input": [], "output": []}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data["instruction"].append(item.get("instruction", ""))
                    data["input"].append(item.get("input", ""))
                    data["output"].append(item.get("output", ""))
        return data
    
    train_data = load_jsonl(train_file)
    
    # 预处理训练数据
    train_dataset = preprocess_function(train_data, tokenizer)
    
    # 验证集
    val_dataset = None
    if os.path.exists(val_file):
        val_data = load_jsonl(val_file)
        val_dataset = preprocess_function(val_data, tokenizer)
    
    return train_dataset, val_dataset

class CustomDataset(torch.utils.data.Dataset):
    """自定义数据集"""
    def __init__(self, data_dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())
        self.length = len(data_dict[self.keys[0]])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {}
        for key in self.keys:
            item[key] = self.data[key][idx]
        return item

def main():
    # 检查文件是否存在
    if not os.path.exists(MODEL_DIR):
        print(f"错误：模型目录不存在: {MODEL_DIR}")
        exit(1)
    
    if not os.path.exists(TRAIN_FILE):
        print(f"错误：训练文件不存在: {TRAIN_FILE}")
        print("请先运行: python scripts/prepare_training_data.py")
        exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载tokenizer（使用修复版本支持中文）
    print("\n步骤1：加载模型和分词器...")
    tokenizer = get_chinese_tokenizer(MODEL_DIR)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # 配置LoRA
    print("\n步骤2：配置LoRA...")
    lora_config = LoraConfig(
        r=16,                  # 保持原有参数
        lora_alpha=32,         # 配合 rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA配置完成")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"总参数: {total_params:,}")
    
    # 加载数据集
    print("\n步骤3：加载训练数据...")
    train_data, val_data = load_and_prepare_dataset(TRAIN_FILE, VAL_FILE, tokenizer)
    
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data) if val_data else None
    
    print(f"训练集: {len(train_dataset)} 条")
    if val_dataset:
        print(f"验证集: {len(val_dataset)} 条")
    
    # DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # 配置训练参数
    print("\n步骤4：配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=20,        # 增加训练轮数，让模型充分学习
        learning_rate=1e-4,        # 降低学习率，更稳定的学习
        warmup_steps=20,            # 增加预热步数
        logging_steps=10,
        save_steps=50,
        save_total_limit=5,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        weight_decay=0.01,          # 添加权重衰减，防止过拟合
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",
    )
    
    # 创建Trainer
    print("\n步骤5：开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    try:
        trainer.train()
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"\n模型已保存到: {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
