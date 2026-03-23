import os
import json
import tempfile
from transformers import PreTrainedTokenizerFast

def get_chinese_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """
    获取支持中文的 DeepSeek tokenizer
    解决 LlamaTokenizer 无法处理中文的问题
    """
    tokenizer_json = os.path.join(model_path, 'tokenizer.json')
    
    # 直接使用 PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    
    # 添加必要的 special tokens
    tokenizer.pad_token = '<｜end▁of▁sentence｜>'
    tokenizer.eos_token = '<｜end▁of▁sentence｜>'
    tokenizer.bos_token = '<｜begin▁of▁sentence｜>'
    
    # 添加 chat template
    tokenizer.chat_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
    )
    
    return tokenizer
