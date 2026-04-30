#!/usr/bin/env python3
"""
测试 DeepSeek tokenizer 中文支持
"""
import sys
import os

# 添加 models 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from tokenizer_fix import get_chinese_tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'deepseek-llm-7b-chat')

def test_chinese_tokenizer():
    """测试中文 tokenizer"""
    print("=" * 60)
    print("DeepSeek Tokenizer 中文支持测试")
    print("=" * 60)
    
    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = get_chinese_tokenizer(MODEL_DIR)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    
    # 测试用例
    tests = [
        # 短中文
        ("你好", "基本问候"),
        ("血压", "医疗术语"),
        ("金融数据", "行业数据"),
        
        # 长中文
        ("数据分类任务是判断数据的敏感等级", "复杂句子"),
        ("这是一条包含中英文混合的文本 text content", "混合语言"),
        
        # 测试你的数据场景
        ("行业：医疗，字段：血压，样本：120/80", "分类任务输入"),
        ("ID类/主键ID", "分类标签"),
        ("第1级/公开", "分级标签"),
    ]
    
    print("\n" + "-" * 60)
    print("分词测试")
    print("-" * 60)
    
    all_passed = True
    for text, description in tests:
        try:
            # 编码
            ids = tokenizer.encode(text, add_special_tokens=False)
            
            # 解码
            decoded = tokenizer.decode(ids)
            
            # 验证
            success = (len(ids) > 0 and decoded == text)
            status = "✓" if success else "✗"
            
            print(f"\n{status} {description}")
            print(f"  原文: {repr(text)}")
            print(f"  Token数: {len(ids)}")
            print(f"  解码: {repr(decoded)}")
            
            if not success:
                all_passed = False
                print(f"  错误: 编码为空或解码不匹配")
                
        except Exception as e:
            all_passed = False
            print(f"\n✗ {description}")
            print(f"  错误: {e}")
    
    # 测试 chat template
    print("\n" + "-" * 60)
    print("Chat Template 测试")
    print("-" * 60)
    
    try:
        messages = [{"role": "user", "content": "请分类这条数据：血压测量值"}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_text = tokenizer.decode(prompt)
        
        print(f"\n✓ Chat template 正常工作")
        print(f"  输入: {messages[0]['content']}")
        print(f"  模板: {prompt_text[:100]}...")
        
    except Exception as e:
        all_passed = False
        print(f"\n✗ Chat template 失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("结果: 全部测试通过 ✓")
        print("Tokenizer 中文支持正常，可以进行训练和预测")
    else:
        print("结果: 部分测试失败 ✗")
        print("请检查 tokenizer_fix.py")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = test_chinese_tokenizer()
    sys.exit(0 if success else 1)
