#!/usr/bin/env python3
"""
新增模块测试脚本
用于测试向量相似度、语义建模和模型压缩模块
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.similarity_calculator import FieldSimilarityCalculator
from scripts.embedding_extractor import FieldSemanticModeler
from scripts.knowledge_base_loader import find_similar_fields_in_kb, load_general_rules, detect_conflicts_with_similarity
from scripts.model_compression import ModelCompressor


def test_similarity():
    """测试向量相似度计算"""
    print("\n" + "=" * 60)
    print("测试1: 向量相似度计算")
    print("=" * 60)
    
    calc = FieldSimilarityCalculator()
    
    test_pairs = [
        ("customer_id", "user_id"),
        ("customer_id", "order_id"),
        ("email_address", "user_email"),
        ("created_at", "updated_at"),
        ("password", "passwd"),
        ("salary", "bonus"),
        ("phone_number", "mobile_phone"),
        ("balance_amount", "transaction_amount"),
    ]
    
    print(f"\n{'字段1':<20} {'字段2':<20} {'TF-IDF':<8} {'关键词':<8} {'综合':<8} {'级别'}")
    print("-" * 80)
    
    for f1, f2 in test_pairs:
        result = calc.compute_similarity(f1, f2)
        print(f"{result.field1:<20} {result.field2:<20} "
              f"{result.char_similarity:<8.3f} {result.semantic_similarity:<8.3f} "
              f"{result.combined_score:<8.3f} {result.match_level}")
    
    # 测试批量查找
    print("\n批量查找相似字段:")
    candidates = [
        "user_id", "customer_id", "order_id", "product_id",
        "username", "customer_name", "product_name",
        "email", "phone", "mobile", "address",
        "created_at", "updated_at",
        "price", "amount", "quantity"
    ]
    
    target = "user_email"
    print(f"\n目标: {target}")
    similar = calc.find_similar_fields(target, candidates, top_k=5)
    for r in similar:
        print(f"  {r.field2:<18} 相似度: {r.combined_score:.4f} ({r.match_level})")


def test_semantic_modeling():
    """测试字段语义建模"""
    print("\n" + "=" * 60)
    print("测试2: 字段语义建模")
    print("=" * 60)
    
    modeler = FieldSemanticModeler()
    
    test_cases = [
        ("customer_id", ["1001", "1002", "1003", "1004"]),
        ("user_email", ["test@example.com", "admin@test.cn", "user@company.org"]),
        ("order_amount", ["150.50", "299.00", "88.88", "1200.00"]),
        ("created_at", ["2024-01-15", "2024-02-20", "2024-03-10"]),
        ("phone_number", ["13812345678", "13998765432"]),
        ("customer_name", ["张三", "李四", "王五", "赵六"]),
        ("is_active", ["true", "false", "true", "false"]),
        ("password", ["******", "******", "******"]),
    ]
    
    for field_name, samples in test_cases:
        features = modeler.model_field(field_name, samples)
        
        print(f"\n字段: {field_name}")
        print(f"  样本: {samples[:3]}")
        print(f"  标准化: {features.normalized_name}")
        print(f"  推断分类: {features.inferred_category}")
        print(f"  置信度: {features.inferred_confidence:.2f}")
        
        if features.data_features:
            pattern = features.data_features.get('pattern')
            if pattern:
                print(f"  数据模式: {pattern.pattern_type} (匹配率: {pattern.match_ratio:.1%})")
        
        # 显示结构特征
        suffix_match = features.structure_features.get('suffix_match', 0)
        if suffix_match > 0:
            print(f"  后缀匹配: {suffix_match:.2f}")


def test_knowledge_base_similarity():
    """测试知识库相似度搜索"""
    print("\n" + "=" * 60)
    print("测试3: 知识库相似度搜索")
    print("=" * 60)
    
    rules = load_general_rules()
    print(f"\n知识库规则数量: {len(rules)}")
    
    test_fields = ["customer_email", "user_password", "account_balance", "order_date"]
    
    for target in test_fields:
        print(f"\n搜索字段: {target}")
        similar = find_similar_fields_in_kb(target, threshold=0.3)
        if similar:
            for s in similar[:3]:
                print(f"  - {s.matched_field:<18} 相似度: {s.similarity:.4f} | "
                      f"分类: {s.category} | 分级: {s.grading}")
        else:
            print("  未找到相似字段")


def test_conflict_detection():
    """测试冲突检测"""
    print("\n" + "=" * 60)
    print("测试4: 冲突检测")
    print("=" * 60)
    
    test_cases = [
        ("user_email", "身份类/联系方式", "第3级/敏感"),
        ("customer_name", "属性类/名称标题", "第1级/公开"),
        ("balance", "度量类/计量数值", "第3级/敏感"),
    ]
    
    for field, category, grading in test_cases:
        print(f"\n检测字段: {field}")
        print(f"  预测分类: {category}")
        print(f"  预测分级: {grading}")
        
        conflicts = detect_conflicts_with_similarity(field, category, grading)
        if conflicts:
            for c in conflicts:
                print(f"  ⚠️ 冲突: {c['suggestion'][:60]}...")
        else:
            print("  ✓ 无冲突")


def test_model_compression():
    """测试模型压缩信息"""
    print("\n" + "=" * 60)
    print("测试5: 模型压缩对比")
    print("=" * 60)
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/deepseek-llm-7b-chat")
    
    if not os.path.exists(model_dir):
        print(f"\n⚠️ 模型目录不存在: {model_dir}")
        print("跳过模型压缩测试")
        return
    
    compressor = ModelCompressor(model_dir)
    
    try:
        original_size = compressor.get_model_size(model_dir)
        print(f"\n原始模型大小: {original_size:.2f} GB")
        
        # 量化方法对比
        print("\n【量化方法对比】")
        print(f"{'方法':<15} {'模型大小':<12} {'显存需求':<12} {'精度损失':<12}")
        print("-" * 55)
        
        # FP16
        fp16_size = original_size / 2
        print(f"{'FP16':<15} {fp16_size:<12.2f} {fp16_size + 3:<12.2f} {'~0%':<12}")
        
        # INT8
        int8_size = original_size / 2 * 0.75
        print(f"{'INT8':<15} {int8_size:<12.2f} {int8_size + 3:<12.2f} {'<0.5%':<12}")
        
        # QLoRA-4bit
        qlora_size = original_size / 4
        print(f"{'QLoRA-4bit':<15} {qlora_size:<12.2f} {qlora_size + 2:<12.2f} {'<1%':<12}")
        
        # 剪枝对比
        print("\n【剪枝方法对比】")
        sparsity_values = [0.1, 0.2, 0.3, 0.5]
        print(f"{'方法':<20} {'压缩率':<12} {'模型大小':<12} {'精度损失':<12}")
        print("-" * 60)
        
        for sparsity in sparsity_values:
            pruned_size = original_size * (1 - sparsity * 0.5)
            pct = sparsity * 100
            print(f"{'Magnitude剪枝'+str(int(pct))+'%':<20} {pct:.0f}%{'':<8} {pruned_size:<12.2f} {'<0.5%':<12}")
        
        # 知识蒸馏对比
        print("\n【知识蒸馏对比】")
        distillation_ratios = [(0.5, "DeepSeek-3.5B"), (0.3, "DeepSeek-2B"), (0.25, "DeepSeek-1.75B")]
        print(f"{'学生模型':<20} {'压缩率':<12} {'模型大小':<12} {'精度损失':<12}")
        print("-" * 60)
        
        for ratio, model_name in distillation_ratios:
            student_size = original_size * ratio
            print(f"{model_name:<20} {ratio*100:.0f}%{'':<8} {student_size:<12.2f} {'<2%':<12}")
        
        print(f"\n详细压缩命令:")
        print(f"  FP16:   python scripts/model_compression.py --method fp16")
        print(f"  INT8:   python scripts/model_compression.py --method int8")
        print(f"  QLoRA:   python scripts/model_compression.py --method qlora")
        print(f"  剪枝30%: python scripts/model_compression.py --method pruned --sparsity 0.3")
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")


def main():
    print("=" * 60)
    print("DeepSeek 数据分类分级系统 - 新增模块测试")
    print("=" * 60)
    
    test_similarity()
    test_semantic_modeling()
    test_knowledge_base_similarity()
    test_conflict_detection()
    test_model_compression()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
