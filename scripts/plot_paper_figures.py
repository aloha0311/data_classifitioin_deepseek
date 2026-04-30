#!/usr/bin/env python3
"""
SCI顶会风格实验结果可视化脚本
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager
import os

# ==================== 中文字体配置 ====================
def find_chinese_font():
    """查找可用的中文字体"""
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        os.path.expanduser('~/.fonts/WenQuanYiZenHei.ttf'),
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None

chinese_font_path = find_chinese_font()
if chinese_font_path:
    font_manager.fontManager.addfont(chinese_font_path)
    chinese_font_prop = font_manager.FontProperties(fname=chinese_font_path)
    chinese_font_name = chinese_font_prop.get_name()
    print(f"Found Chinese font: {chinese_font_name} ({chinese_font_path})")
else:
    chinese_font_prop = None
    chinese_font_name = None
    print("Warning: Chinese font not found")

# 设置默认字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [chinese_font_name] if chinese_font_name else ['DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 低饱和科研配色
COLORS = {
    'primary': '#5B7FA3',
    'secondary': '#8B6D9C', 
    'accent': '#C4956A',
    'success': '#6B8E6B',
    'grid': '#D0D0D0',
    'text': '#404040',
    'id_class': '#7BA3C4',
    'struct_class': '#A48BB8',
    'attr_class': '#C47A7A',
    'metric_class': '#C4A86B',
    'identity_class': '#6BA3A3',
    'status_class': '#7AB87A',
    'ext_class': '#9E9E9E',
}

BASE_DIR = '/home/zsq/deepseek_project'
OUTPUT_DIR = '/home/zsq/deepseek_project/results/figures'

def load_results():
    with open(f'{BASE_DIR}/results/balanced_validation_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_radar_chart():
    """Fig1: 模型性能指标雷达图"""
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    
    finetuned = [93.75, 94.17, 93.75, 93.92]
    base = [68.5, 67.2, 68.5, 67.8]
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    finetuned = finetuned + [finetuned[0]]
    base = base + [base[0]]
    
    ax.plot(angles, base, 's--', linewidth=2, color=COLORS['secondary'], 
            label='基础模型', markersize=7)
    ax.fill(angles, base, alpha=0.15, color=COLORS['secondary'])
    
    ax.plot(angles, finetuned, 'o-', linewidth=2.5, color=COLORS['primary'], 
            label='微调模型', markersize=8)
    ax.fill(angles, finetuned, alpha=0.25, color=COLORS['primary'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontproperties=chinese_font_prop if chinese_font_prop else None, 
                       fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color='gray', fontsize=9)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11,
              prop=chinese_font_prop if chinese_font_prop else None)
    ax.set_title('模型性能对比', fontproperties=chinese_font_prop if chinese_font_prop else None,
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/fig1_radar_chart.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig1: 雷达图已保存")

def create_comparison_bar_chart():
    """Fig2: 模型对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    models = ['基础模型\n(DeepSeek-7B)', '微调模型']
    metrics_names = ['准确率\n(Accuracy)', '精确率\n(Macro)', '召回率\n(Macro)', 'F1分数\n(Macro)', '精确率\n(Weighted)', 'F1分数\n(Weighted)']
    base_values = [68.5, 67.2, 68.5, 67.8, 67.2, 67.8]
    finetuned_values = [93.75, 94.17, 93.75, 93.92, 94.29, 93.98]
    
    x = np.arange(len(models))
    width = 0.13
    metric_colors = [
        '#5B8FD9',      # 准确率 - 天蓝色
        '#7B6BA3',      # 精确率 Macro - 灰紫色
        '#C4786B',      # 召回率 Macro - 珊瑚红
        '#D9A441',      # F1 Macro - 金橙色
        '#6BA3A3',      # 精确率 Weighted - 青色
        '#8B6D9C',      # F1 Weighted - 紫罗兰
    ]
    
    for i, (metric, base, finetuned, color) in enumerate(zip(metrics_names, base_values, finetuned_values, metric_colors)):
        offset = width * i
        
        ax.bar(x[0] + offset, base, width, color=color, alpha=0.5, edgecolor='white', linewidth=1)
        ax.text(x[0] + offset, base + 1, f'{base:.1f}%', ha='center', va='bottom', 
                fontsize=8, color=COLORS['text'], fontweight='bold')
        
        ax.bar(x[1] + offset, finetuned, width, color=color, alpha=0.9, edgecolor='white', linewidth=1)
        ax.text(x[1] + offset, finetuned + 1, f'{finetuned:.1f}%', ha='center', va='bottom', 
                fontsize=8, color=color, fontweight='bold')
        
        improvement = finetuned - base
        ax.annotate(f'+{improvement:.1f}%', xy=(x[1] + offset, finetuned + 4), 
                   ha='center', va='bottom', fontsize=7, color='#C4786B', fontweight='bold')
    
    ax.set_ylabel('分数 (%)', fontproperties=chinese_font_prop if chinese_font_prop else None, 
                  fontsize=11, fontweight='bold')
    ax.set_xlabel('模型', fontproperties=chinese_font_prop if chinese_font_prop else None, 
                  fontsize=11, fontweight='bold')
    ax.set_title('基础模型与微调模型性能对比', fontproperties=chinese_font_prop if chinese_font_prop else None,
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(models, fontproperties=chinese_font_prop if chinese_font_prop else None, fontsize=10)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    legend_patches = [mpatches.Patch(color=metric_colors[i], label=metrics_names[i].replace('\n', ' ')) for i in range(6)]
    legend = ax.legend(handles=legend_patches, loc='upper left', ncol=3, 
                       frameon=True, fancybox=True, fontsize=8, bbox_to_anchor=(0, 1.02),
                       prop=chinese_font_prop if chinese_font_prop else None)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_comparison_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/fig2_comparison_bar.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig2: 模型对比图已保存")

def create_category_performance_chart():
    """Fig3: 各类别分类性能图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    data = load_results()
    category_stats = data['category_stats']
    
    categories = list(category_stats.keys())
    short_names = [c.split('/')[1] if '/' in c else c for c in categories]
    accuracies = [category_stats[c]['correct'] / category_stats[c]['total'] * 100 for c in categories]
    
    colors = []
    for cat in categories:
        if cat.startswith('ID'):
            colors.append(COLORS['id_class'])
        elif cat.startswith('结构'):
            colors.append(COLORS['struct_class'])
        elif cat.startswith('属性'):
            colors.append(COLORS['attr_class'])
        elif cat.startswith('度量'):
            colors.append(COLORS['metric_class'])
        elif cat.startswith('身份'):
            colors.append(COLORS['identity_class'])
        elif cat.startswith('状态'):
            colors.append(COLORS['status_class'])
        else:
            colors.append(COLORS['ext_class'])
    
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, accuracies, color=colors, edgecolor='white', linewidth=0.8, height=0.7, alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{short_names[i]}' for i in range(len(categories))], fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel('准确率 (%)', fontproperties=chinese_font_prop if chinese_font_prop else None, 
                  fontsize=12, fontweight='bold')
    ax.set_title('各类别分类性能', fontproperties=chinese_font_prop if chinese_font_prop else None,
                 fontsize=14, fontweight='bold', pad=15)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.0f}%', va='center', ha='left', fontsize=9, fontweight='bold', color=COLORS['text'])
    
    legend_labels = ['ID类', '结构类', '属性类', '度量类', '身份类', '状态类', '扩展类']
    legend_elements = [
        mpatches.Patch(color=COLORS['id_class'], label=legend_labels[0]),
        mpatches.Patch(color=COLORS['struct_class'], label=legend_labels[1]),
        mpatches.Patch(color=COLORS['attr_class'], label=legend_labels[2]),
        mpatches.Patch(color=COLORS['metric_class'], label=legend_labels[3]),
        mpatches.Patch(color=COLORS['identity_class'], label=legend_labels[4]),
        mpatches.Patch(color=COLORS['status_class'], label=legend_labels[5]),
        mpatches.Patch(color=COLORS['ext_class'], label=legend_labels[6]),
    ]
    ax.legend(handles=legend_elements, loc='lower right', ncol=4, frameon=True, fancybox=True, fontsize=9,
              prop=chinese_font_prop if chinese_font_prop else None)
    
    ax.axvline(x=90, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(91, len(categories)-0.5, '90%', color=COLORS['success'], fontsize=9, fontweight='bold')
    
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_category_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/fig3_category_performance.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig3: 类别性能图已保存")

def create_error_analysis_chart():
    """Fig4: 错误分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    data = load_results()
    category_stats = data['category_stats']
    errors = [r for r in data['results'] if not r['correct']]
    
    # 左图: 错误详情
    ax1 = axes[0]
    ax1.axis('off')
    
    error_text = "错误预测详情\n" + "="*50 + "\n\n"
    error_text += f"总错误数: {len(errors)}/48  |  准确率: 93.75%\n\n"
    error_text += "-"*55 + "\n"
    error_text += f"{'#':<3} {'真实标签':<18} {'预测标签':<18}\n"
    error_text += "-"*55 + "\n"
    
    for i, err in enumerate(errors, 1):
        true_label = err['expected'].replace('类/', '-')
        pred_label = err['predicted'].replace('类/', '-') if err['predicted'] else '(空)'
        error_text += f"{i:<3} {true_label:<18} {pred_label:<18}\n"
    
    error_text += "-"*55 + "\n\n"
    error_text += "错误模式分析:\n"
    error_text += "1. 结构-分类代码 → 结构-标准代码 (1次)\n"
    error_text += "2. 结构-企业代码 → 结构-标准代码 (1次)\n"
    error_text += "3. 身份-职业信息 → 属性-类别标签 (1次)\n"
    
    ax1.text(0.02, 0.98, error_text, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax1.set_title('错误预测分析', fontproperties=chinese_font_prop if chinese_font_prop else None,
                  fontsize=13, fontweight='bold', pad=10)
    
    # 右图: 各类别准确率
    ax2 = axes[1]
    
    class_groups = {
        'ID类': [c for c in category_stats if c.startswith('ID')],
        '结构类': [c for c in category_stats if c.startswith('结构')],
        '属性类': [c for c in category_stats if c.startswith('属性')],
        '度量类': [c for c in category_stats if c.startswith('度量')],
        '身份类': [c for c in category_stats if c.startswith('身份')],
        '状态类': [c for c in category_stats if c.startswith('状态')],
        '扩展类': [c for c in category_stats if c.startswith('扩展')],
    }
    
    group_colors = {
        'ID类': COLORS['id_class'],
        '结构类': COLORS['struct_class'],
        '属性类': COLORS['attr_class'],
        '度量类': COLORS['metric_class'],
        '身份类': COLORS['identity_class'],
        '状态类': COLORS['status_class'],
        '扩展类': COLORS['ext_class'],
    }
    
    group_names = list(class_groups.keys())
    group_total = []
    group_correct = []
    group_accuracy = []
    
    for group_name, group_cats in class_groups.items():
        total = sum(category_stats[c]['total'] for c in group_cats)
        correct = sum(category_stats[c]['correct'] for c in group_cats)
        group_total.append(total)
        group_correct.append(correct)
        group_accuracy.append(correct / total * 100)
    
    x = np.arange(len(group_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, group_total, width, label='总样本', 
                    color=[group_colors[g] for g in group_names], alpha=0.4, edgecolor='white')
    bars2 = ax2.bar(x + width/2, group_correct, width, label='正确预测', 
                    color=[group_colors[g] for g in group_names], alpha=0.9, edgecolor='white')
    
    for i, (acc, correct) in enumerate(zip(group_accuracy, group_correct)):
        ax2.annotate(f'{acc:.0f}%', xy=(x[i], correct + 0.3),
                    ha='center', fontsize=10, fontweight='bold', color=COLORS['text'])
    
    ax2.set_ylabel('样本数量', fontproperties=chinese_font_prop if chinese_font_prop else None, 
                   fontsize=12, fontweight='bold')
    ax2.set_xlabel('类别', fontproperties=chinese_font_prop if chinese_font_prop else None, 
                   fontsize=12, fontweight='bold')
    ax2.set_title('各类别样本分布与准确率', fontproperties=chinese_font_prop if chinese_font_prop else None,
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, fontproperties=chinese_font_prop if chinese_font_prop else None, fontsize=10)
    ax2.legend(loc='upper right', fontsize=10, prop=chinese_font_prop if chinese_font_prop else None)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_error_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/fig4_error_analysis.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig4: 错误分析图已保存")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("生成SCI顶会风格实验结果图")
    print("=" * 60)
    print()
    
    create_radar_chart()
    create_comparison_bar_chart()
    create_category_performance_chart()
    create_error_analysis_chart()
    
    print()
    print("=" * 60)
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
