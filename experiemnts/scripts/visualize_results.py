#!/usr/bin/env python3
"""
可视化训练结果脚本
读取 results.json 并生成多个可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """加载结果 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_class_metrics(data, output_dir):
    """绘制各类别的指标对比图"""
    valid_data = data['class_map']['valid']
    test_data = data['class_map']['test']
    
    # 提取类别名称(排除 'all')
    classes = [item['class'] for item in valid_data if item['class'] != 'all']
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各类别性能指标对比', fontsize=16, fontweight='bold')
    
    # 1. mAP@50:95 对比
    ax1 = axes[0, 0]
    valid_map5095 = [item['map@50:95'] for item in valid_data if item['class'] != 'all']
    test_map5095 = [item['map@50:95'] for item in test_data if item['class'] != 'all']
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, valid_map5095, width, label='Valid', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, test_map5095, width, label='Test', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('mAP@50:95', fontsize=12)
    ax1.set_title('mAP@50:95 对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. mAP@50 对比
    ax2 = axes[0, 1]
    valid_map50 = [item['map@50'] for item in valid_data if item['class'] != 'all']
    test_map50 = [item['map@50'] for item in test_data if item['class'] != 'all']
    
    bars1 = ax2.bar(x - width/2, valid_map50, width, label='Valid', alpha=0.8, color='#3498db')
    bars2 = ax2.bar(x + width/2, test_map50, width, label='Test', alpha=0.8, color='#e74c3c')
    
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('mAP@50', fontsize=12)
    ax2.set_title('mAP@50 对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Precision 对比
    ax3 = axes[1, 0]
    valid_precision = [item['precision'] for item in valid_data if item['class'] != 'all']
    test_precision = [item['precision'] for item in test_data if item['class'] != 'all']
    
    bars1 = ax3.bar(x - width/2, valid_precision, width, label='Valid', alpha=0.8, color='#2ecc71')
    bars2 = ax3.bar(x + width/2, test_precision, width, label='Test', alpha=0.8, color='#f39c12')
    
    ax3.set_xlabel('类别', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision 对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=15)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Recall 对比
    ax4 = axes[1, 1]
    valid_recall = [item['recall'] for item in valid_data if item['class'] != 'all']
    test_recall = [item['recall'] for item in test_data if item['class'] != 'all']
    
    bars1 = ax4.bar(x - width/2, valid_recall, width, label='Valid', alpha=0.8, color='#9b59b6')
    bars2 = ax4.bar(x + width/2, test_recall, width, label='Test', alpha=0.8, color='#1abc9c')
    
    ax4.set_xlabel('类别', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Recall 对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes, rotation=15)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'class_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {output_path}")
    plt.close()

def plot_overall_metrics(data, output_dir):
    """绘制整体指标雷达图"""
    valid_all = [item for item in data['class_map']['valid'] if item['class'] == 'all'][0]
    test_all = [item for item in data['class_map']['test'] if item['class'] == 'all'][0]
    
    # 准备数据
    categories = ['mAP@50:95', 'mAP@50', 'Precision', 'Recall']
    valid_values = [
        valid_all['map@50:95'],
        valid_all['map@50'],
        valid_all['precision'],
        valid_all['recall']
    ]
    test_values = [
        test_all['map@50:95'],
        test_all['map@50'],
        test_all['precision'],
        test_all['recall']
    ]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    valid_values += valid_values[:1]
    test_values += test_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, valid_values, 'o-', linewidth=2, label='Valid', color='#3498db')
    ax.fill(angles, valid_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, test_values, 'o-', linewidth=2, label='Test', color='#e74c3c')
    ax.fill(angles, test_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('整体性能指标雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_path = output_dir / 'overall_metrics_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {output_path}")
    plt.close()

def plot_class_performance_heatmap(data, output_dir):
    """绘制类别性能热力图"""
    valid_data = data['class_map']['valid']
    
    # 提取所有类别(包括 all)
    classes = [item['class'] for item in valid_data]
    metrics = ['mAP@50:95', 'mAP@50', 'Precision', 'Recall']
    
    # 构建数据矩阵
    matrix = []
    for item in valid_data:
        row = [
            item['map@50:95'],
            item['map@50'],
            item['precision'],
            item['recall']
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    
    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标签
    for i in range(len(classes)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Valid 集各类别性能热力图', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('指标值', rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'class_performance_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {output_path}")
    plt.close()

def print_summary(data):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("📊 训练结果摘要")
    print("="*60)
    
    valid_all = [item for item in data['class_map']['valid'] if item['class'] == 'all'][0]
    test_all = [item for item in data['class_map']['test'] if item['class'] == 'all'][0]
    
    print("\n【整体性能 - Valid 集】")
    print(f"  mAP@50:95: {valid_all['map@50:95']:.4f}")
    print(f"  mAP@50:    {valid_all['map@50']:.4f}")
    print(f"  Precision: {valid_all['precision']:.4f}")
    print(f"  Recall:    {valid_all['recall']:.4f}")
    
    print("\n【整体性能 - Test 集】")
    print(f"  mAP@50:95: {test_all['map@50:95']:.4f}")
    print(f"  mAP@50:    {test_all['map@50']:.4f}")
    print(f"  Precision: {test_all['precision']:.4f}")
    print(f"  Recall:    {test_all['recall']:.4f}")
    
    print("\n【各类别性能 - Valid 集】")
    for item in data['class_map']['valid']:
        if item['class'] != 'all':
            print(f"  {item['class']:12s}: mAP@50:95={item['map@50:95']:.4f}, "
                  f"mAP@50={item['map@50']:.4f}, "
                  f"P={item['precision']:.4f}, R={item['recall']:.4f}")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--json', type=str, 
                       default='results/1baseline/results.json',
                       help='结果 JSON 文件路径')
    parser.add_argument('--output', type=str,
                       default='results/1baseline/visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载数据
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"❌ 错误: 文件不存在 {json_path}")
        return
    
    print(f"📂 读取结果文件: {json_path}")
    data = load_results(json_path)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 打印摘要
    print_summary(data)
    
    # 生成可视化图表
    print("\n🎨 生成可视化图表...")
    plot_class_metrics(data, output_dir)
    plot_overall_metrics(data, output_dir)
    plot_class_performance_heatmap(data, output_dir)
    
    print(f"\n✅ 完成! 所有图表已保存到: {output_dir}")
    print(f"   - class_metrics_comparison.png (各类别指标对比)")
    print(f"   - overall_metrics_radar.png (整体性能雷达图)")
    print(f"   - class_performance_heatmap.png (性能热力图)")

if __name__ == '__main__':
    main()
