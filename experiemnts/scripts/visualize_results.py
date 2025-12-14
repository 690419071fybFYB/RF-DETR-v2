#!/usr/bin/env python3
"""
Visualization script for training results
Reads results.json and generates various visualization charts
Supports single model analysis and dual model comparison
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Use English labels to avoid font compatibility issues
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """加载结果 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== 单模型可视化函数 ====================

def plot_class_metrics(data, output_dir):
    """(Single Model) Plot class-wise metrics comparison"""
    valid_data = data['class_map']['valid']
    test_data = data['class_map']['test']
    
    # Extract class names (exclude 'all')
    classes = [item['class'] for item in valid_data if item['class'] != 'all']
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class-wise Performance Metrics Comparison (Valid vs Test)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('mAP@50:95', 'map@50:95'),
        ('mAP@50', 'map@50'),
        ('Precision', 'precision'),
        ('Recall', 'recall')
    ]
    
    x = np.arange(len(classes))
    width = 0.35
    
    for idx, (label, key) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        valid_vals = [item[key] for item in valid_data if item['class'] != 'all']
        test_vals = [item[key] for item in test_data if item['class'] != 'all']
        
        bars1 = ax.bar(x - width/2, valid_vals, width, label='Valid', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8, color='#e74c3c')
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'single_model_class_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {output_path}")
    plt.close()

def plot_overall_radar(data, output_dir):
    """(Single Model) Plot overall metrics radar chart"""
    valid_all = [item for item in data['class_map']['valid'] if item['class'] == 'all'][0]
    test_all = [item for item in data['class_map']['test'] if item['class'] == 'all'][0]
    
    categories = ['mAP@50:95', 'mAP@50', 'Precision', 'Recall']
    valid_values = [valid_all['map@50:95'], valid_all['map@50'], valid_all['precision'], valid_all['recall']]
    test_values = [test_all['map@50:95'], test_all['map@50'], test_all['precision'], test_all['recall']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    valid_values += valid_values[:1]
    test_values += test_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, valid_values, 'o-', linewidth=2, label='Valid', color='#3498db')
    ax.fill(angles, valid_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, test_values, 'o-', linewidth=2, label='Test', color='#e74c3c')
    ax.fill(angles, test_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar Chart (Valid vs Test)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_path = output_dir / 'single_model_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {output_path}")
    plt.close()

# ==================== 双模型对比可视化函数 ====================

def plot_comparison_bar(data1, data2, name1, name2, output_dir, split='test'):
    """(Comparison) Plot dual-model class-wise metrics comparison"""
    dataset1 = data1['class_map'][split]
    dataset2 = data2['class_map'][split]
    
    classes = [item['class'] for item in dataset1 if item['class'] != 'all']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance Comparison ({name1} vs {name2}) - {split.capitalize()} Set', fontsize=16, fontweight='bold')
    
    metrics = [
        ('mAP@50:95', 'map@50:95'),
        ('mAP@50', 'map@50'),
        ('Precision', 'precision'),
        ('Recall', 'recall')
    ]
    
    x = np.arange(len(classes))
    width = 0.35
    
    for idx, (label, key) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        vals1 = [item[key] for item in dataset1 if item['class'] != 'all']
        vals2 = [item[key] for item in dataset2 if item['class'] != 'all']
        
        bars1 = ax.bar(x - width/2, vals1, width, label=name1, alpha=0.8, color='#2ecc71')
        bars2 = ax.bar(x + width/2, vals2, width, label=name2, alpha=0.8, color='#9b59b6')
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05) # Leave space for labels
        
        # Add value labels and improvement rate
        for i, (v1, v2) in enumerate(zip(vals1, vals2)):
            ax.text(x[i] - width/2, v1, f'{v1:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Calculate difference
            diff = v2 - v1
            color = 'red' if diff >= 0 else 'blue'
            sign = '+' if diff >= 0 else ''
            # Show on top of the second bar
            ax.text(x[i] + width/2, v2, f'{v2:.2f}\n({sign}{diff:.2f})', 
                   ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / f'comparison_bar_{split}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存对比图: {output_path}")
    plt.close()

def plot_comparison_radar(data1, data2, name1, name2, output_dir, split='test'):
    """(Comparison) Plot dual-model overall metrics radar chart"""
    all1 = [item for item in data1['class_map'][split] if item['class'] == 'all'][0]
    all2 = [item for item in data2['class_map'][split] if item['class'] == 'all'][0]
    
    categories = ['mAP@50:95', 'mAP@50', 'Precision', 'Recall']
    values1 = [all1['map@50:95'], all1['map@50'], all1['precision'], all1['recall']]
    values2 = [all2['map@50:95'], all2['map@50'], all2['precision'], all2['recall']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values1 += values1[:1]
    values2 += values2[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
    
    # Model 1
    ax.plot(angles, values1, 'o-', linewidth=2, label=name1, color='#2ecc71')
    ax.fill(angles, values1, alpha=0.15, color='#2ecc71')
    
    # Model 2
    ax.plot(angles, values2, 'o-', linewidth=2, label=name2, color='#9b59b6')
    ax.fill(angles, values2, alpha=0.15, color='#9b59b6')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add metric value annotations
    for i, angle in enumerate(angles[:-1]):
        # Model 1 label
        ax.text(angle, values1[i] + 0.05, f"{values1[i]:.2f}", 
               ha='center', va='center', color='#2ecc71', fontweight='bold')
        # Model 2 label (slightly offset if close)
        offset = 0.05 if abs(values2[i] - values1[i]) < 0.1 else 0
        if values2[i] < values1[i]: offset = -0.05
        ax.text(angle, values2[i] + offset, f"{values2[i]:.2f}", 
               ha='center', va='center', color='#9b59b6', fontweight='bold')

    ax.set_title(f'Overall Performance Radar Comparison ({split.capitalize()} Set)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_path = output_dir / f'comparison_radar_{split}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存对比雷达图: {output_path}")
    plt.close()

def print_comparison_summary(data1, data2, name1, name2, split='test'):
    """Print comparison summary table"""
    all1 = [item for item in data1['class_map'][split] if item['class'] == 'all'][0]
    all2 = [item for item in data2['class_map'][split] if item['class'] == 'all'][0]
    
    print("\n" + "="*70)
    print(f"📊 Model Performance Comparison Summary ({split.capitalize()} Set)")
    print("="*70)
    print(f"{'Metric':<15} | {name1:<20} | {name2:<20} | {'Change':<10}")
    print("-" * 70)
    
    metrics = [
        ('mAP@50:95', 'map@50:95'),
        ('mAP@50', 'map@50'),
        ('Precision', 'precision'),
        ('Recall', 'recall')
    ]
    
    for label, key in metrics:
        v1 = all1[key]
        v2 = all2[key]
        diff = v2 - v1
        sign = '+' if diff >= 0 else ''
        print(f"{label:<15} | {v1:.4f}{' '*14} | {v2:.4f}{' '*14} | {sign}{diff:.4f}")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize training results (supports single and dual model comparison)')
    
    # Basic parameters (backward compatible)
    parser.add_argument('--json', dest='json1', type=str, help='First result file path (or single file)')
    parser.add_argument('--json1', type=str, help='First result file path')
    parser.add_argument('--name1', type=str, default='Model A', help='First model name')
    
    # Comparison parameters
    parser.add_argument('--json2', type=str, help='Second result file path (for comparison)')
    parser.add_argument('--name2', type=str, default='Model B', help='Second model name')
    
    parser.add_argument('--output', type=str, default='results/comparison_viz', help='Output directory')
    
    args = parser.parse_args()
    
    # Handle parameter compatibility
    if args.json1 is None:
        print("❌ Error: Please provide at least --json or --json1 parameter")
        return

    json_path1 = Path(args.json1)
    
    # Check file 1
    if not json_path1.exists():
        print(f"❌ Error: File not found {json_path1}")
        return
        
    print(f"📂 Loading Model 1: {json_path1}")
    data1 = load_results(json_path1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine mode
    if args.json2:
        # === Dual model comparison mode ===
        json_path2 = Path(args.json2)
        if not json_path2.exists():
            print(f"❌ Error: File not found {json_path2}")
            return
            
        print(f"📂 Loading Model 2: {json_path2}")
        data2 = load_results(json_path2)
        
        print(f"\n🚀 Generating comparison report: {args.name1} vs {args.name2}")
        
        # 1. Print summary
        print_comparison_summary(data1, data2, args.name1, args.name2, split='test')
        
        # 2. Generate charts (Test set)
        plot_comparison_bar(data1, data2, args.name1, args.name2, output_dir, split='test')
        plot_comparison_radar(data1, data2, args.name1, args.name2, output_dir, split='test')
        
    else:
        # === Single model analysis mode ===
        print("\n🚀 Generating single model analysis report")
        plot_class_metrics(data1, output_dir)
        plot_overall_radar(data1, output_dir)
        
        # Print simple summary
        all_metrics = [x for x in data1['class_map']['test'] if x['class'] == 'all'][0]
        print(f"\n📊 {args.name1} Test Set Performance:")
        print(f"  mAP@50:95: {all_metrics['map@50:95']:.4f}")
        print(f"  mAP@50:    {all_metrics['map@50']:.4f}")

    print(f"\n✨ All done! Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
