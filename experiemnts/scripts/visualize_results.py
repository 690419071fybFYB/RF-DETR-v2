#!/usr/bin/env python3
"""
Visualization script for multi-model training results comparison
Reads multiple results.json files and generates comparative visualization charts.
Supports any number of models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import itertools

# Set font styles
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """Load JSON result file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_color_palette(n):
    """Get distinct colors for n models"""
    # Use tab10 for small N, tab20 for larger N
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        return plt.cm.tab20(np.linspace(0, 1, 20))

def plot_bar_comparison(models_data, output_dir, split='test'):
    """Plot grouped bar chart for class-wise metrics comparison"""
    if not models_data: return

    # Extract classes from the first model (assuming consistent classes across models)
    # Filter out 'all' to only show specific classes
    first_data = models_data[0]['data']['class_map'][split]
    classes = [item['class'] for item in first_data if item['class'] != 'all']
    
    metrics = [
        ('mAP@50:95', 'map@50:95'), 
        ('mAP@50', 'map@50'),
        ('Precision', 'precision'), 
        ('Recall', 'recall')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'Multi-Model Class-wise Comparison ({split.capitalize()} Set)', fontsize=20, fontweight='bold')
    
    n_models = len(models_data)
    colors = get_color_palette(n_models)
    
    x = np.arange(len(classes))
    total_width = 0.8
    bar_width = total_width / n_models
    
    for idx, (label, key) in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        for i, model in enumerate(models_data):
            model_name = model['name']
            model_vals = []
            
            # Retrieve value for each class
            current_split_data = model['data']['class_map'][split]
            for cls_name in classes:
                # Find matching class data
                cls_data = next((item for item in current_split_data if item['class'] == cls_name), None)
                val = cls_data[key] if cls_data else 0.0
                model_vals.append(val)
            
            # Calculate offset for grouped bars
            offset = (i - n_models/2) * bar_width + bar_width/2
            bars = ax.bar(x + offset, model_vals, bar_width, label=model_name, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
            
            # Add value labels on top of each bar
            for bar, val in zip(bars, model_vals):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),  # 2 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7, fontweight='bold',
                           rotation=90)
        
        ax.set_ylabel(label, fontsize=14)
        ax.set_title(f'{label}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
        
        # Only add legend to the first subplot to avoid clutter, or add outside if needed
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
    out_file = output_dir / f'multi_model_bar_{split}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Bar chart saved: {out_file}")
    plt.close()

def plot_radar_comparison(models_data, output_dir, split='test'):
    """Plot radar chart for overall metrics comparison"""
    categories = ['mAP@50:95', 'mAP@50', 'Precision', 'Recall']
    keys = ['map@50:95', 'map@50', 'precision', 'recall']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = get_color_palette(len(models_data))
    
    for i, model in enumerate(models_data):
        # find 'all' category
        all_res = next((item for item in model['data']['class_map'][split] if item['class'] == 'all'), None)
        if not all_res: continue
        
        values = [all_res.get(k, 0.0) for k in keys]
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.05, color=colors[i])
        
        # Add value labels at each point on the radar chart
        for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            # Adjust label position based on angle to avoid overlap
            ha = 'left' if 0 < angle < np.pi else 'right'
            if abs(angle - np.pi/2) < 0.1 or abs(angle - 3*np.pi/2) < 0.1:
                ha = 'center'
            
            # Offset for different models to reduce overlap
            offset_r = 0.03 + i * 0.02
            ax.text(angle, value + offset_r, f'{value:.3f}',
                   ha=ha, va='bottom',
                   fontsize=8, fontweight='bold',
                   color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    ax.set_title(f'Overall Performance Radar ({split.capitalize()})', fontsize=18, fontweight='bold', pad=30)
    
    # Place legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_file = output_dir / f'multi_model_radar_{split}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Radar chart saved: {out_file}")
    plt.close()

def plot_scale_metrics(models_data, output_dir, split='test'):
    """Plot scale-specific metrics (small/medium/large)"""
    if not models_data: return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Scale-Specific Performance ({split.capitalize()} Set)', fontsize=18, fontweight='bold')
    
    colors = get_color_palette(len(models_data))
    
    # Plot 1: Scale-specific mAP
    ax = axes[0]
    scales = ['Small', 'Medium', 'Large']
    keys = ['map_small', 'map_medium', 'map_large']
    
    x = np.arange(len(scales))
    width = 0.8 / len(models_data)
    
    for i, model in enumerate(models_data):
        all_res = next((item for item in model['data']['class_map'][split] if item['class'] == 'all'), None)
        if not all_res: continue
        
        values = [all_res.get(k, 0.0) for k in keys]
        offset = (i - len(models_data)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=model['name'], color=colors[i], alpha=0.85)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Scale-Specific mAP', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: Scale-specific AR
    ax = axes[1]
    ar_keys = ['ar_small', 'ar_medium', 'ar_large']
    
    for i, model in enumerate(models_data):
        all_res = next((item for item in model['data']['class_map'][split] if item['class'] == 'all'), None)
        if not all_res: continue
        
        values = [all_res.get(k, 0.0) for k in ar_keys]
        offset = (i - len(models_data)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=model['name'], color=colors[i], alpha=0.85)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('AR', fontsize=12)
    ax.set_title('Scale-Specific Average Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    out_file = output_dir / f'multi_model_scale_metrics_{split}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Scale metrics chart saved: {out_file}")
    plt.close()

def plot_ar_metrics(models_data, output_dir, split='test'):
    """Plot Average Recall metrics at different IoU thresholds"""
    if not models_data: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Average Recall Comparison ({split.capitalize()} Set)', fontsize=18, fontweight='bold')
    
    colors = get_color_palette(len(models_data))
    
    ar_types = ['AR@1', 'AR@10', 'AR@100']
    ar_keys = ['ar@1', 'ar@10', 'ar@100']
    
    x = np.arange(len(ar_types))
    width = 0.8 / len(models_data)
    
    for i, model in enumerate(models_data):
        all_res = next((item for item in model['data']['class_map'][split] if item['class'] == 'all'), None)
        if not all_res: continue
        
        values = [all_res.get(k, 0.0) for k in ar_keys]
        offset = (i - len(models_data)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=model['name'], color=colors[i], alpha=0.85)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Average Recall', fontsize=12)
    ax.set_title('Average Recall at Different Max Detections', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ar_types)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    out_file = output_dir / f'multi_model_ar_metrics_{split}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ AR metrics chart saved: {out_file}")
    plt.close()

def print_summary(models_data, split='test'):
    """Print a comprehensive comparison table of all available metrics"""
    print("\n" + "="*140)
    print(f"📊 Multi-Model Performance Summary ({split.capitalize()} Set)")
    print("="*140)
    
    # Calculate column width based on max name length
    name_width = max([len(m['name']) for m in models_data] + [10]) + 2
    name_width = min(name_width, 25) # Cap width
    
    header = f"{'Metric':<20}"
    for model in models_data:
        disp_name = (model['name'][:name_width-3] + '..') if len(model['name']) > name_width else model['name']
        header += f" | {disp_name:^{name_width}}"
    print(header)
    print("-" * len(header))
    
    # Get all available metrics from first model's 'all' class
    first_all = next((item for item in models_data[0]['data']['class_map'][split] if item['class'] == 'all'), None)
    if not first_all:
        print("❌ No 'all' class data found")
        return
    
    # Define metric categories and their keys
    metric_groups = [
        ("🎯 Main Metrics", [
            ('mAP@50:95', 'map@50:95'),
            ('mAP@50', 'map@50'),
            ('mAP@75', 'map@75'),
            ('Precision', 'precision'),
            ('Recall', 'recall')
        ]),
        ("📏 Scale-specific mAP", [
            ('mAP (small)', 'map_small'),
            ('mAP (medium)', 'map_medium'),
            ('mAP (large)', 'map_large')
        ]),
        ("🎯 Average Recall (AR)", [
            ('AR@1', 'ar@1'),
            ('AR@10', 'ar@10'),
            ('AR@100', 'ar@100')
        ]),
        ("📏 Scale-specific AR", [
            ('AR (small)', 'ar_small'),
            ('AR (medium)', 'ar_medium'),
            ('AR (large)', 'ar_large')
        ])
    ]
    
    # Print metrics by groups
    for group_name, metrics in metric_groups:
        print(f"\n{group_name}")
        print("-" * len(header))
        
        for label, key in metrics:
            # Check if metric exists
            if key not in first_all:
                continue
                
            row = f"{label:<20}"
            for model in models_data:
                all_res = next((item for item in model['data']['class_map'][split] if item['class'] == 'all'), None)
                val = all_res.get(key, 0.0) if all_res else 0.0
                row += f" | {val:^{name_width}.4f}"
            print(row)
    
    print("="*140 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize training results for multiple models')
    
    # Use nargs='+' to accept list of files
    parser.add_argument('--jsons', nargs='+', required=True, 
                        help='List of result JSON paths. Example: --jsons exp1/res.json exp2/res.json')
    parser.add_argument('--names', nargs='+', 
                        help='List of model names corresponding to the JSONs. If skipped, folder names will be used.')
    parser.add_argument('--output', type=str, default='results/comparison_viz', 
                        help='Directory to save output charts')
    
    args = parser.parse_args()
    
    json_paths = [Path(p) for p in args.jsons]
    
    # Determine model names
    model_names = []
    if args.names:
        if len(args.names) != len(json_paths):
            print(f"❌ Error: Provided {len(args.names)} names for {len(json_paths)} files. Counts must match.")
            return
        model_names = args.names
    else:
        # Auto-generate names from paths
        for p in json_paths:
            # Use parent folder name as model name (often meaningful in experiments)
            # If parent is just 'results', use stem (filename without extension)
            name = p.parent.name if p.parent.name != 'results' else p.stem
            model_names.append(name)
    
    # Load data
    models_data = []
    print(f"🔍 Found {len(json_paths)} models to compare.")
    
    for p, name in zip(json_paths, model_names):
        if not p.exists():
            print(f"⚠️ Warning: File {p} does not exist. Skipping.")
            continue
            
        try:
            print(f"📂 Loading: {name} <- {p}")
            data = load_results(p)
            models_data.append({'name': name, 'data': data, 'path': p})
        except Exception as e:
            print(f"❌ Error loading {p}: {e}")

    if not models_data:
        print("❌ No valid data loaded. Exiting.")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Generating comparison visualizations...")
    
    # 1. Print Summary Table
    print_summary(models_data, split='test')
    
    # 2. Plot Charts
    plot_bar_comparison(models_data, output_dir, split='test')
    plot_radar_comparison(models_data, output_dir, split='test')
    plot_scale_metrics(models_data, output_dir, split='test')
    plot_ar_metrics(models_data, output_dir, split='test')
    
    print(f"\n✨ Comparison Complete! Results saved to: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
