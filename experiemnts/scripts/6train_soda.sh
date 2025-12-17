#!/bin/bash
#
# SODA-A Filtered Subset 训练启动脚本
# 
# 用法:
#   ./train_soda.sh baseline   # 运行基线版本
#   ./train_soda.sh density    # 运行密度引导版本
#   ./train_soda.sh both       # 两个版本都运行（依次）
#

set -e

# 切换到项目目录
cd /root/RF-DETR-v2
# 激活环境（如果需要）
# source /home/fyb/envs/torch-rfdetr-v2/bin/activate

# 检查参数
MODE=${1:-baseline}

echo "=========================================="
echo "SODA-A Filtered Subset 训练"
echo "=========================================="
echo "训练模式: $MODE"
echo "数据集: /home/fyb/datasets/SODA-A_Filtered_Subset"
echo "类别数: 6 (car, tractor, van, pickup, boat, plane)"
echo "=========================================="
echo ""

if [ "$MODE" == "baseline" ]; then
    echo "🚀 启动基线训练..."
    python3 experiemnts/scripts/6train_soda_baseline.py

elif [ "$MODE" == "density" ]; then
    echo "🚀 启动密度引导训练..."
    python3 experiemnts/scripts/6train_soda_density.py

elif [ "$MODE" == "both" ]; then
    echo "🚀 启动基线训练..."
    python3 experiemnts/scripts/6train_soda_baseline.py
    
    echo ""
    echo "✅ 基线训练完成！"
    echo ""
    echo "🚀 启动密度引导训练..."
    python3 experiemnts/scripts/6train_soda_density.py
    
    echo ""
    echo "✅ 所有训练完成！"

else
    echo "❌ 无效的模式: $MODE"
    echo "   支持的模式: baseline, density, both"
    exit 1
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果保存在:"
if [ "$MODE" == "baseline" ] || [ "$MODE" == "both" ]; then
    echo "  - results/soda_filtered_baseline/"
fi
if [ "$MODE" == "density" ] || [ "$MODE" == "both" ]; then
    echo "  - results/soda_filtered_density/"
fi
echo "=========================================="
