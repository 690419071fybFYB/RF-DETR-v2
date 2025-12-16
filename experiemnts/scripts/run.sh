#!/bin/bash

# 设置错误时退出
set -e

echo "🚀 开始运行实验流程..."

echo "[1/3] 运行实验1..."
python /root/RF-DETR-v2/experiemnts/scripts/1baseline.py

echo "[2/3] 运行实验2..."
python /root/RF-DETR-v2/experiemnts/scripts/301_density_guided_RSOD.py

echo "[3/3] 运行实验3..."
python /root/RF-DETR-v2/experiemnts/scripts/303_density_guided_improved_RSOD.py

echo "✅ 所有实验完成！"