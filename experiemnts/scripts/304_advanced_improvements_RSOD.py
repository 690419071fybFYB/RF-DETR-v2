#!/usr/bin/env python3
"""
Advanced Improvements Training Script (方案一 + 方案三 + 方案五)
- 方案一: CIoU Loss - 改进边界框回归
- 方案三: Density-Aware Query Selection - 密度感知Query选择
- 方案五: Per-Image Density Enhancement - 逐图像密度引导增强

Dataset: RSOD
Branch: feat/advanced-improvements
"""

from rfdetr import RFDETRBase

# Create model instance
model = RFDETRBase()

# Train with all three improvements enabled
model.train(
    dataset_file='coco',
    dataset_dir='/root/RSOD_cocoFormat/',
    coco_path='/root/RSOD_cocoFormat/',
    
    # Basic training params
    epochs=50,
    batch_size=12,
    grad_accum_steps=2,
    lr=1e-4,
    num_workers=2,
    
    # ====== 方案一: CIoU Loss ======
    use_ciou_loss=True,              # 启用CIoU Loss
    ciou_loss_coef=2.0,              # CIoU损失权重 (与GIoU相同)
    
    # ====== 方案三 + 方案五: Density Guidance ======
    # 需要 projector_scale 包含 P3, P4, P5 以支持密度预测器
    use_density_guidance=True,       # 启用密度引导 (包含方案三和方案五)
    density_hidden_dim=256,          # 密度预测器隐藏层维度
    density_loss_coef=0.5,           # 密度损失系数
    projector_scale=["P3", "P4", "P5"],  # 多尺度特征用于密度预测
    
    # Output Directory
    output_dir='results/304_advanced_improvements_RSOD',
    
    # Other params
    num_classes=4,  # RSOD: aircraft, oiltank, overpass, playground
)
