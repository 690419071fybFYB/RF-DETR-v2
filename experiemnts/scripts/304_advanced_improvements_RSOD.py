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

# ====== 模型结构参数必须在初始化时传递 ======
model = RFDETRBase(
    # ====== 方案三 + 方案五: Density Guidance (模型结构) ======
    use_density_guidance=True,       # 启用密度引导
    density_hidden_dim=256,          # 密度预测器隐藏层维度
    # projector_scale 使用默认值 ["P4"]，因为预训练权重只支持单尺度
    # 如果要使用多尺度，需要从头训练或使用 pretrain_weights=None
    
    # ====== 方案一: CIoU Loss (模型配置) ======
    use_ciou_loss=True,              # 启用CIoU Loss
)

# ====== 训练参数在 train() 中传递 ======
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
    
    # ====== 损失权重 (训练参数) ======
    ciou_loss_coef=2.0,              # CIoU损失权重 (与GIoU相同)
    density_loss_coef=0.5,           # 密度损失系数
    
    # Output Directory
    output_dir='results/304_advanced_improvements_RSOD',
    
    # Other params
    num_classes=4,  # RSOD: aircraft, oiltank, overpass, playground
)
