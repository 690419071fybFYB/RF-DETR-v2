#!/usr/bin/env python3
"""
VisDrone Training Script with Density Guidance
训练配置针对密集小目标场景优化
Dataset: VisDrone-DET 2019
"""

from rfdetr import RFDETRBase

# ====== Model Configuration ======
model = RFDETRBase(
    num_classes=10,              # VisDrone: 10 类别
    hidden_dim=256,
    use_density_guidance=True,   # 启用密度引导 (适合密集场景)
    density_hidden_dim=256,
)

# ====== Training Configuration ======
model.train(
    # Dataset paths
    dataset_file='coco',
    dataset_dir='/root/datasets/VisDrone',
    coco_path='/root/datasets/VisDrone',
    resolution=616,
    # Basic training params
    epochs=30,
    batch_size=6,               # 降低batch_size以适应VisDrone的大图
    grad_accum_steps=2,          # 梯度累积，有效batch_size=12
    lr=1e-4,
    num_workers=1,
    
    # Density Guidance (针对密集小目标)
    use_density_guidance=True,
    density_hidden_dim=256,
    density_loss_coef=0.5,
    
    
    # Output Directory
    output_dir='results/5_visdrone_density',
    
    # VisDrone specific
    num_classes=10,  # pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
)
