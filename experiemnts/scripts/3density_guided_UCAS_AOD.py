#!/usr/bin/env python3
"""
测试密度引导查询增强模块的训练脚本
用于在UCAS-AOD数据集上进行初步验证
"""

from rfdetr import RFDETRBase

# 创建模型实例
model = RFDETRBase()

# 启用密度引导的训练配置
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/UCAS_AOD_COCO',
    coco_path='/home/fyb/datasets/UCAS_AOD_COCO',
    
    # 基础训练参数
    epochs=5,
    batch_size=4,
    grad_accum_steps=1,
    lr=1e-4,
    num_workers=2,
    
    # 密度引导参数
    use_density_guidance=True,          # 启用密度引导
    density_hidden_dim=256,              # 密度预测器隐藏维度
    density_loss_coef=0.5,               # 密度损失权重
    
    # 输出目录
    output_dir='results/3density_guided_UCAS_AOD',
    
    # 其他参数
    num_classes=2,  # UCAS-AOD has 2 classes (car, plane)
)
