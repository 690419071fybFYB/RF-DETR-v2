#!/usr/bin/env python3
"""
测试自适应密度-尺度损失加权的训练脚本
"""

from rfdetr import RFDETRBase
import argparse

def main():
    # 创建模型实例
    model = RFDETRBase()

    # 启用自适应损失加权的训练配置
    model.train(
        dataset_file='coco',
        dataset_dir='/root/RSOD_cocoFormat/',
        coco_path='/root/RSOD_cocoFormat/',
        
        # 基础训练参数
        epochs=50,
        batch_size=12,
        grad_accum_steps=2,
        lr=1e-4,
        num_workers=2,
        
        # 密度引导参数 (必须启用密度引导才能使用密度因子)
        use_density_guidance=True,
        density_hidden_dim=256,
        density_loss_coef=0.5,
        
        # 自适应损失加权参数
        enable_adaptive_loss_weighting=True,
        scale_coef=1.0,    # 尺度权重系数
        density_coef=0.5,  # 密度权重系数
        normalize_weights=True,
        
        # 输出目录
        output_dir='results/4adaptive_loss_weighting_RSOD',
        
        # 其他参数
        num_classes=4,
    )

if __name__ == '__main__':
    main()
