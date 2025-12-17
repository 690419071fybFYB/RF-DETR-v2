"""
SODA-A Filtered Subset 训练脚本 - 密度引导版本

数据集: SODA-A (6类筛选子集)
- 只包含: car, tractor, van, pickup, boat, plane
- 训练集: 3,515 张切片 (640x640), 183,742 标注
- 验证集: 1,656 张切片 (640x640), 147,574 标注
- 密度: 平均 52.3 目标/图 (训练), 89.1 目标/图 (验证)
- 小目标占比: 95.4%

硬件: 4090 24GB

特性:
- 使用密度引导初始化
- 针对小目标密集场景优化
"""

from rfdetr import RFDETRBase

# 创建模型
model = RFDETRBase(
    num_classes=6,  # car, tractor, van, pickup, boat, plane
    use_density_guidance=True,  # 使用密度引导
)

# 训练
model.train(
    # 数据集配置
    dataset_file='coco',
    dataset_dir='/root/datasets/SODA-A_Filtered_Subset',
    coco_path='/root/datasets/SODA-A_Filtered_Subset',
    num_classes=6,
    
    # 训练配置
    epochs=30,
    batch_size=4,
    grad_accum_steps=2,  # 有效 batch_size = 16
    num_workers=2,
    
    # 模型配置
    resolution=560,  # 切片已是640，560足够
    multi_scale=True,
    expanded_scales=True,
    
    # 优化器配置
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
    
    # 输出配置
    output_dir='results/6soda_filtered_density',
    
    # 其他配置
    device='cuda',
    seed=42,
)
