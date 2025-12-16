#!/usr/bin/env python3
"""
VisDrone Baseline Training Script (Without Density Guidance)
用于对比实验的基线模型
Dataset: VisDrone-DET 2019
"""

from rfdetr import RFDETRBase

# ====== Baseline Model (No Density Guidance) ======
model = RFDETRBase(
    num_classes=10,              # VisDrone: 10 classes
    hidden_dim=256,
    use_density_guidance=False,  # Disable density guidance for baseline
)

# ====== Training Configuration ======
model.train(
    # Dataset paths
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/VisDrone/coco_format',
    coco_path='/home/fyb/datasets/VisDrone/coco_format',
    
    # Basic training params (same as density-guided version)
    epochs=50,
    batch_size=6,
    grad_accum_steps=2,
    lr=1e-4,
    num_workers=4,
    
    # Output Directory
    output_dir='results/visdrone_baseline',
    
    # VisDrone specific
    num_classes=10,
)
