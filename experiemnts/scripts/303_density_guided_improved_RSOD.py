#!/usr/bin/env python3
"""
Density Guided Training (Improved Version)
Fixes the batch averaging issue in query enhancement to allow per-image density guidance.
Dataset: RSOD
"""

from rfdetr import RFDETRBase

# Create model instance
model = RFDETRBase()

# Train with density guidance enabled
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
    
    # Density Guidance Params
    use_density_guidance=True,           # Enable density guidance
    density_hidden_dim=256,              # Density predictor hidden dim
    density_loss_coef=0.5,               # Density loss coefficient
    
    # Output Directory
    output_dir='results/303_density_guided_improved_RSOD',
    
    # Other params
    num_classes=2,  # Using 2 classes as per previous RSOD configuration
)
