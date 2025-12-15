#!/usr/bin/env python3
"""
Smoke Test for Advanced Improvements (方案一 + 方案三 + 方案五)
验证新功能是否可以正常初始化和前向传播
"""

import sys
import os
# Ensure we import local rfdetr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from rfdetr import RFDETRBase
from rfdetr.util.misc import NestedTensor

def test_advanced_improvements():
    print("=" * 60)
    print("🚀 Smoke Test: Advanced Improvements")
    print("=" * 60)
    
    # 1. Test Model Initialization
    print("\n[1/4] Testing Model Initialization...")
    try:
        # 使用 P4 单尺度匹配预训练权重, 或者跳过预训练权重
        model_wrapper = RFDETRBase(
            hidden_dim=256, 
            use_density_guidance=True,
            density_hidden_dim=256,
            projector_scale=["P4"],  # 使用默认单尺度以匹配预训练权重
            use_ciou_loss=True,  # 方案一
        )
        model = model_wrapper.model.model 
        model.train()
        print("   ✅ Model initialization successful!")
        
        # Check if all components are present
        print("   - use_density_guidance:", model.use_density_guidance)
        print("   - density_predictor:", hasattr(model, 'density_predictor'))
        print("   - density_query_enhancer:", hasattr(model, 'density_query_enhancer'))
        print("   - density_query_selector:", hasattr(model, 'density_query_selector'))
        
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Test Forward Pass
    print("\n[2/4] Testing Forward Pass...")
    try:
        device = next(model.parameters()).device
        print(f"   Model is on device: {device}")
        
        batch_size = 2
        channels = 3
        height, width = 560, 560  # Must be divisible by 56 for DINOv2
        
        images = torch.randn(batch_size, channels, height, width).to(device)
        masks = torch.zeros((batch_size, height, width), dtype=torch.bool).to(device)
        samples = NestedTensor(images, masks)
        
        with torch.no_grad():
            outputs = model(samples)
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        print(f"   ✅ Forward pass successful!")
        print(f"   - pred_logits shape: {pred_logits.shape}")
        print(f"   - pred_boxes shape: {pred_boxes.shape}")
        
        if 'pred_density' in outputs:
            print(f"   - pred_density shape: {outputs['pred_density'].shape}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Test CIoU Loss Function
    print("\n[3/4] Testing CIoU Loss Function...")
    try:
        from rfdetr.util import box_ops
        
        # Create test boxes
        boxes1 = torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]], dtype=torch.float32)
        boxes2 = torch.tensor([[110, 110, 210, 210], [60, 60, 160, 160]], dtype=torch.float32)
        
        giou = box_ops.generalized_box_iou(boxes1, boxes2)
        ciou = box_ops.complete_box_iou(boxes1, boxes2)
        diou = box_ops.distance_box_iou(boxes1, boxes2)
        
        print(f"   ✅ IoU functions work correctly!")
        print(f"   - GIoU diagonal: {torch.diag(giou).tolist()}")
        print(f"   - CIoU diagonal: {torch.diag(ciou).tolist()}")
        print(f"   - DIoU diagonal: {torch.diag(diou).tolist()}")
        
    except Exception as e:
        print(f"   ❌ CIoU Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test DensityAwareQuerySelector
    print("\n[4/4] Testing DensityAwareQuerySelector...")
    try:
        from rfdetr.models.density_module import DensityAwareQuerySelector
        
        selector = DensityAwareQuerySelector(
            num_queries=300,
            hidden_dim=256,
            min_activation=0.3,
            temperature=1.0
        )
        
        # Test input
        query_feat = torch.randn(2, 300, 256)  # [B, N, D]
        density_map = torch.rand(2, 1, 35, 35)  # [B, 1, H, W]
        
        weighted_query = selector(query_feat, density_map)
        
        print(f"   ✅ DensityAwareQuerySelector works correctly!")
        print(f"   - Input shape: {query_feat.shape}")
        print(f"   - Output shape: {weighted_query.shape}")
        print(f"   - Activation range: [{weighted_query.min():.4f}, {weighted_query.max():.4f}]")
        
    except Exception as e:
        print(f"   ❌ DensityAwareQuerySelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 All tests passed! The code is ready for training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_advanced_improvements()
    sys.exit(0 if success else 1)
