#!/usr/bin/env python3
"""
RF-DETR SAHI 滑动窗口推理示例
============================

本脚本演示如何使用 SAHI 对大图进行滑动窗口切片推理，
提升 RF-DETR 对小目标的检测效果。

使用前请先安装 SAHI:
    pip install sahi

用法:
    python sahi_inference_demo.py --image path/to/large_image.jpg
    python sahi_inference_demo.py --image path/to/large_image.jpg --weights path/to/weights.pt
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="RF-DETR SAHI 滑动窗口推理示例")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径 (可选)")
    parser.add_argument("--model-type", type=str, default="base", 
                       choices=["base", "large", "small", "medium", "nano"],
                       help="模型类型")
    parser.add_argument("--num-classes", type=int, default=80, help="类别数")
    parser.add_argument("--slice-height", type=int, default=640, help="切片高度")
    parser.add_argument("--slice-width", type=int, default=640, help="切片宽度")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="切片重叠比例")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    parser.add_argument("--compare", action="store_true", help="同时对比普通推理和切片推理")
    args = parser.parse_args()
    
    # 导入相关库
    try:
        from sahi.predict import get_sliced_prediction, get_prediction
        from sahi.utils.cv import read_image
    except ImportError:
        print("错误: SAHI 未安装。请运行: pip install sahi")
        return
    
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall, RFDETRMedium, RFDETRNano
    from rfdetr.sahi_adapter import RFDETRDetectionModel
    
    # 选择模型类
    model_classes = {
        "base": RFDETRBase,
        "large": RFDETRLarge,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "nano": RFDETRNano,
    }
    ModelClass = model_classes[args.model_type]
    
    print(f"{'='*60}")
    print(f"RF-DETR SAHI 滑动窗口推理")
    print(f"{'='*60}")
    print(f"图片: {args.image}")
    print(f"模型类型: {args.model_type}")
    print(f"切片尺寸: {args.slice_width}x{args.slice_height}")
    print(f"重叠比例: {args.overlap_ratio}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"{'='*60}")
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    t0 = time.time()
    
    rfdetr_model = ModelClass(num_classes=args.num_classes)
    
    if args.weights:
        import torch
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "model" in checkpoint:
            rfdetr_model.model.model.load_state_dict(checkpoint["model"], strict=False)
        else:
            rfdetr_model.model.model.load_state_dict(checkpoint, strict=False)
        print(f"   已加载权重: {args.weights}")
    
    # 创建 SAHI 适配模型
    detection_model = RFDETRDetectionModel(
        model=rfdetr_model,
        num_classes=args.num_classes,
        confidence_threshold=args.conf_threshold,
        device=args.device,
    )
    print(f"   模型加载完成 ({time.time()-t0:.2f}s)")
    
    # 读取图片
    print("\n[2/4] 读取图片...")
    image = read_image(args.image)
    h, w = image.shape[:2]
    print(f"   图片尺寸: {w}x{h}")
    
    # 计算切片数量
    num_slices_h = int(np.ceil((h - args.slice_height * args.overlap_ratio) / 
                               (args.slice_height * (1 - args.overlap_ratio))))
    num_slices_w = int(np.ceil((w - args.slice_width * args.overlap_ratio) / 
                               (args.slice_width * (1 - args.overlap_ratio))))
    print(f"   预计切片数: {num_slices_h * num_slices_w} ({num_slices_w}x{num_slices_h})")
    
    # 执行切片推理
    print("\n[3/4] 执行切片推理...")
    t0 = time.time()
    
    sliced_result = get_sliced_prediction(
        image=args.image,
        detection_model=detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_ratio,
        overlap_width_ratio=args.overlap_ratio,
        postprocess_type="GREEDYNMM",  # NMS 后处理
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )
    
    sliced_time = time.time() - t0
    sliced_count = len(sliced_result.object_prediction_list)
    print(f"   切片推理完成 ({sliced_time:.2f}s)")
    print(f"   检测到 {sliced_count} 个目标")
    
    # 对比普通推理
    if args.compare:
        print("\n[3.5/4] 执行普通推理 (对比)...")
        t0 = time.time()
        
        normal_result = get_prediction(
            image=args.image,
            detection_model=detection_model,
        )
        
        normal_time = time.time() - t0
        normal_count = len(normal_result.object_prediction_list)
        print(f"   普通推理完成 ({normal_time:.2f}s)")
        print(f"   检测到 {normal_count} 个目标")
        
        print(f"\n   {'='*40}")
        print(f"   对比结果:")
        print(f"   {'='*40}")
        print(f"   普通推理: {normal_count} 个目标, {normal_time:.2f}s")
        print(f"   切片推理: {sliced_count} 个目标, {sliced_time:.2f}s")
        print(f"   目标增加: {sliced_count - normal_count} 个 ({(sliced_count/max(normal_count,1)-1)*100:.1f}%)")
        print(f"   {'='*40}")
    
    # 可视化结果
    print("\n[4/4] 保存可视化结果...")
    
    # 在图片上绘制检测框
    result_image = image.copy()
    
    for pred in sliced_result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        score = pred.score.value
        category = pred.category.name
        
        # 绘制边框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{category}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 保存结果
    if args.output is None:
        input_path = Path(args.image)
        output_path = input_path.parent / f"{input_path.stem}_sahi_result{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    cv2.imwrite(str(output_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"   结果已保存到: {output_path}")
    
    # 打印检测统计
    print(f"\n{'='*60}")
    print(f"检测统计:")
    print(f"{'='*60}")
    
    # 按类别统计
    category_counts = {}
    for pred in sliced_result.object_prediction_list:
        cat = pred.category.name
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print(f"\n总计: {sliced_count} 个目标")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
