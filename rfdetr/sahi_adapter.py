"""
RF-DETR SAHI Adapter
===================
为 RF-DETR 提供 SAHI（Slicing Aided Hyper Inference）支持，
实现滑动窗口切片推理，提升大图小目标检测效果。

用法示例:
    from rfdetr.sahi_adapter import RFDETRDetectionModel
    from sahi.predict import get_sliced_prediction

    # 加载模型
    detection_model = RFDETRDetectionModel(
        model_path="path/to/weights.pt",
        confidence_threshold=0.3,
        device="cuda"
    )

    # 滑动窗口推理
    result = get_sliced_prediction(
        image="large_image.jpg",
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# 尝试导入 SAHI
try:
    from sahi.models.base import DetectionModel
    from sahi.prediction import ObjectPrediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    DetectionModel = object  # Fallback for type hints
    logger.warning("SAHI is not installed. Please install it with: pip install sahi")


class RFDETRDetectionModel(DetectionModel):
    """
    RF-DETR 的 SAHI 适配器类。
    
    支持从预训练权重或已初始化的模型加载。
    
    Args:
        model_path: 模型权重路径 (.pt 文件)
        model: 已初始化的 RFDETRBase/RFDETRLarge 等模型实例
        model_type: 模型类型，可选 'base', 'large', 'small', 'medium', 'nano'
        num_classes: 类别数量
        confidence_threshold: 置信度阈值
        device: 设备 ('cuda' 或 'cpu')
        category_mapping: 类别 ID 到名称的映射
        category_remapping: 类别 ID 重映射
        load_at_init: 是否在初始化时加载模型
        image_size: 模型输入分辨率 (如果使用自定义分辨率)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        model_type: str = "base",
        num_classes: int = 80,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: Optional[int] = None,
        **kwargs,
    ):
        if not SAHI_AVAILABLE:
            raise ImportError(
                "SAHI is not installed. Please install it with: pip install sahi"
            )
        
        self._model_type = model_type
        self._num_classes = num_classes
        self._image_size = image_size
        self._rfdetr_model = model  # 可以直接传入已初始化的模型
        
        super().__init__(
            model_path=model_path,
            model=model,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            **kwargs,
        )
    
    def check_dependencies(self) -> None:
        """检查依赖项是否已安装。"""
        if not SAHI_AVAILABLE:
            raise ImportError(
                "SAHI is not installed. Please install it with: pip install sahi"
            )
    
    def load_model(self) -> None:
        """加载 RF-DETR 模型。"""
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall, RFDETRMedium, RFDETRNano
        
        # 如果已经提供了模型实例，直接使用
        if self._rfdetr_model is not None:
            self.model = self._rfdetr_model
            logger.info("Using provided RF-DETR model instance")
        else:
            # 根据类型选择模型类
            model_classes = {
                "base": RFDETRBase,
                "large": RFDETRLarge,
                "small": RFDETRSmall,
                "medium": RFDETRMedium,
                "nano": RFDETRNano,
            }
            
            if self._model_type not in model_classes:
                raise ValueError(
                    f"Unknown model type: {self._model_type}. "
                    f"Available types: {list(model_classes.keys())}"
                )
            
            ModelClass = model_classes[self._model_type]
            
            # 创建模型实例
            self.model = ModelClass(num_classes=self._num_classes)
            
            # 加载权重
            if self.model_path:
                self._load_weights(self.model_path)
                logger.info(f"Loaded RF-DETR weights from: {self.model_path}")
        
        # 设置设备
        if self.device:
            self.model.model.model.to(self.device)
        
        # 设置分辨率 (如果指定)
        if self._image_size:
            self.model.model.resolution = self._image_size
        
        # 优化推理
        self.model.model.model.eval()
        
        # 设置类别名称
        self.category_names = self.model.class_names
        
        logger.info(f"RF-DETR model loaded successfully. "
                   f"Resolution: {self.model.model.resolution}, "
                   f"Classes: {self._num_classes}")
    
    def _load_weights(self, weights_path: str) -> None:
        """加载模型权重。"""
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # 处理可能的 key 前缀问题
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        
        self.model.model.model.load_state_dict(state_dict, strict=False)
    
    def set_model(self, model: Any) -> None:
        """设置模型实例。"""
        self.model = model
    
    @property
    def num_categories(self) -> int:
        """返回类别数量。"""
        return self._num_classes
    
    @property
    def has_mask(self) -> bool:
        """RF-DETR 是否支持分割 mask。"""
        return False
    
    @property
    def category_names(self) -> Dict[int, str]:
        """返回类别名称映射。"""
        if hasattr(self, "_category_names") and self._category_names:
            return self._category_names
        if hasattr(self.model, "class_names"):
            return self.model.class_names
        return {}
    
    @category_names.setter
    def category_names(self, value: Dict[int, str]) -> None:
        """设置类别名称映射。"""
        self._category_names = value
    
    def perform_inference(self, image: np.ndarray) -> List[Dict]:
        """
        对单张图片执行推理。
        
        Args:
            image: numpy 数组格式的图片 (H, W, C)，BGR 或 RGB 格式
            
        Returns:
            推理结果列表
        """
        # 确保图像是 RGB 格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image)
        
        # 执行推理
        with torch.no_grad():
            detections = self.model.predict(
                pil_image, 
                threshold=self.confidence_threshold
            )
        
        # 存储原始检测结果供后续处理
        self._original_predictions = detections
        
        return [detections]
    
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = None,
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        将 RF-DETR 的检测结果转换为 SAHI 的 ObjectPrediction 格式。
        
        Args:
            shift_amount_list: [[shift_x, shift_y]] 切片在原图中的偏移量列表
            full_shape_list: [[height, width]] 原图尺寸列表
        """
        object_prediction_list = []
        
        detections = self._original_predictions
        
        if detections is None or len(detections) == 0:
            self._object_prediction_list_per_image = [object_prediction_list]
            return
        
        # 处理 shift_amount_list 参数 - 可能是 [[x, y]] 或 [x, y] 格式
        if shift_amount_list is None:
            shift_amount = [0, 0]
        elif isinstance(shift_amount_list, list) and len(shift_amount_list) > 0:
            if isinstance(shift_amount_list[0], list):
                # 格式: [[x, y], ...]
                shift_amount = shift_amount_list[0]
            else:
                # 格式: [x, y]
                shift_amount = shift_amount_list
        else:
            shift_amount = [0, 0]
        
        # 处理 full_shape_list 参数
        if full_shape_list is None:
            full_shape = None
        elif isinstance(full_shape_list, list) and len(full_shape_list) > 0:
            if isinstance(full_shape_list[0], list):
                full_shape = full_shape_list[0]
            else:
                full_shape = full_shape_list
        else:
            full_shape = None
        
        # 遍历所有检测结果
        for i in range(len(detections)):
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            score = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            
            # 应用偏移量
            x1 = float(bbox[0]) + shift_amount[0]
            y1 = float(bbox[1]) + shift_amount[1]
            x2 = float(bbox[2]) + shift_amount[0]
            y2 = float(bbox[3]) + shift_amount[1]
            
            # 边界裁剪
            if full_shape is not None:
                x1 = max(0, min(x1, full_shape[1]))
                y1 = max(0, min(y1, full_shape[0]))
                x2 = max(0, min(x2, full_shape[1]))
                y2 = max(0, min(y2, full_shape[0]))
            
            # 跳过无效的框
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 获取类别名称
            category_name = self.category_names.get(class_id, str(class_id))
            
            # 创建 ObjectPrediction
            object_prediction = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                score=score,
                category_id=class_id,
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            
            object_prediction_list.append(object_prediction)
        
        # 设置结果到 _object_prediction_list_per_image (SAHI API 要求)
        self._object_prediction_list_per_image = [object_prediction_list]


def get_rfdetr_sahi_model(
    model_path: Optional[str] = None,
    model: Optional[Any] = None,
    model_type: str = "base",
    num_classes: int = 80,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
    image_size: Optional[int] = None,
) -> RFDETRDetectionModel:
    """
    便捷函数，快速获取 RF-DETR 的 SAHI 适配模型。
    
    Args:
        model_path: 模型权重路径
        model: 已初始化的模型实例
        model_type: 模型类型 ('base', 'large', 'small', 'medium', 'nano')
        num_classes: 类别数
        confidence_threshold: 置信度阈值
        device: 设备
        image_size: 输入分辨率
        
    Returns:
        RFDETRDetectionModel 实例
    """
    return RFDETRDetectionModel(
        model_path=model_path,
        model=model,
        model_type=model_type,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=image_size,
    )
