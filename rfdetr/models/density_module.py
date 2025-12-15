# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
密度引导查询增强模块 (Density-Guided Query Enhancement)
包含密度图生成、密度预测、查询增强和密度感知Query选择四个核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple


class DensityMapGenerator:
    """
    从GT边界框生成高斯密度图,用于监督密度预测器训练
    """
    
    @staticmethod
    def generate_density_map(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        image_size: Tuple[int, int],
        sigma: float = 8.0,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        生成目标密度图
        
        Args:
            boxes: [N, 4] 归一化的边界框 (cx, cy, w, h), 范围 [0, 1]
            labels: [N] 类别标签
            image_size: (H, W) 密度图尺寸
            sigma: 高斯核标准差
            normalize: 是否归一化到 [0, 1]
            
        Returns:
            density_map: [H, W] 密度热力图
        """
        H, W = image_size
        density_map = torch.zeros(H, W, device=boxes.device, dtype=boxes.dtype)
        
        if len(boxes) == 0:
            return density_map
        
        # 将归一化坐标转换为像素坐标
        boxes_pixel = boxes.clone()
        boxes_pixel[:, 0] *= W  # cx
        boxes_pixel[:, 1] *= H  # cy
        boxes_pixel[:, 2] *= W  # w
        boxes_pixel[:, 3] *= H  # h
        
        # 为每个目标生成高斯核
        for box in boxes_pixel:
            cx, cy, bw, bh = box
            
            # 根据目标大小自适应调整sigma
            adaptive_sigma = sigma * min(bw, bh) / 64.0
            adaptive_sigma = max(adaptive_sigma, 1.0)  # 最小sigma为1
            
            # 计算高斯核的有效范围 (3*sigma)
            radius = int(3 * adaptive_sigma)
            
            # 计算高斯核的范围
            x_min = max(0, int(cx - radius))
            x_max = min(W, int(cx + radius + 1))
            y_min = max(0, int(cy - radius))
            y_max = min(H, int(cy + radius + 1))
            
            # 生成网格
            y_grid, x_grid = torch.meshgrid(
                torch.arange(y_min, y_max, device=boxes.device),
                torch.arange(x_min, x_max, device=boxes.device),
                indexing='ij'
            )
            
            # 计算高斯分布
            gaussian = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * adaptive_sigma ** 2))
            
            # 累加到密度图(取最大值以避免重叠区域过度增强)
            density_map[y_min:y_max, x_min:x_max] = torch.maximum(
                density_map[y_min:y_max, x_min:x_max],
                gaussian
            )
        
        # 归一化
        if normalize and density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        return density_map


class DensityPredictor(nn.Module):
    """
    密度预测器:融合多尺度特征预测目标密度图
    输入: S3 (H/8), S4 (H/16), S5 (H/32) 三个尺度的特征
    输出: 密度图 (H/16, W/16)
    """
    
    def __init__(
        self,
        in_channels: List[int] = [384, 384, 384],  # S3, S4, S5的通道数
        hidden_dim: int = 256,
        output_stride: int = 16  # 输出密度图相对于输入图像的步长
    ):
        super().__init__()
        
        self.output_stride = output_stride
        
        # 侧向连接:将不同尺度的特征对齐到相同通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # FPN融合卷积
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels))
        ])
        
        # 密度预测头
        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, C, H, W] 特征图
                      [S3 (H/8, W/8), S4 (H/16, W/16), S5 (H/32, W/32)]
        
        Returns:
            density_map: [B, 1, H/16, W/16] 密度图
        """
        assert len(features) == 3, f"Expected 3 features, got {len(features)}"
        
        # 侧向连接
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # 自顶向下融合 (FPN)
        # S5 -> S4 -> S3
        fpn_features = []
        
        # 从最粗尺度开始
        prev_feat = laterals[2]  # S5
        for i in range(2, -1, -1):
            if i < 2:
                # 上采样并相加
                upsampled = F.interpolate(
                    prev_feat,
                    size=laterals[i].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                prev_feat = laterals[i] + upsampled
            
            # 应用FPN卷积
            fpn_feat = self.fpn_convs[i](prev_feat)
            fpn_features.append(fpn_feat)
        
        # 使用S4尺度的特征预测密度图 (对应H/16)
        density_feature = fpn_features[1]  # S4对应的FPN特征
        
        # 预测密度图
        density_map = self.density_head(density_feature)
        
        return density_map


class DensityGuidedQueryEnhancer(nn.Module):
    """
    密度引导查询增强器:利用密度图增强query的位置编码
    采用位置偏置增强策略
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_feature_levels: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        
        # 密度值到位置偏置的映射
        self.density_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 门控机制:控制密度信息的引入程度
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        query_pos: torch.Tensor,
        density_map: torch.Tensor,
        reference_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_pos: [B, N, D] 查询位置编码
            density_map: [B, 1, H, W] 密度图
            reference_points: [B, N, 2] 或 [B, N, 4] 参考点坐标,归一化到[-1, 1]
        
        Returns:
            enhanced_query_pos: [B, N, D] 增强后的位置编码
        """
        B, N, D = query_pos.shape
        
        # 提取参考点的 (x, y) 坐标
        if reference_points.size(-1) == 4:
            # 如果是 (cx, cy, w, h) 格式,只取 (cx, cy)
            ref_points_xy = reference_points[..., :2]
        else:
            ref_points_xy = reference_points
        
        # 将归一化坐标从 [0, 1] 转换到 [-1, 1] (grid_sample要求)
        ref_points_normalized = ref_points_xy * 2.0 - 1.0
        
        # 从密度图采样参考点位置的密度值
        # grid_sample需要 [..., H, W, 2] 格式
        grid = ref_points_normalized.unsqueeze(2)  # [B, N, 1, 2]
        
        # 采样密度值
        density_values = F.grid_sample(
            density_map,  # [B, 1, H, W]
            grid,  # [B, N, 1, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [B, 1, N, 1]
        
        # 调整形状
        density_values = density_values.squeeze(-1).transpose(1, 2)  # [B, N, 1]
        
        # 将密度值映射为位置偏置
        density_bias = self.density_proj(density_values)  # [B, N, D]
        
        # 门控机制:根据查询位置编码和密度值动态控制增强强度
        gate_input = torch.cat([query_pos, density_values], dim=-1)  # [B, N, D+1]
        gate_weights = self.gate(gate_input)  # [B, N, D]
        
        # 应用门控的密度偏置
        enhanced_query_pos = query_pos + gate_weights * density_bias
        
        return enhanced_query_pos


class DensityAwareQuerySelector(nn.Module):
    """
    密度感知Query选择器 (方案三)
    根据场景密度动态调整query的激活程度
    在密集区域激活更多queries,稀疏区域减少queries
    """
    
    def __init__(
        self,
        num_queries: int = 300,
        hidden_dim: int = 256,
        min_activation: float = 0.3,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.min_activation = min_activation
        self.temperature = temperature
        
        # 密度统计 -> query激活权重
        # 输入: 密度图的全局统计特征 (mean, max, std)
        self.density_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # 生成每个query的激活权重
        self.query_activator = nn.Sequential(
            nn.Linear(128, num_queries),
            nn.Sigmoid()
        )
        
        # 可学习的query重要性先验 (某些queries天然更重要)
        self.query_prior = nn.Parameter(torch.ones(num_queries))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        query_feat: torch.Tensor,
        density_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_feat: [B, N, D] 或 [N, D] 查询特征
            density_map: [B, 1, H, W] 密度图
        
        Returns:
            weighted_query_feat: [B, N, D] 加权后的查询特征
        """
        B = density_map.shape[0]
        
        # 处理 query_feat 维度
        if query_feat.dim() == 2:
            # [N, D] -> [B, N, D]
            query_feat = query_feat.unsqueeze(0).expand(B, -1, -1)
        
        N, D = query_feat.shape[1], query_feat.shape[2]
        
        # 计算密度图的全局统计特征
        density_flat = density_map.view(B, -1)  # [B, H*W]
        density_mean = density_flat.mean(dim=1, keepdim=True)  # [B, 1]
        density_max = density_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
        density_std = density_flat.std(dim=1, keepdim=True)  # [B, 1]
        
        # 组合统计特征
        density_stats = torch.cat([density_mean, density_max, density_std], dim=1)  # [B, 3]
        
        # 编码密度统计
        density_encoded = self.density_encoder(density_stats)  # [B, 128]
        
        # 生成query激活权重
        activation_weights = self.query_activator(density_encoded)  # [B, N]
        
        # 结合可学习的先验
        query_prior_norm = torch.sigmoid(self.query_prior)  # [N]
        activation_weights = activation_weights * query_prior_norm.unsqueeze(0)  # [B, N]
        
        # 确保最小激活度
        activation_weights = self.min_activation + (1 - self.min_activation) * activation_weights
        
        # 温度缩放 (使分布更尖锐或平滑)
        if self.temperature != 1.0:
            activation_weights = activation_weights ** (1.0 / self.temperature)
        
        # 应用激活权重到query特征
        weighted_query_feat = query_feat * activation_weights.unsqueeze(-1)  # [B, N, D]
        
        return weighted_query_feat

