# SODA-A Filtered Subset 训练指南

## 📊 数据集信息

- **数据集路径**: `/home/fyb/datasets/SODA-A_Filtered_Subset`
- **类别数**: 6 (car, tractor, van, pickup, boat, plane)
- **训练集**: 3,515 张图片 (640×640), 183,742 标注
- **验证集**: 1,656 张图片 (640×640), 147,574 标注
- **平均密度**: 52.3 目标/图 (训练), 89.1 目标/图 (验证)
- **小目标占比**: ~95%

## 🚀 快速开始

### 方法1: 使用Shell脚本（推荐）

```bash
cd /home/fyb/mydir/rf-detr-origin/rf-detr

# 运行基线版本
./experiemnts/scripts/train_soda.sh baseline

# 运行密度引导版本
./experiemnts/scripts/train_soda.sh density

# 两个版本都运行
./experiemnts/scripts/train_soda.sh both
```

### 方法2: 直接运行Python脚本

```bash
cd /home/fyb/mydir/rf-detr-origin/rf-detr

# 基线版本
python3 experiemnts/scripts/train_soda_baseline.py

# 密度引导版本
python3 experiemnts/scripts/train_soda_density.py
```

### 方法3: 后台运行（nohup）

```bash
cd /home/fyb/mydir/rf-detr-origin/rf-detr

# 基线版本
nohup python3 experiemnts/scripts/train_soda_baseline.py > train_baseline.log 2>&1 &

# 密度引导版本
nohup python3 experiemnts/scripts/train_soda_density.py > train_density.log 2>&1 &
```

## 📁 训练脚本说明

### 1. `train_soda_baseline.py` - 基线版本

- **功能**: 标准RF-DETR训练，不使用密度引导
- **适用**: 建立性能基准
- **输出**: `results/soda_filtered_baseline/`

### 2. `train_soda_density.py` - 密度引导版本

- **功能**: 使用密度引导初始化，优化小目标检测
- **适用**: 针对密集小目标场景优化
- **输出**: `results/soda_filtered_density/`

### 3. `train_soda.sh` - 启动脚本

- **功能**: 便捷启动训练
- **用法**: `./train_soda.sh [baseline|density|both]`

## ⚙️ 训练配置

### 硬件配置
- GPU: 4090 24GB
- Batch Size: 4
- Gradient Accumulation: 4 (有效batch=16)

### 训练参数
- Epochs: 50
- Resolution: 560
- Learning Rate: 1e-4 (backbone: 1e-5)
- Multi-scale: True
- Workers: 4

### 预计训练时间
- 单次训练: ~3-4 小时 (4090)
- 两个版本: ~6-8 小时

## 📊 监控训练

### 查看日志

```bash
# 实时查看日志
tail -f results/soda_filtered_baseline/train.log

# 查看TensorBoard
tensorboard --logdir results/soda_filtered_baseline/
```

### 检查结果

训练结束后，结果会保存在：
```
results/soda_filtered_baseline/
├── checkpoint_best.pth      # 最佳模型
├── checkpoint_last.pth      # 最新模型
├── results.json             # 评估结果
└── train.log                # 训练日志
```

## 🔍 模型评估

训练完成后，比较两个版本的性能：

```bash
cd /home/fyb/mydir/rf-detr-origin/rf-detr

# 查看结果
cat results/soda_filtered_baseline/results.json
cat results/soda_filtered_density/results.json
```

## 📝 注意事项

1. **数据集已就绪**: 已通过完整性验证，可直接训练
2. **切片数据**: 图片已切成640×640，无需再切片
3. **类别映射**: 6个类别ID从1-6连续编码
4. **显存优化**: 如果显存不足，可降低batch_size或resolution

## 🐛 故障排除

### 问题1: CUDA Out of Memory
```python
# 降低batch_size
batch_size=2
grad_accum_steps=8  # 保持有效batch=16
```

### 问题2: 数据加载慢
```python
# 减少workers
num_workers=2
```

### 问题3: 训练中断
```python
# 从检查点恢复
resume_from='results/soda_filtered_baseline/checkpoint_last.pth'
```

## 📈 下一步

训练完成后：
1. 使用SAHI进行切片推理测试
2. 在完整SODA-A测试集上评估
3. 对比基线和密度引导版本的性能

---

**创建时间**: 2025-12-16
**数据集版本**: SODA-A Filtered Subset (Top 8% dense)
