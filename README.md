# YOLOv11 花粉识别与计数项目

使用YOLOv11训练花粉识别和计数模型的完整项目。

## 项目结构

```
14.计数训练/
├── main.py                 # 主训练脚本
├── prepare_dataset.py      # 数据集准备工具
├── inference.py            # 推理和计数脚本
├── requirements.txt        # 项目依赖
├── README.md              # 本文件
└── dataset/               # 数据集目录（需要创建）
    ├── images/
    │   ├── train/         # 训练集图像
    │   ├── val/           # 验证集图像
    │   └── test/          # 测试集图像
    └── labels/
        ├── train/         # 训练集标注
        ├── val/           # 验证集标注
        └── test/          # 测试集标注
```

## 数据准备

### 1. 数据格式要求

#### 图像格式
- 支持格式：JPG, JPEG, PNG, BMP
- 推荐分辨率：640x640 或更高
- 图像应清晰，花粉特征明显

#### 标注格式（YOLO格式）
每个图像对应一个同名的 `.txt` 标注文件，格式如下：

```
<class_id> <x_center> <y_center> <width> <height>
```

其中：
- `class_id`：类别ID（花粉为0）
- `x_center`：边界框中心点X坐标（归一化到0-1）
- `y_center`：边界框中心点Y坐标（归一化到0-1）
- `width`：边界框宽度（归一化到0-1）
- `height`：边界框高度（归一化到0-1）

示例：
```
0 0.508333 0.531250 0.154167 0.168750
0 0.745833 0.312500 0.120833 0.137500
0 0.275000 0.718750 0.145833 0.162500
```

### 2. 数据标注工具推荐

可以使用以下工具进行标注：

1. **LabelImg**（推荐）
   - 开源免费，支持YOLO格式
   - 下载：https://github.com/heartexlabs/labelImg

2. **Roboflow**（在线工具）
   - 网页端标注，支持自动转换格式
   - 网址：https://roboflow.com/

3. **CVAT**
   - 功能强大，支持团队协作
   - 网址：https://cvat.org/

### 3. 创建数据集

运行准备脚本创建标准目录结构：

```bash
python prepare_dataset.py
```

然后手动将数据放入相应目录：

1. 将花粉图像放入 `dataset/images/train`、`val`、`test` 目录
2. 将对应的标注文件放入 `dataset/labels/train`、`val`、`test` 目录
3. 确保图像和标注文件名一致（例如：`image001.jpg` 对应 `image001.txt`）

### 4. 数据集规模建议

- **最小数据集**：100-200张图像（用于快速实验）
- **推荐数据集**：500-1000张图像（较好效果）
- **理想数据集**：1000+张图像（最佳效果）

数据集分割比例：
- 训练集：70%
- 验证集：20%
- 测试集：10%

## 环境安装

### 1. 创建Python虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "from ultralytics import YOLO; print('安装成功！')"
```

## 训练模型

### 1. 基础训练

```bash
python main.py
```

### 2. 自定义训练参数

编辑 `main.py` 中的参数：

```python
model, results = train_yolo_model(
    data_yaml=data_yaml,
    model_size='n',      # 模型大小：'n', 's', 'm', 'l', 'x'
    epochs=100,          # 训练轮数
    img_size=640,        # 图像尺寸
    batch_size=16        # 批次大小
)
```

### 3. 模型大小选择

| 模型 | 速度 | 精度 | 参数量 | 适用场景 |
|------|------|------|--------|----------|
| YOLOv11n | 最快 | 较低 | 最少 | 快速实验、实时应用 |
| YOLOv11s | 快 | 中等 | 少 | 平衡性能 |
| YOLOv11m | 中等 | 较高 | 中等 | 推荐使用 |
| YOLOv11l | 慢 | 高 | 多 | 高精度要求 |
| YOLOv11x | 最慢 | 最高 | 最多 | 最高精度要求 |

### 4. 训练输出

训练完成后，模型和结果保存在：
```
runs/train/pollen_detection/
├── weights/
│   ├── best.pt       # 最佳模型
│   └── last.pt       # 最后一轮模型
├── results.png       # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
└── val_batch0_pred.jpg   # 验证集预测示例
```

## 模型推理与计数

### 1. 单张图像预测

```python
from inference import PollenCounter

counter = PollenCounter('runs/train/pollen_detection/weights/best.pt')
count, confidences = counter.count_single_image(
    'path/to/image.jpg',
    visualize=True,
    save_path='result.jpg'
)
print(f"检测到 {count} 个花粉")
```

### 2. 批量图像处理

```python
counter = PollenCounter('runs/train/pollen_detection/weights/best.pt')
df = counter.count_batch_images(
    image_dir='path/to/images/',
    output_csv='results.csv',
    save_visualizations=True,
    output_dir='./results'
)
```

### 3. 视频处理

```python
counter = PollenCounter('runs/train/pollen_detection/weights/best.pt')
counter.count_video(
    video_path='input_video.mp4',
    output_video_path='output_video.mp4'
)
```

## 性能优化技巧

### 1. 数据增强
- 已在训练脚本中启用（旋转、缩放、翻转等）
- 可根据需要调整增强参数

### 2. 超参数调优
- 学习率：根据训练曲线调整
- Batch size：根据显存大小调整（GPU显存不足可减小）
- 图像尺寸：更大的尺寸可能提高精度但降低速度

### 3. 硬件加速
- GPU训练：自动使用GPU（如果可用）
- CPU训练：设置 `device='cpu'`（较慢）

## 常见问题

### Q1: 显存不足怎么办？
A: 减小 `batch_size` 或降低 `img_size`

### Q2: 训练速度太慢？
A: 使用更小的模型（如 `yolo11n`）或减少 `epochs`

### Q3: 检测精度不高？
A: 
- 增加训练数据
- 使用更大的模型
- 调整置信度阈值
- 改进数据标注质量

### Q4: 如何导出模型？
A: 使用 `export_model()` 函数导出为ONNX等格式

```python
from main import export_model
from ultralytics import YOLO

model = YOLO('runs/train/pollen_detection/weights/best.pt')
model.export(format='onnx')  # 导出为ONNX格式
```

## 参考资料

- [Ultralytics YOLOv11 文档](https://docs.ultralytics.com/)
- [YOLO格式标注说明](https://docs.ultralytics.com/datasets/detect/)
- [数据增强技术](https://docs.ultralytics.com/modes/train/#augmentation)

## 许可证

本项目仅供学习和研究使用。

## 更新日志

- 2025-11-07: 初始版本，支持YOLOv11训练和推理
