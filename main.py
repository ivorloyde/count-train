"""
YOLOv11 花粉识别与计数模型训练脚本
"""
from ultralytics.models import YOLO
import os
import yaml
from pathlib import Path
from prepare_dataset import split_dataset_sklearn, create_dataset_structure


def create_dataset_yaml(dataset_path='./dataset'):
    """
    创建数据集配置文件
    """
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # 可选
        'names': {
            0: 'pollen'  # 花粉类别，可以根据需要添加更多类别
        },
        'nc': 1  # 类别数量
    }
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, sort_keys=False)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    return yaml_path


def train_yolo_model(data_yaml, model_size='n', epochs=100, img_size=640, batch_size=16):
    """
    训练YOLOv11模型
    
    参数:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        img_size: 输入图像尺寸
        batch_size: 批次大小
    """
    # 加载预训练模型
    model_name = f'yolo11{model_size}.pt'
    model = YOLO(model_name)
    
    print(f"开始训练 {model_name} 模型...")
    print(f"训练参数: epochs={epochs}, img_size={img_size}, batch={batch_size}")
    
    # 训练模型
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='pollen_detection',
        project='runs/train',
        patience=50,  # 早停耐心值
        save=True,
        device=0,  # 使用GPU 0，如果没有GPU则设置为 'cpu'
        # 其他可选参数
        optimizer='AdamW',
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        augment=True,  # 数据增强
        mosaic=1.0,  # 马赛克增强
        mixup=0.0,  # mixup增强
        copy_paste=0.0,  # 复制粘贴增强
    )
    
    print("训练完成！")
    return model, results


def validate_model(model, data_yaml):
    """
    验证模型性能
    """
    print("开始验证模型...")
    metrics = model.val(data=data_yaml)
    
    print(f"验证结果:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def predict_and_count(model, image_path, conf_threshold=0.25, save=True):
    """
    使用训练好的模型进行预测和计数
    
    参数:
        model: 训练好的YOLO模型
        image_path: 图像路径
        conf_threshold: 置信度阈值
        save: 是否保存预测结果
    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save,
        project='runs/predict',
        name='pollen_count'
    )
    
    # 统计检测到的花粉数量
    for result in results:
        boxes = result.boxes
        pollen_count = len(boxes)
        print(f"图像 {image_path} 中检测到 {pollen_count} 个花粉")
        
        # 打印每个检测框的信息
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            print(f"  花粉 {i+1}: 置信度={conf:.2f}, 类别={cls}")
    
    return results


def export_model(model, format='onnx'):
    """
    导出模型为其他格式
    
    支持的格式: onnx, torchscript, tflite, etc.
    """
    print(f"导出模型为 {format} 格式...")
    model.export(format=format)
    print("模型导出完成！")


def main():
    """
    主函数
    """
    # 1. 设置路径
    dataset_path = './dataset'
    # 如果 dataset 下没有划分好的 train/val/test，则尝试使用 sklearn 的 train_test_split
    images_train_dir = os.path.join(dataset_path, 'images', 'train')
    need_split = True
    if os.path.exists(images_train_dir):
        # 检查目录中是否有图像文件
        has_files = any(os.scandir(images_train_dir))
        if has_files:
            need_split = False

    if need_split:
        # 尝试在若干候选源目录中找到原始图片与标注
        candidate_pairs = [
            ('./source_images', './source_labels'),
            ('./raw_images', './raw_labels'),
            ('./data/images', './data/labels'),
            ('./images_all', './labels_all'),
        ]

        found = False
        for src_img, src_lbl in candidate_pairs:
            if os.path.exists(src_img) and os.path.exists(src_lbl):
                print(f"发现原始数据：{src_img} 和 {src_lbl}，使用 sklearn 划分到 {dataset_path}")
                # 确保目标目录存在
                create_dataset_structure(dataset_path)
                # 使用默认比例：val=0.2, test=0.1
                split_dataset_sklearn(src_img, src_lbl, dataset_path=dataset_path, val_size=0.2, test_size=0.1, random_state=42)
                found = True
                break

        if not found:
            print("未找到原始数据用于自动划分；请将原始图像和标注放入其中一个候选目录，或手动创建 dataset 下的 train/val/test 结构。候选目录包括: ./source_images & ./source_labels, ./raw_images & ./raw_labels, ./data/images & ./data/labels, ./images_all & ./labels_all")
    
    # 2. 创建数据集配置文件
    data_yaml = create_dataset_yaml(dataset_path)
    
    # 3. 训练模型
    # model_size 可选: 'n'(nano), 's'(small), 'm'(medium), 'l'(large), 'x'(xlarge)
    model, results = train_yolo_model(
        data_yaml=data_yaml,
        model_size='n',  # 使用nano模型，速度快适合初期实验
        epochs=100,
        img_size=640,
        batch_size=16
    )
    
    # 4. 验证模型
    metrics = validate_model(model, data_yaml)
    
    # 5. 使用最佳模型进行预测
    best_model_path = 'runs/train/pollen_detection/weights/best.pt'
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        
        # 预测单张图像示例
        # predict_and_count(best_model, 'path/to/test/image.jpg', conf_threshold=0.25)
        
        # 预测整个文件夹
        # predict_and_count(best_model, 'path/to/test/images/', conf_threshold=0.25)
    
    # 6. 导出模型（可选）
    # export_model(best_model, format='onnx')
    
    print("所有任务完成！")


if __name__ == '__main__':
    main()
