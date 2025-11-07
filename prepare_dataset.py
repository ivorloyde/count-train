"""
数据集准备工具脚本
用于创建标准的YOLO数据集目录结构
"""
import os
import shutil
from pathlib import Path
import random


def create_dataset_structure(base_path='./dataset'):
    """
    创建YOLO数据集标准目录结构
    
    目录结构:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    directories = [
        os.path.join(base_path, 'images', 'train'),
        os.path.join(base_path, 'images', 'val'),
        os.path.join(base_path, 'images', 'test'),
        os.path.join(base_path, 'labels', 'train'),
        os.path.join(base_path, 'labels', 'val'),
        os.path.join(base_path, 'labels', 'test'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("数据集目录结构创建完成！")
    return base_path


def split_dataset(source_images_dir, source_labels_dir, 
                  dataset_path='./dataset', 
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    将原始数据集按比例分割为训练集、验证集和测试集
    
    参数:
        source_images_dir: 原始图像目录
        source_labels_dir: 原始标注文件目录
        dataset_path: 目标数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(source_images_dir).glob(f'*{ext}'))
        image_files.extend(Path(source_images_dir).glob(f'*{ext.upper()}'))
    
    image_files = [f.stem for f in image_files]
    random.shuffle(image_files)
    
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    # 分割数据
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    print(f"总文件数: {total}")
    print(f"训练集: {len(train_files)} ({train_ratio*100:.1f}%)")
    print(f"验证集: {len(val_files)} ({val_ratio*100:.1f}%)")
    print(f"测试集: {len(test_files)} ({test_ratio*100:.1f}%)")
    
    # 复制文件到相应目录
    def copy_files(file_list, split_name):
        for filename in file_list:
            # 复制图像
            for ext in image_extensions:
                src_img = os.path.join(source_images_dir, f"{filename}{ext}")
                if os.path.exists(src_img):
                    dst_img = os.path.join(dataset_path, 'images', split_name, f"{filename}{ext}")
                    shutil.copy2(src_img, dst_img)
                    break
            
            # 复制标注文件
            src_label = os.path.join(source_labels_dir, f"{filename}.txt")
            if os.path.exists(src_label):
                dst_label = os.path.join(dataset_path, 'labels', split_name, f"{filename}.txt")
                shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print("数据集分割完成！")


def split_dataset_sklearn(source_images_dir, source_labels_dir,
                          dataset_path='./dataset',
                          val_size=0.2, test_size=0.1, random_state=42):
    """
    使用 sklearn.model_selection.train_test_split 对数据集进行随机划分并复制到目标目录。

    参数:
        source_images_dir: 原始图像目录（字符串或 Path）
        source_labels_dir: 原始标注目录（字符串或 Path）
        dataset_path: 目标数据集路径
        val_size: 验证集比例（相对于全部数据）
        test_size: 测试集比例（相对于全部数据）
        random_state: 随机种子

    注意：该函数不做分层（stratify），因为每张图像可能包含多个目标。若需要分层，请自行实现基于标签统计的分层逻辑。
    """
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    source_images_dir = Path(source_images_dir)
    source_labels_dir = Path(source_labels_dir)
    dataset_path = Path(dataset_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(sorted(source_images_dir.glob(f'*{ext}')))
        image_paths.extend(sorted(source_images_dir.glob(f'*{ext.upper()}')))

    image_paths = list(dict.fromkeys(image_paths))  # 去重并保持顺序
    total = len(image_paths)
    if total == 0:
        print('未在 source_images_dir 中找到任何图像文件')
        return

    # 首先划分出 test 集
    trainval_paths, test_paths = train_test_split(
        image_paths, test_size=test_size, random_state=random_state
    )

    # 然后从 trainval 中划分出 val（相对于 trainval 的比例）
    if (1 - test_size) <= 0:
        raise ValueError('test_size 过大，剩余用于 train/val 的数据为 0')
    val_ratio_relative = val_size / (1 - test_size)

    train_paths, val_paths = train_test_split(
        trainval_paths, test_size=val_ratio_relative, random_state=random_state
    )

    # 创建目标目录
    dirs = [
        dataset_path / 'images' / 'train',
        dataset_path / 'images' / 'val',
        dataset_path / 'images' / 'test',
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val',
        dataset_path / 'labels' / 'test',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    def copy_list(paths, split_name):
        for p in paths:
            # 复制图像
            dst_img = dataset_path / 'images' / split_name / p.name
            shutil.copy2(p, dst_img)

            # 复制对应标注文件（.txt）
            label_file = source_labels_dir / f"{p.stem}.txt"
            if label_file.exists():
                dst_label = dataset_path / 'labels' / split_name / f"{p.stem}.txt"
                shutil.copy2(label_file, dst_label)

    copy_list(train_paths, 'train')
    copy_list(val_paths, 'val')
    copy_list(test_paths, 'test')

    print(f"总图像数: {total}")
    print(f"训练集: {len(train_paths)}")
    print(f"验证集: {len(val_paths)}")
    print(f"测试集: {len(test_paths)}")
    print('使用 sklearn 划分完成！')


def convert_labelme_to_yolo(labelme_json_path, output_txt_path, img_width, img_height, class_mapping):
    """
    将LabelMe格式的JSON标注转换为YOLO格式
    
    YOLO格式: <class_id> <x_center> <y_center> <width> <height>
    所有坐标都归一化到[0,1]
    
    参数:
        labelme_json_path: LabelMe JSON文件路径
        output_txt_path: 输出YOLO格式txt文件路径
        img_width: 图像宽度
        img_height: 图像高度
        class_mapping: 类别名称到ID的映射，如 {'pollen': 0}
    """
    import json
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_txt_path, 'w') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_mapping:
                continue
            
            class_id = class_mapping[label]
            points = shape['points']
            
            # 计算边界框
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            # 转换为YOLO格式（归一化）
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # 写入文件
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def create_sample_label(output_path, num_samples=5):
    """
    创建示例标注文件
    """
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            # 随机生成归一化坐标
            x_center = random.uniform(0.1, 0.9)
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.05, 0.2)
            height = random.uniform(0.05, 0.2)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"示例标注文件已创建: {output_path}")


def main():
    """
    主函数 - 演示如何使用
    """
    # 1. 创建数据集目录结构
    dataset_path = create_dataset_structure('./dataset')
    
    # 2. 如果你有原始数据，可以使用split_dataset函数分割
    # split_dataset(
    #     source_images_dir='path/to/original/images',
    #     source_labels_dir='path/to/original/labels',
    #     dataset_path=dataset_path
    # )
    
    # 3. 创建示例标注文件（用于测试）
    create_sample_label(os.path.join(dataset_path, 'labels', 'train', 'sample.txt'))
    
    print("\n数据集准备完成！")
    print("下一步：")
    print("1. 将花粉图像放入 dataset/images/train、val、test 目录")
    print("2. 将对应的YOLO格式标注文件放入 dataset/labels/train、val、test 目录")
    print("3. 确保图像和标注文件名一致（除了扩展名）")
    print("4. 运行 main.py 开始训练")


if __name__ == '__main__':
    main()
