"""
模型推理和计数脚本
用于批量处理图像并统计花粉数量
"""
from ultralytics.models import YOLO
import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime


class PollenCounter:
    """花粉计数器类"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化计数器
        
        参数:
            model_path: 训练好的模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"模型加载成功: {model_path}")
    
    def count_single_image(self, image_path, visualize=False, save_path=None):
        """
        对单张图像进行花粉计数
        
        参数:
            image_path: 图像路径
            visualize: 是否可视化结果
            save_path: 保存可视化结果的路径
        
        返回:
            count: 花粉数量
            confidences: 每个检测框的置信度列表
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        count = len(boxes)
        confidences = [box.conf[0].item() for box in boxes]
        
        if visualize or save_path:
            self.visualize_results(image_path, result, save_path)
        
        return count, confidences
    
    def count_batch_images(self, image_dir, output_csv='pollen_count_results.csv', 
                          save_visualizations=False, output_dir='./results'):
        """
        批量处理图像并统计花粉数量
        
        参数:
            image_dir: 图像目录
            output_csv: 输出CSV文件路径
            save_visualizations: 是否保存可视化结果
            output_dir: 可视化结果保存目录
        """
        if save_visualizations:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        results_data = []
        
        print(f"开始处理 {len(image_files)} 张图像...")
        for i, img_path in enumerate(image_files, 1):
            save_path = None
            if save_visualizations:
                save_path = os.path.join(output_dir, f"{img_path.stem}_result{img_path.suffix}")
            
            count, confidences = self.count_single_image(
                str(img_path), 
                visualize=save_visualizations, 
                save_path=save_path
            )
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            results_data.append({
                '图像名称': img_path.name,
                '花粉数量': count,
                '平均置信度': f"{avg_conf:.3f}",
                '最高置信度': f"{max(confidences):.3f}" if confidences else 'N/A',
                '最低置信度': f"{min(confidences):.3f}" if confidences else 'N/A',
                '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f"[{i}/{len(image_files)}] {img_path.name}: 检测到 {count} 个花粉")
        
        # 保存结果到CSV
        df = pd.DataFrame(results_data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_csv}")
        
        # 打印统计信息
        total_pollen = df['花粉数量'].sum()
        avg_pollen = df['花粉数量'].mean()
        print(f"\n统计信息:")
        print(f"  总图像数: {len(image_files)}")
        print(f"  总花粉数: {total_pollen}")
        print(f"  平均每张图像花粉数: {avg_pollen:.2f}")
        print(f"  最多花粉数: {df['花粉数量'].max()}")
        print(f"  最少花粉数: {df['花粉数量'].min()}")
        
        return df
    
    def visualize_results(self, image_path, result, save_path=None):
        """
        可视化检测结果
        
        参数:
            image_path: 原始图像路径
            result: YOLO检测结果
            save_path: 保存路径
        """
        image = cv2.imread(str(image_path))
        boxes = result.boxes
        
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"Pollen {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加总计数
        total_count = len(boxes)
        cv2.putText(image, f"Total: {total_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow('Pollen Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def count_video(self, video_path, output_video_path='output_video.mp4', 
                    show_frame_count=True):
        """
        对视频进行花粉计数
        
        参数:
            video_path: 输入视频路径
            output_video_path: 输出视频路径
            show_frame_count: 是否在视频上显示帧计数
        """
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_counts = []
        
        print("开始处理视频...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 对当前帧进行预测
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            count = len(boxes)
            total_counts.append(count)
            
            # 绘制检测结果
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加计数信息
            if show_frame_count:
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(frame)
            
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧...")
        
        cap.release()
        out.release()
        
        print(f"\n视频处理完成！")
        print(f"输出视频: {output_video_path}")
        print(f"总帧数: {frame_count}")
        print(f"平均花粉数: {sum(total_counts)/len(total_counts):.2f}")
        print(f"最大花粉数: {max(total_counts)}")


def main():
    """
    主函数 - 使用示例
    """
    # 1. 初始化计数器
    model_path = 'runs/train/pollen_detection/weights/best.pt'  # 替换为你的模型路径
    counter = PollenCounter(model_path, conf_threshold=0.25)
    
    # 2. 单张图像计数示例
    # count, confidences = counter.count_single_image(
    #     'path/to/image.jpg',
    #     visualize=True,
    #     save_path='result.jpg'
    # )
    # print(f"检测到 {count} 个花粉")
    
    # 3. 批量图像计数示例
    # df = counter.count_batch_images(
    #     image_dir='path/to/images/',
    #     output_csv='pollen_count_results.csv',
    #     save_visualizations=True,
    #     output_dir='./results'
    # )
    
    # 4. 视频计数示例
    # counter.count_video(
    #     video_path='path/to/video.mp4',
    #     output_video_path='output_video.mp4'
    # )
    
    print("请根据需要取消注释相应的代码块进行使用")


if __name__ == '__main__':
    main()
