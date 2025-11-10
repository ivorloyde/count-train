import os
import glob
from ultralytics import YOLO

# 假设你想遍历的文件夹路径是 'path_to_folder'
path_to_folder = r"D:\基因组所工作\14.计数训练\runs\detect\train15\weights"

# 使用glob模块找到所有.pt文件
for pt_file in glob.glob(os.path.join(path_to_folder, "best.pt")):
    model = YOLO(pt_file)
    # Export model
    success = model.export(format="onnx")



#命令行输入
#yolo export model=D:/基因组所工作/14.计数训练/runs/detect/train14/weights/best.pt format=onnx