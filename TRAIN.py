# 这是一个用于训练YOLO模型的脚本，使用了自定义的数据集配置文件pollen.yaml。
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")

    model.train(data=r"pollen.yaml", epochs=3, batch=-1, imgsz=640, cache="ram", workers=1)
