from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolo11n.pt')

    model.train(
        data=r"coco8.yaml",
        epochs=2,
        batch=-1,
        imgsz=640,
        cache="ram",
    )