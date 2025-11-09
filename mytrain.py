from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolo11n.pt')

    model.train(
        data=r"pollen.yaml",
        epochs=100,
        batch=-1,
        imgsz=1280,
        cache="ram",
        workers=1
    )