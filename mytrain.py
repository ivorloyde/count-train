from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolo11m.pt')

    model.train(
        data=r"pollenv2.yaml",
        epochs=100,
        batch=-1,
        imgsz=640,
        cache="ram",
        workers=1
    )