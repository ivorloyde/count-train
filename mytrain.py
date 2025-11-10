from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolo12s.pt')

    model.train(
        data=r"pollenv5.yaml",
        epochs=100,
        batch=-1,
        imgsz=640,
        cache="ram",
        workers=1
    )