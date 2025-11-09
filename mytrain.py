from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolov11n.pt')

    model.train(
        data=r"ultralytics/assets",
        epochs=10,
        batch=-1,
        imgsz=1280,
        cache="ram",
        wokers=1,
    )