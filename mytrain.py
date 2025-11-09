from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r'yolo11l.pt')

    model.train(
        data=r"pollen.yaml",
        epochs=250,
        batch=-1,
        imgsz=800,
        cache="ram",
        workers=1
    )