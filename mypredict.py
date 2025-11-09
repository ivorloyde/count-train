from ultralytics.models import YOLO

model = YOLO(r'yolov11n.pt')

model.predict(
    source=r"ultralytics/assets",
    save=True,
    show=False,
)