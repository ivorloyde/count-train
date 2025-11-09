from ultralytics.models import YOLO

model = YOLO(r"yolo11n.pt")

model.predict(
    source=r"ultralytics/assets",
    save=True,
    show=False,
)
