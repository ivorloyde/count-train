# 这是一个使用Ultralytics YOLO模型进行预测的示例代码。
from ultralytics.models import YOLO

model = YOLO(r"yolo11n.pt")

model.predict(
    source=r"ultralytics/assets",
    save=True,
    show=False,
)
