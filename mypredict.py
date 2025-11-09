from ultralytics.models import YOLO

model = YOLO(r'D:\基因组所工作\14.计数训练\runs\detect\train2\weights\best.pt')

model.predict(
    source=r"D:\基因组所工作\数据集\pollen\others",
    save=True,
    show=False,
    save_txt=True,
)