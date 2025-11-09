from ultralytics.models import YOLO

model = YOLO(r'D:\基因组所工作\14.计数训练\runs\detect\train4\weights\best.pt')

model.predict(
    source=r"D:\基因组所工作\数据集\pollen\低倍镜\Tg1-51（5）",
    save=True,
    show=False,
    save_txt=True,
)