from ultralytics import settings

# Update a setting
settings.update({"datasets_dir": r"D:\基因组所工作\14.计数训练\datasets"})

# Update multiple settings
# settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

# Reset settings to default values
# settings.reset()