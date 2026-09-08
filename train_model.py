from ultralytics import YOLO

model = YOLO("runs/detect/onion_quality_v2/weights/last.pt")
results = model.train(resume=True)