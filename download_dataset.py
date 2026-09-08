from roboflow import Roboflow

rf = Roboflow(api_key="TOauylAy88hqUFEMmRrS")

project = rf.workspace("onion-grading-nx").project("veg1-hcqsf-2")
version = project.version(4)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)