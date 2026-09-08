from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = YOLO("runs/detect/onion_quality_v2/weights/best.pt")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image, conf=0.05)

    detections = []
    class_counts = {}

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3)
            })
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "total_detections": len(detections),
        "class_counts": class_counts,
        "detections": detections
    }

@app.get("/")
def health_check():
    return {"status": "AI service running"}