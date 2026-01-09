from ultralytics import YOLO
import cv2

# Load model once
model = YOLO("best.pt")

def run_inference(image_path):
    results = model(image_path)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })

    return detections


if __name__ == "__main__":
    output = run_inference("Screenshot 2026-01-05 211135.png")
    print(output)
