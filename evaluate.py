from ultralytics import YOLO

test_folder = "dataset/test/images"

# Path to your best model weights
best_model_path = "runs/train/exp/weights/best.pt"

# Load YOLOv11 model with the trained weights
model = YOLO(best_model_path)


# Run predictions and save results
results = model.predict(
    source=test_folder,  # folder with test images
    conf=0.25,           # confidence threshold for detection
    save=True,           # saves annotated images
    save_txt=True        # saves YOLO-format predictions
)
