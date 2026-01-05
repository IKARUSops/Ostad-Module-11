from ultralytics import YOLO

def main():
    

    # Load a pretrained model (recommended for transfer learning)
    model = YOLO("yolo11n.pt")  # Load a pretrained YOLOv11n model

    # Train the model with specific parameters
    results = model.train(
        data="dataset/data.yaml", # Path to your data config file
        epochs=100,             # Number of epochs to train for (300+ is a common baseline)
        imgsz=640,              # Input image size
        batch=16,
        patience=20,            # Number of epochs to wait for a decrease in validation loss before early stopping
        device='cpu'               # Specify the GPU device (e.g., 0, 1 or 'cpu')
    )



if __name__ == "__main__":
    main()
