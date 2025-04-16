from ultralytics import YOLO
import os

def main():
    # Load pretrained YOLOv8n model (small and fast)
    model = YOLO("yolov8n.pt")

    # Absolute path to dataset config file
    data_config_path = os.path.abspath("src/config/data.yaml")

    # Train the model
    model.train(
        data=data_config_path,
        epochs=10,
        imgsz=640,
        batch=16,
        name="wood_defects",
        project="runs/train"
    )

if __name__ == "__main__":
    main()
