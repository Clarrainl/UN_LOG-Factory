import os
from ultralytics import YOLO
import pandas as pd

def export_detections_csv(model_path, input_dir, output_csv, conf=0.01):
    model = YOLO(model_path)

    rows = []

    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        result = model(img_path, conf=conf)[0]

        boxes = result.boxes
        if boxes and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf_score = float(b.conf[0].item())
                rows.append({
                    "image": img_name,
                    "class": cls_id,
                    "confidence": round(conf_score, 3)
                })
        else:
            rows.append({
                "image": img_name,
                "class": None,
                "confidence": None
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV exported to {output_csv}")

if __name__ == "__main__":
    export_detections_csv(
        model_path="runs/train/wood_defects2/weights/best.pt",
        input_dir="data/generated_views_T09",
        output_csv="data/view_detections/detection_report_T09.csv",
        conf=0.01
    )
