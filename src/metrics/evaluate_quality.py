import os
import sys
from ultralytics import YOLO

# ------------------ CONFIG ------------------
MODEL_NAME = "T06"
ROOT = f"data/{MODEL_NAME}"
VIEWS_PATH = f"{ROOT}/generated_views"
MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
CONF = 0.01
# --------------------------------------------

def evaluate_quality(detections):
    num_knots = sum(1 for d in detections if d["class"] == 0)
    num_cracks = sum(1 for d in detections if d["class"] == 1)

    if num_knots <= 3 and num_cracks == 0:
        quality = "🟢 Alta"
    elif num_knots + num_cracks <= 6 and num_cracks <= 1:
        quality = "🟡 Media"
    else:
        quality = "🔴 Baja"

    print("\n🧾 Reporte de Calidad:")
    print(f"   - Nudos: {num_knots}")
    print(f"   - Grietas: {num_cracks}")
    print(f"   → Calidad estimada: {quality}")
    return {
        "knot_count": num_knots,
        "crack_count": num_cracks,
        "quality": quality
    }

def main():
    print("🔍 Evaluando calidad del tronco...")
    model = YOLO(MODEL_PATH)
    detections = []

    for fname in sorted(os.listdir(VIEWS_PATH)):
        if not fname.endswith(".png"):
            continue
        result = model(os.path.join(VIEWS_PATH, fname), conf=CONF)[0]
        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                detections.append({"image": fname, "class": cls})

    results = evaluate_quality(detections)

    # Guardar a archivo
    output_path = os.path.join(ROOT, "quality_report.txt")
    with open(output_path, "w") as f:
        f.write("🧾 Quality Evaluation Report\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Nudos detectados: {results['knot_count']}\n")
        f.write(f"Grietas detectadas: {results['crack_count']}\n")
        f.write(f"Calidad estimada: {results['quality']}\n")
    
    print(f"💾 Reporte guardado en: {output_path}")

if __name__ == "__main__":
    main()
