import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
IMAGE_FOLDER = "data/T06/generated_views"
SAVE_PATH = "data/T06/visuals/bar_chart.png"
CONF = 0.01

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def main():
    print("📦 Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    class_names = {0: "Knot", 1: "Crack"}
    counts = {0: 0, 1: 0}

    print("🔍 Running detections...")
    for file in sorted(os.listdir(IMAGE_FOLDER)):
        if not file.endswith(".png"):
            continue

        result = model(os.path.join(IMAGE_FOLDER, file), conf=CONF)[0]
        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls in counts:
                    counts[cls] += 1

    # Plot
    print("📊 Generating bar chart...")
    labels = [class_names[i] for i in counts.keys()]
    values = [counts[i] for i in counts.keys()]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["blue", "red"])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha="center", fontsize=12)

    plt.title("Total Detections by Class")
    plt.ylabel("Number of Detections")
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Bar chart saved at {SAVE_PATH}")

if __name__ == "__main__":
    main()
