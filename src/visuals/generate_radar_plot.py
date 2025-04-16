import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re

# --- CONFIG ---
MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
IMAGE_FOLDER = "data/T06/generated_views"
SAVE_PATH = "data/T06/visuals/radar_plot.png"
NUM_ANGLES = 36
CONF = 0.01

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def extract_angle(filename):
    match = re.search(r"angle_(\d+)", filename)
    return int(match.group(1)) if match else None

def main():
    print("📡 Loading model...")
    model = YOLO(MODEL_PATH)

    angles = np.linspace(0, 360, NUM_ANGLES, endpoint=False)
    knot_counts = np.zeros(NUM_ANGLES)
    crack_counts = np.zeros(NUM_ANGLES)

    print("🔍 Analyzing images...")
    for file in sorted(os.listdir(IMAGE_FOLDER)):
        if not file.endswith(".png"):
            continue
        angle = extract_angle(file)
        if angle is None:
            continue
        idx = angle // 10

        result = model(os.path.join(IMAGE_FOLDER, file), conf=CONF)[0]
        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls == 0:
                    knot_counts[idx] += 1
                elif cls == 1:
                    crack_counts[idx] += 1

    print("📈 Plotting radar chart...")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    radians = np.deg2rad(angles)

    ax.plot(radians, knot_counts, label="Knots", color="blue", linewidth=2)
    ax.fill(radians, knot_counts, alpha=0.2, color="blue")

    ax.plot(radians, crack_counts, label="Cracks", color="red", linewidth=2)
    ax.fill(radians, crack_counts, alpha=0.2, color="red")

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_title("Detections by Viewing Angle", pad=20)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Radar plot saved at: {SAVE_PATH}")

if __name__ == "__main__":
    main()
