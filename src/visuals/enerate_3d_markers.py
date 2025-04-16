import os
import numpy as np
import vedo
from ultralytics import YOLO

# --- CONFIG ---
MODEL_NAME = "T06"
ROOT = f"data/{MODEL_NAME}"
OBJ_PATH = f"{ROOT}/{MODEL_NAME}.obj"
TEXTURE_FOLDER = f"{ROOT}/textures"
TEXTURE_NAME = next((f for f in os.listdir(TEXTURE_FOLDER) if f.endswith(".jpg")), None)
TEXTURE_PATH = os.path.join(TEXTURE_FOLDER, TEXTURE_NAME)
GENERATED_VIEWS = f"{ROOT}/generated_views"
MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
CONF = 0.01

def main():
    print("🧠 Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("📦 Loading 3D model...")
    mesh = vedo.load(OBJ_PATH)
    if os.path.exists(TEXTURE_PATH):
        mesh.texture(TEXTURE_PATH)
        print(f"🎨 Texture applied from {TEXTURE_PATH}")
    else:
        print("⚠️ No texture found.")

    mesh.scale(10)
    mesh.lighting("plastic")

    print("🔍 Running detections...")
    spheres = []

    for fname in sorted(os.listdir(GENERATED_VIEWS)):
        if not fname.endswith(".png"):
            continue
        result = model(os.path.join(GENERATED_VIEWS, fname), conf=CONF)[0]

        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                x, y = float(box.xywhn[0][0]), float(box.xywhn[0][1])
                px = (x - 0.5) * 100
                py = (0.5 - y) * 100
                pt = np.array([px, py, 0])
                closest = mesh.closest_point(pt)
                color = "blue" if cls == 0 else "red"
                label = "knot" if cls == 0 else "crack"

                s = vedo.Sphere(pos=closest, r=0.8, c=color, alpha=0.7)
                t = vedo.Text3D(label, pos=closest + [1, 0.5, 1], s=2, c=color)
                spheres.extend([s, t])

    print(f"✅ Total markers: {len(spheres)//2}")
    vedo.show(mesh, *spheres, axes=1, viewup='z', bg='white', title=f"{MODEL_NAME} - 3D Detection Map")

if __name__ == "__main__":
    main()
