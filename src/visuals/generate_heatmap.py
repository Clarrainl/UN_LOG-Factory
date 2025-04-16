import os
import numpy as np
import vedo
from ultralytics import YOLO

# ------------ CONFIG ------------
MODEL_NAME = "T06"
ROOT = f"data/{MODEL_NAME}"
OBJ_PATH = f"{ROOT}/{MODEL_NAME}.obj"
TEXTURE_DIR = os.path.join(ROOT, "textures")
VIEW_FOLDER = os.path.join(ROOT, "generated_views")

MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
OUTPUT_DIR = f"data/{MODEL_NAME}_viz/heatmap"
CONF = 0.01
SPHERE_RADIUS = 0.05
HEAT_RADIUS = 2.0  # Distance around which to accumulate influence
# --------------------------------

def find_texture(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".jpg"):
            return os.path.join(folder, f)
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Loading mesh...")
    mesh = vedo.load(OBJ_PATH)
    texture = find_texture(TEXTURE_DIR)
    if texture:
        mesh.texture(texture)
        print(f"🎨 Texture applied from {texture}")
    mesh.scale(10)
    mesh.lighting("plastic")

    print("🔍 Running detection...")
    model = YOLO(MODEL_PATH)
    image_files = sorted([f for f in os.listdir(VIEW_FOLDER) if f.endswith(".png")])
    all_points = []

    for f in image_files:
        result = model(os.path.join(VIEW_FOLDER, f), conf=CONF)[0]
        if result.boxes:
            for b in result.boxes:
                x, y = float(b.xywhn[0][0]), float(b.xywhn[0][1])
                px = (x - 0.5) * 100
                py = (0.5 - y) * 100
                pt = np.array([px, py, 0])
                closest = mesh.closest_point(pt)
                all_points.append(closest)

    if not all_points:
        print("⚠️ No detections found!")
        return

    print(f"✅ Found {len(all_points)} detections")

    print("🔥 Calculating heat values...")
    heat_values = []
    all_points = np.array(all_points)
    for v in mesh.points:
        dists = np.linalg.norm(all_points - v, axis=1)
        influence = np.exp(-dists**2 / (2 * HEAT_RADIUS**2))
        heat_values.append(np.sum(influence))
    heat_values = np.array(heat_values)
    heat_values = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min() + 1e-5)

    print("🖌️ Applying heatmap to mesh...")
    mesh.cmap("hot", heat_values, on="points")

    print("🧊 Building 3D view...")
    spheres = [vedo.Sphere(p, r=SPHERE_RADIUS, c='blue') for p in all_points]

    vedo.show(
        mesh, *spheres,
        axes=1,
        bg="white",
        title=f"{MODEL_NAME} - Heatmap of Defects",
        viewup="z"
    )

    print(f"📁 Done! You can optionally save mesh or screenshot in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
