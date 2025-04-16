import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import vedo
import trimesh

# ------------------ CONFIG ------------------
MODEL_NAME = "T06"  # 🔁 Change this ONLY to analyze a different model

ROOT = f"data/{MODEL_NAME}"
OBJ_PATH = f"{ROOT}/{MODEL_NAME}.obj"
TEXTURE_FOLDER = os.path.join(ROOT, "textures")

# Optional: force a specific texture name here, or set to None to autodetect
FORCED_TEXTURE_NAME = None  # e.g. "ce9f5a5858b49134f3df039ff7b246e7.jpg"

GENERATED_VIEWS = os.path.join(ROOT, "generated_views")
VIEW_DETECTIONS = os.path.join(ROOT, "view_detections")
CSV_REPORT = os.path.join(ROOT, "detection_report.csv")
GRID_IMAGE = os.path.join(VIEW_DETECTIONS, "detection_grid.png")

MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
CONF = 0.005  # Confidence threshold for YOLOv8
NUM_VIEWS = 36
# --------------------------------------------


def find_texture(texture_dir, forced_name=None):
    """Find texture JPG file in given directory, or use forced name if given."""
    if forced_name:
        full_path = os.path.join(texture_dir, forced_name)
        if os.path.exists(full_path):
            return full_path
        else:
            print(f"⚠️ Forced texture '{forced_name}' not found.")
    else:
        if os.path.exists(texture_dir):
            for file in os.listdir(texture_dir):
                if file.lower().endswith(".jpg"):
                    return os.path.join(texture_dir, file)
    return None


def step1_render_views(num_views=12):
    os.makedirs(GENERATED_VIEWS, exist_ok=True)
    mesh = trimesh.load(OBJ_PATH)
    views = {
        f"angle_{i * (360 // num_views)}": [0, i * (360 // num_views), 0]
        for i in range(num_views)
    }
    for view_name, angles in views.items():
        scene = mesh.scene()
        scene.set_camera(angles=angles)
        png = scene.save_image(resolution=[640, 640])
        if png:
            with open(os.path.join(GENERATED_VIEWS, f"{view_name}.png"), "wb") as f:
                f.write(png)
    print("✅ Step 1: Rendered views")


def step2_run_yolo():
    os.makedirs(VIEW_DETECTIONS, exist_ok=True)
    model = YOLO(MODEL_PATH)
    images = sorted(f for f in os.listdir(GENERATED_VIEWS) if f.endswith(".png"))
    for img in images:
        img_path = os.path.join(GENERATED_VIEWS, img)
        result = model(img_path, conf=CONF)[0]
        rendered = result.plot()
        cv2.imwrite(os.path.join(VIEW_DETECTIONS, img), rendered)
    print("✅ Step 2: YOLOv8 detections saved")


def step3_export_csv():
    model = YOLO(MODEL_PATH)
    rows = []
    images = sorted(f for f in os.listdir(GENERATED_VIEWS) if f.endswith(".png"))
    for img in images:
        result = model(os.path.join(GENERATED_VIEWS, img), conf=CONF)[0]
        boxes = result.boxes
        if boxes and len(boxes) > 0:
            for b in boxes:
                rows.append({
                    "image": img,
                    "class": int(b.cls[0].item()),
                    "confidence": round(float(b.conf[0].item()), 3)
                })
        else:
            rows.append({"image": img, "class": None, "confidence": None})
    pd.DataFrame(rows).to_csv(CSV_REPORT, index=False)
    print(f"✅ Step 3: CSV exported to {CSV_REPORT}")


def step4_generate_grid(cols=4, image_size=(320, 320)):
    images = sorted(f for f in os.listdir(VIEW_DETECTIONS) if f.endswith(".png"))
    rows = int(np.ceil(len(images) / cols))
    grid_images = []
    for i in range(rows * cols):
        if i < len(images):
            img = cv2.imread(os.path.join(VIEW_DETECTIONS, images[i]))
            img = cv2.resize(img, image_size)
        else:
            img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        grid_images.append(img)
    grid = [np.hstack(grid_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    collage = np.vstack(grid)
    cv2.imwrite(GRID_IMAGE, collage)
    print(f"✅ Step 4: Collage saved at {GRID_IMAGE}")


def step5_project_on_3d():
    model = YOLO(MODEL_PATH)
    mesh = vedo.load(OBJ_PATH)

    texture_path = find_texture(TEXTURE_FOLDER, forced_name=FORCED_TEXTURE_NAME)
    if texture_path and os.path.exists(texture_path):
        mesh.texture(texture_path)
        print(f"🎨 Texture applied from {texture_path}")
    else:
        print("⚠️ Texture not found or not applied.")

    mesh.lighting("plastic")
    mesh.scale(10)

    spheres = []
    images = sorted(f for f in os.listdir(GENERATED_VIEWS) if f.endswith(".png"))

    for img in images:
        result = model(os.path.join(GENERATED_VIEWS, img), conf=CONF)[0]
        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                x, y = float(box.xywhn[0][0]), float(box.xywhn[0][1])
                px = (x - 0.5) * 100
                py = (0.5 - y) * 100
                pz = 0
                pt = np.array([px, py, pz])
                closest = mesh.closest_point(pt)

                # Visual elements
                color = "red" if cls == 0 else "blue"
                label = "knot" if cls == 0 else "crack"
                r = 0.05  # sphere radius

                s = vedo.Sphere(closest, r=r, c=color)
                t = vedo.Text3D(label, pos=closest + [0, r * 1.5, 0], s=r * 2.5, c=color)

                spheres.extend([s, t])

    vedo.show(mesh, *spheres, axes=1, bg='white', title=f"{MODEL_NAME} - Detections on 3D")


# ----------------- RUN ALL ------------------
if __name__ == "__main__":
    step1_render_views(num_views=NUM_VIEWS)
    step2_run_yolo()
    step3_export_csv()
    step4_generate_grid()
    step5_project_on_3d()
