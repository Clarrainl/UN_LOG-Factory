import os
from ultralytics import YOLO
import vedo
import numpy as np

def get_center_pixel(box):
    """Return normalized x, y center from YOLO bounding box"""
    return float(box[0]), float(box[1])

def annotate_model_with_detections(model_path, mesh_path, texture_path, images_dir, conf=0.01):
    # Load 3D model
    mesh = vedo.load(mesh_path)

    # Apply texture if available
    if texture_path and os.path.exists(texture_path):
        mesh.texture(texture_path)
        print(f"🎨 Texture applied: {texture_path}")
    else:
        print("⚠️ No texture applied (not found or not specified).")

    # Improve rendering
    mesh.lighting("plastic")
    mesh.scale(10)  # Optional: scale up if mesh looks too small

    # Load YOLO model
    model = YOLO(model_path)

    spheres = []
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        result = model(img_path, conf=conf)[0]

        if result.boxes and len(result.boxes) > 0:
            print(f"📦 Detected {len(result.boxes)} object(s) in {img_name}")

        for box in result.boxes:
            cls = int(box.cls[0].item())
            x_norm, y_norm = get_center_pixel(box.xywhn[0])

            # Estimación de posición 3D basada en vista 2D
            x = (x_norm - 0.5) * 100
            y = (0.5 - y_norm) * 100
            z = 0

            # Proyectar sobre superficie del tronco
            estimated_point = np.array([x, y, z])
            closest = mesh.closest_point(estimated_point)

            color = "red" if cls == 0 else "blue"
            label = "knot" if cls == 0 else "crack"

            s = vedo.Sphere(pos=closest, r=0.05, c=color, alpha=1.0)
            t = vedo.Text3D(label, pos=closest + [1, 1, 1], s=0.05, c=color)
            spheres.extend([s, t])

    print(f"✅ {len(spheres)//2} detections projected on the 3D model.")
    vedo.show(mesh, *spheres, axes=1, bg='white', title="Detected Defects on Textured Log")

if __name__ == "__main__":
    annotate_model_with_detections(
        model_path="runs/train/wood_defects2/weights/best.pt",
        mesh_path="data/T09/T09.obj",
        texture_path="data/T09/textures/",  # cambia si tu textura se llama distinto
        images_dir="data/generated_views_T09",
        conf=0.01
    )
