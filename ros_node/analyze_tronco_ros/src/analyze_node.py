#!/usr/bin/env python3
import rospy
import os
import numpy as np
from ultralytics import YOLO
import trimesh
import vedo

def run_detection():
    try:
        rospy.loginfo("analyze_tronco_node started.")

        # --- Configuración de paths ---
        MODEL_NAME = "T06"
        BASE_PATH = f"/root/final_software/data/{MODEL_NAME}"
        OBJ_PATH = f"{BASE_PATH}/{MODEL_NAME}.obj"
        TEXTURE_DIR = os.path.join(BASE_PATH, "textures")
        VIEWS_PATH = os.path.join(BASE_PATH, "generated_views")
        OUTPUT_IMAGE = f"/root/final_software/output/{MODEL_NAME}_3d_projection.png"
        YOLO_MODEL_PATH = "/root/final_software/runs/train/wood_defects2/weights/best.pt"

        os.makedirs("/root/final_software/output", exist_ok=True)

        # --- Cargar textura automáticamente ---
        texture_file = next((f for f in os.listdir(TEXTURE_DIR) if f.endswith('.jpg')), None)
        texture_path = os.path.join(TEXTURE_DIR, texture_file) if texture_file else None

        # --- Cargar malla ---
        rospy.loginfo("Loading mesh...")
        mesh = vedo.load(OBJ_PATH)
        mesh.lighting("plastic")
        mesh.scale(10)

        if texture_path and os.path.exists(texture_path):
            mesh.texture(texture_path)
            rospy.loginfo("Texture applied.")
        else:
            rospy.logwarn("Texture not found!")

        # --- Cargar modelo YOLO ---
        rospy.loginfo("Running YOLO detection...")
        model = YOLO(YOLO_MODEL_PATH)
        images = sorted(f for f in os.listdir(VIEWS_PATH) if f.endswith(".png"))

        spheres = []
        for img in images:
            result = model(os.path.join(VIEWS_PATH, img), conf=0.01)[0]
            if result.boxes:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    x, y = float(box.xywhn[0][0]), float(box.xywhn[0][1])
                    px = (x - 0.5) * 100
                    py = (0.5 - y) * 100
                    pz = 0
                    pt = np.array([px, py, pz])
                    closest = mesh.closest_point(pt)
                    color = "red" if cls == 0 else "blue"
                    spheres.append(vedo.Sphere(closest, r=0.05, c=color))

        # --- Mostrar en offscreen y guardar imagen ---
        vedo.settings.offscreen = True
        vp = vedo.Plotter(offscreen=True)
        vp.show(mesh, *spheres, bg='white', axes=1)
        vp.screenshot(OUTPUT_IMAGE)
        vp.close()

        rospy.loginfo(f"✅ Saved projection: {OUTPUT_IMAGE}")

    except Exception as e:
        rospy.logerr(f"Error during processing: {e}")

if __name__ == "__main__":
    rospy.init_node("analyze_tronco_node")
    run_detection()
