import os
import cv2
from ultralytics import YOLO

# ------------------ CONFIG ------------------
IMAGE_PATH = "data/test01.png"  # Cambia esto si tu imagen tiene otro nombre o formato
MODEL_PATH = "runs/train/wood_defects2/weights/best.pt"
CONFIDENCE = 0.25  # Puedes bajarlo si quieres más detecciones
OUTPUT_DIR = "results/single_prediction"
# --------------------------------------------

def predict_image():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cargar modelo YOLO
    print("🔍 Cargando modelo...")
    model = YOLO(MODEL_PATH)

    # Ejecutar detección
    print(f"📷 Analizando imagen: {IMAGE_PATH}")
    result = model(IMAGE_PATH, conf=CONFIDENCE)[0]

    # Mostrar resultados en consola
    for box in result.boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = model.names[cls]
        print(f"✅ Detectado: {label} ({conf:.2f})")

    # Renderizar y guardar imagen con detecciones
    img_rendered = result.plot()
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(IMAGE_PATH))
    cv2.imwrite(output_path, img_rendered)
    print(f"🖼️ Imagen guardada con resultados: {output_path}")

    # (Opcional) Mostrar en ventana
    cv2.imshow("YOLOv8 - Detección", img_rendered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_image()
