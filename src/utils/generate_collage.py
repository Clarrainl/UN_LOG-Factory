import os
import cv2
import math
import numpy as np

def generate_collage(input_dir, output_path, cols=4, image_size=(320, 320)):
    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    if not images:
        print("❌ No images found in the input directory.")
        return

    rows = math.ceil(len(images) / cols)
    grid_images = []

    for idx in range(rows * cols):
        if idx < len(images):
            img_path = os.path.join(input_dir, images[idx])
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
        else:
            # Rellenar con negro si no hay suficientes imágenes
            img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        grid_images.append(img)

    # Agrupar en filas
    rows_imgs = [
        np.hstack(grid_images[i * cols:(i + 1) * cols]) for i in range(rows)
    ]

    # Combinar todas las filas
    final_image = np.vstack(rows_imgs)

    # Guardar la imagen final
    cv2.imwrite(output_path, final_image)
    print(f"✅ Collage saved to: {output_path}")

if __name__ == "__main__":
    generate_collage(
        input_dir="data/view_detections_T09",
        output_path="data/view_detections/detection_grid_T09.png",
        cols=4,
        image_size=(320, 320)
    )
