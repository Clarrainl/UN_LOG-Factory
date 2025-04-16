# MRAC01(24/25): Software III - Log Analysis on Wooden Logs using Deep Learning

## Overview

This project focuses on the detection and spatial analysis of wood defects—specifically **knots** and **cracks**—on 3D scanned logs. Using **deep learning** with **YOLOv8**, the pipeline performs automated inspection from texture mapping to object detection and 3D projection.

The primary goal is to enhance digital fabrication and material optimization in architecture by automating visual inspections of natural wood materials.

---

## Machine Learning Model

### What kind of ML is used?

We use a **Deep Learning** model based on **YOLOv8 (You Only Look Once version 8)**, which belongs to the class of **Convolutional Neural Networks (CNNs)**. This model is pre-trained and fine-tuned to detect *knots* and *cracks* on wood textures.

### Why Deep Learning?

Deep learning is chosen due to:
- High performance in image classification and object detection
- Ability to generalize from complex textures
- Real-time detection capability

Unlike shallow learning (e.g. SVM or decision trees), deep learning models extract hierarchical features using multiple **hidden layers**, enabling better performance on unstructured data like images.

### Network Architecture (YOLOv8)

Here's a conceptual breakdown:

```
Input Layer
↓
Backbone (CSP-Darknet): feature extraction
↓
Neck (PANet): feature aggregation
↓
Head (YOLO Layer): bounding box regression + classification
↓
Output:
    - Class: "knot" or "crack"
    - Confidence score
    - Bounding box (x, y, w, h)
```

---

## Dataset & Training

The YOLOv8 model was fine-tuned using a custom dataset:

- Collected images of wood logs with manually annotated defects
- Two classes: `knot` and `crack`
- Format: YOLO bounding box annotations

Training was performed using the `ultralytics` library with configuration like:

```bash
yolo task=detect mode=train model=yolov8n.pt data=wood_defects.yaml epochs=100 imgsz=640
```

The trained model weights were saved in:

```
runs/train/wood_defects2/weights/best.pt
```

---

## Pipeline Script: `analyze_tronco.py`

This script automates the full inspection process on a 3D object.

### Steps

1. **Render Views**  
   - Uses `trimesh` to rotate the mesh and generate `N` evenly spaced views.
   - Saves each view as a `.png` in a folder:  
     `data/<MODEL_NAME>/generated_views/`

2. **Run YOLO Detection**  
   - Loads the trained YOLOv8 model and performs detection on each rendered image.
   - Saves annotated results in `view_detections/`

3. **CSV Report**  
   - Creates a `.csv` file logging detection results per view.
   - Contains columns: `image`, `class`, `confidence`

4. **Image Grid**  
   - Combines all detection views into a collage for quick visual inspection.

5. **3D Projection**  
   - Projects detected points (from 2D images) back to the 3D mesh.
   - Places labeled spheres for each detection (e.g., `knot`, `crack`)
   - Uses `vedo` for 3D visualization and optionally exports `3d_projection.png` in offscreen mode.

---

## Outputs

After running the script, you'll get:

| Output                          | Description                                  |
|---------------------------------|----------------------------------------------|
| `generated_views/`              | Rendered cylindrical views of the object     |
| `view_detections/`             | YOLO-annotated detection views               |
| `detection_report.csv`         | Tabular summary of all detections            |
| `detection_grid.png`           | Image collage of all detections              |
| `3d_projection.png`            | Projection of detections on the 3D mesh      |

---

## Getting Started

### Prerequisites
Ensure you have:
- Ubuntu 20.04+ or Docker (ROS + Python environment)
- Python 3.8+
- A virtual environment (recommended)

### Dependencies
Install core dependencies:

```bash
pip install numpy pandas opencv-python ultralytics trimesh vedo
```

If running in ROS/Docker, additional setup is required (see ROS section).

---

## Demo

To run the full analysis:

```bash
python analyze_tronco.py
```

To run the ROS node version:

```bash
rosrun analyze_tronco_ros analyze_node.py
```

---

## Authors
- [Charlie Larraín](https://github.com/Clarrainl)

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Trimesh Documentation](https://trimsh.org/)
- [Vedo 3D Viewer](https://vedo.embl.es/)

## Credits
- MRAC-IAAC 2025 - Advanced Digital Design in Robotics

#### Acknowledgements

- [Marita Georganta](https://www.linkedin.com/in/marita-georganta/) — GitHub Template
- [Nestor ]

```

---
