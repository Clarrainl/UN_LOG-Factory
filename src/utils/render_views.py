import trimesh
import os

def render_views(obj_path, output_dir, num_views=12):
    # Load 3D model
    mesh = trimesh.load(obj_path)

    if not isinstance(mesh, trimesh.Scene):
        scene = mesh.scene()
    else:
        scene = mesh

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate views around Y axis (azimuthal rotation)
    views = {
        f"angle_{i * (360 // num_views)}": [0, i * (360 // num_views), 0]
        for i in range(num_views)
    }

    for view_name, angles in views.items():
        try:
            print(f"\n🔄 Rendering {view_name} with angles {angles}...")

            # New scene for each view
            scene = mesh.scene()
            scene.set_camera(angles=angles)

            png_data = scene.save_image(resolution=[640, 640], visible=True)

            if png_data is None:
                print(f"❌ Rendering failed for {view_name}: no image data.")
                continue

            image_path = os.path.join(output_dir, f"{view_name}.png")
            with open(image_path, 'wb') as f:
                f.write(png_data)

            print(f"✅ Saved view: {image_path}")

        except Exception as e:
            print(f"❌ Error rendering {view_name}: {e}")

# Example usage
if __name__ == "__main__":
    # Change the path if needed
    render_views("data/T09/T09.obj", "data/generated_views_T09", num_views=12)
