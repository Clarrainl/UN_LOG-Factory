import bpy
import os

# === CONFIGURACIÓN ===
model_path = "/home/charlie/Desktop/final_software/data/T03/T03.obj"
texture_path = "/home/charlie/Desktop/final_software/data/T03/textures/5d20d1aa6eda225aa4147425e0ce78ff.jpg"
output_image_path = "/home/charlie/Desktop/final_software/data/T03/unwrapped_texture.png"
image_size = 2048

# === RESET SCENE ===
bpy.ops.wm.read_factory_settings(use_empty=True)

# === IMPORT OBJ ===
bpy.ops.import_scene.obj(filepath=model_path)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# === AGREGAR LUZ Y CÁMARA MÍNIMAS (requeridas por Blender para bakear) ===
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
bpy.ops.object.camera_add(location=(0, -5, 2))

# === CREAR MATERIAL CON EMISSION (no depende de luces externas) ===
mat = bpy.data.materials.new(name="AutoLitMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Limpiar nodos por si acaso
for n in nodes:
    nodes.remove(n)

# Crear nodos del shader
tex_node = nodes.new(type='ShaderNodeTexImage')
tex_node.image = bpy.data.images.load(texture_path)
emit_node = nodes.new(type='ShaderNodeEmission')
output_node = nodes.new(type='ShaderNodeOutputMaterial')

# Conectar nodos
links.new(tex_node.outputs['Color'], emit_node.inputs['Color'])
links.new(emit_node.outputs['Emission'], output_node.inputs['Surface'])

# Asignar material
if not obj.data.materials:
    obj.data.materials.append(mat)
else:
    obj.data.materials[0] = mat

# === CREAR IMAGEN PARA EL BAKE ===
bake_img = bpy.data.images.new("BakeImage", width=image_size, height=image_size)
bake_node = nodes.new('ShaderNodeTexImage')
bake_node.image = bake_img
nodes.active = bake_node

# === UV MAP + UNWRAP CILÍNDRICO ===
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.cylinder_project(direction='VIEW_ON_EQUATOR', scale_to_bounds=True)
bpy.ops.object.mode_set(mode='OBJECT')

# === CONFIGURAR RENDER Y BAKE ===
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.bake.use_clear = True
bpy.context.scene.render.bake.use_selected_to_active = False
bpy.context.scene.render.bake.use_cage = False
bpy.context.scene.render.bake.target = 'IMAGE_TEXTURES'

# === ASEGURAR SELECCIÓN SOLO DEL MESH ===
for obj_ in bpy.context.scene.objects:
    obj_.select_set(False)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# === HORNEAR (BAKE) EMISSION ===
bpy.ops.object.bake(type='EMIT')

# === GUARDAR LA TEXTURA GENERADA ===
bake_img.filepath_raw = output_image_path
bake_img.file_format = 'PNG'
bake_img.save()
print("✅ Bake completado y exportado:", output_image_path)
