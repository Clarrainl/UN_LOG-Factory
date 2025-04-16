import os
import shutil
import random

# Get absolute paths to everything
BASE_DIR = os.path.abspath("data/dataset")
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
LABEL_DIR = os.path.join(BASE_DIR, "labels_raw")

DEST_IMAGE_TRAIN = os.path.join(BASE_DIR, "images/train")
DEST_IMAGE_VAL = os.path.join(BASE_DIR, "images/val")
DEST_LABEL_TRAIN = os.path.join(BASE_DIR, "labels/train")
DEST_LABEL_VAL = os.path.join(BASE_DIR, "labels/val")

# Create destination directories
os.makedirs(DEST_IMAGE_TRAIN, exist_ok=True)
os.makedirs(DEST_IMAGE_VAL, exist_ok=True)
os.makedirs(DEST_LABEL_TRAIN, exist_ok=True)
os.makedirs(DEST_LABEL_VAL, exist_ok=True)

# Collect image filenames
image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
print(f"📷 Found {len(image_filenames)} images")

random.shuffle(image_filenames)
split_ratio = 0.8
split_index = int(len(image_filenames) * split_ratio)
train_files = image_filenames[:split_index]
val_files = image_filenames[split_index:]

def move_files(img_list, img_dest, lbl_dest):
    for fname in img_list:
        base = os.path.splitext(fname)[0]
        img_src = os.path.join(IMAGE_DIR, fname)
        lbl_src = os.path.join(LABEL_DIR, base + ".txt")

        img_dst = os.path.join(img_dest, fname)
        lbl_dst = os.path.join(lbl_dest, base + ".txt")

        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

        # Copy image
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)

        # Copy label with debug
        if os.path.exists(lbl_src):
            with open(lbl_src, 'r') as f:
                content = f.read()
                print(f"📝 Label: {lbl_src} | Content preview: {content[:50]}")

            shutil.copy2(lbl_src, lbl_dst)
            print(f"✅ Copied label: {lbl_src} -> {lbl_dst}")
        else:
            print(f"⚠️ Label not found for: {base}")

move_files(train_files, DEST_IMAGE_TRAIN, DEST_LABEL_TRAIN)
move_files(val_files, DEST_IMAGE_VAL, DEST_LABEL_VAL)

print(f"\n✅ Done. Train: {len(train_files)} images, Val: {len(val_files)} images")
print(f"✅ Labels: {len(os.listdir(DEST_LABEL_TRAIN))} train / {len(os.listdir(DEST_LABEL_VAL))} val")
