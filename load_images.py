
import os
import cv2
import pickle
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 50
DATASET_DIR = "gestures"

images = []
labels = []

folders = sorted(
    [f for f in glob(os.path.join(DATASET_DIR, "*"))
     if os.path.isdir(f)],
    key=lambda p: int(os.path.basename(p))
)

if len(folders) < 2:
    raise ValueError("Collect at least 2 gesture classes before training.")

class_ids = [int(os.path.basename(f)) for f in folders]
if class_ids != list(range(1, len(folders) + 1)):
    raise ValueError("Gesture folders must be numbered consecutively from 1.")

for folder in folders:
    gesture_id = int(os.path.basename(folder))
    class_index = gesture_id - 1

    files = sorted(glob(os.path.join(folder, "*.jpg")))
    if not files:
        raise ValueError(f"No JPG images found in {folder}")

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Skipping unreadable image:", path)
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(class_index)

X = np.asarray(images, dtype=np.uint8)
y = np.asarray(labels, dtype=np.int32)

if len(X) == 0:
    raise ValueError("No valid images found.")

# 70% training, 15% validation, 15% testing.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

for name, data in [
    ("train_images", X_train), ("train_labels", y_train),
    ("val_images", X_val), ("val_labels", y_val),
    ("test_images", X_test), ("test_labels", y_test)
]:
    with open(name, "wb") as f:
        pickle.dump(data, f)

print("\nDataset prepared successfully.")
print("Gesture classes:", len(folders))
print("Total images:", len(X))
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))
print("\nClass mapping:")
for folder in folders:
    gid = int(os.path.basename(folder))
    print(f"Class index {gid - 1} -> gesture ID {gid}")
