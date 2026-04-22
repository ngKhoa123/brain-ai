from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import os

# ===== giữ nguyên config =====
IMG_SIZE = (128, 128)
ORIENTATIONS = 12
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)


# ===== HOG =====
def extract_hog(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img)

    img = rgb2gray(img)
    img = resize(img, IMG_SIZE, anti_aliasing=True)

    features = hog(
        img,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        transform_sqrt=True
    )

    return features   # ✔ fix indent (quan trọng)


# ===== BUILD DATASET (GIỮ NGUYÊN) =====
# ===== FIX NHẸ =====
def build_dataset(folder):
    X, y = [], []

    for label in sorted(os.listdir(folder)):  # ✔ fix
        label_path = os.path.join(folder, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(label_path, file)

                feat = extract_hog(path)
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


# ===== THÊM DUY NHẤT CHO APP =====
def extract_hog_single(path):
    """
    Dùng cho predict 1 ảnh trong app
    KHÔNG đụng vào hàm cũ
    """
    feat = extract_hog(path)
    return feat.reshape(1, -1)

import cv2
import numpy as np


# =========================
# ===== DL PREPROCESS (GIỮ NGUYÊN LOGIC) =====
# =========================
def preprocess_image_dl(path, size=(224, 224)):
    """
    Crop background + pad + resize
    (giống y code bạn train)
    """

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # ===== remove background =====
    mask = img > 10
    if np.any(mask):
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img[y_min:y_max+1, x_min:x_max+1]

    # ===== padding to square =====
    h, w = img.shape
    size_pad = max(h, w)

    padded = np.zeros((size_pad, size_pad), dtype=img.dtype)

    y_offset = (size_pad - h) // 2
    x_offset = (size_pad - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    img = padded

    # ===== resize =====
    img = cv2.resize(img, size)

    return img

from PIL import Image

def preprocess_dl_for_model(path, transform):
    img = preprocess_image_dl(path)  # cv2 preprocessing của bạn

    if img is None:
        return None


    img = Image.fromarray(img)

    # dùng đúng transform bạn đã train
    img = transform(img)

    # thêm batch dimension
    img = img.unsqueeze(0)

    return img