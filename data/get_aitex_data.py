import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from os import path

aitex_root = path.join(path.pardir(__file__), "aitex")

raw_img = path.join(aitex_root, "Defect_images")
raw_no_defect = path.join(aitex_root, "NODefect_images")
raw_masks = path.join(aitex_root, "Mask_images")

aitex_processed = path.join(aitex_root, "processed")
aitex_images = path.join(aitex_processed, "images")
aitex_labels = path.join(aitex_processed, "labels")

os.mkdir(aitex_processed)
os.mkdir(aitex_images)
os.mkdir(aitex_labels)

def save_mask(img_id:str, tensor: np.ndarray):
    np.save(path.join(aitex_labels, f"{img_id}.npy"), tensor)

def save_image(img_id:str, tensor: np.ndarray):
    np.save(path.join(aitex_images, f"{img_id}.npy"), tensor)

def load_image(fpath):
    img = Image.open(fpath)
    img = np.as_array(img)
    return img

def process_defect(defect_img_name:str):
    img_id = defect_img_name.split(".")[0]

    mask_name = f"{img_id}_mask.png"

    mask = load_image(path.join(raw_masks, mask_name))
    print(mask.shape)
    save_mask(img_id, mask)

    img = load_image(path.join(raw_img, defect_image_name))
    save_img(img_id, img)

process_defect("0001_002_00.png")