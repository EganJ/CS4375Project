import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from os import path

aitex_root = path.join(path.dirname(__file__), "aitex")

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
    img = np.asarray(img)
    img = img.reshape((1, *img.shape))
    return img

def process_defect(defect_img_name:str):
    img_id = defect_img_name.split(".")[0]

    mask_name = f"{img_id}_mask.png"

    mask = load_image(path.join(raw_masks, mask_name))
    save_mask(img_id, mask)

    img = load_image(path.join(raw_img, defect_img_name))
    save_image(img_id, img)

def process_no_defect(img_name:str):
    img_id = img_name.split(".")[0]

    img = load_image(path.join(raw_no_defect, img_name))
    save_image(img_id, img)

    mask = np.zeros_like(img)
    save_mask(img_id, mask)

print("Processing defect images...")
for defect_img in tqdm(os.listdir(raw_img)):
    process_defect(defect_img)

print("Processing no-defect images...")
for no_defect_img in tqdm(os.listdir(raw_no_defect)):
    process_no_defect(no_defect_img)