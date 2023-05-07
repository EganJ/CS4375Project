import kaggle
import zipfile
import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from os import path

kaggle.api.authenticate()

root = os.path.dirname(__file__)

steel_dim = (1600, 256)


def decode_mask(encoded_pixels):
    """
        Converts the encoded pixels into masks. This code was taken from
        https://dev.to/vijethrai/steel-defect-detection-86i
    """
    counts = []
    mask = np.zeros(steel_dim[0] * steel_dim[1], dtype=np.int8)
    pre_mask = np.asarray([int(point) for point in encoded_pixels.split()])
    for index, count in enumerate(pre_mask):
        if (index % 2 != 0):
            counts.append(count)
        i = 0
    for index, pixel in enumerate(pre_mask):
        if (index % 2 == 0):
            if (i == len(counts)):
                break
            mask[pixel:pixel+counts[i]] = 1
            i += 1
    mask = np.reshape(mask, steel_dim).T
    # Not sure if this line was important...
    # mask = cv2.resize(mask, (256, 1600)).T
    return mask


def read_or_create_target_image(imgname):
    """
        Reads the 4-channel tensor target for the image, or returns a tensor
        of zeroes if it doesn't exist.
    """
    if path.exists(imgname):
        return np.load(imgname)
    else:
        return np.zeros((steel_dim[1], steel_dim[0], 4))


steel_defect_dir = path.join(root, "steel_defect")
if path.exists(steel_defect_dir):
    print("Steel defect directory already found, skipping...")
else:
    os.mkdir(steel_defect_dir)
    print("Downloading steel defect dataset from kaggle...")
    kaggle.api.competition_download_files("severstal-steel-defect-detection",
                                            steel_defect_dir,
                                            quiet=False)

    print("Unzipping steel defect dataset...")
    with zipfile.ZipFile(path.join(steel_defect_dir,
                                    "severstal-steel-defect-detection.zip"),
                            "r") as zipped:
        zipped.extractall(steel_defect_dir)

    # Process images and targets into numpy files for quick and easy loading
    print("Processing steel defect dataset...")

    steel_processed_dir = path.join(steel_defect_dir, "processed")
    os.mkdir(steel_processed_dir)
    processed_img_dir = path.join(steel_processed_dir, "images")
    os.mkdir(processed_img_dir)
    processed_label_dir = path.join(steel_processed_dir, "labels")
    os.mkdir(processed_label_dir)

    with open(path.join(steel_defect_dir, "train.csv"), "r") as label_data:
        n_lines = len([1 for _ in label_data]) - 1

    with open(path.join(steel_defect_dir, "train.csv"), "r") as label_data:
        rows = csv.reader(label_data)
        # skip header
        next(rows)
        for row in tqdm(rows, total=n_lines):
            # row = ["imgname.jpg", "class_no", "encoded pixels"]
            img_id = row[0].split(".")[0]
            img_origin_file = path.join(
                steel_defect_dir, "train_images", row[0])
            channel = int(row[1]) - 1
            defect_mask = decode_mask(row[2])

            image_fpath = path.join(processed_img_dir, f"{img_id}.npy")
            if not path.exists(image_fpath):
                img_tensor = np.asarray(Image.open(img_origin_file))
                np.save(image_fpath, img_tensor)

            label_fpath = path.join(processed_label_dir, f"{img_id}.npy")
            defect_tensor = read_or_create_target_image(label_fpath)
            defect_tensor[:, :, channel] = defect_mask
            np.save(label_fpath, defect_tensor)
