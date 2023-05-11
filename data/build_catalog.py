"""
    Test-train split catalog of steel data
"""

prop_test = 0.2

import os
import random
from os import path

steel_dir = path.join(path.dirname(__file__), "steel_defect")

images = path.join(steel_dir, "processed", "images")

names = os.listdir(images)
random.shuffle(names)

split_ind = int(len(names) * prop_test)
test_set = names[:split_ind]
train_set = names[split_ind:]

with open(path.join(steel_dir, "test_catalog.txt"), "w") as test_catalog:
    test_catalog.write("\n".join(test_set))

with open(path.join(steel_dir, "train_catalog.txt"), "w") as train_catalog:
    train_catalog.writelines("\n".join(train_set))