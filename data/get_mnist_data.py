""""
    Breaks the mnist file into equally-sized chunks.

    Segmentation mnist files from https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist 
"""
import numpy as np
import os
from os import path

root = path.dirname(__file__)
mnist_dir = path.join(root, "mnist")

train_blocks = 45
test_blocks = 5

n_blocks = train_blocks + test_blocks

# normalize x to [0, 1]
x = np.load(path.join(mnist_dir, "combined.npy")) / 255
x = np.reshape(x, (5000, 1, 64, 84))
y = np.load(path.join(mnist_dir, "segmented.npy"))
y = y.transpose(0, 3, 1, 2)

print(np.max(x))
print(np.max(y))

x = np.split(x, n_blocks)
y = np.split(y, n_blocks)

print(len(x))
print(len(y))
print(x[-1].shape)
print(y[-1].shape)

for d1 in ("train", "test"):
    os.mkdir(path.join(mnist_dir, d1))
    for d2 in ("input", "label"):
        os.mkdir(path.join(mnist_dir, d1, d2))

for i, (xblock, yblock) in enumerate(zip(x, y)):
    if i < train_blocks:
        d1 = "train"
    else:
        d1 = "test"

    np.save(path.join(mnist_dir, d1, "input", f"block{i}.npy"), xblock)
    np.save(path.join(mnist_dir, d1, "label", f"block{i}.npy"), yblock)