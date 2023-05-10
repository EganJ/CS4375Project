import torch
import random
import numpy as np
import threading
from torch.utils.data import Dataset, DataLoader
import os
from os import path

steel_root = path.join(path.dirname(__file__), "steel_defect")
steel_img_dir = path.join(steel_root, "processed", "images")
steel_label_dir = path.join(steel_root, "processed", "labels")

with open(path.join(steel_root, "train_catalog.txt"), "r") as train_catalog:
    steel_train_catalog = train_catalog.readlines()
    steel_train_catalog = [n.strip() for n in steel_train_catalog]

with open(path.join(steel_root, "test_catalog.txt"), "r") as test_catalog:
    steel_test_catalog = test_catalog.readlines()
    steel_test_catalog = [n.strip() for n in steel_test_catalog]


aitex_root = path.join(path.dirname(__file__), "aitex")
aitex_img_dir = path.join(aitex_root, "processed", "images")
aitex_label_dir = path.join(aitex_root, "processed", "labels")

with open(path.join(aitex_root, "train_catalog.txt"), "r") as train_catalog:
    aitex_train_catalog = train_catalog.readlines()
    aitex_train_catalog = [n.strip() for n in aitex_train_catalog]

with open(path.join(aitex_root, "test_catalog.txt"), "r") as test_catalog:
    aitex_test_catalog = test_catalog.readlines()
    aitex_test_catalog = [n.strip() for n in aitex_test_catalog]


mnist_root = path.join(path.dirname(__file__), "mnist")
mnist_test = path.join(mnist_root, "test")
mnist_train = path.join(mnist_root, "train")

class AitexDataset(Dataset):
    def __init__(self, train = True):
        if train:
            self.catalog = aitex_train_catalog.copy()
        else:
            self.catalog = aitex_test_catalog.copy()

    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, index):
        name = self.catalog[index]

        x = torch.Tensor(np.load(path.join(aitex_img_dir, name)))
        y = torch.Tensor(np.load(path.join(aitex_label_dir, name)))

        return x, y
    

class SegmentationMnistDataset(Dataset):
    def __init__(self, train = True):
        if train:
            base_dir = mnist_train
        else:
            base_dir = mnist_test

        xdir = path.join(base_dir, "input")
        ydir = path.join(base_dir, "label")

        x_blocks = []
        y_blocks = []

        for blockname in os.listdir(xdir):
            x_blocks.append(np.load(path.join(xdir, blockname)))
            y_blocks.append(np.load(path.join(ydir, blockname)))
        
        self.x = np.concatenate(x_blocks)
        self.y = np.concatenate(y_blocks)

        self.n = len(self.x)

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return torch.Tensor(self.x[index]), torch.Tensor(self.y[index])


class SteelDataset(Dataset):
    """ Do NOT use with shuffle = True"""
    def __init__(self, train = True, chunksize = 40, threads = 4):
        self.chunksize = chunksize
        self.threads = threads

        if train:
            self.catalog = steel_train_catalog.copy()
        else:
            self.catalog = steel_test_catalog.copy()
        
        self.loaded = [ None ] * len(self.catalog)
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, index):
        item = self.loaded[index]
        if item is None:
            self.load_starting_from(index)
            item = self.loaded[index]
        
        return item
        

    def reset_loaded(self):
        # erase previous to avoid eating up memory
        del self.loaded
        self.loaded = [None] * len(self.catalog)

    def load_starting_from(self, start_ind):
        """
            Loads a large block of items into memory using multiple threads.
            Erases previous loaded items.
            The first loaded item is start_ind
        """
        
        self.reset_loaded()

        n = min(self.chunksize, len(self.catalog) - start_ind)

        breaks = [start_ind + (n * j) // self.threads
                  for j in range(self.threads + 1)]

        threads = [threading.Thread(target=self._load_chunk, args = (breaks[i], breaks[i+1]))
                   for i in range(self.threads)]

        for t in threads:
            t.start()
        
        for t in threads:
            t.join()

    def _load_chunk(self, i_start, i_end):
        for i in range(i_start, i_end):
            name = self.catalog[i]
            self.loaded[i] = [torch.Tensor(np.load(path.join(steel_img_dir, name))),
                              torch.Tensor(np.load(path.join(steel_label_dir, name)))]

    
class SteelLoader(DataLoader):
    def __init__(self, train = True, batch_size = 30):
        self.dataset = SteelDataset(train)
        super().__init__(self.dataset, shuffle = False, batch_size= batch_size)

    def shuffle(self):
        """
            Shuffles the dataset in a way that still allows for performant
            disk access
        """
        random.shuffle(self.dataset.catalog)
