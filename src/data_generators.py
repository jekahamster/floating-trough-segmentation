import os 
import numpy as np
import cv2
import imageio
import tensorflow as tf

from pathlib import Path
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 images_path, 
                 masks_path, 
                 image_type="jpg", 
                 mask_type="png", 
                 batch_size=32, 
                 shuffle=True, 
                 height=128, 
                 width=128, 
                 mask_channels=1,
                 transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_type = image_type
        self.mask_type = mask_type
        self.fnames_stem = [Path(fname).stem for fname in os.listdir(images_path)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.mask_channels = mask_channels
        self.transform = transform
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.fnames_stem)

    def __len__(self):
        return len(self.fnames_stem) // self.batch_size

    def __getitem__(self, index):
        X, y = self.get_batch(index)
        return X, y

    def get_batch(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        X = np.empty((self.batch_size, self.height, self.width, 3), dtype=float)
        y = np.empty((self.batch_size, self.height, self.width, self.mask_channels), dtype=float)

        for i, sample_index in enumerate(range(start_index, end_index)):
            img, mask = self.get_single(sample_index)
            X[i,] = img
            y[i,] = mask

        X = tf.convert_to_tensor(X)
        y = tf.convert_to_tensor(y)

        return X, y
    
    def get_single(self, index):
        fname_stem = self.fnames_stem[index]
        img_path = self.images_path / f"{fname_stem}.{self.image_type}"
        mask_path = self.masks_path / f"{fname_stem}.{self.mask_type}"

        img = imageio.imread(img_path)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        img = np.array(img, dtype=np.float32) / 255.0

        mask = imageio.imread(mask_path)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return img, mask