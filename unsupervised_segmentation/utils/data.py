#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img

import unsupervised_segmentation.utils.ncuts as ncuts

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]


class Dataset(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        indeces, vals = ncuts.gaussian_neighbor((img_size, img_size))
        self.indeces = tf.constant(indeces)
        self.vals = tf.constant(vals, dtype=tf.float32)
        self.weight_shapes = np.prod((img_size, img_size)).astype(np.int64)
        self.weight_size = tf.constant([self.weight_shapes, self.weight_shapes])

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros(
            (self.batch_size, self.img_size, self.img_size, 1), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = (
                np.array(
                    load_img(
                        path,
                        color_mode="grayscale",
                        target_size=(self.img_size, self.img_size),
                    )
                )
                / 255
            )
            x[j] = np.expand_dims(img, 2)
        w = ncuts.process_weight_multi(
            x, self.indeces, self.vals, self.weight_shapes, self.weight_size
        )
        return tf.constant(x), w
