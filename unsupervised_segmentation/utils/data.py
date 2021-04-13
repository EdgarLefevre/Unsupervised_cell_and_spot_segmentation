#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import progressbar
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img

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


def cal_weight(raw_data, shape, radius=5, sigmaI=10, sigmaX=4):
    # According to the weight formula, when Euclidean distance < r,the weight is 0,
    # so reduce the dissim matrix size to radius-1 to save time and space.
    dissim = cp.zeros(
        (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            (radius - 1) * 2 + 1,
            (radius - 1) * 2 + 1,
        )
    )
    data = cp.asarray(raw_data)
    padded_data = cp.pad(
        data,
        ((0, 0), (0, 0), (radius - 1, radius - 1), (radius - 1, radius - 1)),
        "constant",
    )
    for m in range(2 * (radius - 1) + 1):
        for n in range(2 * (radius - 1) + 1):
            dissim[:, :, :, :, m, n] = (
                data - padded_data[:, :, m : shape[2] + m, n : shape[3] + n]
            )
    temp_dissim = cp.exp(-cp.power(dissim, 2).sum(1, keepdims=True) / sigmaI ** 2)
    dist = cp.zeros((2 * (radius - 1) + 1, 2 * (radius - 1) + 1))
    for m in range(1 - radius, radius):
        for n in range(1 - radius, radius):
            if m ** 2 + n ** 2 < radius ** 2:
                dist[m + radius - 1, n + radius - 1] = cp.exp(
                    -(m ** 2 + n ** 2) / sigmaX ** 2
                )
    res = cp.multiply(temp_dissim, dist)
    return res


def get_weights(raw_data):
    weights = []
    shape = raw_data.shape
    for img in raw_data:
        img = cp.expand_dims(img, axis=0)
        sigI = cp.sum((img - cp.mean(img)) ** 2) / (shape[1] * shape[2])
        tmp_weight = cal_weight(img, img.shape, sigmaI=sigI)
        weight = cp.asnumpy(tmp_weight)
        weights.append(weight)
        del tmp_weight
        del weight
    cp.get_default_memory_pool().free_all_blocks()
    return weights


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

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.

        For wnet -> should return only x+weight map (couple)
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
        w = get_weights(x)
        return x, w


class Dataset2(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.

        For wnet -> should return only x+weight map (couple)
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
        return x
