#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img

import unsupervised_segmentation.utils.utils as utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def inference(dataset, path_model):
    model_seg = keras.models.load_model(
        path_model,
    )
    res = model_seg.predict(dataset)
    return tf.argmax(res, 3).numpy().reshape(128, 128)


if __name__ == "__main__":
    img_list = utils.list_files_path("/home/edgar/Documents/Datasets/JB/good/")
    for img_path in img_list:
        print(img_path)
        img = np.array(
            load_img(img_path, color_mode="grayscale", target_size=(128, 128))
        ).reshape(1, 128, 128, 1)
        img = img / 255
        print(np.shape(img))
        mask = inference(
            img,
            "/home/edgar/Documents/Projects/Unsupervised_cell_and_spot_segmentation/gen",  # noqa
        )
        fig = plt.figure(figsize=(15, 10))
        columns = 2
        rows = 1
        ax = []
        ax.append(fig.add_subplot(rows, columns, 1))
        ax[-1].set_title("Input")
        plt.imshow((img * 255).reshape(128, 128), cmap="gray")
        ax.append(fig.add_subplot(rows, columns, 2))
        ax[-1].set_title("Mask")
        plt.imshow(mask * 255, cmap="gray")
        plt.show()
        break
