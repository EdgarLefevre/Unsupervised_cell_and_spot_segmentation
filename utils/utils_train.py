#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tensorflow as tf


def get_checkpoint(gen_opti, wnet_opti, gen, model_wnet):  # todo : use this
    checkpoint_dir = "./saved_models"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=gen_opti,
        wnet_optimizer=wnet_opti,
        generator=gen,
        wnet=model_wnet,
    )
    return checkpoint, checkpoint_prefix


def visualize(gen, model_wnet, image, k, opt):
    if k % 5 == 0 or k == 1:
        pred = gen(image)
        output = model_wnet(image)
        image = (image[0] * 255).astype(np.uint8).reshape(opt.size, opt.size)
        argmax = tf.expand_dims(tf.argmax(pred, 3), 3)
        pred, output = (
            (argmax[0] * 255).numpy().astype(np.uint8).reshape(opt.size, opt.size, 1),
            (output[0] * 255).numpy().astype(np.uint8).reshape(opt.size, opt.size),
        )
        io.imsave("result/image" + str(k) + ".png", image)
        io.imsave("result/pred" + str(k) + ".png", pred)
        io.imsave("result/output" + str(k) + ".png", output)


def plot(train, test):
    fig, axes = plt.subplots(2, figsize=(12, 8))
    fig.suptitle("Training Metrics")

    axes[0].set_ylabel("Train Loss", fontsize=14)
    axes[0].plot(train)

    axes[1].set_ylabel("Test Loss", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(test)
    fig.savefig("plots/plot.png")
    plt.close(fig)
