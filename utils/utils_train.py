#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os
import sys

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
    if k % 2 == 0 or k == 1:
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


def is_nan(train_loss, test_loss, epoch):
    if np.isnan(train_loss) or np.isnan(test_loss):
        print(
            "Loss is Not a Number. Train_loss: {}.  Test_loss: {}.\nEpoch {}.".format(
                train_loss, test_loss, epoch + 1
            )
        )
        sys.exit()


def reduce_lr(epoch, decay, gen, wnet, freq=10):
    if (epoch + 1) % freq == 0:
        gen.learning_rate = gen.learning_rate.numpy() / decay
        wnet.learning_rate = wnet.learning_rate.numpy() / decay
        print("\nLearning rate is {}".format(gen.leanring_rate.numpy()))
