#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
        image = (image.numpy() * 255).astype(np.uint8).reshape(-1, opt.size, opt.size)
        argmax = tf.expand_dims(tf.argmax(pred, 3), 3)
        pred, output = (
            (argmax * 255).numpy().astype(np.uint8).reshape(-1, opt.size, opt.size),
            (output * 255).numpy().astype(np.uint8).reshape(-1, opt.size, opt.size),
        )
        # io.imsave("result/image" + str(k) + ".png", image)
        # io.imsave("result/pred" + str(k) + ".png", pred)
        # io.imsave("result/output" + str(k) + ".png", output)
        plot_images(image, pred, output, k, opt.size)


def plot_images(imgs, pred, output, k, size):
    fig = plt.figure(figsize=(15, 10))
    columns = 3
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="gray")
        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j].reshape((size, size)), cmap="gray")
        ax.append(fig.add_subplot(rows, columns, i + 3))
        ax[-1].set_title("Output")
        plt.imshow(output[j].reshape((size, size)), cmap="gray")
        i += 3
        if i >= 15:
            break
    plt.savefig("result/epoch_" + str(k) + ".png")
    plt.close()


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
        print("\nLearning rate is {}".format(gen.learning_rate.numpy()))
