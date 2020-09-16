#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import argparse
import math
import os

import progressbar
import tensorflow as tf
import tensorflow.keras as keras

import models.losses as losses
import models.wnet as wnet
import utils.data as data
import utils.utils as utils
import utils.utils_train as utrain

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed


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

PATH_SAVE_gen = "gen.h5"
PATH_SAVE_wnet = "wnet.h5"
BEST_LOSS = math.inf


def loss(gen, wnet, image, wei):
    loss_ncut = losses.soft_n_cut_loss2
    loss_recons = keras.metrics.mse
    pred = gen(image)
    gen_loss = tf.abs(
        loss_ncut(pred, wei)
    )  # todo : ici abs pas dans le papier, neg value
    output = wnet(image)
    wnet_loss = tf.cast(
        loss_recons(keras.backend.flatten(image), keras.backend.flatten(output)),
        dtype=tf.double,
    )
    # print("gen loss : {}\treconstruction loss : {}".format(gen_loss, wnet_loss))
    return gen_loss, wnet_loss


def _grad(gen, wnet, image, wei):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as wnet_tape:
        gen_loss, wnet_loss = loss(gen, wnet, image, wei)
    return (
        gen_loss,
        wnet_loss,
        gen_tape.gradient(gen_loss, gen.trainable_variables),
        wnet_tape.gradient(wnet_loss, wnet.trainable_variables),
    )


def save(gen, _wnet, loss, path_gen, path_wnet):
    global BEST_LOSS
    if loss < BEST_LOSS:
        utils.print_gre("Model saved.")
        BEST_LOSS = loss
        gen.save(path_gen)
        _wnet.save(path_wnet)


def _step(
    gen,
    model_wnet,
    x,
    w,
    optimizer_gen,
    optimizer_wnet,
    train=True,
):
    epoch_loss_avg = tf.keras.metrics.Mean()
    gen_loss, wnet_loss, gen_grads, wnet_grads = _grad(
        gen,
        model_wnet,
        x,
        w,
    )
    if train:
        optimizer_gen.apply_gradients(zip(gen_grads, gen.trainable_variables))
        optimizer_wnet.apply_gradients(zip(wnet_grads, model_wnet.trainable_variables))
    return epoch_loss_avg(gen_loss + wnet_loss)


def run_epoch(
    dataset,
    len_dataset,
    gen,
    model_wnet,
    o_gen,
    o_wnet,
    epoch,
    train=True,
):
    loss = 0
    with progressbar.ProgressBar(
        max_value=len_dataset / opt.batch_size, widgets=widgets
    ) as bar:
        for i, (x, w) in enumerate(dataset):
            bar.update(i)
            loss += _step(
                gen,
                model_wnet,
                x,
                w,
                o_gen,
                o_wnet,
                train,
            )
    if not train:
        x = tf.convert_to_tensor(
            x.numpy()[0].reshape(-1, 128, 128, 1), dtype=tf.float32
        )
        utrain.visualize(gen, model_wnet, x, epoch + 1, opt)
    return loss


def train():
    dataset_train, dataset_test, len_train, len_test = data.get_dataset(
        opt.img_path, opt
    )
    gen, model_wnet = wnet.wnet(input_shape=(opt.size, opt.size, 1))
    optimizer_gen = tf.keras.optimizers.Adam(opt.lr)
    optimizer_wnet = tf.keras.optimizers.Adam(opt.lr)
    train_loss_list = []
    test_loss_list = []
    utils.print_gre("Training....")
    for epoch in range(opt.n_epochs):
        utils.print_gre("Epoch {}/{}:".format(epoch + 1, opt.n_epochs))
        utils.print_gre("Training data:")
        train_loss = run_epoch(
            dataset_train,
            len_train,
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
        )
        utils.print_gre("Testing data:")
        test_loss = run_epoch(
            dataset_test,
            len_test,
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
            False,
        )
        utils.print_gre(
            "Epoch {:03d}/{:03d}: Loss: {:.3f} Test_Loss: {:.3f}".format(
                epoch + 1, opt.n_epochs, train_loss, test_loss
            )
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        utrain.is_nan(train_loss, test_loss, epoch)
        utrain.reduce_lr(epoch, 10, optimizer_gen, optimizer_wnet)
        save(gen, model_wnet, test_loss, PATH_SAVE_gen, PATH_SAVE_wnet)
        utrain.plot(train_loss_list, test_loss_list)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=12, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--size", type=int, default=128, help="Size of the image, one number"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Set patience value for early stopper"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="/home/elefevre/Datasets/Unsupervised_cell_and_spot_segmentation/Data_Eduardo/cell/patched_128/",  # noqa
        help="Path to get imgs",
    )
    args = parser.parse_args()
    utils.print_red("Args: " + str(args))
    return args


if __name__ == "__main__":
    opt = get_args()
    train()
