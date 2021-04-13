#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import math
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed

import progressbar
import sklearn.model_selection as sk
import tensorflow as tf
import tensorflow.keras as keras

import unsupervised_segmentation.models.losses as losses
import unsupervised_segmentation.models.wnet as wnet
import unsupervised_segmentation.utils.data as data
import unsupervised_segmentation.utils.utils as utils
import unsupervised_segmentation.utils.utils_train as utrain




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
    gen,
    model_wnet,
    o_gen,
    o_wnet,
    epoch,
    train=True,
):
    loss = 0
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
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
        x = tf.convert_to_tensor(x[0].reshape(-1, 128, 128, 1), dtype=tf.float32)
        utrain.visualize(gen, model_wnet, x, epoch + 1, opt)
    return loss


def loss2(gen, wnet, image):
    loss_ncut = losses.soft_n_cut_loss
    loss_recons = keras.metrics.mse
    pred = gen(image)
    gen_loss = tf.abs(loss_ncut(image, pred))
    output = wnet(image)
    wnet_loss = tf.cast(
        loss_recons(keras.backend.flatten(image), keras.backend.flatten(output)),
        dtype=tf.double,
    )
    return gen_loss, wnet_loss


def _grad2(gen, wnet, image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as wnet_tape:
        gen_loss, wnet_loss = loss2(gen, wnet, image)
    return (
        gen_loss,
        wnet_loss,
        gen_tape.gradient(gen_loss, gen.trainable_variables),
        wnet_tape.gradient(wnet_loss, wnet.trainable_variables),
    )


def _step2(
    gen,
    model_wnet,
    x,
    optimizer_gen,
    optimizer_wnet,
    train=True,
):
    epoch_loss_avg = tf.keras.metrics.Mean()
    gen_loss, wnet_loss, gen_grads, wnet_grads = _grad2(
        gen,
        model_wnet,
        x,
    )
    if train:
        optimizer_gen.apply_gradients(zip(gen_grads, gen.trainable_variables))
        optimizer_wnet.apply_gradients(zip(wnet_grads, model_wnet.trainable_variables))
    return epoch_loss_avg(gen_loss + wnet_loss)


def run_epoch2(
    dataset,
    gen,
    model_wnet,
    o_gen,
    o_wnet,
    epoch,
    train=True,
):
    loss = 0
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i, x in enumerate(dataset):
            bar.update(i)
            loss += _step2(
                gen,
                model_wnet,
                x,
                o_gen,
                o_wnet,
                train,
            )
    if not train:
        x = tf.convert_to_tensor(x[0].reshape(-1, 128, 128, 1), dtype=tf.float32)
        utrain.visualize(gen, model_wnet, x, epoch + 1, opt)
    return loss


def train2():
    img_path_list = utils.list_files_path(opt.img_path)
    # not good if we need to do metrics
    img_train, img_test = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )

    dataset_train = data.Dataset2(opt.batch_size, opt.size, img_train)
    dataset_test = data.Dataset2(opt.batch_size, opt.size, img_test)
    gen, model_wnet = wnet.wnet(input_shape=(opt.size, opt.size, 1))
    optimizer_gen = tf.keras.optimizers.Adam(opt.lr)
    optimizer_wnet = tf.keras.optimizers.Adam(opt.lr)
    train_loss_list = []
    test_loss_list = []
    utils.print_gre("Training....")
    for epoch in range(opt.n_epochs):
        utils.print_gre("Epoch {}/{}:".format(epoch + 1, opt.n_epochs))
        utils.print_gre("Training data:")
        train_loss = run_epoch2(
            dataset_train,
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
        )
        utils.print_gre("Testing data:")
        test_loss = run_epoch2(
            dataset_test,
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


def train():
    img_path_list = utils.list_files_path(opt.img_path)
    # not good if we need to do metrics
    img_train, img_test = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )

    dataset_train = data.Dataset(opt.batch_size, opt.size, img_train)
    dataset_test = data.Dataset(opt.batch_size, opt.size, img_test)
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
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
        )
        utils.print_gre("Testing data:")
        test_loss = run_epoch(
            dataset_test,
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


if __name__ == "__main__":
    opt = utils.get_args()
    train()
