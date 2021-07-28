#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import math
import os

import progressbar
import sklearn.model_selection as sk
import tensorflow as tf
import tensorflow.keras as keras
import wandb

import unsupervised_segmentation.models.attention_wnet as wnet
import unsupervised_segmentation.utils.data as data
import unsupervised_segmentation.utils.ncuts as ncuts
import unsupervised_segmentation.utils.utils as utils
import unsupervised_segmentation.utils.utils_train as utrain

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version: ", tf.__version__)



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

PATH_SAVE_gen = "weights/gen"
PATH_SAVE_wnet = "weights/wnet"
BEST_LOSS = math.inf


def ssim(ytrue, ypred):
    """
    - ssim because in ssim +1 = best and -1 worst ; -ssim+1 -> 0 best and +2 worst
    :param ytrue:
    :type ytrue:
    :param ypred:
    :type ypred:
    :return:
    :rtype:
    """
    return -tf.image.ssim(ytrue, ypred, max_val=1) + 1


def loss(gen, wnet, image, wei):
    loss_ncut = ncuts.soft_ncut
    loss_recons = ssim  # keras.metrics.mse
    pred = gen(image)
    gen_loss = tf.reduce_mean(tf.abs(loss_ncut(image, pred, wei)))
    output = wnet(image)
    # wnet_loss = tf.cast(
    #     loss_recons(keras.backend.flatten(image), keras.backend.flatten(output)),
    #     dtype=tf.double,
    # )
    wnet_loss = tf.reduce_mean(tf.cast(
        loss_recons(image, output),
        dtype=tf.double,
    ))
    # utils.print_gre(
    # "\nTrain: Gen Loss: {:.3f} Reconstruction Loss: {:.3f}".format(gen_loss, wnet_loss)
    # )
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
    gen_loss, wnet_loss, gen_grads, wnet_grads = _grad(
        gen,
        model_wnet,
        x,
        w,
    )
    if train:
        optimizer_gen.apply_gradients(zip(gen_grads, gen.trainable_variables))
        optimizer_wnet.apply_gradients(zip(wnet_grads, model_wnet.trainable_variables))
    return gen_loss, wnet_loss


def run_epoch(
        dataset,
        gen,
        model_wnet,
        o_gen,
        o_wnet,
        epoch,
        train=True,
):
    loss_gen = 0
    loss_recons = 0
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        for i, (x, w) in enumerate(dataset):
            bar.update(i)
            l_gen, l_recons = _step(
                gen,
                model_wnet,
                x,
                w,
                o_gen,
                o_wnet,
                train,
            )
            loss_gen += l_gen
            loss_recons += l_recons
    if not train:
        x = tf.reshape(x, (-1, opt.size, opt.size, 1))
        utrain.visualize(gen, model_wnet, x, epoch + 1, opt)
    return loss_gen, loss_recons


def train():
    img_path_list = utils.list_files_path(opt.img_path)
    # not good if we need to do metrics
    img_train, img_test = sk.train_test_split(
        img_path_list, test_size=0.2, random_state=42
    )

    dataset_train = data.Dataset(opt.batch_size, opt.size, img_train)
    dataset_test = data.Dataset(opt.batch_size, opt.size, img_test)
    # dataset_train, dataset_test = data.get_new_dataset(opt.img_path)
    gen, model_wnet = wnet.wnet(input_shape=(opt.size, opt.size, 1))
    optimizer_gen = tf.keras.optimizers.Adam(opt.lr)
    optimizer_wnet = tf.keras.optimizers.Adam(opt.lr)
    train_loss_list = []
    test_loss_list = []
    utils.print_gre("Training....")
    for epoch in range(opt.n_epochs):
        utils.print_gre("Epoch {}/{}:".format(epoch + 1, opt.n_epochs))
        utils.print_gre("Training data:")
        gen_loss, recons_loss = run_epoch(
            dataset_train,
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
        )
        utils.print_gre(
            "Train: Gen Loss: {:.3f} Reconstruction Loss: {:.3f}".format(
                gen_loss.numpy().mean(), recons_loss.numpy().mean()
            )
        )
        utils.print_gre("Testing data:")
        gen_loss_t, recons_loss_t = run_epoch(
            dataset_test,
            gen,
            model_wnet,
            optimizer_gen,
            optimizer_wnet,
            epoch,
            False,
        )
        utils.print_gre(
            "Test: Gen Loss: {:.3f} Reconstruction Loss: {:.3f}".format(
                gen_loss_t.numpy().mean(), recons_loss_t.numpy().mean()
            )
        )
        # wandb.log({"train_gen_loss": gen_loss.numpy(),
        #            "train_recons_loss": recons_loss.numpy(),
        #            "test_gen_loss": gen_loss_t.numpy(),
        #            "test_recons_loss": recons_loss_t.numpy(),
        #            })
        train_loss_list.append(gen_loss)
        test_loss_list.append(gen_loss_t)
        utrain.is_nan(gen_loss, gen_loss_t, epoch)
        # utrain.reduce_lr(epoch, 10, optimizer_gen, optimizer_wnet)
        save(gen, model_wnet, gen_loss_t, PATH_SAVE_gen, PATH_SAVE_wnet)
        utrain.plot(train_loss_list, test_loss_list)


if __name__ == "__main__":
    opt = utils.get_args()
    # wandb.login(key="2087297064263382243a621b1bcdd37fcf1c6bb4")
    # wandb.init(
    #     project="wnet",
    #     entity="elefevre",
    #     config=opt,
    #     name="test-tf.ssim"
    # )

    train()
