#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import argparse
import os

import progressbar
import tensorflow as tf
import tensorflow.keras as keras

import models.losses as losses
import models.wnet as wnet
import utils.data as data
import utils.utils as utils
import utils.utils_train as utrain

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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

PATH_SAVE = "./wnet.h5"
BEST_LOSS = 9999999999999999999999999999


def loss(gen, wnet, image, wei, loss1, loss2, size):
    pred = gen(image)
    gen_loss = tf.abs(loss1(pred, wei))  # todo : ici abs pas dans le papier, neg value
    output = wnet(image)
    wnet_loss = (size ** 2) * tf.cast(
        loss2(keras.backend.flatten(image), keras.backend.flatten(output)),
        dtype=tf.double,
    )
    # print("gen loss : {} \t reconstruction loss : {}".format(gen_loss, wnet_loss))
    return gen_loss, wnet_loss


def grad(gen, wnet, image, wei, loss1, loss2, opt):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as wnet_tape:
        gen_loss, wnet_loss = loss(gen, wnet, image, wei, loss1, loss2, opt.size)
    return (
        gen_loss / opt.batch_size,
        wnet_loss / opt.batch_size,
        gen_tape.gradient(gen_loss, gen.trainable_variables),
        wnet_tape.gradient(wnet_loss, wnet.trainable_variables),
    )


def save(model, loss):
    global BEST_LOSS
    if loss < BEST_LOSS:
        utils.print_gre("Model saved")
        BEST_LOSS = loss
        model.save(PATH_SAVE)


def train_step(
    gen,
    model_wnet,
    x,
    w,
    loss_ncut,
    loss_recons,
    opt,
    optimizer_gen,
    optimizer_wnet,
    e_l_avg,
    strat,
):
    # with strat.scope():
    gen_loss, wnet_loss, gen_grads, wnet_grads = grad(
        gen, model_wnet, x, w, loss_ncut, loss_recons, opt
    )
    optimizer_gen.apply_gradients(zip(gen_grads, gen.trainable_variables))
    optimizer_wnet.apply_gradients(zip(wnet_grads, model_wnet.trainable_variables))
    return e_l_avg(gen_loss + wnet_loss)


def test_step(
    gen, model_wnet, x, w, loss_ncut, loss_recons, opt, epoch_test_loss_avg, epoch
):
    gen_loss, wnet_loss, gen_grads, wnet_grads = grad(
        gen, model_wnet, x, w, loss_ncut, loss_recons, opt
    )
    return epoch_test_loss_avg(gen_loss + wnet_loss)


# @tf.function
def distributed_train_step(
    gen,
    model_wnet,
    x,
    w,
    loss_ncut,
    loss_recons,
    opt,
    strat,
    optimizer_gen,
    optimizer_wnet,
    e_l_avg,
):
    per_replica_losses = strat.run(
        train_step,
        args=(
            gen,
            model_wnet,
            x,
            w,
            loss_ncut,
            loss_recons,
            opt,
            optimizer_gen,
            optimizer_wnet,
            e_l_avg,
            strat,
        ),
    )
    return strat.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


def distributed_test_step(
    gen,
    model_wnet,
    x,
    w,
    loss_ncut,
    loss_recons,
    opt,
    strat,
    epoch_test_loss_avg,
    epoch,
):
    return strat.run(
        test_step,
        args=(
            gen,
            model_wnet,
            x,
            w,
            loss_ncut,
            loss_recons,
            opt,
            epoch_test_loss_avg,
            epoch,
        ),
    )


def train(path_imgs, opt):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        dataset_train, dataset_test, len_train, len_test = data.get_dataset(
            path_imgs, opt, mirrored_strategy
        )
        shape = (opt.size, opt.size, 1)
        gen, model_wnet = wnet.wnet(input_shape=shape)

        optimizer_gen = tf.keras.optimizers.Adam(opt.lr)
        optimizer_wnet = tf.keras.optimizers.Adam(opt.lr)
        checkpoint, checkpoint_path = utrain.get_checkpoint(
            optimizer_gen, optimizer_wnet, gen, model_wnet
        )
        loss_ncut = losses.soft_n_cut_loss2
        loss_recons = keras.metrics.mse
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_test_loss_avg = tf.keras.metrics.Mean()
        train_loss_list = []
        test_loss_list = []

        utils.print_gre("Training....")
        for epoch in range(opt.n_epochs):
            train_loss = 0.0
            test_loss = 0.0
            utils.print_gre("Epoch {}/{}:".format(epoch + 1, opt.n_epochs))
            utils.print_gre("Training data:")
            with progressbar.ProgressBar(
                max_value=len_train / opt.batch_size, widgets=widgets
            ) as bar:
                for i, (x, w) in enumerate(dataset_train):
                    bar.update(i)
                    train_loss += distributed_train_step(
                        gen,
                        model_wnet,
                        x,
                        w,
                        loss_ncut,
                        loss_recons,
                        opt,
                        mirrored_strategy,
                        optimizer_gen,
                        optimizer_wnet,
                        epoch_loss_avg,
                    )
            utils.print_gre("Testing data:")
            with progressbar.ProgressBar(
                max_value=len_test / opt.batch_size, widgets=widgets
            ) as bar2:
                for j, (x, w) in enumerate(dataset_test):
                    bar2.update(j)
                    test_loss += distributed_test_step(
                        gen,
                        model_wnet,
                        x,
                        w,
                        loss_ncut,
                        loss_recons,
                        opt,
                        mirrored_strategy,
                        epoch_test_loss_avg,
                        epoch,
                    )
                images = mirrored_strategy.experimental_local_results(x)
                utrain.visualize(gen, model_wnet, images[0].numpy(), epoch + 1, opt)
            # checkpoint.save(file_prefix=checkpoint_path)
            utils.print_gre(
                "Epoch {:03d}/{:03d}: Loss: {:.3f} Test_Loss: {:.3f}".format(
                    epoch + 1, opt.n_epochs, train_loss, test_loss
                )
            )
            train_loss_list.append(epoch_loss_avg.result())
            test_loss_list.append(epoch_test_loss_avg.result())
            # save(model, epoch_test_loss_avg.result())
            utrain.plot(train_loss_list, test_loss_list)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="number of epochs of training"
    )
    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--size", type=int, default=256, help="Size of the image, one number"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Set patience value for early stopper"
    )
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    opt = get_args()
    train("/home/elefevre/Data_Eduardo/cell/patched/", opt)
