#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
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

PATH_SAVE_gen = "gen_test.h5"
PATH_SAVE_wnet = "wnet_test.h5"
BEST_LOSS = math.inf


def save_model(gen, _wnet, loss, path_gen, path_wnet):
    global BEST_LOSS
    if loss < BEST_LOSS:
        utils.print_gre("Model saved.")
        BEST_LOSS = loss
        gen.save(path_gen)
        _wnet.save(path_wnet)


strat = tf.distribute.MirroredStrategy()
with strat.scope():
    test_loss = tf.keras.metrics.Mean(name="test_loss")

    def loss_wnet(image, pred):
        loss_recons = keras.metrics.mse
        return tf.cast(
            loss_recons(keras.backend.flatten(image), keras.backend.flatten(pred)),
            dtype=tf.float64,
        )

    def loss_gen(pred, wei):
        loss_ncut = losses.soft_n_cut_loss2_multi
        return tf.abs(loss_ncut(pred, wei))

    def _grad(gen, _wnet, image, wei):
        with tf.GradientTape() as wnet_tape:
            pred_wnet = _wnet(image)
            wnet_loss = loss_wnet(image, pred_wnet)
        wnet_grads = wnet_tape.gradient(wnet_loss, _wnet.trainable_variables)
        with tf.GradientTape() as gen_tape:
            pred_gen = gen(image)
            gen_loss = loss_gen(pred_gen, wei)
        gen_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
        return (gen_loss, wnet_loss, gen_grads, wnet_grads)

    def train_step(
        x,
        w,
        gen,
        _wnet,
        optimizer_gen,
        optimizer_wnet,
    ):
        gen_loss, wnet_loss, gen_grads, wnet_grads = _grad(gen, _wnet, x, w)
        optimizer_gen.apply_gradients(zip(gen_grads, gen.trainable_variables))
        optimizer_wnet.apply_gradients(zip(wnet_grads, _wnet.trainable_variables))
        return gen_loss + wnet_loss

    def test_step(
        x,
        w,
        gen,
        _wnet,
    ):
        pred_gen = gen(x)
        pred_wnet = _wnet(x)
        test_loss.update_state(loss_gen(pred_gen, w) + loss_wnet(x, pred_wnet))

    @tf.function
    def distributed_train_step(img, w, gen, _wnet, optimizer_gen, optimizer_wnet):
        per_replica_losses = strat.run(
            train_step, args=(img, w, gen, _wnet, optimizer_gen, optimizer_wnet)
        )
        return strat.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(img, w, gen, _wnet):
        return strat.run(test_step, args=(img, w, gen, _wnet))

    def run_epoch(
        dataset_train,
        dataset_test,
        len_dataset_train,
        len_dataset_test,
        gen,
        model_wnet,
        optimizer_gen,
        optimizer_wnet,
        epoch,
    ):
        loss_train = 0
        with progressbar.ProgressBar(
            max_value=len_dataset_train / opt.batch_size, widgets=widgets
        ) as bar:
            for i, (x, w) in enumerate(dataset_train):
                bar.update(i)
                loss_train += distributed_train_step(
                    x, w, gen, model_wnet, optimizer_gen, optimizer_wnet
                )

        utils.print_gre("Testing data:")
        with progressbar.ProgressBar(
            max_value=len_dataset_test / opt.batch_size, widgets=widgets
        ) as bar:
            for i, (x, w) in enumerate(dataset_test):
                bar.update(i)
                distributed_test_step(x, w, gen, model_wnet)
        try:
            x = strat.experimental_local_results(x)[0].numpy()[0]
            x = tf.convert_to_tensor(x.reshape(-1, 128, 128, 1), dtype=tf.float32)
            utrain.visualize(gen, model_wnet, x, epoch + 1, opt)
        except Exception as e:
            print(e)
        return loss_train

    def train():
        dataset_train, dataset_test, len_train, len_test = data.get_dataset(
            opt.img_path, opt
        )
        dataset_train = strat.experimental_distribute_dataset(dataset_train)
        dataset_test = strat.experimental_distribute_dataset(dataset_test)
        # manquera distributed step
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
                dataset_test,
                len_train,
                len_test,
                gen,
                model_wnet,
                optimizer_gen,
                optimizer_wnet,
                epoch,
            )
            utils.print_gre(
                "Epoch {:03d}/{:03d}: Loss: {:.3f} Test_Loss: {:.3f}".format(
                    epoch + 1, opt.n_epochs, train_loss, test_loss.result()
                )
            )
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss.result())
            utrain.is_nan(train_loss, test_loss.result(), epoch)
            utrain.reduce_lr(epoch, 10, optimizer_gen, optimizer_wnet)
            save_model(
                gen, model_wnet, test_loss.result(), PATH_SAVE_gen, PATH_SAVE_wnet
            )
        utrain.plot(train_loss_list, test_loss_list)

    opt = utils.get_args()
    train()
