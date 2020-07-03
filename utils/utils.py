#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import tensorflow.keras.backend as K


def print_red(skk):
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    print("\033[92m{}\033[00m".format(skk))


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def MSE_BCE(y_true, y_pred, alpha=1000, beta=10):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * mse + beta * bce
