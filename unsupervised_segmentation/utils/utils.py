#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
import os

import tensorflow.keras.backend as K


def print_red(skk):
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    print("\033[92m{}\033[00m".format(skk))


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_files_path(path):
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=150, help="number of epochs of training"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
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
        default="/home/edgar/Documents/Datasets/JB/good/",  # noqa
        help="Path to get imgs",
    )
    args = parser.parse_args()
    print_red("Args: " + str(args))
    return args


def MSE_BCE(y_true, y_pred, alpha=1000, beta=10):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * mse + beta * bce
