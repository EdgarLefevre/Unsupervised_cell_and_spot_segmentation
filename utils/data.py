#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import skimage.io as io
import numpy as np
import utils.utils as utils
import random
import cupy as cp
import tensorflow as tf
import sklearn.model_selection as sk
import progressbar

# widget list for the progress bar
widgets = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


def get_dataset(path_imgs, args, strat):
    utils.print_gre("Creating dataset...")
    dataset = []
    file_list = utils.list_files(path_imgs)  # add [:10] to reduce dataset size
    for file in file_list:
        img = io.imread(path_imgs + file, plugin="tifffile")
        img = np.array(img / 255).reshape(-1, args.size, args.size, 1).astype('float32')
        img = np.rollaxis(img, 3, 1)
        dataset.append(img)
    n = range(np.shape(dataset)[0])
    n_sample = random.sample(list(n), len(n))
    dataset = np.array(dataset)
    dataset = dataset.reshape(-1, args.size, args.size, 1)[n_sample]
    weights = get_weights(dataset)
    dataset_train, dataset_test, weights_train, weights_test = sk.train_test_split(dataset, weights, test_size=0.2,
                                                                                   random_state=42)
    ds_train = tf.data.Dataset.from_tensor_slices((dataset_train, weights_train)).shuffle(10000).batch(args.batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((dataset_test, weights_test)).shuffle(10000).batch(args.batch_size)
    utils.print_gre("Dataset created !")
    return strat.experimental_distribute_dataset(ds_train), strat.experimental_distribute_dataset(ds_test), len(dataset_train), len(dataset_test)


def cal_weight(raw_data, shape, radius=5, sigmaI=10, sigmaX=4):
    # According to the weight formula, when Euclidean distance < r,the weight is 0,
    # so reduce the dissim matrix size to radius-1 to save time and space.
    dissim = cp.zeros(
        (shape[0], shape[1], shape[2], shape[3], (radius - 1) * 2 + 1, (radius - 1) * 2 + 1))
    data = cp.asarray(raw_data)
    padded_data = cp.pad(data, (
        (0, 0), (0, 0), (radius - 1, radius - 1), (radius - 1, radius - 1)), 'constant')
    for m in range(2 * (radius - 1) + 1):
        for n in range(2 * (radius - 1) + 1):
            dissim[:, :, :, :, m, n] = data - padded_data[:, :, m:shape[2] + m, n:shape[3] + n]
    temp_dissim = cp.exp(-cp.power(dissim, 2).sum(1, keepdims=True) / sigmaI ** 2)
    dist = cp.zeros((2 * (radius - 1) + 1, 2 * (radius - 1) + 1))
    for m in range(1 - radius, radius):
        for n in range(1 - radius, radius):
            if m ** 2 + n ** 2 < radius ** 2:
                dist[m + radius - 1, n + radius - 1] = cp.exp(-(m ** 2 + n ** 2) / sigmaX ** 2)
    res = cp.multiply(temp_dissim, dist)
    return res


def get_weights(raw_data):
    utils.print_gre("Calculating weights...")
    weights = []
    # raw_data = np.rollaxis(raw_data, 3, 1)
    shape = raw_data.shape
    with progressbar.ProgressBar(max_value=shape[0], widgets=widgets) as bar:
        for batch_id in range(0, shape[0]):
            bar.update(batch_id)
            batch = raw_data[batch_id:min(shape[0], batch_id + 1)]
            tmp_weight = cal_weight(batch, batch.shape)
            weight = cp.asnumpy(tmp_weight)
            weights.append(weight)
            del tmp_weight
    cp.get_default_memory_pool().free_all_blocks()
    utils.print_gre("Weights calculated.")
    return weights
