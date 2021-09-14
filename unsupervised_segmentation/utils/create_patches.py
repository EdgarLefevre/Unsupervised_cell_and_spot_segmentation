#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import multiprocessing
import os

import numpy as np
import skimage.io as io

BASE_PATH = "/home/edgar/Documents/Datasets/JB/"
PATH_IMG = BASE_PATH + "raw/8bits/cells/"
PATH_SLICES = BASE_PATH + "slices/"


def list_files(path):
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def multi_process_fun(file_list, function):
    list_size = len(file_list)
    num_workers = 8
    worker_amount = int(list_size / num_workers)
    processes = []
    for worker_num in range(num_workers):
        process = multiprocessing.Process(
            target=function,
            args=(
                [
                    file_list[
                        worker_amount * worker_num : worker_amount * worker_num
                        + worker_amount
                    ]
                ]
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def create_patch_name(path, i):
    img_name = path.split("/")[-1]
    return img_name.split(".")[0] + "_" + str(i) + ".png"


def wrapped_patch(im_list, patch_size=512):
    for img_path in im_list:
        im = io.imread(img_path)
        # im = np.uint8((im / np.amax(im)) * 255)
        list_patches = patch(im, patch_size, 16)
        for i, patch_ in enumerate(list_patches):
            name = create_patch_name(img_path, i)
            io.imsave(BASE_PATH + "/patches/" + name, patch_)


def patch(im, size, nb_patch):
    patch_list = []
    ymin = 0
    ymax = size
    xmin = 0
    xmax = size
    i = 0
    x_im = np.shape(im)[0]
    y_im = np.shape(im)[1]
    while i < nb_patch - 1:
        patch_list.append(im[xmin:xmax, ymin:ymax])
        xmin += size
        xmax += size
        if xmax > x_im:
            ymin += size
            ymax += size
            xmin = 0
            xmax = size
        if ymax > y_im:
            print("too much patches")
            break
        i += 1
    return patch_list


def img_to_slices(img_list):
    for im in img_list:
        img = io.imread(im, plugin="tifffile")
        for i, slice_img in enumerate(img):
            # slice_img = (slice_img / np.amax(slice_img)) * 255
            io.imsave(PATH_SLICES + create_patch_name(im, i), slice_img)


if __name__ == "__main__":
    # img_list = list_files(PATH_IMG)
    # img_to_slices(img_list)
    slice_list = list_files(PATH_SLICES)
    multi_process_fun(slice_list, wrapped_patch)
