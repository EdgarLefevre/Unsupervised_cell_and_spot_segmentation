#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import multiprocessing
import os

import numpy as np
import skimage.io as io
import sklearn.feature_extraction.image as image

PATH_IMG = "/home/edgar/Desktop/dypfish/datasets/Edouardo_xist/spot/"


def list_files(path):
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def multi_process_fun(file_list, function):
    list_size = len(file_list)
    num_workers = 4
    worker_amount = int(list_size / num_workers)
    print(len(file_list[worker_amount * 0 : worker_amount * 0 + worker_amount]))
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
    return img_name.split(".")[0] + "_" + str(i) + ".tiff"


def wrapped_patch(im_list, patch_size=(256, 256)):
    for img_path in im_list:
        im = io.imread(img_path, plugin="tifffile")
        im = (im / np.amax(im)) * 255
        list_patches = patch_img(np.uint8(im), patch_size)
        for i, patch in enumerate(list_patches):
            if np.amax(patch) != 0:
                name = create_patch_name(img_path, i)
                io.imsave(PATH_IMG + "/patched/" + name, patch)


def patch_img(img, patch_size):
    return image.extract_patches_2d(img, patch_size, max_patches=15)


if __name__ == "__main__":
    img_list = list_files(PATH_IMG)
    multi_process_fun(img_list, wrapped_patch)
