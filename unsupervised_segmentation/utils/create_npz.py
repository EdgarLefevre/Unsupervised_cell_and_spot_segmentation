import os

import ncuts
import numpy as np
import utils


def create_npz_name(filepath, img_size):
    file_name = os.path.basename(filepath).split(".")[0]
    img_path = os.path.dirname(filepath)
    img_path += "/"+str(img_size)+"/"
    return img_path+file_name

def save_indeces_and_vals(names, img_size=128):
    for name in names:
        indeces, vals = ncuts.gaussian_neighbor((img_size, img_size))
        file_name = create_npz_name(name, img_size)
        np.savez(file_name, indeces=indeces, vals=vals)


if __name__ == "__main__":
    img_path_list = utils.list_files_path("/home/edgar/Documents/Datasets/JB/good/")
    save_indeces_and_vals(img_path_list)


    # npzfile = np.load("/home/edgar/Desktop/test.npz")
    # print(npzfile.files)
    # vals_ = npzfile["vals"]
    # print(vals_.shape)
    # assert vals.all() == vals_.all()
