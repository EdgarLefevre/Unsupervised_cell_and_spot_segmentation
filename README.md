# Unsupervised cell and spot segmentation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


Wnet implementation with tensorflow 2, experiment to segment cell and spot data without labels.

## Installation

```sh
conda env create -f environment.yml
pre-commit install # for devs
```

## Usage

```sh
python train_singlegpu.py  # if you want to run on a singlegpu
python train_multigpu.py  # if you want to run on several gpus
```

## Setting up gpu

To use a gpu you have to set the system variable `CUDA_VISIBLE_DEVICES` and assign the ids of the gpus you want to use:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu id
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # gpu ids
```
This line of code is at the begining of the train scripts, change it in function of your architecture and the number of gpus you want to use.

## Future Work

- [x] soft n cut loss
- [x] multigpu
- [ ] improve stability
- [ ] result folder as path
- [ ] inference
