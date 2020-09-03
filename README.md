# Unsupervised cell and spot segmentation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


Experiments to segment cell and spot images without labels.
## Installation

```sh
conda env create -f env.yaml
pre-commit install
```

## Usage

```sh
python train.py
```

## Future Work

- [x] code crf
- [ ] multigpu
- [x] clean training loop
- [x] Images path as arg
- [ ] improve stability
