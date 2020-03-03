Adaptive Bayesian Reticulum
===========================
Overview
--------
This is an implementation of the paper [Adaptive Bayesian Reticulum](https://arxiv.org/abs/1912.05901)
by Nuti et al. The model is a binary and multiclass classification tree with soft margins and a novel
tree construction method.


Installation
------------
To install you can either use _conda_ or _pip_:

#### Conda
```
git clone https://github.com/UBS-IB/adaptive-bayesian-reticulum
cd adaptive-bayesian-reticulum
conda build conda.recipe
conda install --use-local adaptive-bayesian-reticulum
```

#### PIP
```
git clone https://github.com/UBS-IB/adaptive-bayesian-reticulum
cd adaptive-bayesian-reticulum
pip install -e .
```

## Usage
Please see the [Demo Scripts](examples) for usage examples.

Note that the model is fully compatible with scikit-learn, so you can use it for
e.g. cross-validation or performance evaluation with scikit-learn classes and
functions.


## TODO
- Update links to paper once published in a journal
