import autograd.numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(x / (1 - x))