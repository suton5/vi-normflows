"""Implementations of planar and radial flows.
"""
from typing import Union

from autograd import numpy as np


def planar_flow(z: np.ndarray,
                w: np.ndarray,
                u: np.ndarray,
                b: Union[int, float],
                h=np.tanh) -> np.ndarray:
    """Apply a planar flow to each element of `samples`

    :param z: numpy array, samples to be transformed
        Shape: (n_samples, n_dim)
    :param u: numpy array, parameter of flow
    :param w: numpy array, parameter of flow
    :param b: numeric, parameter of flow
    :param h: callable, non-linear function (default tanh)
    :returns: numpy array, transformed samples

    Transforms given samples according to the planar flow
    :math:`f(z) = z + uh(w^Tz + b)`
    """
    u = _get_uhat(u, w)
    assert np.dot(u, w) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
    assert np.dot(u, w) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
    return z + np.outer(np.tanh(np.matmul(z, w) + b), u)


def _get_uhat(u, w):
    return u + (m(np.dot(w, u)) - np.dot(w, u)) * (w / (np.linalg.norm(w) ** 2))


def m(x):
    return -1 + np.log(1 + np.exp(x))
