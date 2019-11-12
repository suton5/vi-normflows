"""Implementations of planar and radial flows.
"""
from autograd import numpy as np


def planar_flow(z: np.ndarray,
                u: np.ndarray,
                w: np.ndarray,
                b: np.ndarray,
                h=np.tanh) -> np.ndarray:
    """Apply a planar flow to each element of `samples`

    :param z: numpy array, samples to be transformed
    :param u: numpy array, parameter of flow
    :param w: numpy array, parameter of flow
    :param b: numpy array, parameter of flow
    :param h: callable, non-linear function (default tanh)
    :returns: numpy array, transformed samples

    Transforms given samples according to the planar flow
    :math:`f(z) = z + uh(w^Tz + b)`
    """
    return z + u * h(np.dot(w, z) + b)
