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
    if h == np.tanh:
        print("prev u", u)
        u = _get_uhat(u, w)
<<<<<<< HEAD
        assert np.dot(u, w) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
=======
        print("new u", u)
    print("w" ,w)

    assert np.dot(u, w) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
>>>>>>> 05f3a657ddcb2a5b7bfddc9201345dc7fec07d37

    w = w.flatten()
    u = u.flatten()

    assert z.ndim == 2, f'z has incorrect number of dimensions ({z.ndim}).'

    N = z.shape[0]
    d = z.shape[1]

    assert d == w.shape[0], f'Dimensions of z and w are not aligned ({d} != {w.shape[0]}).'
    assert d == u.shape[0], f'Dimensions of z and u are not aligned ({d} != {u.shape[0]}).'

    dotprod = np.dot(w, z.T)
    h_arg = dotprod + b
    h_res = np.repeat(h(h_arg).reshape(-1, 1), d, axis=1)
    u_mult = u * h_res
    res = z + u_mult.reshape(N, d)
    return res


def _get_uhat(u, w):
    return u + (m(np.dot(w, u)) - np.dot(w, u)) * (w/np.linalg.norm(w))#np.divide(w, np.dot(w, w))


def m(x):
    return -1 + np.log(1 + np.exp(x))
