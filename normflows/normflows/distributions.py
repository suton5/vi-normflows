"""Various distribution PDFs and sampling functions
"""
import autograd.numpy as np
import autograd.scipy as sp

from .transformations import sigmoid
from .flows import planar_flow

def mvn(Z, mu, sigma_diag):
    """Multivariate normal distribution

    :param Z: np.ndarray -- Samples from MVN. Shape (n_samples, dim_z)
    :param mu: np.ndarray -- Shape (n_samples | 1, dim_z)
    :param sigma_diag: np.ndarray -- Shape (dim_z,)
    :return: log_prob (scalar)
    """
    N, D = Z.shape
    mu = mu.reshape(-1, D)
    if len(sigma_diag) == 1:
        siginv = 1 / sigma_diag
        det_sigma = sigma_diag
    else:
        siginv = np.linalg.inv(sigma_diag)
        det_sigma = np.linalg.det(sigma_diag)

    zdiff = Z - mu

    const = (2 * np.pi) ** (-D / 2) * det_sigma ** (-1 / 2)
    prod = np.dot(zdiff, siginv).reshape(N, D)
    prob = const * np.exp(-0.5 * np.sum(prod * zdiff, axis=1)).reshape(N, 1)
    return prob


def log_mvn(Z, mu, log_sigma_diag):
    """Log of Multivariate normal distribution

    :param Z: np.ndarray -- Samples from MVN. Shape (n_samples, dim_z)
    :param mu: np.ndarray -- Shape (n_samples | 1, dim_z)
    :param log_sigma_diag: np.ndarray -- Shape (dim_z,)
    :return: log_prob (scalar)
    """
    N, D = Z.shape
    mu = mu.reshape(-1, D)
    logdet_sigma = np.sum(log_sigma_diag)
    if len(log_sigma_diag) == 1:
        siginv = np.exp(1 - log_sigma_diag)
    else:
        siginv = np.linalg.inv(np.diag(np.exp(log_sigma_diag)))
    const = -D / 2 * np.log(2 * np.pi) - 0.5 * logdet_sigma
    zdiff = Z - mu

    logprob = const - 0.5 * np.sum(np.matmul(zdiff, siginv) * zdiff, axis=1)
    return logprob


def prob_gm(Z, mu, sigma_diag, pi):
    G = pi.shape[0] + 1
    prob = 0
    for g in range(G):
        if g == G - 1:
            pi_g = 1 - np.sum(pi)
        else:
            pi_g = pi[g]
        prob = prob + mvn(Z, mu[g], sigma_diag[g]) * pi_g
    return prob


def log_prob_gm(Z, mu, log_sigma_diag, logit_pi):
    # assert np.sum(pi) < 1, f'probabilities of GMM do not sum to less than 1 ({np.sum(pi)})'
    # assert np.all(pi >= 0), f'Probabilities of GMM cannot be negative'
    #TODO: Constrain this!
    pi = sigmoid(logit_pi)
    # if np.sum(pi) > 1:
    #     pi /= np.sum(pi)
    assert mu.shape[0] == pi.shape[0] + 1, f'Number of means does not match number of components'
    assert Z.shape[1] == mu.shape[1], f'Dimensions of random variable and mean vector not aligned'

    sigma_diag = np.exp(log_sigma_diag)
    prob = prob_gm(Z, mu, sigma_diag, pi)
    return np.log(prob)


def sample_from_pz(mu, log_sigma_diag, W, U, b, K):
    N, D = mu.shape
    sd = np.sqrt(np.exp(log_sigma_diag))

    Z = np.zeros((K + 1, N, D))
    z = np.random.randn(N, D) * sd + mu

    Z[0] = z
    for k in range(K):
        Z[k + 1] = planar_flow(z, W[k], U[k], b[k])
    return Z
