"""Various distribution PDFs and sampling functions
"""
import autograd.numpy as np


def log_mvn(Z, mu, log_sigma_diag):
    """Multivariate normal distribution

    #TODO: SPEED THIS UP!!!

    :param Z: np.ndarray -- Samples from MVN. Shape (n_samples, dim_z)
    :param mu: np.ndarray -- Shape (n_samples | 1, dim_z)
    :param log_sigma_diag: np.ndarray -- Shape (dim_z,)
    :return: log_prob (scalar)
    """
    N, D = Z.shape
    mu = mu.reshape(-1, D)
    logdet_sigma = np.sum(log_sigma_diag)
    siginv = np.linalg.inv(np.exp(log_sigma_diag).reshape(D, D))
    const = -D / 2 * np.log(2 * np.pi) - 0.5 * logdet_sigma
    zdiff = Z - mu

    logprob = 0

    # Vectorize this boi. This is why it takes an hour you dingus
    for n in range(Z.shape[0]):
        logprob += const - 0.5 * np.dot(np.dot((zdiff[n]), siginv), (zdiff[n]))
    return logprob


def log_prob_gm(Z, mu, log_sigma_diag, pi):
    assert np.sum(pi) < 1, f'probabilities of GMM do not sum to less than 1 ({np.sum(pi)})'

    G =mu.shape[0]

    log_prob = 0
    for g in range(G):
        # Multivariate gaussian
        if g == G - 1:
            pi_g = 1 - np.sum(pi)
        else:
            pi_g = pi[g]
        log_prob += log_mvn(Z, mu[g], log_sigma_diag[g]) * pi_g
    return log_prob
