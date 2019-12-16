"""Trying to learn a simple mixture of Gaussians (1D and 2D)
"""
from autograd import numpy as np
import autograd.numpy.random as npr

import matplotlib.pyplot as plt

from normflows import flows, distributions, transformations, optimization, nn_models


rs = npr.RandomState(0)
nn = nn_models.Feedforward(architecture=nn_models.default_architecture, random=rs)


def make_samples_z(X, weights, W, U, b, K, D, N):
    """Sample from latent distribution

    :param X: {np.ndarray} -- Observed variables (N, D)
    :param weights: {np.ndarray} -- Weights of inference network (1, D)
    :param W: {np.ndarray} -- flow parameter (K, D)
    :param U: {np.ndarray} -- flow parameter (K, D)
    :param b: {np.ndarray} -- flow parameter (K,)
    :param K: {int} -- Number of flows
    :param D: {int} -- Dimension of Z
    :param N: {int} -- Number of samples
    :return: {np.ndarray} -- samples
    """
    q0_params = nn.forward(weights, X.T).reshape(N, -1)
    mu, log_sigma_diag = q0_params[:, :D], q0_params[:, D:]
    return distributions.sample_from_pz(mu, log_sigma_diag, W, U, b, K)


def sample_gaussian_mixture(mus, Sigma_diags, probs, n_samples):
    """Sample `n_samples` points from a Gaussian mixture model.

    Arguments:
        mus {np.ndarray} -- means for gaussians
            shape: (n_mixtures, dim)
        Sigma_diags {np.ndarray} -- diagonals for covariance matrices
            shape: (n_mixtures, dim)
        probs {np.ndarray} -- probabilities of being from each Gaussian
            shape: (n_mixtures,)

    Returns:
        samples {np.ndarray} -- samples from the GMM
            shape: (n_samples, dim)
    """
    assert np.abs(np.sum(probs) - 1) < 1e-6, 'Probabilities do not add up to 1'
    dists = np.random.choice(len(probs), size=n_samples, p=probs)
    samps = np.zeros((n_samples, mus.shape[1]))
    for i in range(n_samples):
        dist = dists[i]
        Sigma = np.diag(Sigma_diags[dist])
        mu = mus[dist]
        samps[i] = rs.multivariate_normal(mu, Sigma, size=1)
    return samps


def plot_samples(Z, ax):
    dim = Z.shape[1]
    if dim == 1:
        ax.hist(Z, bins=25, edgecolor='k')
    elif dim == 2:
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.5)
    return ax


def get_gmm_samples(n_samples=10000):
    """Simple GMM with two modes
    """
    mus = np.array([2, 7]).reshape(-1, 1)
    Sigma_diags = np.array([1, 1]).reshape(-1, 1)
    probs = np.array([0.7, 0.3])
    samps = sample_gaussian_mixture(mus, Sigma_diags, probs, n_samples)
    return samps


def f_true(Z):
    """Defining a simple affine transformation to transform Z into X.
    """
    mus = 2 * Z + 3.5
    Sigma = np.array([[1]])
    X = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        X[i, :] = np.random.multivariate_normal(mus[i], Sigma, size=1)
    return X


def f_pred(Z, slope, intercept):
    """Predicting observed variables given the latent variables.
    """
    mus = transformations.affine(Z, slope, intercept)
    Sigma = np.array([[1]])  # Assuming the likelihood has variance 1
    X = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        X[i, :] = rs.multivariate_normal(mus[i], Sigma, size=1)
    return X


def make_unpack_params(D, K, G):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        D {int} -- dimension of latent variables
        K {int} -- number of flows
        G {int} -- Number of groups in the gaussian mixture

    Returns
        'unpack_params'
    """
    def unpack_phi(phi):
        nw = nn.D  # Number of weights
        # mu0 = phi[:D]
        # log_sigma_diag0 = phi[D:2 * D]
        weights, flow_params = phi[:nw].reshape(1, -1), phi[nw:]
        W = flow_params[:K * D].reshape(K, D)
        U = flow_params[K * D:2 * K * D].reshape(K, D)
        b = flow_params[-K:]

        return weights, W, U, b

    def unpack_theta(theta):
        mu_z = theta[:D * G].reshape(G, D)
        log_sigma_diag_pz = theta[D * G: 2 * D * G].reshape(G, D)
        logit_pi = theta[2 * D * G:2 * D * G + G - 1]
        A = theta[-(D ** 2 + 2 * D):-2 * D].reshape(D, D)
        B = theta[-2 * D:-D]
        log_sigma_diag_lklhd = theta[-D:]

        return mu_z, log_sigma_diag_pz, logit_pi, A, B, log_sigma_diag_lklhd
        # return A, B, log_sigma_diag_lklhd

    def unpack_params(params):
        phi = params[:nn.D + 2 * (K * D) + K]
        theta = params[nn.D + 2 * (K * D) + K:]

        phi = unpack_phi(phi)
        theta = unpack_theta(theta)

        return phi, theta

    return unpack_params

def get_init_params(D, K, G):
    # --- Initializing --- #
    # phi
    # init_mu0 = np.ones(D) * 1
    # init_log_sigma0 = np.ones(D) * 0
    init_weights = np.ones(nn.D) * 2
    init_W = np.ones((K, D)) * 1
    init_U = np.ones((K, D)) * 1
    init_b = np.ones(K) * 0

    init_phi = np.concatenate([
            init_weights,
            init_W.flatten(),
            init_U.flatten(),
            init_b
        ])

    # theta
    init_mu_z = np.ones((G, D)) * 4
    init_log_sigma_z = np.ones((G, D)) * 1
    init_logit_pi = transformations.logit(np.array([0.5]))
    init_A = np.eye(D)
    init_B = np.zeros(D)
    init_log_sigma_lklhd = np.zeros(D)  # Assuming diagonal covariance for likelihood

    init_theta = np.concatenate([
            init_mu_z.flatten(),
            init_log_sigma_z.flatten(),
            init_logit_pi,
            init_A.flatten(),
            init_B,
            init_log_sigma_lklhd
        ])

    init_params = np.concatenate((init_phi, init_theta))

    return init_params


def logp(X, Z, theta):
    """Joint likelihood for Gaussian mixture model
    """
    # Maybe reshape these bois
    mu_z, log_sigma_diag_pz, logit_pi, A, B, log_sigma_diag_lklhd = theta
    # A, B, log_sigma_diag_lklhd = theta
    log_prob_z = distributions.log_prob_gm(Z, mu_z, log_sigma_diag_pz, logit_pi)

    mu_x = transformations.affine(Z, A, B)
    log_prob_x = distributions.log_mvn(X, mu_x, log_sigma_diag_lklhd)

    return log_prob_x + log_prob_z


def run_optimization(X, K, D, init_params, unpack_params, max_iter=10000, N=1000, step_size=1e-4):
    return optimization.optimize(logp, X, D, K, N,
                                 init_params, unpack_params, max_iter, step_size,
                                 verbose=True)


def main():
    K = 4
    D = 1
    G = 2
    n_samples = 500

    Z = get_gmm_samples(n_samples=n_samples)
    X = f_true(Z)

    unpack_params = make_unpack_params(D, K, G)
    init_params = get_init_params(D, K, G)
    phi, theta = run_optimization(X, K, D, init_params, unpack_params,
                                  max_iter=2000, N=n_samples, step_size=5e-3)

    print(f"Variational params: {phi}")
    print(f"Generative params: {theta}")

    Zhat = make_samples_z(X, *phi, K, D, n_samples)
    print(Zhat.shape)
    ZK = Zhat[K, :, :]

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True)
    plot_samples(Z, axs[0, 0])
    axs[0, 0].set_title('Latent')
    plot_samples(ZK, axs[1, 0])
    axs[1, 0].set_title('Variational latent')

    mu_z, log_sigma_diag_pz, logit_pi, A, B, log_sigma_diag_lklhd = theta
    # A, B, log_sigma_diag_lklhd = theta
    Xhat = f_pred(ZK, A, B)

    plot_samples(X, axs[0, 1])
    axs[0, 1].set_title('Observed')
    plot_samples(Xhat, axs[1, 1])
    axs[1, 1].set_title("Variational observed")

    plt.show()


if __name__ == '__main__':
    #TODO: Go back to log pi instead of logit pi.
    main()
