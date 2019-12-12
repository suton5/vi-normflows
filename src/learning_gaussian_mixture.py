"""Trying to learn a simple mixture of Gaussians (1D and 2D)
"""
from autograd import grad
from autograd.misc.optimizers import adam, rmsprop, sgd
from autograd import numpy as np
import autograd.numpy.random as npr

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt

from normflows import flows, distributions


rs = npr.RandomState(0)


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
    mus = np.array([0, 4]).reshape(-1, 1)
    Sigma_diags = np.array([1, 1]).reshape(-1, 1)
    probs = np.array([0.3, 0.7])
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
    mus = slope * Z + intercept
    Sigma = np.array([[1]])  # Assuming the likelihood has variance 1
    X = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        X[i, :] = rs.multivariate_normal(mus[i], Sigma, size=1)
    return X


def sample_from_pz(mu, log_sigma_diag, W, U, b, K, num_samples):
    Sigma = np.diag(np.exp(log_sigma_diag))
    dim_z = len(mu)

    Z = np.zeros((K + 1, num_samples, dim_z))
    z = rs.multivariate_normal(mu, Sigma, num_samples)

    Z[0] = z
    for k in range(K):
        Z[k + 1] = flows.planar_flow(z, W[k], U[k], b[k])
    return Z


def gradient_create(F, D, X, K, G, N):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        F {callable} -- Energy function (to be minimized)
        D {int} -- dimension of latent variables
        X {np.ndarray} -- Observed data
        K {int} -- number of flows
        G {int} -- Number of groups in the gaussian mixture
        N {int} -- Number of samples to draw

    Returns
        'variational_objective', 'gradient', 'unpack_params'
    """
    def unpack_params(params):
        phi = params[:2 * D * (1 + K) + K]
        theta = params[2 * D * (1 + K) + K:]

        phi = unpack_phi(phi)
        theta = unpack_theta(theta)

        return phi, theta

    def unpack_phi(phi):
        mu0 = phi[:D]
        log_sigma_diag0 = phi[D:2 * D]
        W = phi[2 * D:2 * D + K * D].reshape(K, D)
        U = phi[2 * D + K * D:2 * (D + K * D)].reshape(K, D)
        b = phi[-K:]

        return mu0, log_sigma_diag0, W, U, b

    def unpack_theta(theta):
        mu_z = theta[:D * G].reshape(G, D)
        log_sigma_diag_pz = theta[D * G: 2 * D * G].reshape(G, D)
        pi = theta[2 * D * G:2 * D * G + G - 1]
        A = theta[-(D ** 2 + 2 * D):-2 * D].reshape(D, D)
        B = theta[-2 * D:-D]
        log_sigma_diag_lklhd = theta[-D:]

        return mu_z, log_sigma_diag_pz, pi, A, B, log_sigma_diag_lklhd

    def variational_objective(params, t):
        phi, theta = unpack_params(params)
        mu0, log_sigma_diag0, W, U, b = phi

        z0 = rs.randn(N, D) * np.sqrt(np.exp(log_sigma_diag0)) + mu0
        free_energy = F(z0, X, phi, theta, K)
        return -free_energy

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def optimize(logp, X, D, G, K, N,
             max_iter, step_size,
             verbose=True):
    """Run the optimization for a mixture of Gaussians

    Arguments:
        logp {callable} -- Joint log-density of Z and X
        D {int} -- Dimension of Z
        G {int} -- Number of Gaussians in GMM
        N {int} -- Number of samples to draw
        K {int} -- Number of flows
        max_iter {int} -- Maximum iterations of optimization
        step_size {float} -- Learning rate for optimizer
    """
    def logq0(z):
        """Just a standard Gaussian
        """
        return -D / 2 * np.log(2 * np.pi) - 0.5 * np.sum(z ** 2, axis=0)

    def hprime(x):
        return 1 - np.tanh(x) ** 2

    def logdet_jac(w, z, b):
        return np.outer(w.T, hprime(np.matmul(w, z) + b))

    def F(z0, X, phi, theta, K):
        eps = 1e-7
        mu0, log_sigma_diag0, W, U, b = phi

        zk = z0
        running_sum = 0
        for k in range(K):
            # Unsure if abs value is necessary
            running_sum += np.log(eps + np.abs(1 + np.dot(U[k], logdet_jac(W[k], zk.T, b[k]))))
            zk = flows.planar_flow(zk, W[k], U[k], b[k])

        first = np.mean(logq0(z0))
        second = np.mean(logp(X, zk, theta))
        third = np.mean(running_sum)

        return first - second - third

    objective, gradient, unpack_params = gradient_create(F, D, X, K,
                                                         G, N)
    pbar = tqdm(total=max_iter)

    def callback(params, t, g):
        pbar.update()
        if verbose:
            if t % 100 == 0:
                grad_mag = np.linalg.norm(gradient(params, t))
                tqdm.write(f"Iteration {t}; gradient mag: {grad_mag:.3f}")
                phi, theta = unpack_params(params)
                print(f"Phi: {phi}")
                print(f"Theta: {theta}")

    # --- Initializing --- #
    # phi
    init_mu0 = np.zeros(D)
    init_log_sigma0 = np.zeros(D)
    init_W = np.ones((K, D))
    init_U = np.ones((K, D))
    init_b = np.zeros(K)

    init_phi = np.concatenate([
            init_mu0,
            init_log_sigma0,
            init_W.flatten(),
            init_U.flatten(),
            init_b
        ])

    # theta
    init_mu_z = np.zeros((D, G))
    init_log_sigma_z = np.zeros((D, G))
    init_pi = np.ones(G - 1) * 0.6
    init_A = np.eye(D)
    init_B = np.zeros(D)
    init_log_sigma_lklhd = np.zeros(D)  # Assuming diagonal covariance for likelihood

    init_theta = np.concatenate([
            init_mu_z.flatten(),
            init_log_sigma_z.flatten(),
            init_pi,
            init_A.flatten(),
            init_B,
            init_log_sigma_lklhd
        ])

    init_params = np.concatenate((init_phi, init_theta))

    variational_params = adam(gradient, init_params, step_size=step_size, callback=callback, num_iters=max_iter)

    pbar.close()

    return unpack_params(variational_params)


def logp(X, Z, theta):
    """Joint likelihood for Gaussian mixture model
    """
    # Maybe reshape these bois
    mu_z, log_sigma_diag_pz, pi, A, B, log_sigma_diag_lklhd = theta
    log_prob_z = distributions.log_prob_gm(Z, mu_z, log_sigma_diag_pz, pi)

    mu_x = np.matmul(A, Z.T).T + B
    log_prob_x = distributions.log_mvn(X, mu_x, log_sigma_diag_lklhd)

    return log_prob_x + log_prob_z

def run_optimization(X, K, D, G, max_iter=5000, N=1000, step_size=1e-4):
    return optimize(logp, X, D, G, K, N,
                    max_iter, step_size,
                    verbose=True)


def main():
    K = 3
    D = 1
    G = 2
    n_samples = 500

    Z = get_gmm_samples(n_samples=n_samples)
    X = f_true(Z)

    phi, theta = run_optimization(X, K, D, G, max_iter=5000, N=n_samples)

    print(f"Variational params: {phi}")

    Zhat = sample_from_pz(*phi, K, n_samples)

    fig, axs = plt.subplots(ncols=2)
    plot_samples(Z, axs[0])
    axs[0].set_title('Latent')
    plot_samples(Zhat[-1, :], axs[1])
    axs[1].set_title('Variational latent')

    # plot_samples(Zhat)

    plt.show()


if __name__ == '__main__':
    main()
