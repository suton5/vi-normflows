from tqdm import tqdm

from autograd import numpy as np
from autograd.numpy import random as npr
from autograd import grad
from autograd.misc.optimizers import adam, rmsprop
import matplotlib.pyplot as plt

from normflows import flows
from .distributions import sample_from_pz
from .plotting import plot_samples


rs = npr.RandomState(0)


def gradient_create(F, D, K, N, unpack_params):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        F {callable} -- Energy function (to be minimized)
        D {int} -- dimension of latent variables
        K {int} -- number of flows
        N {int} -- Number of samples to draw
        unpack_params {callable} -- Parameter unpacking function

    Returns
        'variational_objective', 'gradient', 'unpack_params'
    """

    def variational_objective(params, t):
        phi, theta = unpack_params(params)
        z0 = rs.randn(N, D)  # Gaussian noise here. Will add back in mu and sigma in F
        free_energy = F(z0, phi, theta, K, t)
        return free_energy

    gradient = grad(variational_objective)

    return variational_objective, gradient


def optimize(logp, X, D, K, N, init_params, unpack_params, max_iter, step_size, verbose=True):
    """Run the optimization for a mixture of Gaussians

    Arguments:
        logp {callable} -- Joint log-density of Z and X
        X {np.ndarray} -- Observed data
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

    def F(z0, phi, theta, K, t):
        eps = 1e-7
        mu0, log_sigma_diag0, W, U, B = phi
        beta_t = np.min(np.array([1, 0.001 + t / 10000]))

        sd = np.sqrt(np.exp(log_sigma_diag0))
        zk = z0 * sd + mu0

        running_sum = 0.
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            # u_hat = -1 + np.log(1 + np.exp((np.dot(w, u) - np.dot(w, u)) * (w / np.linalg.norm(w)))) + u
            # affine = np.outer(hprime(np.matmul(zk, w) + b), w)
            # running_sum += np.log(eps + np.abs(1 + np.matmul(affine, u)))
            # zk = zk + np.outer(np.tanh(np.matmul(zk, w) + b), u_hat)

            running_sum += np.log(eps + np.abs(1 + np.dot(u, logdet_jac(w, zk.T, b))))
            zk = flows.planar_flow(zk, w, u, b)

        # Unsure if this should be z0 or z1 (after adding back in mean and sd)
        first = np.mean(logq0(z0))
        second = np.mean(logp(X, zk, theta)) * beta_t  # Play with temperature
        third = np.mean(running_sum)

        return first - second - third

    objective, gradient = gradient_create(F, D, K, N, unpack_params)
    pbar = tqdm(total=max_iter)

    param_trace = []

    def callback(params, t, g):
        pbar.update()
        param_trace.append(params)
        if verbose:
            if t % 100 == 0:
                grad_t = gradient(params, t)
                grad_mag = np.linalg.norm(grad_t)
                tqdm.write(f"Iteration {t}; objective: {objective(params, t)} gradient mag: {grad_mag:.3f}")
                # tqdm.write(f"Gradient: {grad_t}")

    variational_params = rmsprop(gradient, init_params, step_size=step_size, callback=callback, num_iters=max_iter)
    pbar.close()

    param_trace = np.vstack(param_trace).T
    return unpack_params(variational_params)
