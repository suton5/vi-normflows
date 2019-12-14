from tqdm import tqdm

from autograd import numpy as np
from autograd.numpy import random as npr
from autograd import grad
from autograd.misc.optimizers import adam

from normflows import transformations, flows


rs = npr.RandomState(0)


def gradient_create(F, D, X, K, N, unpack_params):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        F {callable} -- Energy function (to be minimized)
        D {int} -- dimension of latent variables
        X {np.ndarray} -- Observed data
        K {int} -- number of flows
        N {int} -- Number of samples to draw
        unpack_params {callable} -- Parameter unpacking function

    Returns
        'variational_objective', 'gradient', 'unpack_params'
    """

    def variational_objective(params, t):
        phi, theta = unpack_params(params)
        z0 = rs.randn(N, D)  # Gaussian noise here. Will add back in mu and sigma in F
        free_energy = F(z0, X, phi, theta, K)
        return -free_energy

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def optimize(logp, X, D, G, K, N, init_params, unpack_params, max_iter, step_size, verbose=True):
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

        zk = z0 * np.sqrt(np.exp(log_sigma_diag0)) + mu0

        running_sum = 0
        for k in range(K):
            # Unsure if abs value is necessary
            running_sum += np.log(eps + np.abs(1 + np.dot(U[k], logdet_jac(W[k], zk.T, b[k]))))
            zk = flows.planar_flow(zk, W[k], U[k], b[k])

        # Unsure if this should be z0 or z1 (after adding back in mean and sd)
        first = np.mean(logq0(z0))
        second = np.mean(logp(X, zk, theta))  # Playing with temperature
        third = np.mean(running_sum)

        return first - second - third

    objective, gradient, unpack_params = gradient_create(F, D, X, K, N, unpack_params)
    pbar = tqdm(total=max_iter)

    param_trace = []

    def callback(params, t, g):
        pbar.update()
        param_trace.append(params)
        if verbose:
            if t % 100 == 0:
                grad_t = gradient(params, t)
                # grad_mag = np.linalg.norm(grad_t)
                # tqdm.write(f"Iteration {t}; gradient mag: {grad_mag:.3f}")
                tqdm.write(f"Gradient: {grad_t}")

    variational_params = adam(gradient, init_params, step_size=step_size, callback=callback, num_iters=max_iter)
    pbar.close()

    param_trace = np.vstack(param_trace).T
    return unpack_params(variational_params)
