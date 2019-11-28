from autograd import numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam, sgd, rmsprop
from autograd import scipy as sp
from scipy.special import logsumexp
import numpy
import scipy
import matplotlib.pyplot as plt
import sys

def trial1(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2)
    return np.exp(-u)

def p1(z):
    z1, z2 = z[:, 0], z[:, 1]
    first = (np.linalg.norm(z, 2, 1) - 2)/0.4
    exp1 = np.exp(-0.5*((z1 - 2)/0.6)**2)
    exp2 = np.exp(-0.5*((z1 + 2)/0.6)**2)
    u = 0.5*first**2 - np.log(exp1 + exp2)
    return np.exp(-u)

def p2(z):
    z1, z2 = z[:, 0], z[:, 1]
    w1 = lambda x: np.sin(2 * np.pi * x/4)
    u = 0.5 * ((z2 - w1(z1))/0.4) ** 2
    dummy = np.ones(u.shape) * 1e7
    u = np.where(np.abs(z1) <= 4, u, dummy)
    return np.exp(-u)

m = lambda x: -1 + np.log(1 + np.exp(x))
h = lambda x: np.tanh(x)
h_prime = lambda x: 1 - np.tanh(x)**2


def gradient_create(target, eps, dim_z, num_samples, K):

    def unpack_params(params):
        W = params[:K*dim_z].reshape(K,dim_z)
        U = params[K*dim_z:2*K*dim_z].reshape(K,dim_z)
        B = params[-K:]
        return W,U,B

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        W,U,B = unpack_params(params)
        z0 = np.random.multivariate_normal(np.zeros(dim_z), np.eye(dim_z),
                num_samples)
        z_prev = z0
        sum_log_det_jacob = 0.
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
            affine = np.outer(h_prime(np.matmul(z_prev, w) + b), w)
            sum_log_det_jacob += np.log(eps + np.abs(1 + np.matmul(affine, u)))
            z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
        z_K = z_prev
        #log_q_K = sp.stats.multivariate_normal.pdf(z0, np.zeros(2), np.eye(2))
        #- sum_log_det_jacob
        log_q_K = -0.5 * np.sum(np.log(2*np.pi) + z0**2, 1) - sum_log_det_jacob
        log_p = np.log(eps + target(z_K))
        return np.mean(log_q_K - log_p)

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params

K = 16
dim_z = 2
num_samples = 10000

objective, gradient, unpack_params = gradient_create(p1, 1e-7, dim_z, num_samples, K)

def callback(params, t, g):
    if t%100 == 0:
        print("Iteration {}; Gradient mag: {}; Objective: {}".format(t,
            np.linalg.norm(gradient(params, t)), objective(params, t)))


init_W = 1*np.ones((K, dim_z))
init_U = 1*np.ones((K, dim_z))
init_b = 1*np.ones((K))
init_params = np.concatenate((init_W.flatten(), init_U.flatten(), init_b.flatten()))

variational_params = adam(gradient, init_params, callback, 2000, 1e-2)
W, U, B = unpack_params(variational_params)
z0 = np.random.randn(num_samples, dim_z)
z_prev = z0
for k in range(K):
    plt.figure(figsize=(10,8))
    plt.scatter(z_prev[:,0], z_prev[:,1])
    plt.show()
    w, u, b = W[k], U[k], B[k]
    u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
    z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
z_K = z_prev

plt.figure(figsize=(10,8))
plt.scatter(z_K[:,0], z_K[:,1])
plt.show()
