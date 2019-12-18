import autograd.numpy.random as npr
import matplotlib.pyplot as plt

from .config import figs, rs, figname
from .distributions import make_samples_z, sample_from_pz
from .transformations import affine
from .plotting import plot_obs_latent, plot_mnist


def clear_figs():
    [f.unlink() for f in figs.glob('*')]


def get_samples_from_params(phi, theta, X, K):
    N, D = X.shape
    Zhat = make_samples_z(*phi, K)
    ZK = Zhat[K, :, :]
    mu_z, log_sigma_diag_pz, logit_pi, A, B = theta
    Xhat = affine(ZK, A, B) + rs.randn(N, D)
    return Xhat, ZK


def compare_reconstruction(phi, theta, x_true, encode, decode, t):
    mu0, log_sigma_diag0, W, U, b = encode(phi, x_true)
    K = W.shape[0]
    z = sample_from_pz(mu0, log_sigma_diag0, W, U, b, K)
    logits = decode(theta, z)
    xhat = npr.binomial(1, logits)

    x_true_im = x_true.reshape(28, 28)
    xhat_im = xhat.reshape(28, 28)

    plot_mnist(x_true_im, xhat_im)
    plt.savefig(figname.format(t))
    plt.close()
