from .config import figs, rs
from .distributions import make_samples_z
from .transformations import affine
from .plotting import plot_obs_latent


def clear_figs():
    [f.unlink() for f in figs.glob('*')]


def get_samples_from_params(phi, theta, X, K):
    N, D = X.shape
    Zhat = make_samples_z(X, *phi, K, D, N)
    ZK = Zhat[K, :, :]
    mu_z, log_sigma_diag_pz, logit_pi, A, B = theta
    Xhat = affine(ZK, A, B) + rs.randn(N, D)
    return Xhat, ZK
