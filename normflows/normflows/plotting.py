import matplotlib.pyplot as plt


def plot_samples(Z, ax=None):
    if ax is None:
        plotter = plt
    else:
        plotter = ax
    dim = Z.shape[1]
    if dim == 1:
        return plotter.hist(Z, bins=25, edgecolor='k')
    elif dim == 2:
        return plotter.scatter(Z[:, 0], Z[:, 1], alpha=0.5)


def plot_obs_latent(X, Z, Xhat, Zhat):
    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True)
    plot_samples(Z, axs[0, 0])
    axs[0, 0].set_title('Latent')
    plot_samples(Zhat, axs[1, 0])
    axs[1, 0].set_title('Variational latent')

    plot_samples(X, axs[0, 1])
    axs[0, 1].set_title('Observed')
    plot_samples(Xhat, axs[1, 1])
    axs[1, 1].set_title("Variational observed")
