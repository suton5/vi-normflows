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
