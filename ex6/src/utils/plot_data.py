import matplotlib.pyplot as pyplot


def plot_data(X, y, grid=False):
    """
    plot_data Plots the data points X and y into a new figure
        plot_data(x,y) plots the data points with + for the positive examples
        and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    """

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    pyplot.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')
    pyplot.grid(grid)
