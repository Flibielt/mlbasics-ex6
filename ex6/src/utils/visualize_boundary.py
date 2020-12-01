from matplotlib import pyplot
import numpy as np

from .svm_predict import svm_predict


def visualize_boundary(X, y, model):
    """
    visualize_boundary plots a non-linear decision boundary learned by the SVM
        visualize_boundary(X, y, model) plots a non-linear decision
        boundary learned by the SVM and overlays the data on it
    """

    # make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.stack((X1[:, i], X2[:, i]), axis=1)
        vals[:, i] = svm_predict(model, this_X)

    pyplot.contour(X1, X2, vals, colors='y', linewidths=2)
    pyplot.pcolormesh(X1, X2, vals, cmap='YlGnBu', alpha=0.25, edgecolors='None', lw=0)
    pyplot.grid(False)
