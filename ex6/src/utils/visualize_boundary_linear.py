from matplotlib import pyplot
import numpy as np

from .plot_data import plot_data


def visualize_boundary_linear(X, y, model):
    """
    visualize_boundary_linear plots a linear decision boundary learned by the SVM
        visualize_boundary_linear(X, y, model) plots a linear decision boundary
        learned by the SVM and overlays the data on it
    """

    w, b = model['w'], model['b']
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0] * xp + b) / w[1]

    plot_data(X, y)
    pyplot.plot(xp, yp, '-b')
