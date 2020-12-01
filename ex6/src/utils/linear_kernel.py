import numpy as np


def linear_kernel(x1, x2):
    """
    linear_kernel returns a linear kernel between x1 and x2
        sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
        and returns the value in sim
    """

    return np.dot(x1, x2)
