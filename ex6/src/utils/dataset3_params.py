import numpy as np

from .svm_train import svm_train
from .svm_predict import svm_predict
from .gaussian_kernel import gaussian_kernel


def dataset3_params(X, y, Xval, yval):
    """
    dataset3_params returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
       [C, sigma] = dataset3_params(X, y, Xval, yval) returns your choice of C and
       sigma. You should complete this function to return the optimal C and
       sigma based on a cross-validation set.
    """

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    """
    ====================== YOUR CODE HERE ======================
    Instructions: Fill in this function to return the optimal C and sigma
        learning parameters found using the cross validation set.
        You can use svmPredict to predict the labels on the cross
        validation set. For example, 
          predictions = svmPredict(model, Xval);
        will return the predictions on the cross validation set.
    
    Note: You can compute the prediction error using 
        mean(double(predictions ~= yval))
    """

    C_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    err_array = np.zeros([C_array.size, sigma_array.size])

    for i in np.arange(C_array.size):
        for j in np.arange(sigma_array.size):
            model = svm_train(X, y, C_array[i], gaussian_kernel, args=(sigma_array[j],))
            predictions = svm_predict(model, Xval)
            pred_error = np.mean(predictions != yval)

            err_array[i, j] = pred_error

    ind = np.unravel_index(np.argmin(err_array, axis=None), err_array.shape)
    C = C_array[ind[0]]
    sigma = sigma_array[ind[1]]

    # ============================================================
    return C, sigma

