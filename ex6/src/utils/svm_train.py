def svm_train(x, y, c, kernel_function, tol, max_passes):
    """
        svm_train Trains an SVM classifier using a simplified version of the SMO
    algorithm.
    [model] = svm_train(X, Y, C, kernelFunction, tol, max_passes) trains an
    SVM classifier and returns trained model. X is the matrix of training
    examples.  Each row is a training example, and the jth column holds the
    jth feature.  Y is a column matrix containing 1 for positive examples
    and 0 for negative examples.  C is the standard SVM regularization
    parameter.  tol is a tolerance value used for determining equality of
    floating point numbers. max_passes controls the number of iterations
    over the dataset (without changes to alpha) before the algorithm quits.

    Note: This is a simplified version of the SMO algorithm for training
        SVMs. In practice, if you want to train an SVM classifier, we
        recommend using an optimized package such as:

            LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
            SVMLight (http://svmlight.joachims.org/)
    """

    # todo
