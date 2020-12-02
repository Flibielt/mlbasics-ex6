from matplotlib import pyplot
from scipy.io import loadmat
import numpy as np
import os

from .utils import plot_data, gaussian_kernel, svm_train, linear_kernel, visualize_boundary_linear, \
    visualize_boundary, dataset3_params


def ex6():
    """
    Exercise 6 | Support Vector Machines

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

        gaussian_kernel.py
        dataset3_params.py
        process_email.py
        email_features.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    =============== Part 1: Loading and Visualizing Data ================
    We start the exercise by first loading and visualizing the dataset. 
    The following code will load the dataset into your environment and plot
    the data.
    """

    print('Loading and Visualizing Data ...\n')

    # Load from ex6data1
    # You will have X, y as keys in the dict data
    data1_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex6data1.mat'
    data1_path = data1_path.replace('\\', '/')
    data = loadmat(data1_path)
    X, y = data['X'], data['y'][:, 0]

    # Plot training data
    plot_data(X, y)

    input('Program paused. Press enter to continue.\n')

    """
    ==================== Part 2: Training Linear SVM ====================
    The following code will train a linear SVM on the dataset and plot the
    decision boundary learned.
    """

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1

    model = svm_train(X, y, C, linear_kernel, 1e-3, 20)
    visualize_boundary_linear(X, y, model)
    pyplot.show(block=False)

    input('Program paused. Press enter to continue.\n')

    """
    =============== Part 3: Implementing Gaussian Kernel ===============
    You will now implement the Gaussian kernel to use
    with the SVM. You should complete the code in gaussian_kernel.py
    """

    print('\nEvaluating the Gaussian Kernel ...\n')

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2

    sim = gaussian_kernel(x1, x2, sigma)

    print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
          '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

    """
    =============== Part 4: Visualizing Dataset 2 ================
    The following code will load the next dataset into your environment and 
    plot the data. 
    """
    print('Loading and Visualizing Data ...\n')

    # Load from ex6data2
    # You will have X, y as keys in the dict data
    data2_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex6data2.mat'
    data2_path = data2_path.replace('\\', '/')
    data = loadmat(data2_path)
    X, y = data['X'], data['y'][:, 0]

    # Plot training data
    plot_data(X, y)

    input('Program paused. Press enter to continue.\n')

    """
    ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    After you have implemented the kernel, we can now use it to train the 
    SVM classifier.
    """

    print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

    # SVM Parameters
    C = 1
    sigma = 0.1

    # We set the tolerance and max_passes lower here so that the code will run
    # faster. However, in practice, you will want to run the training to
    # convergence.
    model = svm_train(X, y, C, gaussian_kernel, args=(sigma,))
    visualize_boundary(X, y, model)
    pyplot.show(block=False)

    """
    =============== Part 6: Visualizing Dataset 3 ================
    The following code will load the next dataset into your environment and 
    plot the data. 
    """
    print('Loading and Visualizing Data ...\n')

    # Load from ex6data3
    # You will have X, y, Xval, yval as keys in the dict data
    data3_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex6data3.mat'
    data3_path = data3_path.replace('\\', '/')
    data = loadmat(data3_path)
    X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

    # Plot training data
    plot_data(X, y)

    input('Program paused. Press enter to continue.\n')

    """
    ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
    This is a different dataset that you can use to experiment with. Try
    different values of C and sigma here.
    """

    # Try different SVM Parameters here
    C, sigma = dataset3_params(X, y, Xval, yval)

    # Train the SVM
    # model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
    model = svm_train(X, y, C, gaussian_kernel, args=(sigma,))
    visualize_boundary(X, y, model)
    pyplot.show(block=False)
    print(C, sigma)
