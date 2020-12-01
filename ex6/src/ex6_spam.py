from scipy.io import loadmat
import numpy as np
import os

from .utils import process_email, email_features, svm_train, linear_kernel, svm_predict, get_vocab_list


def ex6_spam():
    """
    Exercise 6 | Spam Classification with SVMs

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

        gaussianKernel.m
        dataset3Params.m
        processEmail.m
        emailFeatures.m

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    ==================== Part 1: Email Preprocessing ====================
    To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    to convert each email into a vector of features. In this part, you will
    implement the preprocessing steps for each email. You should
    complete the code in processEmail.m to produce a word indices vector
    for a given email.
    """

    print('\nPreprocessing sample email (emailSample1.txt)\n')

    # Extract Features
    with open(os.path.join('Data', 'emailSample1.txt')) as fid:
        file_contents = fid.read()

    word_indices = process_email(file_contents)

    # Print stats
    print('Word Indicies: %d' % word_indices)

    input('Program paused. Press enter to continue.\n')

    """
    ==================== Part 2: Feature Extraction ====================
    Now, you will convert each email into a vector of features in R^n. 
    You should complete the code in emailFeatures.m to produce a feature
    vector for a given email.
    """
    print('\nExtracting features from sample email (emailSample1.txt)\n')

    features = email_features(word_indices)

    # Print Stats
    print('\nLength of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))

    input('Program paused. Press enter to continue.\n')

    """
    =========== Part 3: Train Linear SVM for Spam Classification ========
    In this section, you will train a linear classifier to determine if an
    email is Spam or Not-Spam.
    """

    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = loadmat(os.path.join('Data', 'spamTrain.mat'))
    X, y = data['X'].astype(float), data['y'][:, 0]

    print('Training Linear SVM (Spam Classification)')
    print('This may take 1 to 2 minutes ...\n')

    C = 0.1
    model = svm_train(X, y, C, linear_kernel)

    # Compute the training accuracy
    p = svm_predict(model, X)

    print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))

    """
    =================== Part 4: Test Spam Classification ================
    After training the classifier, we can evaluate it on a test set. We have
    included a test set in spamTest.mat
    """

    # Load the test dataset
    # You will have Xtest, ytest in your environment
    data = loadmat(os.path.join('Data', 'spamTest.mat'))
    Xtest, ytest = data['Xtest'].astype(float), data['ytest'][:, 0]

    print('Evaluating the trained Linear SVM on a test set ...')
    p = svm_predict(model, Xtest)

    print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))

    input('\nProgram paused. Press enter to continue.\n')

    """
    ================= Part 5: Top Predictors of Spam ====================
    Since the model we are training is a linear SVM, we can inspect the
    weights learned by the model to understand better how it is determining
    whether an email is spam or not. The following code finds the words with
    the highest weights in the classifier. Informally, the classifier
    'thinks' that these words are the most likely indicators of spam.
    """

    # Sort the weights and obtain the vocabulary list
    idx = np.argsort(model['w'])
    top_idx = idx[-15:][::-1]
    vocab_list = get_vocab_list()

    print('Top predictors of spam:')
    print('%-15s %-15s' % ('word', 'weight'))
    print('----' + ' ' * 12 + '------')
    for word, w in zip(np.array(vocab_list)[top_idx], model['w'][top_idx]):
        print('%-15s %0.2f' % (word, w))

    input('\nProgram paused. Press enter to continue.\n')

    """
    =================== Part 6: Try Your Own Emails =====================
    Now that you've trained the spam classifier, you can use it on your own
    emails! In the starter code, we have included spamSample1.txt,
    spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
    The following code reads in one of these emails and then uses your 
    learned SVM classifier to determine whether the email is Spam or 
    Not Spam
    """

    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = os.path.join('Data', 'emailSample1.txt')

    with open(filename) as fid:
        file_contents = fid.read()

    word_indices = process_email(file_contents, verbose=False)
    x = email_features(word_indices)
    p = svm_predict(model, x)

    print('\nProcessed %s\nSpam Classification: %s' % (filename, 'spam' if p else 'not spam'))
    print('(1 indicates spam, 0 indicates not spam)\n\n')
