import numpy as np
import os


def get_vocab_list():
    """
    get_vocab_list reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words
        vocabList = get_vocab_list() reads the fixed vocabulary list in vocab.txt
        and returns a cell array of the words in vocabList.
    """

    vocab_path = os.path.dirname(os.path.realpath(__file__)) + '/vocab.txt'
    vocab_path = vocab_path.replace('\\', '/')
    vocab_path = vocab_path.replace('utils', 'data')
    vocab_list = np.genfromtxt(vocab_path, dtype=object)
    return list(vocab_list[:, 1].astype(str))
