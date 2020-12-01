import numpy as np
import os


def get_vocab_list():
    """
    get_vocab_list reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words
        vocabList = get_vocab_list() reads the fixed vocabulary list in vocab.txt
        and returns a cell array of the words in vocabList.
    """

    vocab_list = np.genfromtxt(os.join('Data', 'vocab.txt'), dtype=object)
    return list(vocab_list[:, 1].astype(str))
