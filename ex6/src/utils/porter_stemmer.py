def porter_stemmer(in_string):
    """
    Applies the Porter Stemming algorithm as presented in the following
    paper:
    Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
        no. 3, pp 130-137

    Original code modeled after the C version provided at:
    http://www.tartarus.org/~martin/PorterStemmer/c.txt

    The main part of the stemming algorithm starts here. b is an array of
    characters, holding the word to be stemmed. The letters are in b[k0],
    b[k0+1] ending at b[k]. In fact k0 = 1 in this demo program (since
    matlab begins indexing by 1 instead of 0). k is readjusted downwards as
    the stemming progresses. Zero termination is not in fact used in the
    algorithm.

    To call this function, use the string to be stemmed as the input
    argument.  This function returns the stemmed word as a string.
    """

    # todo
