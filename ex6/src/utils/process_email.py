import re

from .porter_stemmer import PorterStemmer
from .get_vocab_list import get_vocab_list


def process_email(email_contents, verbose=True):
    """
    process_email preprocesses a the body of an email and
    returns a list of word_indices
        word_indices = PROCESSEMAIL(email_contents) preprocesses
        the body of an email and returns a list of indices of the
        words contained in the email.
    """

    # Load Vocabulary
    vocab_list = get_vocab_list()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)

    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)

    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]

    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #       word_indices if it is in the vocabulary. At this point
        #       of the code, you have a stemmed word from the email in
        #       the variable str. You should look up str in the
        #       vocabulary list (vocabList). If a match exists, you
        #       should add the index of the word to the word_indices
        #       vector. Concretely, if str = 'action', then you should
        #       look up the vocabulary list to find where in vocabList
        #       'action' appears. For example, if vocabList{18} =
        #       'action', then, you should add 18 to the word_indices
        #       vector (e.g., word_indices = [word_indices ; 18]; ).
        #
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.

        try:
            word_indices.append(vocab_list.index(word))
        except ValueError:
            pass

        # =============================================================

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices
