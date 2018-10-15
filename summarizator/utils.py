"""Utility methods."""
import os
from sklearn.model_selection import train_test_split
import _pickle as pickle

import config
from constants import EMPTY_VOCAB_IDX, EOS_VOCAB_IDX, VOCAB_EMBEDDINGS_FILENAME


def join_ingredients(ingredients_listlist):
    """Join multiple lists of ingredients with ' , '."""
    return [' , '.join(i) for i in ingredients_listlist]


def get_flat_ingredients_list(ingredients_joined_train):
    """Flatten lists of ingredients encoded as a string into a single list."""
    return ' , '.join(ingredients_joined_train).split(' , ')


def section_print():
    """Memorized function keeping track of section number."""
    section_number = 0

    def inner(message):
        """Print section number."""
        global section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: initializing section function'.format(section_number))
    return inner


def is_filename_char(x):
    """Return True if x is an acceptable filename character."""
    if x.isalnum():
        return True
    if x in ['-', '_']:
        return True
    return False


def url_to_filename(filename):
    """Map a URL string to filename by removing unacceptable characters."""
    return "".join(x for x in filename if is_filename_char(x))


def prt(label, word_idx, idx2word):
    """Map `word_idx` list to words and print it with its associated `label`."""
    words = [idx2word[word] for word in word_idx]
    print('{}: {}\n'.format(label, ' '.join(words)))


def str_shape(x):
    """Format the dimension of numpy array `x` as a string."""
    return "(" + 'x'.join([str(element) for element in x.shape]) + ")"


def load_embedding(nb_unknown_words):
    """Read word embeddings and vocabulary from disk."""

    with open(os.path.join(config.path_data, '{}.pkl'.format(VOCAB_EMBEDDINGS_FILENAME)), 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)

    vocab_size, embedding_size = embedding.shape

    print('dimension of embedding space for words: {:,}'.format(embedding_size))

    print('vocabulary size: {:,} the last {:,} words can be used as place holders for unknown/oov words'.
          format(vocab_size, nb_unknown_words))

    print('total number of different words: {:,}'.format(len(idx2word)))

    print('number of words outside vocabulary which we can substitute using glove similarity: {:,}'.
          format(len(glove_idx2idx)))

    print('number of words that will be regarded as unknown (unk) /out-of-vocabulary(oov): {:,}'.
          format(len(idx2word) - vocab_size - len(glove_idx2idx)))

    return embedding, idx2word, word2idx, glove_idx2idx


def load_data():
    """Read recipe data from disk."""
    with open(os.path.join(config.path_data, '{}.data.pkl'.format(VOCAB_EMBEDDINGS_FILENAME)), 'rb') as fp:
        X, Y = pickle.load(fp)

    print('number of examples', len(X), len(Y))

    return X, Y


def process_vocab(idx2word, vocab_size, oov0, nb_unknown_words):
    """Update vocabulary to account for unknown words."""
    # reserve vocabulary space for unknown words
    for i in range(nb_unknown_words):
        idx2word[vocab_size - 1 - i] = '<{}>'.format(i)

    # mark words outside vocabulary with ^ at their end
    for i in range(oov0, len(idx2word)):
        idx2word[i] = idx2word[i] + '^'

    # add empty word and end-of-sentence to vocab
    idx2word[EMPTY_VOCAB_IDX] = '_'
    idx2word[EOS_VOCAB_IDX] = '~'

    return idx2word


def load_split_data(nb_val_samples, seed):
    """Create train-test split."""
    # load data and create train test split
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)

    del X, Y  # free up memory by removing X and Y

    return X_train, X_test, Y_train, Y_test
