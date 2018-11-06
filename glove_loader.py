import numpy as np

GLOVE_EMBEDDING_SIZE = 100


def load_glove(data_dir_path=None):
    if data_dir_path is None:
        data_dir_path = '.'

    _word2em = {}
    glove_model_path = data_dir_path + "/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
    file = open(glove_model_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


def glove_zero_emb():
    return np.zeros(shape=GLOVE_EMBEDDING_SIZE)


class Glove(object):

    word2em = None

    GLOVE_EMBEDDING_SIZE = GLOVE_EMBEDDING_SIZE

    def __init__(self, glove_data_path):
        self.word2em = load_glove(data_dir_path=glove_data_path)

