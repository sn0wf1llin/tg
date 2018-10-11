import pickle
from collections import Counter


import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer

default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your chose


def clean_text(text, ):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ') #strip whitespaes
    text = text.lower() #lowercase
    text = stem_text(text) #stemming
    text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_stopwords(text) #remove stopwords
    #text.strip(' ') # strip white spaces again?

    return text


FN = 'vocabulary-embedding'
seed = 42
vocab_size = 40000
embedding_dim = 100
FN0 = 'tokens_sample'  # this is the name of the data file which I assume you already have

fname = 'glove.6B.{}.txt'.format(embedding_dim)

with open('%s.pkl' % FN0, 'rb') as fp:
    heads, descs, keywords = pickle.load(fp)  # keywords are not used in this project

clean_heads, clean_descs = [clean_text(h) for h in heads], [clean_text(d) for d in descs]


def get_vocab(lst):
    vocab_counter = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocab_counter.items(), key=lambda x: -x[1]))

    return vocab, vocab_counter


vocab, vocab_counter = get_vocab(clean_descs + clean_heads)

empty = 0  # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos + 1  # first real word


def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos

    idx2word = dict((idx, word) for word, idx in word2idx.items())

    return word2idx, idx2word


word2idx, idx2word = get_idx(vocab, vocabcount)
