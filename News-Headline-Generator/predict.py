from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, SpatialDropout1D, Dropout, RepeatVector  # , Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import pickle
import numpy as np
import random
import keras.backend as K
import h5py
import sys
import Levenshtein


FN = 'predict'
# os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'
FN0 = 'vocabulary-embedding'
FN1 = 'train'

maxlend = 50  # 0 - if we dont want to use description at all
maxlenh = 25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3  # match FN1
batch_norm = False
activation_rnn_size = 40 if maxlend else 0
# training parameters
seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
batch_size = 64
nb_train_samples = 30000
nb_val_samples = 3000

empty = 0
eos = 1






with open('%s.pkl' % FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp, encoding='latin1')

vocab_size, embedding_size = embedding.shape
nb_unknown_words = 10

print('dimension of embedding space for words', embedding_size)
print('vocabulary size', vocab_size,
      'the last %d words can be used as place holders for unknown/oov words' % nb_unknown_words)
print('total number of different words', len(idx2word), len(word2idx))
print('number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx))
print('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',
      len(idx2word) - vocab_size - len(glove_idx2idx))

for i in range(nb_unknown_words):
    idx2word[vocab_size - 1 - i] = '<%d>' % i

for i in range(vocab_size - nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i] + '^'


idx2word[empty] = '_'
idx2word[eos] = '~'


def prt(label, x):
    print(label + ':')
    s = ""
    for w in x:
        s += idx2word[w] + " "
    print(s)

# seed weight initialization
random.seed(seed)
np.random.seed(seed)
regularizer = l2(weight_decay) if weight_decay else None
rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
                        name='embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size,
                return_sequences=True,  # batch_norm=batch_norm,
                name='lstm_%d' % (i + 1),
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                bias_regularizer=regularizer,
                dropout=p_W,
                recurrent_dropout=p_U,)
    rnn_model.add(lstm)
    rnn_model.add(SpatialDropout1D(p_dense, name='dropout_%d' % (i + 1)))


def str_shape(x):
    return 'x'.join(map(str, x.shape))


def inspect_model(model):
    print(model.name)
    for i, l in enumerate(model.layers):
        print(i, 'cls=%s name=%s' % (type(l).__name__, l.name))
        weights = l.get_weights()
        wshapes = ""

        for weight in weights:
            wshapes += str_shape(weight) + " "
        print(wshapes)


def load_weights(model, filepath):
    """Modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print('Loading', filepath, 'to', model.name)
    flattened_layers = model.layers

    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except Exception as e:
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)

    return weight_values


weights = load_weights(rnn_model, '%s.hdf5' % FN1)
# [w.shape for w in weights]
context_weight = K.variable(1.)
head_weight = K.variable(1.)
cross_weight = K.variable(0.)


def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:, :maxlend], X[:, maxlend:]
    head_activations, head_words = head[:, :, :n], head[:, :, n:]
    desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2], [2]))
    # make sure we dont use description words that are masked out
    assert mask.ndim == 2
    activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, -1e20)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies, (-1, maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=([2], [1]))
    return K.concatenate((context_weight * desc_avg_word, head_weight * head_words))


class SimpleContext(Lambda):
    def __init__(self, **kwargs):
        super(SimpleContext, self).__init__(simple_context, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2 * (rnn_size - activation_rnn_size)
        return nb_samples, maxlenh, n


model = Sequential()
model.add(rnn_model)

if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))

# we are not going to fit so we dont care about loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

n = 2 * (rnn_size - activation_rnn_size)


# out very own softmax
def output2probs(output):
    output = np.dot(output, weights[0]) + weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output


def output2probs1(output):
    output0 = np.dot(output[:n // 2], weights[0][:n // 2, :])
    output1 = np.dot(output[n // 2:], weights[0][n // 2:, :])
    output = output0 + output1  # + output0 * output1
    output += weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output


def lpadd(x, maxlend=maxlend, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty] * (maxlend - n) + x + [eos]


samples = [lpadd([3] * 26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')

np.all(data[:, maxlend] == eos)

data.shape, map(len, samples)

probs = model.predict(data, verbose=0, batch_size=1)
print(probs.shape)


# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
def beamsearch(predict, start=[empty] * maxlend + [eos], avoid=None, avoid_score=1,
               k=1, maxsample=maxlen, use_unk=True, oov=vocab_size - 1, empty=empty, eos=eos, temperature=1.0):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """

    def sample(energy, n, temperature=temperature):
        """sample at most n different elements according to their energy"""
        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in range(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.  # make sure we select each element only once
        return res

    dead_samples = []
    dead_scores = []
    live_samples = [list(start)]
    live_scores = [0]

    while live_samples:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples, empty=empty)
        assert vocab_size == probs.shape[1]

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_scores[:, empty] = 1e20
        if not use_unk and oov is not None:
            cand_scores[:, oov] = 1e20
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i, a[n]] += avoid_score
        live_scores = list(cand_scores.flatten())

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        dead_scores = [dead_scores[r] for r in ranks if r < n]
        dead_samples = [dead_samples[r] for r in ranks if r < n]

        live_scores = [live_scores[r - n] for r in ranks if r >= n]
        live_samples = [live_samples[(r - n) // vocab_size] + [(r - n) % vocab_size] for r in ranks if r >= n]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        def is_zombie(s):
            return s[-1] == eos or len(s) > maxsample

        # add zombies to the dead
        dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
        dead_samples += [s for s in live_samples if is_zombie(s)]

        # remove zombies from the living 
        live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
        live_samples = [s for s in live_samples if not is_zombie(s)]

    return dead_samples, dead_scores


def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    """for every sample, calculate probability for every possible label
    you need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = map(len, samples)
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array(
        [output2probs(prob[sample_length - maxlend - 1]) for prob, sample_length in zip(probs, sample_lengths)])


def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < vocab_size - nb_unknown_words else glove_idx2idx.get(x, x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= vocab_size - nb_unknown_words])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
    xs = [outside.get(x, x) for x in xs]
    return xs


def vocab_unfold(desc, xs):
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= vocab_size - nb_unknown_words:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x, x) for x in xs]


def gensamples(X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size,
               short=True, temperature=1., use_unk=True):
    if X is None or isinstance(X, int):
        if X is None:
            i = random.randint(0, len(X_test) - 1)
        else:
            i = X
        print('HEAD %d:' % i, ' '.join(idx2word[w] for w in Y_test[i]))
        print('DESC:', ' '.join(idx2word[w] for w in X_test[i]))
        sys.stdout.flush()
        x = X_test[i]
    else:
        x = [word2idx[w.rstrip('^')] for w in X.split()]

    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid, str) or isinstance(avoid[0], int):
            avoid = [avoid]
        avoid = [a.split() if isinstance(a, str) else a for a in avoid]
        avoid = [vocab_fold([w if isinstance(w, int) else word2idx[w] for w in a])
                 for a in avoid]

    print('HEADS:')
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend, len(x)), max(maxlend, len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
                                   k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s, start, scr) for s, scr in zip(sample, score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
            if distance > -0.6:
                print(score, ' '.join(words))
        #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
        else:
            print(score, ' '.join(words))
        codes.append(code)
    return samples


seed = 8
random.seed(seed)
np.random.seed(seed)

X = "* Billy Joel is looking for a buyer in Sagaponack^ . Now that he and wife Katie Lee Joel are splitting up , the singer is planning to sell the two oceanfront^ properties he bought for her in 2007 . The four-bedroom mansion ( No . 1 ) and smaller beach bungalow^ ( No . 2 ) will be listed with Corcoran 's Biana^ Stepanian^ for a combined $ 35 million . * Richard Bressler^ , the former CFO of Viacom and now a managing"
Y = "Billy Joel Lists in Sagaponack^"

samples = gensamples(X=X, skips=2, batch_size=batch_size, k=10, temperature=1.)

X = "18 Cake GIFs That 'll Make You Moist"
Y = "Is it 350degF^ in here or is it just me ?"

samples = gensamples(X, skips=2, batch_size=batch_size, k=10, temperature=1.)

X = "President Barack Obama 's re-election campaign is fundraising off of comments on Obama 's birth certificate by Mitt Romney 's son Matt ."

gensamples(X, skips=2, batch_size=batch_size, k=10, temperature=1, use_unk=True, short=False)

X = "What have you been listening to this year ? If you want to find out using cold , hard evidence , then Spotify 's new Year in Music tool will tell you ."
Y = "Spotify Will Make You Smarter for Your App"

samples = gensamples(X, skips=2, batch_size=batch_size, k=10, temperature=1)

print(samples)

headline = samples[0][0][len(samples[0][1]):]

' '.join(idx2word[w] for w in headline)

avoid = headline

samples = gensamples(X, avoid=avoid, avoid_score=.1, skips=2, batch_size=batch_size, k=10, temperature=1.)

print(samples)

avoid = samples[0][0][len(samples[0][1]):]

samples = gensamples(X, avoid=avoid, avoid_score=.1, skips=2, batch_size=batch_size, k=10, temperature=1.)

print(samples)

print(len(samples))


def wsimple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:, :maxlend], X[:, maxlend:]
    head_activations, head_words = head[:, :, :n], head[:, :, n:]
    desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2], [2]))
    # make sure we dont use description words that are masked out
    assert mask.ndim == 2
    activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, -1e20)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies, (-1, maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

    return activation_weights


class WSimpleContext(Lambda):
    def __init__(self):
        super(WSimpleContext, self).__init__(wsimple_context)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2 * (rnn_size - activation_rnn_size)
        return nb_samples, maxlenh, n


wmodel = Sequential()
wmodel.add(rnn_model)

wmodel.add(WSimpleContext())

wmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)

seed = 8
random.seed(seed)
np.random.seed(seed)

context_weight.set_value(np.float32(1.))
head_weight.set_value(np.float32(1.))

X = "Representatives of the groups depicted in The Revenant^ spoke with BuzzFeed News about the actor 's Golden Globes speech calling on listeners to `` protect ... indigenous lands . ''"
Y = "Native American Groups Officially Respond To Leonardo DiCaprio 's Call To Action"

samples = gensamples(X, skips=2, batch_size=batch_size, k=10, temperature=1.)

print(samples)

sample = samples[0][0]

print(' '.join([idx2word[w] for w in sample]))

print()
data = sequence.pad_sequences([sample], maxlen=maxlen, value=empty, padding='post', truncating='post')
print(data.shape)

weights = wmodel.predict(data, verbose=0, batch_size=1)
print(weights.shape)

startd = np.where(data[0, :] != empty)[0][0]
lenh = np.where(data[0, maxlend + 1:] == eos)[0][0]
print(startd, lenh)
