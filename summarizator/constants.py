"""Define constant variables."""

# define empty and end-of-sentence vocab idx
EMPTY_VOCAB_IDX = 0
EOS_VOCAB_IDX = 1

# input data (X) is made from maxlend description words followed by eos followed by
# headline words followed by eos if description is shorter than maxlend it will be
# left padded with empty if entire data is longer than maxlen it will be clipped and
# if it is shorter it will be right padded with empty.
# labels (Y) are the headline
# words followed by eos and clipped or padded to maxlenh. In other words the input is
# made from a maxlend half in which the description is padded from the left and a
# maxlenh half in which eos is followed by a headline followed by another eos if there
# is enough space. The labels match only the second half and the first label matches
# the eos at the start of the second half (following the description in the first half)
MAXLEND = 100
MAXLENH = 15
MAXLEN = MAXLEND + MAXLENH
activation_rnn_size = 40 if MAXLEND else 0
nb_unknown_words = 10

# function names
VOCAB_EMBEDDINGS_FILENAME = 'vocabulary-embedding'  # filename of vocab embeddings
WEIGHTS_FILENAME = 'weights_v'  # filename of model weights

# training variables
seed = 42
optimizer = 'adam'
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
regularizer = None
