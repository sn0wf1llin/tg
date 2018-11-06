import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
import tensorflow as tf
#tf.keras.utils.to_categorical
# from keras.utils import plot_model
import numpy as np
from tg.config import *


def init_nltk_data_path(nltk_data_path):
	nltk.data.path.append(nltk_data_path)

def get_vocabulary(data):
	# build a list of all description strings
	all_desc = set()
	for article_obj in data:
		h, d = article_obj['HEAD'], article_obj['DESC']

		[all_desc.update(hs.split()) for hs in h]
		[all_desc.update(ds.split()) for ds in d]

	# for UNKnown words
	all_desc.update("<unk>")

	return all_desc


def split_data(X, Y, test_data_percent, seed):
	"""Create train-test split."""
	# load working_dir and create train test split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_data_percent, random_state=seed)

	del X, Y  # free up memory by removing X and Y

	return X_train, X_test, Y_train, Y_test


def fit_data(vocab, glove_embeddings):
	word2idx, idx2word, idx2embedding = {}, {}, {}
	start_index = 0

	for w in vocab:
		try:
			word2idx[w] = start_index
			idx2word[start_index] = w
			idx2embedding[start_index] = glove_embeddings.word2em[w]
			start_index += 1
		except Exception as e:
			pass
			# print("{} for {}".format(e, w))

	return word2idx, idx2word, idx2embedding


def text_as_idx(text_as_sent_list, word2idx, max_length):
	list_of_idx = []

	for ds in text_as_sent_list:
		for dw in ds.split():
			if len(list_of_idx) == max_length:
				return list_of_idx

			list_of_idx.append(word2idx[dw])

	return list_of_idx


def create_tokenizer(descriptions):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(descriptions)
	return tokenizer


def create_sequences(tokenizer, vocab_size, descriptions, headlines):

	sequences_X = tokenizer.texts_to_sequences(descriptions)
	sequences_y = tokenizer.texts_to_sequences(headlines)

	data = pad_sequences(sequences_X, maxlen=src_txt_length)
	labels = pad_sequences(sequences_y, maxlen=sum_txt_length)

	return data, labels
	#
	# data_categorical = list()
	# labels_categotical = list()
	#
	# for d in data:
	# 	data_categorical.append(to_categorical(d, num_classes=vocab_size))
	#
	# for l in labels:
	# 	labels_categotical.append(to_categorical(l, num_classes=vocab_size))
	#
	# return np.array(data_categorical), np.array(labels_categotical)


# def word_for_id(integer, tokenizer):
# 	for word, index in tokenizer.word_index.items():
# 		if index == integer:
# 			return word
# 	return None
#
#
# # generate a description for an image
# def generate_desc(model, tokenizer, photo, max_length):
# 	# seed the generation process
# 	in_text = 'startseq'
# 	# iterate over the whole length of the sequence
# 	for i in range(max_length):
# 		# integer encode input sequence
# 		sequence = tokenizer.texts_to_sequences([in_text])[0]
# 		# pad input
# 		sequence = pad_sequences([sequence], maxlen=max_length)
# 		# predict next word
# 		yhat = model.predict([photo, sequence], verbose=0)
# 		# convert probability to integer
# 		yhat = argmax(yhat)
# 		# map integer to word
# 		word = word_for_id(yhat, tokenizer)
# 		# stop if we cannot map the word
# 		if word is None:
# 			break
# 		# append as input for generating the next word
# 		in_text += ' ' + word
# 		# stop if we predict the end of the sequence
# 		if word == 'endseq':
# 			break
# 	return in_text