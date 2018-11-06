from keras import Input, Model
from keras.layers import Embedding, LSTM, Lambda, Dense, Dropout, Concatenate
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
from termcolor import colored
from tg.config import *
import numpy as np
import keras.backend as K


def my_to_categorical(d, num_classes):
	d = K.transpose(d)
	return np.apply_along_axis(to_categorical, 0, d, {'num_classes': num_classes})


def build_model(vocab_size):
	# source text input model
	inputs1 = Input(shape=(src_txt_length, ))
	am1 = Embedding(vocab_size, 256)(inputs1)
	doam1 = Dropout(0.5)(am1)
	am2 = LSTM(256)(doam1)

	# # summary input model
	inputs2 = Input(shape=(sum_txt_length, ))
	sm1 = Embedding(vocab_size, 256)(inputs2)
	dosm1 = Dropout(0.5)(sm1)
	sm2 = LSTM(256)(dosm1)

	# # decoder output model
	decoder1 = Concatenate()([am2, sm2])
	dec2cat_lambda1 = Lambda(my_to_categorical, arguments={'num_classes': vocab_size})(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(dec2cat_lambda1)

	# # tie it together [article, summary] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	print(model.summary())
	plot_model(model, show_shapes=True)

	return model

def inspect_model(model):
	"""Print the structure of Keras `model`."""
	layers_weights_shapes = []
	layers_names = []

	for i, l in enumerate(model.layers):
		layers_names.append('{} cls:{:30} name:{:30}'.format(i, colored(type(l).__name__, 'green'), colored(l.name, 'red')))

		_l_weights = l.get_weights()
		_l_shapes = []
		for i in _l_weights:
			_l_shapes.append(i.shape)
		layers_weights_shapes.append(_l_shapes)

	print("-" * 88)
	print("{:20}".format("Model Description By Layers"))
	print("-" * 88)
	for ln, lw in zip(layers_names, layers_weights_shapes):
		_w_string = ""
		for w in lw:
			_w_string += str(w) + " -- "

		print("{} {}".format(ln, _w_string[:-4]))
	print("-" * 88)