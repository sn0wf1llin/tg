from tg.model import build_model, inspect_model
from tg.data_loader import load_dataset, save_obj, load_obj, get_vocabulary
from psettings import DEFAULT_PROJECT_PATH
from tg.glove_loader import Glove
from tg.utils import create_tokenizer, create_sequences, split_data, fit_data
import numpy as np
from tg.config import *

# data = load_dataset(DEFAULT_PROJECT_PATH + "tg/data/independent.csv")
# save_obj("independent_cleaned", data, folder=DEFAULT_PROJECT_PATH + "tg/data/")
#

# data = [{
#            'HEAD': ['sent'],
# 		     'DESC': ['sent0', 'sent1', ... , 'sentN']}, ... ]

data = load_obj("independent_cleaned", folder=DEFAULT_PROJECT_PATH + "tg/data/")

# for i in data:
# 	print(i['HEAD'])
# 	print(i['DESC'])
# exit()

# vocab = get_vocabulary(data)
# save_obj("vocab", vocab, folder=DEFAULT_PROJECT_PATH + "tg/data/")
# exit()

vocab = load_obj(name="vocab", folder=DEFAULT_PROJECT_PATH + "tg/data/")
vocab_size = len(vocab)
print("vocab size: {}".format(vocab_size))

descriptions, headlines = list(), list()
for article_dict in data:
	descriptions.append(article_dict['DESC'])
	headlines.append(article_dict['HEAD'])

tokenizer = create_tokenizer(descriptions + headlines)
X, y = create_sequences(tokenizer, vocab_size, descriptions, headlines)

X_train, X_test, Y_train, Y_test = split_data(X, y, 0.1, np.random.randint(1488))

# glove_embeddings_obj = Glove(DEFAULT_PROJECT_PATH + "tg/glove")
#
# word2idx, idx2word, idx2embedding = fit_data(vocab, glove_embeddings_obj)

# for k, v in glove_embeddings_obj.word2em.items():
# 	print(k, v)
# 	exit()
model = build_model(vocab_size)
inspect_model(model)

batch_size = 32

epochs = 100
print(X_train.shape, Y_train.shape)

model.fit(x=[X_train, Y_train], y=Y_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1)

pred = model.predict(X_test[0])
print("{}, {}".format(pred, X_test[0]))
exit()
try:
	model_name = "start.hpy5"
	model.save_weights(DEFAULT_PROJECT_PATH + "tg/models/" + model_name)  # , overwrite=True)
	print('Model weights saved to {}'.format(DEFAULT_PROJECT_PATH + "tg/models/" + model_name))

except Exception as e:
	print("Unable to save model weights to file {}".format(DEFAULT_PROJECT_PATH + "tg/models/" + model_name))
	print(e)


# model = model.load_weights(DEFAULT_PROJECT_PATH + "tg/models/" + model_name)
# print('Model weights loaded from {}'.format(DEFAULT_PROJECT_PATH + "tg/models/" + model_name))
