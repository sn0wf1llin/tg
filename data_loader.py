import pickle
import string
from tg.utils import *
from psettings import DEFAULT_PROJECT_PATH
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import json
import gzip
import pickle
from tg.config import *
init_nltk_data_path(DEFAULT_PROJECT_PATH + "nltk_data")



def load_signalmedia_json_gz():
	data = []

	count = 0
	file = gzip.open(DEFAULT_PROJECT_PATH + 'tg/data/signalmedia-1m.jsonl.gz')

	for each_line in file:
		record = json.loads(each_line)
		_k = record['title']
		_v = record['content']
		hl = 0

		if hl > sum_txt_length:
			data.append({
			'HEAD': _k,
			'DESC': _v
		})

		count += 1
		if count >= 500000:
			break

	return data


def process_text(text, cleanit=False):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)

	for line in sent_tokenize(text):
		if cleanit:
			# convert to lower case
			line = [word.lower() for word in word_tokenize(line) if len(word) > 2]
			# remove punctuation from each token
			line = [w.translate(table) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
		else:
			line = [word for word in word_tokenize(line) if len(word) > 2]

		cleaned.append(' '.join(line))

	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]

	return cleaned


def load_dataset(dpath, cleanit=True, num_examples=10000):
	df = pd.read_csv(dpath)

	titles = df.title.apply(process_text, args=(cleanit, ))
	contents = df.content.apply(process_text, args=(cleanit, ))

	print("Loaded {} samples from {}.\n".format(df.shape[0], dpath))

	data = []
	for index, (k, v) in enumerate(zip(titles, contents)):
		# take only 'long' headers
		hl = 0
		for i in k:
			hl += len(i.split())

		if hl > sum_txt_length:
			_k = " ".join(k)
			_v = " ".join(v)

			data.append({
				'HEAD': _k,
				'DESC': _v
			})

		if num_examples is not None:
			if index == num_examples:
				return data

	return data


def save_obj(name, obj, folder="./"):
	try:
		with open(folder + name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		print("success: write data to {}".format(folder + name + '.pkl'))
	except Exception as e:
		print("error {} occured".format(e))


def load_obj(name, folder="./"):
	try:
		with open(folder + name + '.pkl', 'rb') as f:
			print("success: read data from {}".format(folder + name + '.pkl'))
			return pickle.load(f)
	except Exception as e:
		print("error {} occured".format(e))

