import time
import datetime
from parameters_extractor.metrics.content_check import is_text
from psettings import *
# from peewee_classes import Resources, Articles
import pandas as pd
import numpy as np


def get_articles_from_db(resource_id, last_added_only, period=None, as_query=False):
	# todo add period processing here

	if last_added_only:
		# last_added_only = True => take data only with
		# created_at >= now() - 1 day
		one_day_ago = datetime.datetime.now() - datetime.timedelta(days=6)

		arts = Articles.select().where((Articles.resource == resource_id) and (Articles.created_at >= one_day_ago))
	else:
		arts = Articles.select().where(Articles.resource == resource_id)

	if as_query:
		return arts

	arts = arts.execute().iterator()
	arts = (article for article in arts if (is_text(article, None)[0]))

	return arts


def get_articles_from_csv(resource, csv_file_path, idf='id', contentf='content', titlef='title', leadf='lead', social_interactionsf='interactions'):
	df = pd.read_csv(csv_file_path)
	# lead title content ---- order
	ID = df[idf]
	L = df[leadf]
	T = df[titlef]
	C = df[contentf]
	SI = df[social_interactionsf]

	articles = ((i[0], i[1], i[2], resource, int(i[3])) for i in zip(L, T, C, ID) if
	            (is_text(i, None, as_tuple=True, as_tuple_content_index=2)[0]))

	return articles


def get_parameters_from_csv(resource, csv_file_path):
	df = pd.read_csv(csv_file_path)
	# read id & parameters columns
	# l t c columns not need
	df_useful_columns = [i for i in df.columns if 'Unnamed' not in i]
	df = df[df_useful_columns]

	for i in df.iterrows():
		yield dict(i[1])


def my_print(text):
	if not QUIET:
		print(text)


def merge_parameters_with_in_csv(input_file_path, save_to_file_path, df, on='id', how='outer'):
	try:
		df_in = pd.read_csv(input_file_path)
		
		# if df_in.index.name is None or df_in.index.name != on:
		# 	df_in.set_index([on], inplace=True)

		# if df.index.name is None or df.index.name != on:
		# 	df.set_index([on], inplace=True)

		dfinal = pd.merge(df_in, df, on=on, how=how)
		no_unnamed_columns = [i for i in dfinal.columns if "Unnamed" not in i]

		dfinal = dfinal[no_unnamed_columns]

		dfinal.to_csv(save_to_file_path)
		my_print("{} Parameters saved to [ {} ]".format(SUCCESS_FLAG, save_to_file_path))
	except Exception as e:
		my_print("{} {}".format(EXCEPTION_FLAG, e))
		my_print("{} Cant save parameters to [ {} ]".format(ERROR_FLAG, save_to_file_path))


def replace_ltc_nan(data):
	if type(data) is dict:
		new_ = {}
		for k, v in data.items():
			if type(v) is not str:
				new_[k] = ""
			else:
				new_[k] = v

	elif type(data) is tuple or type(data) is list:
		new_ = []
		for i in data:
			if type(i) is not str:
				new_.append("")
			else:
				new_.append(i)
	else:
		raise Exception('Cant parse the given data type (replace_ltc_nan)')

	return new_

def is_resource_exists(r):
	return Resources.select().where(Resources.resource == r).exists()


def get_resource_lang(r):
	rlang = Resources.select(Resources.lang).where(Resources.resource == r).get()

	return rlang.__data__['lang'].lower()


def timeit(func):
	def timed(*args, **kwargs):
		ts = time.time()
		result = func(*args, **kwargs)
		te = time.time()

		my_print('\t\t\t\t\t\t\t %r  %2.2f ms' % (func.__name__, (te - ts) * 1000))

		return result

	return timed


def raise_error4code(code, details=None):
	if details is None:
		details = "Processing error."

	return {
		'Status': {
			"code": code,
			"details": details,
		},
	}
