from datetime import datetime
from putils import my_print
import pandas as pd
import numpy as np
from itertools import chain, tee
from parameters_extractor.metrics.simple import SimpleMetricsCallback
from parameters_extractor.metrics.additional_options import AdditionalMetricsCallback
from parameters_extractor.metrics.sentiment import PolarityMetricsCallback
from parameters_extractor.metrics.sentiment import SubjectivityCallback
from peewee_classes import *
import multiprocessing
import argparse
from putils import timeit, get_articles_from_db, get_articles_from_csv, merge_parameters_with_in_csv
import datetime as dt
from parameters_extractor.metrics.parameters_correlation import get_correlation_metric

FULL_CONTENT_CALLBACKS = (
	AdditionalMetricsCallback(),
)

CONTENT_CALLBACKS = (
	SimpleMetricsCallback(),
	PolarityMetricsCallback(),
	SubjectivityCallback(),
)


def get_text_parameters(text, element_type):
	params = {'element_type': element_type}

	if text is not None:
		if type(text) is str:
			if text != '':
				params = {k: v for k, v in chain.from_iterable([callback(text) for callback in CONTENT_CALLBACKS])}

				params['element_type'] = element_type
				params['updated_at'] = dt.datetime.today().date(),

	return params


# @timeit
def process_for_additional(a):
	additional_parameters_all = {}

	try:
		lead, title, content = a.lead, a.title, a.content
	except AttributeError:
		try:
			lead, title, content = a['lead'], a['title'], a['content']
		except Exception:
			lead, title, content = a[0], a[1], a[2]

	acs = (callback(lead, title, content) for callback in FULL_CONTENT_CALLBACKS)

	for c in acs:
		for (k, v) in c:
			additional_parameters_all[k] = v

	return additional_parameters_all


def process_for_simple(a, mcnsw_sep=False):
	try:
		lead, title, content = a.lead, a.title, a.content
	except AttributeError:
		try:
			lead, title, content = a['lead'], a['title'], a['content']
		except Exception:
			lead, title, content = a[0], a[1], a[2]

	t_element_type, l_element_type, c_element_type = TITLE_ELEMENT_TYPE, LEAD_ELEMENT_TYPE, CONTENT_ELEMENT_TYPE

	title_p = get_text_parameters(title, element_type=t_element_type)

	content_p = get_text_parameters(content, element_type=c_element_type)

	lead_p = get_text_parameters(lead, element_type=l_element_type)

	if mcnsw_sep:
		if lead_p is not None and title_p is not None and content_p is not None:
			return (lead_p, title_p, content_p), (
				lead_p['most_common_non_stop_words'],
				title_p['most_common_non_stop_words'],
				content_p['most_common_non_stop_words'])
		else:
			return None

	return lead_p, title_p, content_p


# @database.execution_context()
def save_parameters(art, params):
	params['article'] = art

	_p_obj = Parameters.select(Parameters.parameters_pack).where(
		(Parameters.article == art.article) & (Parameters.element_type == params['element_type']))

	if _p_obj.exists():
		p_obj = _p_obj.get()
		params['updated_at'] = dt.datetime.today().date()

		for k, v in params.items():
			setattr(p_obj, k, v)

		p_obj.save()

	else:
		# except DoesNotExist:
		params['created_at'] = dt.datetime.today().date()
		p = Parameters.create(**params)


def __create_or_update_additional_parameters(art, params):
	_ap_obj = AdditionalParameters.select().where(AdditionalParameters.article == art.article)

	if _ap_obj.exists():
		ap_obj = _ap_obj.get()
		params['updated_at'] = dt.datetime.today().date()

		for k, v in params.items():
			setattr(ap_obj, k, v)

		ap_obj.save()

	else:
		params['created_at'] = dt.datetime.today().date()
		ap = AdditionalParameters.create(**params)


# @database.execution_context()
def save_additional_parameters(type_params, art):
	type_params['article'] = art

	__create_or_update_additional_parameters(art, type_params)


def run(corr_calc, base_calc, add_calc, resource, last_added_only, data_type, csv_data_input_file_path=None, csv_data_output_file_path=None):
	n_cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(n_cores)

	if data_type == 'db':
		articles = get_articles_from_db(resource_id=resource, last_added_only=last_added_only)

	elif data_type == 'csv':
		articles = get_articles_from_csv(resource=resource, csv_file_path=csv_data_input_file_path)
	else:
		raise Exception("Cant read data <articles>. Exiting ...")

	articles_simple, articles_additional, articles_for_data_params_simple, articles_for_data_params_additional = tee(
		articles, 4)

	if base_calc:

		my_print("{}Going to calculate simple parameters for [ {} ] ...".format(INFO_FLAG,
																						   resource if resource is not None else "All resources"))

		time_start = datetime.now()

		# extracted_simple_parameters = (process_for_simple(a_data) for a_data in articles_for_data_params_simple)

		if data_type == 'db':
			extracted_simple_parameters = pool.imap(process_for_simple, articles_for_data_params_simple)

			for art, ltc_params in zip(articles_simple, extracted_simple_parameters):
				for param in ltc_params:
					if param is not None:
						save_parameters(art, param)

		elif data_type == 'csv':
			extracted_simple_parameters = (process_for_simple(i) for i in articles_for_data_params_simple)

			if csv_data_output_file_path is not None:
				output_file = csv_data_output_file_path
			else:
				output_file = "{}_simple_parameters.csv".format(resource)

			csv_resource_simple_parameters_df = pd.DataFrame()

			for art, ltc_params in zip(articles_simple, extracted_simple_parameters):
				a_id = art[-1]

				p00 = ltc_params[0]
				p11 = ltc_params[1]
				p22 = ltc_params[2]

				p00_d = {'element_type_{}_'.format(p00['element_type']) + k: v for k, v in p00.items()}
				p11_d = {'element_type_{}_'.format(p11['element_type']) + k: v for k, v in p11.items()}
				p22_d = {'element_type_{}_'.format(p22['element_type']) + k: v for k, v in p22.items()}

				tmp = {'id': a_id, **p00_d, **p11_d, **p22_d}

				csv_resource_simple_parameters_df = csv_resource_simple_parameters_df.append([tmp])

			csv_resource_simple_parameters_df.set_index(['id'], inplace=True)

			merge_parameters_with_in_csv(csv_data_input_file_path, output_file, csv_resource_simple_parameters_df)

		my_print("{}Resources: [ {} ]; Simple parameters calculated in {}".format(SUCCESS_FLAG,
																				  resource if resource is not None else "All resources",
																				  datetime.now() - time_start))

	if corr_calc:
		my_print("{}Going to calculate articles parameters correlation for [ {} ] ...".format(INFO_FLAG,
																							  resource if resource is not None else "All resources"))
		time_start = datetime.now()

		get_correlation_metric(resource, csv_data_file_path=csv_data_input_file_path, data_type=data_type)

		my_print("{}Resources: [ {} ]; Correlation calculated in {}".format(SUCCESS_FLAG,
																			resource if resource is not None else "All resources",
																			datetime.now() - time_start))

	if add_calc:
		my_print("{}Going to calculate additional parameters for [ {} ] ...".format(INFO_FLAG,
																					resource if resource is not None else "All resources"))

		time_start = datetime.now()

		if data_type == 'db':
			extracted_additional_parameters = pool.imap(process_for_additional, articles_for_data_params_additional)

			for art, params in zip(articles_additional, extracted_additional_parameters):

				if params is not None:
					save_additional_parameters(params, art)

		elif data_type == 'csv':
			extracted_additional_parameters = (process_for_additional(i) for i in articles_for_data_params_additional)

			if csv_data_output_file_path is not None:
				output_file = csv_data_output_file_path
			else:
				output_file = "{}_additional_parameters.csv".format(resource)

			csv_resource_additional_parameters_df = pd.DataFrame()#columns=['id', "n_title_symbols", "n_title_numbers", "n_title_letters", "n_title_words", "n_title_mean_letters_in_words", "title_words_diff_emotions", "title_angry", "title_anticipation", "title_disgust", "title_fear", "title_joy", "title_sadness", "title_surprise", "title_trust", "title_neg", "title_pos", "most_frequent_title_word_len", "most_frequent_title_word_count", "title_max_term_length", "n_lead_symbols", "n_lead_numbers", "n_lead_letters", "n_lead_words", "n_lead_mean_letters_in_words", "lead_words_diff_emotions", "lead_angry", "lead_anticipation", "lead_disgust", "lead_fear", "lead_joy", "lead_sadness", "lead_surprise", "lead_trust", "lead_neg", "lead_pos", "most_frequent_lead_word_len", "most_frequent_lead_word_count", "lead_max_term_length", "content_dots_count", "content_commas_count", "content_exclamation_marks_count", "content_question_marks_count", "n_content_symbols", "n_content_numbers", "n_content_letters", "n_content_words", "n_content_mean_letters_in_words", "content_mean_words_count", "content_sentences_count", "max_count_words_in_sent_content", "min_count_words_in_sent_content", "content_total_words_count", "max_freq_of_term_in_content", "min_freq_of_term_in_content", "max_term_length_content", "content_sum_emotionality", "content_mean_emotionality", "content_max_emotionality_sentences", "content_min_emotionality_sentences", "content_mean_emo_of_sentences", "content_angry", "content_anticipation", "content_disgust", "content_fear", "content_joy", "content_sadness", "content_surprise", "content_trust", "content_neg", "content_pos", "title_uniq_wd", "title_complx", "title_snt_len", "title_syll_ct", "title_flesch", "lead_uniq_wd", "lead_complx", "lead_snt_len", "lead_syll_ct", "lead_flesch", "content_ari", "title_ari", "lead_ari", "content_coleman", "content_db1", "content_db2", "content_db_grade", "content_ds", "content_herdan", "content_cttr", "content_hdd", "content_yueles_k", "content_maas_1", "content_mtld", "content_rld", "content_sld", "content_ttr", "title_ttr", "lead_ttr", "content_count_of_types", "content_count_of_tokens", "title_count_of_types", "title_count_of_tokens", "lead_count_of_types", "lead_count_of_tokens", "content_uber", "content_growth_vocabl"])

			for art, params in zip(articles_additional, extracted_additional_parameters):

				a_id = art[-1]

				tmp = {'id': a_id, **params}

				csv_resource_additional_parameters_df = csv_resource_additional_parameters_df.append([tmp])

			csv_resource_additional_parameters_df.set_index('id', inplace=True)

			merge_parameters_with_in_csv(csv_data_input_file_path, output_file, csv_resource_additional_parameters_df)

		my_print("{}Additional parameters for {} calculated in {}".format(SUCCESS_FLAG,
																		  resource if resource is not None else "All resources",
																		  datetime.now() - time_start))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	subparsers = parser.add_subparsers(title='dtype', help='database or csv-file')

	# add db subparser
	parser_db = subparsers.add_parser('db')

	# db add defaults
	parser_db.set_defaults(resource=None,
						   which='db',
						   base_calculate=False,
						   correlation_calculate=False,
						   additional_calculate=False)

	parser_db.add_argument('--method', '-m', required=True, dest='method',
						   help='Choose method | train | evaluate |\n\nevaluate - you have a pre-trainded model in '
								'/lda_data with all useful stuff (dictionary saved, etc.,) and you want to predict '
								'topic for a new article;\n\ntrain    - means that you haven''t pre-trained model, '
								'it''s create lda_data directory (if you have no), and also this one in according '
								'directory with all useful stuff.\n')
	parser_db.add_argument('--resource', '-r', dest='resource')

	parser_db.add_argument('--base', '-b', action='store_true', dest='base_calculate')
	parser_db.add_argument('--corr', '-c', action='store_true', dest='correlation_calculate')
	parser_db.add_argument('--add', '-a', action='store_true', dest='additional_calculate')

	# new --last-added-only
	parser_db.add_argument('--last-added-only', action='store_true', dest='last_added_only', default=False)

	# add csv subparser
	parser_csv = subparsers.add_parser('csv')

	# csv add defaults
	parser_csv.set_defaults(input_file=None,
							output_file=None,
							resource_lang_csv=None,
							which='csv',
							parallelize=False,
							)

	parser_csv.add_argument('--method', '-m', required=True, dest='method',
							help='Choose method | train | evaluate |\n\nevaluate - you have a pre-trainded model in /lda_data with all useful stuff (dictionary saved, etc.,) and you want to predict topic for a new article;\n\ntrain    - means that you haven''t pre-trained model, it''s create lda_data directory (if you have no), and also this one in according directory with all useful stuff.\n')

	parser_csv.add_argument('--base', '-b', action='store_true', dest='base_calculate')
	parser_csv.add_argument('--corr', '-c', action='store_true', dest='correlation_calculate')
	parser_csv.add_argument('--add', '-a', action='store_true', dest='additional_calculate')

	parser_csv.add_argument('--resource', '-r', dest='resource')
	parser_csv.add_argument('--resource-lang-csv', '-rlc', dest='resource_lang_csv')

	parser_csv.add_argument('--input_file', '-in', dest='input_file', help='Input csv file with data')
	parser_csv.add_argument('--output_file', '-out', dest='output_file', help='Output csv file name')
	parser_csv.add_argument('--parallelize', '-p', action='store_true', dest='parallelize')

	args = parser.parse_args()

	if args.which == 'csv':
		if args.input_file is None:
			raise Exception("Input_data parameter cant be None. Set --input-file <file> or -in <file>")
		else:
			"""
				CSV file must be in format ID == i[1][2] Lead i[1][1] Title i[1][0] Content i[1][2]

			"""
			data_file_path = args.input_file

			if args.method == 'evaluate':
				if args.input_file is None:
					raise Exception("Input file cant be None. Set --input-file <file> or -in <file> parameter.")

				run(corr_calc=args.correlation_calculate,
				    add_calc=args.additional_calculate,
				    resource=args.resource,
				    base_calc=args.base_calculate,
				    csv_data_input_file_path=args.input_file,
					csv_data_output_file_path=args.output_file,
			        data_type=args.which,
				    last_added_only=False)


	elif args.which == 'db':
		if args.method == 'evaluate':
			run(corr_calc=args.correlation_calculate,
				add_calc=args.additional_calculate,
				resource=args.resource,
				base_calc=args.base_calculate,
				last_added_only=args.last_added_only,
			    data_type=args.which)
