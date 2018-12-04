from putils import my_print, is_resource_exists
import nltk
import pandas as pd
import datetime as dt
# from topics_extractor.topics_truncate import truncate_topics_tables
from itertools import chain, tee
from topics_extractor.lda_mw_handler import LDAMWHandler
import argparse
import datetime
from parameters_extractor.metrics.content_check import is_text
# from peewee_classes import *
from psettings import *
from putils import get_articles_from_db, get_articles_from_csv
from multiprocessing import Pool

nltk.data.path.append(DEFAULT_PROJECT_PATH + 'nltk_data/')

CONTENT_CALLBACKS = (
	LDAMWHandler(),
)


def get_topic(text, element_type, resource_name, is_csv=False):
	params = {'topic': None, 'element_type': element_type}

	if text is not None:
		if type(text) is str:
			if text != '':
				params = {k: v for k, v in chain.from_iterable(
					[callback(text, resource_name, is_csv=is_csv) for callback in CONTENT_CALLBACKS])}

				params['element_type'] = element_type

	return params


def topic_ltc_by_resource(lead_title_content_resource, is_csv=False):
	try:
		lead, title, content, res, aid = lead_title_content_resource
	except Exception as e:
		lead, title, content, res = lead_title_content_resource
		aid = None

	l_params, t_params, c_params = LEAD_ELEMENT_TYPE, TITLE_ELEMENT_TYPE, CONTENT_ELEMENT_TYPE

	lead_params_topic = get_topic(lead, l_params, res, is_csv=is_csv)
	title_params_topic = get_topic(title, t_params, res, is_csv=is_csv)
	content_params_topic = get_topic(content, c_params, res, is_csv=is_csv)

	if aid is not None:
		return aid, lead_params_topic, title_params_topic, content_params_topic

	return lead_params_topic, title_params_topic, content_params_topic


def save_parameters(params, art):
	params['article'] = art

	try:
		t_id = Topics.select(Topics.topic).where(Topics.value == params['topic']).get().topic

		params['topic'] = t_id

		p_obj_if_exists = Parameters.select().where((
			Parameters.article == art.article) & (Parameters.element_type == params['element_type'])).get()

		params['updated_at'] = dt.datetime.today().date()

		for k, v in params.items():
			setattr(p_obj_if_exists, k, v)

		p_obj_if_exists.save()

	except Parameters.DoesNotExist:
		params['created_at'] = dt.datetime.today().date()
		Parameters.create(**params)


# def _process_group_parameters(callback, params, texts):
#     for params, result in zip(params, callback(texts)):
#         k, v = result
#         params[k] = v
#         yield params

def save_topics_to_csv(save_to_file_path, df):
	try:
		df.to_csv(save_to_file_path)
		my_print("{} Topics saved to [ {} ]".format(SUCCESS_FLAG, save_to_file_path))
	except Exception as e:
		my_print("{} Cant save topics to [ {} ]".format(ERROR_FLAG, save_to_file_path))


def merge_topics_with_in_csv(input_file_path, save_to_file_path, df, on='id', how='outer'):
	try:
		df_in = pd.read_csv(input_file_path)

		if df_in.index.name is None or df_in.index.name != on:
			df_in.set_index([on], inplace=True)

		dfinal = df_in.merge(df, on=on, how=how)
		no_unnamed_columns = [i for i in dfinal.columns if "Unnamed" not in i]

		dfinal = dfinal[no_unnamed_columns]

		dfinal.to_csv(save_to_file_path)
		my_print("{} Topics saved to [ {} ]".format(SUCCESS_FLAG, save_to_file_path))
	except Exception as e:
		my_print("{} {}".format(EXCEPTION_FLAG, e))
		my_print("{} Cant save topics to [ {} ]".format(ERROR_FLAG, save_to_file_path))



def run(resource=None, period=None, last_added_only=False, data_type=None, csv_data_input_file_path=None, csv_data_output_file_path=None):
	gtime_start = datetime.datetime.now()

	if data_type == 'db':
		if resource is None:
			rdata = Resources.select().iterator()
			resources_iterator = [elem.__data__['resource'] for elem in rdata]
		else:
			if is_resource_exists(resource):
				resources_iterator = [resource]
			else:
				my_print("{}Resource [ {} ] not found. Exiting ...".format(ERROR_FLAG, resource))

		ps_resources = (get_articles_from_db(resource_id=r_id, period=period, last_added_only=last_added_only) for r_id in resources_iterator)

		for ps, res in zip(ps_resources, resources_iterator):
			ltime_start = datetime.datetime.now()

			ps, articles_s, data = tee(ps, 3)

			# check content; process if it's not too short or empty
			data = ((p.lead, p.title, p.content, res) for p in data if (is_text(p, None)[0]))

			pool = Pool()

			params = pool.map(topic_ltc_by_resource, data)

			for art, prms in zip(articles_s, params):
				for par in prms:
					if par is not None:
						save_parameters(par, art)

			my_print("{} [ {} ] :: LDA topics calculated in {}".format(SUCCESS_FLAG, res, datetime.datetime.now() - ltime_start))

			del pool

		if len(resources_iterator) != 1:
			my_print("{}{} :: calculated in {}".format(SUCCESS_FLAG, " ".join(resources_iterator), datetime.datetime.now() - gtime_start))

	elif data_type == 'csv':
		if resource is None:
			raise Exception("Resource cant be undefined for csv data_type.")

		ps_csv_resource = get_articles_from_csv(resource, csv_data_input_file_path)

		ltime_start = datetime.datetime.now()

		ps, articles_s, data = tee(ps_csv_resource, 3)

		if csv_data_output_file_path is not None:
			output_file = csv_data_output_file_path
		else:
			output_file = "{}_topics.csv".format(resource)

		csv_resource_topics_df = pd.DataFrame()

		pool = Pool()
		a_tmps = (pool.map(_process_csv_pool, data))

		for tmp in a_tmps:
			csv_resource_topics_df = csv_resource_topics_df.append([tmp])

		csv_resource_topics_df.set_index('id', inplace=True)

		my_print(
			"{} [ {} ] :: LDA topics calculated in {}".format(SUCCESS_FLAG, resource, datetime.datetime.now() - ltime_start))

		# save_topics_to_csv(output_file, csv_resource_topics_df)
		merge_topics_with_in_csv(csv_data_input_file_path, output_file, csv_resource_topics_df)
	else:
		pass


def _process_csv_pool(data):
	for a_tuple in data:
		a_id = a_tuple[-1]
		prms = topic_ltc_by_resource(a_tuple, is_csv=True)

		tmp = {
			'id': a_id,
			'element_type_{}_topic'.format(prms[1]['element_type']): prms[1]['topic'],
			'element_type_{}_topic'.format(prms[2]['element_type']): prms[2]['topic'],
			'element_type_{}_topic'.format(prms[3]['element_type']): prms[3]['topic'],
		}

		ret tmp


def train_models_for_resources(data_type, resources, resource_lang_csv=None, csv_data_file_path=None):
	resources_names_list = []

	if data_type == 'db':
		if resources is None:
			resources_all = Resources.select(Resources.resource).iterator()
			resources_names_list = [i.__data__['resource'] for i in resources_all]
		else:
			resources_names_list = [resources]

	elif data_type == 'csv':
		resources_names_list = [resources]

	if len(resources_names_list) == 0:
		raise Exception("Resources not defined. Set -r <resource> or --resource <resource> variable.")

	for resource_name in resources_names_list:
		LDAMWHandler().train(data_type=data_type, resource=resource_name, res_lang=resource_lang_csv, csv_data_file_path=csv_data_file_path)

	my_print("{}Train finished.\n".format(SUCCESS_FLAG))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	subparsers = parser.add_subparsers(title='dtype')

	# add db subparser
	parser_db = subparsers.add_parser('db')

	# db add defaults
	parser_db.set_defaults(resource=None, updkw=False, period=None, which='db')

	parser_db.add_argument('--method', '-m', required=True, dest='method')
	parser_db.add_argument('--resource', '-r', dest='resource')
	parser_db.add_argument('--period', '-p', dest='period')

	# new --last-added-only
	parser_db.add_argument('--last-added-only', action='store_true', dest='last_added_only', default=False)

	# add csv subparser
	parser_csv = subparsers.add_parser('csv')

	# csv add defaults
	parser_csv.set_defaults(input_file=None, resource_lang_csv=None, output_file=None, which='csv')

	parser_csv.add_argument('--method', '-m', required=True, dest='method')
	parser_csv.add_argument('--resource', '-r', dest='resource')
	parser_csv.add_argument('--resource-lang-csv', '-rlc', dest='resource_lang_csv')
	parser_csv.add_argument('--input-file', '-in', dest='input_file', help='Input csv file with data')
	parser_csv.add_argument('--output-file', '-out', dest='output_file', help='Output csv file name')
	parser_csv.add_argument('--parallelize', '-p', nargs='?', default=True, dest='parallelize',
							help='Add a parallelization opportunity')

	args = parser.parse_args()

	if args.which == 'csv':
		if args.input_file is None:
			raise Exception("Input_data parameter cant be None. Set --input-file <file> or -in <file>")
		else:
			"""
				CSV file must be in format ID == i[1][2] Lead i[1][1] Title i[1][0] Content i[1][2]
				
			"""
			data_file_path = args.input_file

			if args.method == 'train':
				train_models_for_resources(data_type='csv', resources=args.resource, resource_lang_csv=args.resource_lang_csv, csv_data_file_path=data_file_path)

			elif args.method == 'evaluate':
				run(resource=args.resource, data_type=args.which, csv_data_input_file_path=args.input_file, csv_data_output_file_path=args.output_file)


	elif args.which == 'db':
		if args.method == 'train':
			truncate_topics_tables(args.resource)
			train_models_for_resources(data_type='db', resources=args.resource)

		elif args.method == 'evaluate':
			run(resource=args.resource, period=args.period, last_added_only=args.last_added_only, data_type=args.which)
