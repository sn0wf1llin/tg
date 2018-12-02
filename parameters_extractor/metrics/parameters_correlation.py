from putils import my_print, is_resource_exists, get_parameters_from_csv
import numpy as np
from sklearn import preprocessing
from math import *
# from peewee_classes import *
from topics_extractor.lda_mw_handler import *
from itertools import tee
import datetime
import pandas as pd
from putils import merge_parameters_with_in_csv, replace_ltc_nan


def _vectors_equality(vector_0, vector_1, vector_2, mini=False):
	mini_N_value = 100

	# vector_0 - lead
	# vector_1 - title
	# vector_2 - content
	vector_0 = [i if i is not None else np.nan for i in vector_0]
	vector_1 = [i if i is not None else np.nan for i in vector_1]

	vector_0 = np.nan_to_num(np.array(vector_0))
	vector_1 = np.nan_to_num(np.array(vector_1))
	vector_2 = np.nan_to_num(np.array(vector_2))

	if not mini:
		euc_sim_01 = np.linalg.norm(vector_0 - vector_1)
		euc_sim_02 = np.linalg.norm(vector_0 - vector_2)
		euc_sim_12 = np.linalg.norm(vector_1 - vector_2)

		varr = np.array([euc_sim_01, euc_sim_02, euc_sim_12])

		return np.median(preprocessing.minmax_scale(varr.T).T)

	else:
		a = vector_0.tolist()
		b = vector_1.tolist()
		c = vector_2.tolist()

		try:
			mini_N_value = max(len(a), len(b), len(c)) 
			if mini_N_value == 0:
				mini_N_value = 100
		except Exception as e:
			mini_N_value = 100

		sab = set(a).intersection(b)
		sbc = set(b).intersection(c)
		sac = set(a).intersection(c)
		sets_abc = set(a).intersection(b).intersection(c)

		m = np.median(np.array([len(i) for i in [sab, sbc, sac, sets_abc]]))
		
		return 1.0 * m / mini_N_value


def _lda_result_topic_equality(lda_res_topics):
	lt, tt, ct = lda_res_topics
	lt, tt, ct = map(lambda x: x[1:-1], [lt, tt, ct])

	d_equality = {
		0: 0.0,
		1: 0.5,
		2: 0.7,
		3: 1.0,
	}

	def common_elements(list1, list2, list3):
		return len(list(set(list1) & set(list2) & set(list3)))

	lt_words, tt_words, ct_words = map(lambda x: x.split(), [lt, tt, ct])

	eq_words_count = common_elements(ct_words, lt_words, tt_words)

	if eq_words_count > 3:
		return 1.0

	return d_equality[eq_words_count]


def thematic_equality(art_ltc, resource, data_type):
	mini_m = LDAMWHandler(mtype='single_ltc', resource=resource)

	if data_type == 'db':
		l, t, c = art_ltc.get().lead, art_ltc.get().title, art_ltc.get().content

	elif data_type == 'csv':
		l, t, c = art_ltc['lead'], art_ltc['title'], art_ltc['content']

	res = mini_m(article_pack=(l, t, c), is_single_ltc=True)

	if res is None:
		return 0.0

	result_topics, other_topics_ids_vectors = list(i[0] for i in res), list(i[1] for i in res)

	result_other_topics_eq_value = _vectors_equality(*other_topics_ids_vectors, mini=True)

	result_topic_eq_value = _lda_result_topic_equality(result_topics)

	return np.mean(np.array([result_other_topics_eq_value, result_topic_eq_value]))


def _calculate_corr(factors_ltc_all, data_type):
	# len of factors_ltc_all must be 4 as equals to factors_lead, factors_title, factors_content AND
	# lead & title & content as separate dict

	v = 0.0

	# factors_ltc_all - tuple of publication metrics and it's title lead content as dict
	# ((metrics), {
	#               'lead': l,
	#               'title': t,
	#               'content': c,
	#             })
	#  <- example

	if data_type == 'csv':
		fmetrics, ltc, resource, a_id = factors_ltc_all
	elif data_type == 'db':
		fmetrics, ltc, resource = factors_ltc_all

	# ltc = replace_ltc_nan(ltc)

	if len(fmetrics) == 3:
		l_params, t_params, c_params = fmetrics

		use_keys = c_params.keys()

		if len(l_params) == 1:
			l_params = {k: None for k in use_keys}

		if len(t_params) == 1:
			t_params = {k: None for k in use_keys}

		topic_local_eq_value = thematic_equality(ltc, resource, data_type)

		try:
			l_v, t_v, c_v = map(
				lambda d: [v_ for k_, v_ in d.items() if
						   k_ not in ['topic',
									  'theme',
									  'most_common_non_stop_words',
									  'id',
									  'article_id',
									  'created_at',
									  'updated_at']],
				[l_params, t_params, c_params])
		except AttributeError:
			l_v, t_v, c_v = l_params, t_params, c_params

		# remove a md5 article_id and index of row from comparing vectors
		parameters_equality = _vectors_equality(l_v[2:], t_v[2:], c_v[2:])

		v = parameters_equality * topic_local_eq_value

	if v is None:
		return {'correlation_value': 0.0}

	return {'correlation_value': v}


def get_params_arts_data_from_db(resource):
	arts = Articles.select().where(Articles.resource == resource)

	for a in arts:
		try:
			yield ([Parameters.select().where(
				(Parameters.article == a) & (Parameters.element_type == LEAD_ELEMENT_TYPE)).get().__data__,
					Parameters.select().where(
						(Parameters.article == a) & (Parameters.element_type == TITLE_ELEMENT_TYPE)).get().__data__,
					Parameters.select().where(
						(Parameters.article == a) & (Parameters.element_type == CONTENT_ELEMENT_TYPE)).get().__data__],
				   a, resource)

		except Parameters.DoesNotExist:
			pass


def save_parameters_params(p, art):
	p['article'] = art
	p['updated_at'] = datetime.datetime.today().date(),

	try:
		par_by_art_id_check = ParametersCorrelation.select().where(ParametersCorrelation.article == art).get()

		for k, v in p.items():
			setattr(par_by_art_id_check, k, v)

		par_by_art_id_check.save()

	except Exception:
		ParametersCorrelation.create(**p)


def prepare_csv_data(csv_data, resource):

	for i in csv_data:

		a_id = i['id']
		ltc = {'lead': i['lead'], 'title': i['title'], 'content': i['content']}

		# get all simple metrics from dict
		# modify keys = from element_type_1_... to ...

		l_simple_metrics = {k[15:]: v for k, v in i.items() if 'element_type_{}'.format(LEAD_ELEMENT_TYPE) in k}
		t_simple_metrics = {k[15:]: v for k, v in i.items() if 'element_type_{}'.format(TITLE_ELEMENT_TYPE) in k}
		c_simple_metrics = {k[15:]: v for k, v in i.items() if 'element_type_{}'.format(CONTENT_ELEMENT_TYPE) in k}

		# l_additional_metrics = {k: v for k, v in i.items() if
		# 						'element_type_' not in k and ('lead_' in k or '_lead' in k)}
		# t_additional_metrics = {k: v for k, v in i.items() if
		# 						'element_type_' not in k and ('title_' in k or '_title' in k)}
		# c_additional_metrics = {k: v for k, v in i.items() if
		# 						'element_type_' not in k and ('content_' in k or '_content' in k)}
		#
		# if not keys_got:
		# 	use_keys = ['id', 'lead', 'title', 'content'] + list(l_simple_metrics.keys()) + list(t_simple_metrics.keys()) + list(c_simple_metrics.keys()) + list(l_additional_metrics.keys()) + list(t_additional_metrics.keys()) + list(c_additional_metrics.keys())
		# 	left_keys = [i for i in i.keys() if i not in use_keys]
		#
		# left_data = {left_k: i[left_k] for left_k in left_keys}

		fmetrics = (l_simple_metrics, t_simple_metrics, c_simple_metrics)
		yield (fmetrics, ltc, resource, a_id)


def _get_correlation_metric_from_resource(resource, csv_data_file_path, data_type):
	if data_type == 'db':
		p_a_gen = get_params_arts_data_from_db(resource)

	elif data_type == 'csv':
		p_a_gen = get_parameters_from_csv(resource=resource, csv_file_path=csv_data_file_path)
		
		# ((fmetrics=(lmetrics, tmetrics, cmetrics)), ltc=(l, t, c), resource)
		p_a_gen = prepare_csv_data(p_a_gen, resource)

	else:
		raise Exception("Parameters_correlation error: data_type is wrong.")

	params_ltc, ps = tee(p_a_gen)

	params_eq = (_calculate_corr(p_ltc, data_type) for p_ltc in params_ltc)

	if data_type == 'db':
		for par, art_obj in zip(params_eq, ps):
			if par is not None:
				art = art_obj[1].article
				save_parameters_params(par, art)
	elif data_type == 'csv':

		correlation_df = pd.DataFrame()

		for par, art in zip(params_eq, ps):
			a_id = art[-1]
			tmp = {
				'id': a_id,
				'correlation_value': par['correlation_value']
			}
			correlation_df = correlation_df.append([tmp])

		# correlation_df.set_index('id', inplace=True)
		
		merge_parameters_with_in_csv(csv_data_file_path, '{}_correlation_value.csv'.format(resource), correlation_df)


def get_correlation_metric(resource, csv_data_file_path, data_type):
	resources_iterator = []

	if data_type == 'db':
		if resource is None:
			rdata = Resources.select().iterator()
			resources_iterator = [elem.__data__['resource'] for elem in rdata]
		else:
			if is_resource_exists(resource):
				resources_iterator = [resource]
			else:
				my_print("{}Resource [ {} ] not found. Exiting ...".format(ERROR_FLAG, resource))

	elif data_type == 'csv':
		resources_iterator = [resource]

	gstart_time = datetime.datetime.now()

	for _resource in resources_iterator:
		lstart_time = datetime.datetime.now()

		_get_correlation_metric_from_resource(_resource, csv_data_file_path=csv_data_file_path, data_type=data_type)

		my_print("{}Correlation for [ {} ] calculated in {}".format(INFO_FLAG, _resource, datetime.datetime.now() - lstart_time))

	my_print(
		"{}Correlation for [ {} ] calculated in {}".format(INFO_FLAG, "All resources", datetime.datetime.now() - gstart_time))
