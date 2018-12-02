from operator import itemgetter
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from multiprocessing import Pool
import pandas as pd
from psettings import *
import numpy as np
from nltk_data.stop_words_data.stop_word_processing import get_stop_words
from topics_extractor.topics_truncate import truncate_topics_tables

import datetime as dt

import nltk
import re
import os
# from peewee_classes import Articles, Resources, Topics, TopicsResources

ENGLISH_CHARS = re.compile('[^\W_]', re.IGNORECASE)
ALL_CHARS = re.compile('[^\W_]+', re.IGNORECASE | re.UNICODE)


class LDAMWBase:
	def __init__(self,
				 mtype='multiple',
				 resource=None,
				 lda_work_folder=None,
				 lda_model_filename=None,
				 lda_dict_filename=None,
				 lda_topic_word_count=0,
				 lda_topics_count=0,
				 resource_language=None,
				 data_type=None):

		#
		# todo Deutsch Lemmatizer / Stemmer !!!
		#

		self.p_stemmer = PorterStemmer()
		self.wn_lemmatizer = WordNetLemmatizer()

		if resource is not None:
			# resource_lang == 'en' as default
			resource_lang = 'en'

			# hope that resource is correct and exists
			if data_type == 'db':
				resource_lang = Resources.select(Resources.lang).where(Resources.resource == resource).get()
				resource_lang = resource_lang.__data__['lang'].lower()

			elif data_type == 'csv':
				if resource_language is None:
					raise Exception("Resource language must be defined for csv data type.")
				else:
					resource_lang = resource_language
			else:
				pass

			self.stop_words = get_stop_words(resource_lang)

		self.resource_identifier_name = resource

		def _create_model_deps(model_name, twordscount, tcount, mini=False, mini_path=None):

			if not mini:
				mp = DEFAULT_PROJECT_PATH + 'topics_extractor/lda_data' + '/' + model_name
			else:
				mp = DEFAULT_PROJECT_PATH + 'topics_extractor/lda_data' + '/' + mini_path

			mn = 'lda_model' + '_' + model_name
			md = 'dictionary' + '_' + model_name
			ltwordscount = twordscount
			ltcount = tcount

			_short_model_report = "{}{}: {} \n{}{}: {}\n{}{}: {}\n{}{}: {}\n{}{}: {}\n{}".format(
					INFO_FLAG, colored("Model path", 'red', None, ['bold']), mp,
					INFO_FLAG, colored("Model name", 'red', None, ['bold']), mn,
					INFO_FLAG, colored("Model dictionary", 'red', None, ['bold']), md,
					INFO_FLAG, colored("Topic words count", 'red', None, ['bold']), ltwordscount,
					INFO_FLAG, colored("Topics count", 'red', None, ['bold']), ltcount,
					"-" * 88
				)
			if model_name != 'mini':
				print(_short_model_report)

			return mp, mn, md, ltwordscount, ltcount

		if mtype == 'multiple':
			if resource is not None:
				mpath, mname, mdict, lda_topic_word_count, lda_topics_count = _create_model_deps(
					self.resource_identifier_name, LDA_TOPIC_WORD_COUNT, LDA_TOPICS_COUNT)
			else:
				raise Exception("{}Resource must be defined. Exiting... \n".format(EXCEPTION_FLAG))

		elif mtype == 'single_ltc':
			mpath, mname, mdict, lda_topic_word_count, lda_topics_count = _create_model_deps(
				"mini", MINI_LDA_TOPIC_WORD_COUNT, MINI_LDA_TOPICS_COUNT, mini=True, mini_path=self.resource_identifier_name + "/mini")

		if lda_work_folder is None:
			self.lda_work_folder = mpath
		else:
			self.lda_work_folder = lda_work_folder

		if not os.path.exists(self.lda_work_folder):
			os.mkdir(self.lda_work_folder)

		if lda_model_filename is None:
			self.lda_model_filename = os.path.join(self.lda_work_folder, mname)
		else:
			self.lda_model_filename = os.path.join(self.lda_work_folder, lda_model_filename)

		if lda_dict_filename is None:
			self.lda_dict_filename = os.path.join(self.lda_work_folder, mdict)
		else:
			self.lda_dict_filename = os.path.join(self.lda_work_folder, lda_dict_filename)

		self.lda_topics_count = lda_topics_count
		self.lda_topic_word_count = lda_topic_word_count

		self.dictionary = None
		self.lda_model = None
		self.lda_topics = []

	@staticmethod
	def load_csv_data(csv_file):
		df = pd.read_csv(csv_file)
		train_documents = df['content'].values

		return train_documents

	@staticmethod
	def load_single_ltc(ltc_data):
		train_documents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', ltc_data)

		return train_documents

	@staticmethod
	def load_db_data(resource=None):
		# if resource is None:
		#     art_content_stream = Articles.select()
		# else:
		art_content_stream = Articles.select().where(Articles.resource == resource)

		train_documents = (acs.content for acs in art_content_stream if acs.content is not None)

		return train_documents

	def save_model(self, as_name=None, save_on_disk=True, save_topics_into_db=False):
		if save_on_disk:
			print(" \t-> Model was saved as [ {} ]".format(as_name))
			if as_name is not None:
				self.lda_model.save(as_name)
			else:
				self.save_model(self.lda_model_filename)

		if save_topics_into_db:
			truncate_topics_tables(resource=self.resource_identifier_name)

			print(" \t-> Topics will be saved in database for [ {} ]".format(
				self.resource_identifier_name))

			model_numbers_topics = self._get_topics()

			try:
				for topic_info in model_numbers_topics:
					tnum = topic_info[0]
					tresourceid = topic_info[1]
					tname = topic_info[2]

					_topic = {
						'ident_number': tnum,
						'value': tname,
						'created_at': dt.datetime.today().date()
					}

					t = Topics.create(**_topic)

					t_id = t.__data__['topic']

					_topic_resource = {
						'resource': tresourceid,
						'topic': t_id,
						'created_at': dt.datetime.today().date()
					}

					tr = TopicsResources.create(**_topic_resource)

				print("{}[ {} ]".format(SUCCESS_FLAG, self.resource_identifier_name))
			except Exception as e:
				print("{}{}".format(EXCEPTION_FLAG, e))
				print("{}Failure: [ {} ]".format(ERROR_FLAG, self.resource_identifier_name))

	def train_model(self, data_type,
	                resource,
	                single_ltc_data=None,
	                data_file_path=None,
	                train_corpus=None,
					train_dictionary=None,
					save_model_as=None,
					chunksize=LDA_CHUNKSIZE,
					passes=LDA_PASSES):

		if train_corpus is not None:
			corpus = train_corpus

		elif data_type == 'db':
			corpus = self._make_corpus(data_type=data_type, resource=resource)

		elif data_type == 'single_ltc' and single_ltc_data is not None:
			corpus = self._make_corpus(data_type=data_type, ltc=single_ltc_data, resource=resource)

		elif data_type == 'csv' and data_file_path is not None:
			corpus = self._make_corpus(data_type=data_type, data_file_path=data_file_path, resource=resource)

		else:
			raise Exception("{}Corpus is None".format(EXCEPTION_FLAG))

		if train_dictionary is not None:
			dictionary = train_dictionary
		else:
			dictionary = self.dictionary

		"""
			id2word parameter need to get words in topics instead of their indexes in dict
		"""
		_tcount = self.lda_topics_count

		# self.lda_model = LdaModel(corpus=corpus, num_topics=_tcount, id2word=dictionary, passes=passes, chunksize=chunksize)
		self.lda_model = LdaMulticore(corpus=corpus, num_topics=_tcount, id2word=dictionary, passes=passes,
									  chunksize=chunksize)

		if save_model_as is not None and not single_ltc_data:
			self.save_model(save_model_as, save_on_disk=True, save_topics_into_db=False)

		elif single_ltc_data:
			self.save_model(self.lda_model_filename, save_on_disk=True, save_topics_into_db=False)
		elif data_type == 'csv':
			self.save_model(self.lda_model_filename, save_on_disk=True, save_topics_into_db=False)

		else:
			self.save_model(self.lda_model_filename, save_on_disk=True, save_topics_into_db=True)

		print("{}Trained".format(SUCCESS_FLAG))

	def load_model(self, model_file_path=None, dict_file_path=None):
		"""
			load model and dictionary from file (need to save them in train function)
			uses to update model on another corpus
		"""

		if model_file_path is not None and os.path.exists(model_file_path):
			self.lda_model = LdaMulticore.load(model_file_path)
			# self.lda_model = LdaModel.load(model_file_path)
			self.dictionary = Dictionary.load(dict_file_path)
			print(" \t-> Loaded: [ {} ]".format(model_file_path))

		elif model_file_path is None and os.path.exists(self.lda_model_filename):
			self.lda_model = LdaMulticore.load(self.lda_model_filename)
			# self.lda_model = LdaModel.load(self.lda_model_filename)
			self.dictionary = Dictionary.load(self.lda_dict_filename)
			print(" \t-> Loaded: [ {} ]".format(self.lda_model_filename))

		else:
			print("{}Filepath you gave is incorrect. \n     Give another one and retry."
				  "\n     Exiting...".format(ERROR_FLAG))
			exit()

		for i in range(self.lda_model.num_topics):
			terms_id = self.lda_model.get_topic_terms(i, self.lda_topic_word_count)

			terms = [self.dictionary.get(x[0]) for x in terms_id]

			self.lda_topics.append(' '.join(terms))

	def update_model(self, ondata_file_path=None, resource=None, data_type='db'):
		if ondata_file_path is not None and data_type == 'csv':
			corpus = self._make_corpus(data_file_path=ondata_file_path, data_type=data_type, resource=resource)
		elif data_type == 'db':
			corpus = self._make_corpus(data_file_path=None, data_type=data_type, resource=resource)
		else:
			raise Exception("{}Corpus is None".format(EXCEPTION_FLAG))

		self.lda_model.update(corpus)

	def process_record(self, text, data_type):
		"""
			data_type - db / csv / single_ltc
		"""

		if data_type == 'single_ltc':
			try:
				self.load_model()
			except Exception as e:
				print("{}{}".format(EXCEPTION_FLAG, e))
				pass

		elif self.lda_model is None:

			try:
				self.load_model()
			except Exception as e:
				print("{}{}".format(EXCEPTION_FLAG, e))
				pass

		if data_type == 'db':
			if self.lda_model is None:
				return dict()

			doc = self._prepare_single_document(text)

			if doc is not None:
				topics = self._get_document_topics(doc)

				top_topic = topics[0]

				return [('topic', self.lda_topics[top_topic])]

			return [('topic', "")]

		elif data_type  == 'csv':
			doc = self._prepare_single_document(text)
			topics_in_count_by_ids = self._get_document_topics(doc)
			current_doc_topic_id, current_doc_other_topics = topics_in_count_by_ids[0], topics_in_count_by_ids[1:]

			result_topic_word_descr = re.sub('[^A-Za-z]+', ' ', self._get_topic_by_id(current_doc_topic_id))

			return [('topic', result_topic_word_descr),
			        ('other_topics', current_doc_other_topics)]

		elif data_type == 'single_ltc':
			doc = self._prepare_single_document(text)
			topics_in_count_by_ids = self._get_document_topics(doc)
			if topics_in_count_by_ids is not None:
				current_doc_topic_id, current_doc_other_topics = topics_in_count_by_ids[0], topics_in_count_by_ids[1:]

				result_topic_word_descr = re.sub('[^A-Za-z]+', ' ', self._get_topic_by_id(current_doc_topic_id))

				return result_topic_word_descr, current_doc_other_topics
			else:
				return "", []


	def _get_metric_fields(self):
		if self.lda_model is None:
			return []

		else:
			return ['topic']

	def _get_document_topics(self, doc, count=5):
		if doc is not None:
			bow = self.dictionary.doc2bow(doc)
			topics = self.lda_model.get_document_topics(bow, minimum_probability=0.0)
			topics_in_count = list(
				ident_number for (ident_number, prob) in sorted(topics, key=itemgetter(1), reverse=True)[:count])

			return topics_in_count

	def _get_document_topic(self, doc_topics):
		topic_id_probs = {}

		for t_prob in doc_topics:
			topic_id_probs[t_prob[0]] = t_prob[1]

		doc_topic_id = sorted(topic_id_probs, key=topic_id_probs.get, reverse=True)[0]
		doc_topic_prob = topic_id_probs[doc_topic_id]

		return [doc_topic_id, doc_topic_prob]

	def _prepare_single_document(self, sd):
		if sd is None or type(sd) == np.float:
			return None

		try:
			sd = sd.lower()
			sd = nltk.tokenize.word_tokenize(sd)
			sd = (word for word in sd if word.isalpha() and len(word) > 2)
			stopped_sd = (word for word in sd if word not in self.stop_words)

			lemmatized_doc = [self.wn_lemmatizer.lemmatize(word) for word in stopped_sd]

			return lemmatized_doc

		except AttributeError as e:
			print("{}{}".format(EXCEPTION_FLAG, e))
			return None

	def _make_bow(self, text):
		if text is not None:
			d = self._prepare_single_document(text)

			return self.dictionary.doc2bow(d)

	def _make_corpus(self, data_type, resource, data_file_path=None, save_train_dict=True, save_dict_as=None, ltc=None):
		"""
			data type can be csv or db # or new - single_ltc
		"""
		if data_type == 'db':
			documents = self.load_db_data(resource=resource)

		elif data_type == 'csv' and data_file_path is not None:
			documents = self.load_csv_data(data_file_path)

		elif data_type == 'single_ltc' and ltc is not None:

			ltc_text = " ".join(e if type(e) is str else "" for e in ltc)
			documents = self.load_single_ltc(ltc_text)

		else:
			documents = None

			print("{}documents is None. Exiting ... \n".format(ERROR_FLAG))
			exit()

		with Pool() as pool:
			processed_docs = pool.imap(self._prepare_single_document, documents)
			pool.close()
			pool.join()

		processed_docs = (i for i in processed_docs if i is not None)
		self.dictionary = Dictionary(processed_docs)

		if save_train_dict and save_dict_as is None:
			self.dictionary.save(self.lda_dict_filename)
		else:
			self.dictionary.save(save_dict_as)

		corpus = [self.dictionary.doc2bow(proc_doc) for proc_doc in processed_docs]

		return corpus

	def _get_topic_by_id(self, topic_id):
		if self.lda_topic_word_count is not None:
			return self.lda_model.print_topic(topic_id, self.lda_topic_word_count)

		else:
			return self.lda_model.print_topic(topic_id, 6)

	def _get_topics(self, default_view=False, for_db=True):
		"""
			2-tuples (probability * word) of most probable words in topics
			num_topics=-1 <--- to print all topics
		"""

		def _get_words(probabilities_words_string):
			_pre_topic_with_digits_trash = " ".join(re.findall(ALL_CHARS, probabilities_words_string))
			probaply_clean_topic = re.sub(r'\b\d+(?:\.\d+)?\s+', "", _pre_topic_with_digits_trash)

			return probaply_clean_topic  # " ".join(re.findall('[a-zA-Z]+', probabilities_words_string))

		if default_view:
			return self.lda_model.print_topics(num_topics=-1)

		if for_db:
			resource_id = Resources.select().where(Resources.resource == self.resource_identifier_name).first()
			resource_id = resource_id.__data__['resource']

			return [(elem[0], resource_id, _get_words(elem[1])) for elem in
					self.lda_model.print_topics(num_topics=self.lda_topics_count, num_words=self.lda_topic_word_count)]

		return [(elem[0], _get_words(elem[1])) for elem in self.lda_model.print_topics(num_topics=self.lda_topics_count, num_words=self.lda_topic_word_count)]


