from topics_extractor.lda_mw import LDAMWBase
from psettings import *


class LDAMWHandler:
	def __init__(self, mtype='multiple', resource=None):
		"""
			type = ['single_ltc', 'multiple']
		"""

		self.mtype = mtype
		self.VALID_LENGTH = VALID_TEXT_LENGTH

		if mtype == 'multiple':
			self.lda_mw_model = None

		elif mtype == 'single_ltc':
			self.lda_mw_model = LDAMWBase(mtype=mtype, resource=resource)

	@staticmethod
	def metric_fields():
		return ['topic']

	def check_validity_of_data(self, data_to_check):
		l, t, c = data_to_check

		if len(c) < self.VALID_LENGTH:
			return False
		return True

	def __call__(self, article_pack, resource=None, is_single_ltc=False, is_csv=False):
		if not is_single_ltc:
			if self.lda_mw_model is None or self.lda_mw_model.resource_identifier_name != resource:
				self.load(resource=resource)

			if is_csv:
				return self.lda_mw_model.process_record(article_pack, data_type='csv')
			else:
				return self.lda_mw_model.process_record(article_pack, data_type='db')

		else:
			is_data_valid = self.check_validity_of_data(article_pack)

			if is_data_valid:
				self.train(data_type='single_ltc', train_ltc_data=article_pack, resource=resource)

				single_record_ltc_data = list(self.lda_mw_model.process_record(elem, data_type='single_ltc') for elem in
											  article_pack)

				return single_record_ltc_data

			else:
				return None

	def load(self, resource):
		if resource is None:
			self.lda_mw_model = LDAMWBase()
		else:
			self.lda_mw_model = LDAMWBase(resource=resource)

		self.lda_mw_model.load_model()
		print("\n \t-> [ {} ] model loaded \n".format(resource))

	def train(self, data_type, resource=None, train_ltc_data=None, res_lang=None, csv_data_file_path=None):
		if data_type == 'single_ltc':
			if train_ltc_data is not None:
				self.lda_mw_model.train_model(data_type=data_type, single_ltc_data=train_ltc_data, resource=resource)
			else:
				return None

		elif data_type == 'db':
			if self.lda_mw_model is None:
				self.lda_mw_model = LDAMWBase(resource=resource)

			self.lda_mw_model.train_model(data_type=data_type, resource=resource)

		elif data_type == 'csv':
			if self.lda_mw_model is None:
				self.lda_mw_model = LDAMWBase(resource=resource, data_type=data_type, resource_language=res_lang)

			self.lda_mw_model.train_model(data_type=data_type, resource=resource, data_file_path=csv_data_file_path)
