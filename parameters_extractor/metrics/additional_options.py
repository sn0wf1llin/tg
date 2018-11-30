__author__ = 'MA573RWARR10R'
import nltk
from textstat.textstat import *
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import requests
import json
import numpy as np
from operator import itemgetter

from psettings import *

BAD_VALUES = [float('inf'), float('-inf'), float('nan')]


def avg(a, b):
	return a / b if b != 0 else 0


class AdditionalMetricsCallback:
	def __init__(self, R_plumber_service_address=None):

		if R_plumber_service_address is None:
			self.R_plumber_service_address = R_PLUMBER_SERVICE_ADDRESS
		else:
			self.R_plumber_service_address = R_plumber_service_address

	def n_sentences(self, text, min_words_count_per_sentence, details=False):
		sent_detector = nltk.tokenize.punkt.PunktSentenceTokenizer()
		cleared = ((1, self.n_words(sent)) if self.n_words(sent) > min_words_count_per_sentence else (0, 0) for sent in
				   sent_detector.tokenize(text.strip()))

		if details:
			return list(i[1] for i in cleared)

		return sum(i[0] for i in cleared)

	@staticmethod
	def n_symbols(text, numbers=False, letters=False, symbols=False, others=False):
		if numbers:

			return sum(c.isdigit() for c in text)
		elif letters:

			return sum(c.isalpha() for c in text)
		elif symbols:

			return sum(c.isalpha() for c in text) + sum(c.isspace() for c in text)
		elif others:
			numbers = sum(c.isdigit() for c in text)
			words = sum(c.isalpha() for c in text)
			spaces = sum(c.isspace() for c in text)
			others = len(text) - numbers - words - spaces

			return others
		else:
			return None

	@staticmethod
	def n_words(text):
		tokenizer = RegexpTokenizer(r'\w+')

		return len(tokenizer.tokenize(text))

	def types_tokens_on_R(self, txt, url_to='typestokens'):
		enames = ["count_of_tokens_text", "count_of_types_text"]

		if (type(txt) is str) and (len(txt) > 0):
			url_to = self.R_plumber_service_address + url_to

			data = json.dumps({'txt': txt})
			result = requests.post(url_to, data)
			result = json.loads(result.text)

			return {k: v for (k, v) in zip(enames, result)}

		return {k: v for (k, v) in zip(enames, [0 for i in range(len(enames))])}

	@staticmethod
	def check_inf_field(d, field_name, from_list, to_list):
		for f, t in zip(from_list, to_list):
			if d[field_name] == f:
				d[field_name] = t

		return d

	def metrics_on_R(self, txt, url_to='metrics'):
		url_to = self.R_plumber_service_address + url_to
		data = json.dumps({'txt': txt})
		result = requests.post(url_to, data)
		result = json.loads(result.text)

		enames = ["DB1", "DB2", "DB_grade", "DS", "Herdan", "CTTR", "HDD", "yueles_K", "maas_1", "MTLD", "RLd", "SLd",
				  "uber", "growth_vocabl"]

		metrics = {k: v for (k, v) in zip(enames, result)}
		metrics = self.check_inf_field(metrics, 'uber', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])
		metrics = self.check_inf_field(metrics, 'MTLD', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])
		metrics = self.check_inf_field(metrics, 'RLd', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])
		metrics = self.check_inf_field(metrics, 'SLd', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])
		metrics = self.check_inf_field(metrics, 'maas_1', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])
		metrics = self.check_inf_field(metrics, 'yueles_K', from_list=['-Inf', 'Inf'], to_list=[-1e10, 1e10])

		return metrics

	def emotions_on_R(self, txt, url_to='emotions'):
		url_to = self.R_plumber_service_address + url_to

		enames = ["words_diff_emotions", "angry", "anticipation", "disgust", "fear", "joy",
				  "sadness", "surprise", "trust", "neg", "pos"]
		if len(txt) != 0:
			data = json.dumps({'txt': txt})
			result = requests.post(url_to, data)
			result = json.loads(result.text)

			return {k: v for (k, v) in zip(enames, result)}

		return {k: v for (k, v) in zip(enames, [0 for i in range(len(enames))])}

	# def all_on_R(self, lead, title, content, url_to='sum'):
	#
	#     url_to = self.R_all_service_address + url_to
	#
	#     ltc_data = {
	#         'leadd': lead,
	#         'titled': title,
	#         'contentd': content,
	#     }
	#
	#     emotions_as_str = requests.post(url=url_to, data=ltc_data).text
	#
	#     return json.loads(emotions_as_str[1:-1])

	@staticmethod
	def get_most_freq_term(txt, with_antipode=False):
		words = [w for w in nltk.tokenize.word_tokenize(txt) if w.isalpha()]
		freqs = nltk.FreqDist(words)
		freqs = sorted(freqs.items(), key=itemgetter(1), reverse=True)

		if not with_antipode:
			try:
				return freqs[0]
			except IndexError:
				return ('-', 0)

		return freqs[0], freqs[-1]

	@staticmethod
	def get_max_term_length(txt):
		words_lengths = [(w, len(w)) for w in nltk.tokenize.word_tokenize(txt) if w.isalpha()]

		try:
			return sorted(words_lengths, key=itemgetter(1), reverse=True)[0][1]
		except IndexError as e:
			return 0

	def get_additional_for(self, element):
		try:
			n_element_symbols = self.n_symbols(element, symbols=True)
			n_element_numbers = self.n_symbols(element, numbers=True)
			n_element_letters = self.n_symbols(element, letters=True)
			n_element_others = self.n_symbols(element, others=True)
			n_element_words = self.n_words(element)

			try:
				n_element_mean_letters_in_words = n_element_letters / float(n_element_words)
			except ZeroDivisionError:
				n_element_mean_letters_in_words = 0.0

			element_emotionality = self.emotions_on_R(element)
			if element_emotionality['words_diff_emotions'] == 'error':
				raise Exception("R returns 'error' ... ")

			most_frequent_element_word_len, most_frequent_element_word_count = self.get_most_freq_term(element)

			element_max_term_length = self.get_max_term_length(element)

			return n_element_symbols, n_element_numbers, n_element_letters, n_element_others, n_element_words, \
				   n_element_mean_letters_in_words, element_emotionality, most_frequent_element_word_len, \
				   most_frequent_element_word_count, element_max_term_length

		except Exception as e:
			return 0, \
				   0, \
				   0, \
				   0, \
				   0, \
				   0, \
				   {
					   'words_diff_emotions': 0,
					   'angry': 0,
					   'anticipation': 0,
					   'disgust': 0,
					   'fear': 0,
					   'joy': 0,
					   'sadness': 0,
					   'surprise': 0,
					   'trust': 0,
					   'neg': 0,
					   'pos': 0
				   }, \
				   '', \
				   0, \
				   0

	def count_words_per_sentence(self, txt, min_words_count_per_sentence=1):
		if txt == "":
			return {
				'number_of_sentences': 0,
				'wps': 0,
				'mean': 0,
				'total_words_count': 0
			}

		words_count = self.n_words(txt)
		sent_count = self.n_sentences(txt, min_words_count_per_sentence)
		words_per_sentence = self.n_sentences(txt, min_words_count_per_sentence, True)
		mean_wps = np.mean(words_per_sentence)

		return {
			'number_of_sentences': sent_count,
			'wps': words_per_sentence,
			'mean': mean_wps,
			'total_words_count': words_count
		}

	def get_text_features(self, txt):
		if type(txt) is not str or len(txt) == 0:
			return 0, 0, 0, 0, 0

		uniqWd = self.get_unique_words(txt, count_only=True)

		try:
			complx = uniqWd / float(self.n_words(txt))
		except ZeroDivisionError:
			complx = -1

		sntCt = textstat.sentence_count(txt)

		try:
			sntLen = uniqWd / float(sntCt)
		except ZeroDivisionError:
			sntLen = -1

		try:
			syllCt = textstat.syllable_count(txt) / float(self.n_words(txt))
		except ZeroDivisionError:
			syllCt = -1
		# charCt = 0
		# lttrCt = 0
		# FOG = textstat.gunning_fog(txt)
		try:
			flesch = textstat.flesch_reading_ease(txt)
		except Exception:
			flesch = -1

		return uniqWd, complx, sntLen, syllCt, flesch

	def get_unique_words(self, txt, count_only=False):
		words = [w for w in nltk.tokenize.word_tokenize(txt)]

		if count_only:
			return len(Counter(words))

		return Counter(words)

	def __call__(self, l, t, c):
		"""
			Process title
		"""

		# emotionality -- dict with keys ["words_diff_emotions", "angry", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "neg", "pos"]

		n_title_symbols, n_title_numbers, n_title_letters, n_title_others, n_title_words, n_title_mean_letters_in_words, \
		title_emotionality, most_frequent_title_word_len, most_frequent_title_word_count, \
		title_max_term_length = self.get_additional_for(t)

		"""
			Process lead
		"""

		n_lead_symbols, n_lead_numbers, n_lead_letters, n_lead_others, n_lead_words, n_lead_mean_letters_in_words, \
		lead_emotionality, most_frequent_lead_word_len, most_frequent_lead_word_count, \
		lead_max_term_length = self.get_additional_for(l)

		"""
			Process content
		"""
		content_dots_count = c.count('.')
		content_commas_count = c.count(',')
		content_exclamation_marks_count = c.count('!')
		content_question_marks_count = c.count('?')

		n_content_symbols, n_content_numbers, n_content_letters, n_content_others, n_content_words, n_content_mean_letters_in_words, \
		content_emotionality, most_frequent_content_word_len, most_frequent_content_word_count, \
		content_max_term_length = self.get_additional_for(c)

		countwordpersentence = self.count_words_per_sentence(c)

		content_sentences_count = countwordpersentence['number_of_sentences']
		content_wps_counts = countwordpersentence['wps']
		content_mean_words_count = countwordpersentence['mean']
		content_total_words_count = countwordpersentence['total_words_count']

		max_count_words_in_sent_content = max(content_wps_counts)
		min_count_words_in_sent_content = min(content_wps_counts)

		# max & min freqs of the text
		_max_min_f_res = self.get_most_freq_term(c, with_antipode=True)
		max_freq_of_term, min_freq_of_term = _max_min_f_res[0][1], _max_min_f_res[1][1]
		max_term_length_content = len(_max_min_f_res[1][0])

		_content_emo_vector_values = [content_emotionality[k] for k in
									  ["words_diff_emotions", "angry", "anticipation", "disgust", "fear", "joy",
									   "sadness", "surprise", "trust", "neg", "pos"]]

		content_sum_emotionality = sum(_content_emo_vector_values)

		content_mean_emotionality = np.mean(_content_emo_vector_values)

		content_max_emotionality_sentences = max(_content_emo_vector_values)

		content_min_emotionality_sentences = min(_content_emo_vector_values)

		try:
			content_mean_emo_of_sentences = content_sum_emotionality / float(content_sentences_count)
		except ZeroDivisionError:
			content_mean_emo_of_sentences = -1

		# -------------------------------
		# Features for content
		# -------------------------------

		content_uniqWd, content_complx, content_sntLen, content_syllCt, content_flesch = self.get_text_features(c)

		# -------------------------------
		# Features for title
		# -------------------------------

		title_uniqWd, title_complx, title_sntLen, title_syllCt, title_flesch = self.get_text_features(t)

		# -------------------------------
		# Features for lead
		# -------------------------------

		lead_uniqWd, lead_complx, lead_sntLen, lead_syllCt, lead_flesch = self.get_text_features(l)

		# -------------------------------
		# ARI for content
		# -------------------------------

		content_ARI = textstat.automated_readability_index(c)

		# -------------------------------
		# ARI for title
		# -------------------------------

		title_ARI = textstat.automated_readability_index(t)

		# -------------------------------
		# ARI for lead
		# -------------------------------

		if type(l) is not str or len(l) == 0:
			lead_ARI = None
		else:
			lead_ARI = textstat.automated_readability_index(l)

		# -------------------------------
		# Coleman for content
		# -------------------------------

		content_coleman = textstat.coleman_liau_index(c)

		# -------------------------------
		# Daniel Brawn & others
		# for content from R
		# -------------------------------

		_content_metrics_R = self.metrics_on_R(c)
		content_DB1, content_DB2, content_DB_grade, content_DS, content_Herdan, content_CTTR, content_HDD, \
		content_yueles_K, content_maas_1, content_MTLD, content_Rld, content_Sld, content_uber, \
		content_growth_vocabl = (_content_metrics_R[k] for k in
								 ["DB1", "DB2", "DB_grade", "DS", "Herdan", "CTTR", "HDD", "yueles_K", "maas_1", "MTLD",
								  "RLd", "SLd", "uber", "growth_vocabl"])

		content_TTR, lead_TTR, title_TTR = content_complx, lead_complx, title_complx

		_content_types_tokens = self.types_tokens_on_R(c)
		content_count_of_types, content_count_of_tokens = (_content_types_tokens[k] for k in
														   ["count_of_tokens_text", "count_of_types_text"])

		_lead_types_tokens = self.types_tokens_on_R(l)
		lead_count_of_types, lead_count_of_tokens = (_lead_types_tokens[k] for k in
													 ["count_of_tokens_text", "count_of_types_text"])

		_title_types_tokens = self.types_tokens_on_R(t)
		title_count_of_types, title_count_of_tokens = (_title_types_tokens[k] for k in
													   ["count_of_tokens_text", "count_of_types_text"])

		return (
			("n_title_symbols", n_title_symbols),
			("n_title_numbers", n_title_numbers),
			("n_title_letters", n_title_letters),
			("n_title_words", n_title_words),
			("n_title_mean_letters_in_words", n_title_mean_letters_in_words),
			("title_words_diff_emotions", title_emotionality["words_diff_emotions"]),
			("title_angry", title_emotionality["angry"]),
			("title_anticipation", title_emotionality["anticipation"]),
			("title_disgust", title_emotionality["disgust"]),
			("title_fear", title_emotionality["fear"]),
			("title_joy", title_emotionality["joy"]),
			("title_sadness", title_emotionality["sadness"]),
			("title_surprise", title_emotionality["surprise"]),
			("title_trust", title_emotionality["trust"]),
			("title_neg", title_emotionality["neg"]),
			("title_pos", title_emotionality["pos"]),
			("most_frequent_title_word_len", len(most_frequent_title_word_len)),
			("most_frequent_title_word_count", most_frequent_title_word_count),
			("title_max_term_length", title_max_term_length),
			("n_lead_symbols", n_lead_symbols),
			("n_lead_numbers", n_lead_numbers),
			("n_lead_letters", n_lead_letters),
			("n_lead_words", n_lead_words),
			("n_lead_mean_letters_in_words", n_lead_mean_letters_in_words),
			("lead_words_diff_emotions", lead_emotionality["words_diff_emotions"]),
			("lead_angry", lead_emotionality["angry"]),
			("lead_anticipation", lead_emotionality["anticipation"]),
			("lead_disgust", lead_emotionality["disgust"]),
			("lead_fear", lead_emotionality["fear"]),
			("lead_joy", lead_emotionality["joy"]),
			("lead_sadness", lead_emotionality["sadness"]),
			("lead_surprise", lead_emotionality["surprise"]),
			("lead_trust", lead_emotionality["trust"]),
			("lead_neg", lead_emotionality["neg"]),
			("lead_pos", lead_emotionality["pos"]),
			("most_frequent_lead_word_len", len(most_frequent_lead_word_len)),
			("most_frequent_lead_word_count", most_frequent_lead_word_count),
			("lead_max_term_length", lead_max_term_length),
			("content_dots_count", content_dots_count),
			("content_commas_count", content_commas_count),
			("content_exclamation_marks_count", content_exclamation_marks_count),
			("content_question_marks_count", content_question_marks_count),
			("n_content_symbols", n_content_symbols),
			("n_content_numbers", n_content_numbers),
			("n_content_letters", n_content_letters),
			("n_content_words", n_content_words),
			("n_content_mean_letters_in_words", n_content_mean_letters_in_words),
			("content_mean_words_count", content_mean_words_count),
			("content_sentences_count", content_sentences_count),
			("max_count_words_in_sent_content", max_count_words_in_sent_content),
			("min_count_words_in_sent_content", min_count_words_in_sent_content),
			("content_total_words_count", content_total_words_count),
			("max_freq_of_term_in_content", max_freq_of_term),
			("min_freq_of_term_in_content", min_freq_of_term),
			("max_term_length_content", max_term_length_content),
			("content_sum_emotionality", content_sum_emotionality),
			("content_mean_emotionality", content_mean_emotionality),
			("content_max_emotionality_sentences", content_max_emotionality_sentences),
			("content_min_emotionality_sentences", content_min_emotionality_sentences),
			("content_mean_emo_of_sentences", content_mean_emo_of_sentences),
			("content_angry", content_emotionality["angry"]),
			("content_anticipation", content_emotionality["anticipation"]),
			("content_disgust", content_emotionality["disgust"]),
			("content_fear", content_emotionality["fear"]),
			("content_joy", content_emotionality["joy"]),
			("content_sadness", content_emotionality["sadness"]),
			("content_surprise", content_emotionality["surprise"]),
			("content_trust", content_emotionality["trust"]),
			("content_neg", content_emotionality["neg"]),
			("content_pos", content_emotionality["pos"]),
			("title_uniq_wd", title_uniqWd),
			("title_complx", title_complx),
			("title_snt_len", title_sntLen),
			("title_syll_ct", title_syllCt),
			("title_flesch", title_flesch),
			("lead_uniq_wd", lead_uniqWd),
			("lead_complx", lead_complx),
			("lead_snt_len", lead_sntLen),
			("lead_syll_ct", lead_syllCt),
			("lead_flesch", lead_flesch),
			("content_ari", content_ARI),
			("title_ari", title_ARI),
			("lead_ari", lead_ARI),
			("content_coleman", content_coleman),
			("content_db1", content_DB1),
			("content_db2", content_DB2),
			("content_db_grade", content_DB_grade),
			("content_ds", content_DS),
			("content_herdan", content_Herdan),
			("content_cttr", content_CTTR),
			("content_hdd", content_HDD),
			("content_yueles_k", content_yueles_K),
			("content_maas_1", content_maas_1),
			("content_mtld", content_MTLD),
			("content_rld", content_Rld),
			("content_sld", content_Sld),
			("content_ttr", content_TTR),
			("title_ttr", title_TTR),
			("lead_ttr", lead_TTR),
			("content_count_of_types", content_count_of_types),
			("content_count_of_tokens", content_count_of_tokens),
			("title_count_of_types", title_count_of_types),
			("title_count_of_tokens", title_count_of_tokens),
			("lead_count_of_types", lead_count_of_types),
			("lead_count_of_tokens", lead_count_of_tokens),
			("content_uber", content_uber),
			("content_growth_vocabl", content_growth_vocabl),
		)

# if __name__ == "__main__":
#     t = "EXCLUSIVE: PDP in fresh crisis crisis as .Makarfi dissolves Anambra excos ahead of election."
#     l = "EXCLUSIVE: Fresh crisis hits PDP as. Makarfi dissolves Anambra excos ahead of election."
#     c = "- There is a fresh crisis within the largest opposition party Peoples Democratic Party (PDP)   - The crisis started after the Ahmed Makarfi caretaker committee dissolved the Anambra state PDP officials   - The crisis, NAIJ.com gathered, is not unrelated to the forthcoming gubernatorial election in state   A fresh crisis on Thursday, July 27, hit Nigeria&amp;#39;s largest opposition party Peoples Democratic Party following the decision of the Ahmed Makarfi caretaker committee to dissolve Anambra party officials. The caretaker committee successfully dissolved the Ken Emekayi-led officials in Anambra state which has started brewing fresh crisis within the party.  NAIJ.com gathered that the crisis cannot be unrelated to the forthcoming Anambra state gubernatorial election on November 18. It was also gathered that the House of Representatives caucus , prior to the crisis, warned the Makarfi-led caretaker committee against its decision to dissolve the executives in the state.  READ ALSO:  We will show you what we are made of on November 18 - IPOB mocks Obi of Onitsha over statement concerning Anambra elections-boycott   The House caucus in a letter dated July 27, said the dissolution of the duly election Anambra chapter officials would be counter-productive ahead of election in the state. The House while congratulating Makarfi on the party&amp;#39;s victory at the Supreme Court said it cannot feign ignorance of  bickering in certain quarters of the party which could plunge the PDP into another level of crisis. The caucus said it would be a wrong step taken in the wrong direction if the &amp;quot;first litmus test or key assignment facing the reinforced National caretaker committee in the Anambra state gubernatorial election&amp;quot; is mismanaged.  READ ALSO:  Presidential aide reveals root cause of Buhari&rsquo;s illness   The caucus letter seen by NAIJ.com read in parts: &amp;quot;That since 2003, Anambra state PDP has been running a battle over whether the PDP as a party should be privatized or solely owned by a power broker or  group thereof resulting to internal wrangling, conflicts bickering, indiscipline and anti-party activities of different dimension.&amp;quot;  The caucus also warned that any move towards dissolution of the officials in the state would position the party as a privatized venture. It also warned that constituting a caretaker committee in the state would &amp;quot;raise dust that would not settle soon, divide the party and create an undue advantage to our adversaries&amp;quot;. The caucus while expressing worries that it was not carried along said the decision of the Makarfi-led committee is &amp;quot;causing bad blood in the party&amp;quot;.   READ ALSO:  JUST IN: Buratai, Olonishakin get relocation order from Osinbajo over Boko Haram   The PDP House members further added that: &amp;quot;As a caucus, it is our humble call and prayer to all who love and support our great party to sheath their swords as election time is not best favourable time to actualize a dissolution and or a caretaker, but a time for forgiveness, and reconciliation, unity, re-mobilization and onward together to win the election ahead.&amp;quot; Meanwhile following the sack of the officials, a source within the PDP told NAIJ.com that there were jubilation in the Anambra state Government House over the news of the dissolution of the state party executive. The source said the jubilation mostly from members of the All Progressive Grand Alliance (APGA) has raised suspicions that the executives could have been sabotaged in favour of the incumbent governor of the state Willie Obiano who is also in the race for the governorship seat.  PAY ATTENTION:  Read the news on Nigeria&rsquo;s #1 new app  The source said: &amp;quot;The jubilation is raising suspicions that the dissolution may have been hatched in collaboration with Anambra Government to sabotage the electoral fortunes of the PDP to favour Governor Willie Obiano.&amp;quot;   NAIJ.com earlier reported that former President Goodluck Jonathan had said Nigerians still believe in the PDP.  Jonathan while speaking at the party&amp;#39;s first meeting after the Supreme Court ruling against PDP&amp;#39;s national chairman Ali Modu Sheriff said despite its challenges within the past 14 months, it is heart warming to know that Nigerians still trust the largest opposition party. He also called for party members  to work together towrads ensuring that the PDP is strengthened to take over power from the ruling All Progressives Congress in 2019. You can watch this NAIJ.com video of PDP women lamenting the state of the nation:"
#
#     ad = AdditionalMetricsCallback()
#     p = ad(l, t, c)
#     print(len(p))
#     for (k, v) in p:
#         print(f"{v is None} -- {v} {k}")
