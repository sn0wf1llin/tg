import nltk
import numpy as np
from nltk.data import load
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import demo_subjectivity
from psettings import *
from langdetect import detect
from putils import my_print


class PolarityMetricsCallback(object):
    analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + np.exp(-x)) if x != 0 else 0

    def __call__(self, text):
        try:
            words = nltk.tokenize.word_tokenize(text)

            pos_valuences = list(filter(lambda x: x > 0, [self.analyzer.lexicon.get(word.lower(), 0) for word in words]))
            neg_valuences = list(filter(lambda x: x < 0, [self.analyzer.lexicon.get(word.lower(), 0) for word in words]))

            nwords, npos, nneg = len(words), len(pos_valuences), len(neg_valuences)

            polarity = self.analyzer.polarity_scores(text)

            return (
                ('global_negative_polarity', polarity['neg']),
                ('global_positive_polarity', polarity['pos']),
                ('global_neutral_polarity', polarity['neu']),
                ('global_sentiment_polarity', polarity['compound']),

                ('global_rate_positive_words', self._sigmoid(npos / nwords) if nwords else self._sigmoid(npos)),
                ('global_rate_negative_words', self._sigmoid(nneg / nwords) if nwords else self._sigmoid(nneg)),
                ('rate_positive_words', self._sigmoid(npos / (npos + nneg)) if npos or nneg else self._sigmoid(npos)),
                ('rate_negative_words', self._sigmoid(nneg / (npos + nneg)) if npos or nneg else self._sigmoid(nneg)),

                ('avg_positive_polarity',
                 self._sigmoid(sum(pos_valuences) / npos) if npos else self._sigmoid(sum(pos_valuences))),
                ('min_positive_polarity', self._sigmoid(min(pos_valuences)) if pos_valuences else 0.0),
                ('max_positive_polarity', self._sigmoid(max(pos_valuences)) if pos_valuences else 0.0),

                ('avg_negative_polarity',
                 self._sigmoid(sum(neg_valuences) / nneg) if nneg else self._sigmoid(sum(neg_valuences))),
                ('min_negative_polarity', self._sigmoid(min(neg_valuences)) if neg_valuences else 0.0),
                ('max_negative_polarity', self._sigmoid(max(neg_valuences)) if neg_valuences else 0.0),

            )
        except Exception as e:
            my_print("{}{}".format(EXCEPTION_FLAG, e))
            return (
                ('global_negative_polarity', 0),
                ('global_positive_polarity', 0),
                ('global_neutral_polarity', 0),
                ('global_sentiment_polarity', 0),

                ('global_rate_positive_words', 0),
                ('global_rate_negative_words', 0),
                ('rate_positive_words', 0),
                ('rate_negative_words', 0),

                ('avg_positive_polarity', 0),
                ('min_positive_polarity', 0),
                ('max_positive_polarity', 0),

                ('avg_negative_polarity', 0),
                ('min_negative_polarity', 0),
                ('max_negative_polarity', 0),

            )


# class PolarityMetricsWord2VecCallback(object):
#     DATA_FOLDER = "models_data"
#     MODEL_PATH = os.path.join(DATA_FOLDER, "glove.twitter.27B.100d.txt")
#
#     NEUTRAL_DELTA = 0.05
#
#     GOOD_WORD = "excellent"
#     BAD_WORD = "horrible"
#
#     def __init__(self):
#         self.model = Word2Vec.load_word2vec_format(self.MODEL_PATH, binary=False)
#         self.tokenizer = nltk.treebank.TreebankWordTokenizer()
#
#     def _filter_unknown_words(self, words):
#         return [word for word in words if word in self.model.vocab]
#
#     def _tokenize(self, text):
#         # return [word.lower() for word in nltk.tokenize.word_tokenize(text)]
#         return [word.lower() for word in self.tokenizer.tokenize(text)]
#
#     def _compute_polarity(self, word):
#         similarity_with_good = self.model.similarity(word, self.GOOD_WORD)
#         similarity_with_bad = self.model.similarity(word, self.BAD_WORD)
#         return similarity_with_good - similarity_with_bad
#
#     def extract_polarities(self, text):
#         words = self._tokenize(text)
#         words = self._filter_unknown_words(words)
#         return [self._compute_polarity(word) for word in words]
#
#     def extract_features(self, text):
#         polarities = self.extract_polarities(text)
#         if not polarities:
#             return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
#
#         pos_polarities, neu_polarities, neg_polarities = [], [], []
#         for polarity in polarities:
#             if polarity > self.NEUTRAL_DELTA:
#                 pos_polarities.append(polarity)
#             elif polarity < -self.NEUTRAL_DELTA:
#                 neg_polarities.append(polarity)
#             else:
#                 neu_polarities.append(polarity)
#
#         pos_sum, neg_sum, neu_count = sum(pos_polarities), sum(neg_polarities), len(neu_polarities)
#
#         compound = sum(polarities)
#         total = pos_sum + math.fabs(neg_sum)
#         if total != 0:
#             pos = math.fabs(pos_sum / total)
#             neg = math.fabs(neg_sum / total)
#         else:
#             pos, neg = 0, 0
#         neu = math.fabs(neu_count / len(polarities))
#
#         sentiment_dict = \
#             {"neg": round(neg, 3),
#              "neu": round(neu, 3),
#              "pos": round(pos, 3),
#              "compound": round(compound, 4)}
#
#         return sentiment_dict
#
#     def __call__(self, row):
#         try:
#             polarities = self.extract_polarities(row['content'])
#
#             pos_polarities = list(filter(lambda x: x > 0, polarities))
#             neg_polarities = list(filter(lambda x: x < 0, polarities))
#
#             n_all, n_pos, n_neg = len(polarities), len(pos_polarities), len(neg_polarities)
#
#             title_polarity = self.extract_features(row['title'])
#             content_polarity = self.extract_features(row['content'])
#             return (
#                 ('global_negative_polarity', content_polarity['neg']),
#                 ('global_positive_polarity', content_polarity['pos']),
#                 ('global_neutral_polarity', content_polarity['neu']),
#                 ('global_sentiment_polarity', content_polarity['compound']),
#
#                 ('global_rate_positive_words', n_pos / n_all),
#                 ('global_rate_negative_words', n_neg / n_all),
#                 ('rate_positive_words', n_pos / (n_pos + n_neg) if n_pos or n_neg else 0),
#                 ('rate_negative_words', n_neg / (n_pos + n_neg) if n_pos or n_neg else 0),
#
#                 ('avg_positive_polarity', sum(pos_polarities) / n_pos if n_pos else 0),
#                 ('min_positive_polarity', min(pos_polarities) if pos_polarities else 0),
#                 ('max_positive_polarity', max(pos_polarities) if pos_polarities else 0),
#
#                 ('avg_negative_polarity', sum(neg_polarities) / n_neg if n_neg else 0),
#                 ('min_negative_polarity', min(neg_polarities) if neg_polarities else 0),
#                 ('max_negative_polarity', max(neg_polarities) if neg_polarities else 0),
#
#                 ('title_negative_polarity', title_polarity['neg']),
#                 ('title_positive_polarity', title_polarity['pos']),
#                 ('title_neutral_polarity', title_polarity['neu']),
#                 ('title_sentiment_polarity', title_polarity['compound']),
#             )
#         except Exception as e:
#             print(e)
#             return (
#                 ('global_negative_polarity', 0),
#                 ('global_positive_polarity', 0),
#                 ('global_neutral_polarity', 0),
#                 ('global_sentiment_polarity', 0),
#
#                 ('global_rate_positive_words', 0),
#                 ('global_rate_negative_words', 0),
#                 ('rate_positive_words', 0),
#                 ('rate_negative_words', 0),
#
#                 ('avg_positive_polarity', 0),
#                 ('min_positive_polarity', 0),
#                 ('max_positive_polarity', 0),
#
#                 ('avg_negative_polarity', 0),
#                 ('min_negative_polarity', 0),
#                 ('max_negative_polarity', 0),
#
#                 ('title_negative_polarity', 0),
#                 ('title_positive_polarity', 0),
#                 ('title_neutral_polarity', 0),
#                 ('title_sentiment_polarity', 0),
#             )


def get_subjectivity_analyzer(lang):
    try:
        sa_subj_data_file_path = 'nltk_data/sa_subjectivity.pickle'

        sentim_analyzer = load(DEFAULT_PROJECT_PATH + sa_subj_data_file_path)

    except LookupError:
        my_print('{}Cannot find the sentiment analyzer you want to load.'.format(WARNING_FLAG))
        my_print('{}Training & save a new one using NaiveBayesClassifier.'.format(WARNING_FLAG))

        sentim_analyzer = demo_subjectivity(NaiveBayesClassifier.train, True)

    return sentim_analyzer


class SubjectivityCallback(object):

    def __call__(self, text):
        try:
            lang = detect(text)

            self.analyzer = get_subjectivity_analyzer(lang)

            words = [word.lower() for word in nltk.tokenize.word_tokenize(text)]
            # words = [word.lower() for word in nltk.tokenize.word_tokenize(text, lang)]

            return (
                ('global_subjectivity', int(self.analyzer.classify(words) == 'obj')),
            )
        except Exception as e:
            return (
                ('global_subjectivity', 0),
            )
