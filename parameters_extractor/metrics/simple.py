import nltk
from datetime import datetime
from nltk_data.stop_words_data.stop_word_processing import get_stop_words
from string import whitespace
from collections import Counter
from langdetect import detect


def avg(a, b):
    return a / b if b != 0 else 0


class SimpleMetricsCallback(object):
    sent_detector = nltk.tokenize.punkt.PunktSentenceTokenizer()

    def timedelta(self, creation_time):
        """ return days between the article publication
            and the dataset acquisition."""
        creation = datetime.strptime(creation_time[:19], '%Y-%m-%d %H:%M:%S')
        now = datetime.utcnow()
        delta = now - creation
        return delta.days

    @staticmethod
    def n_symbols(text, ignore_spaces=False):
        if ignore_spaces:
            return len([c for c in text if c not in whitespace])
        else:
            return len(text)

    @staticmethod
    def n_syllables(words):
        count = 0
        vowels = 'aeiouy'

        for word in words:
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1

        return count

    def n_sentences(self, text):
        return len(self.sent_detector.tokenize(text.strip()))

    @staticmethod
    def most_common_words(words, count=5):
        words = Counter(words)
        most_common = words.most_common(count)
        if most_common:
            return ', '.join('"{}": {}'.format(k, v) for k, v in most_common)
        else:
            return '-'

    def __call__(self, text):
        if text == "":
            return (
                ('n_symbols', 0),
                ('n_symbols_no_space', 0),
                ('n_syllables', 0),
                ('n_sentences', 0),
                ('n_tokens_content', 0),
                ('n_unique_tokens', 0),
                ('n_non_stop_words', 0),
                ('n_non_stop_unique_tokens', 0),
                ('average_sentence_length', 0),
                ('average_token_length', 0),
                ('average_token_length_syllables', 0),
                ('most_common_non_stop_words', 0),
            )

        try:
            text_lang = detect(text)
        except Exception as e:
            text_lang = 'en'

        n_symbols = self.n_symbols(text)
        n_symbols_no_space = self.n_symbols(text, ignore_spaces=True)
        n_sentences = self.n_sentences(text)
        words = [w for w in nltk.tokenize.word_tokenize(text) if w.isalpha()]

        if text_lang == 'de':
            self.stop_words = get_stop_words('de')
        else:
            # english stopwords by default
            self.stop_words = get_stop_words('en')

        non_stop_words = [word for word in words if word not in self.stop_words]
        n_syllables = self.n_syllables(words)

        return (
            ('n_symbols', n_symbols),
            ('n_symbols_no_space', n_symbols_no_space),
            ('n_syllables', n_syllables),
            ('n_sentences', n_sentences),
            ('n_tokens_content', len(words)),
            ('n_unique_tokens', len(set(words))),
            ('n_non_stop_words', len(non_stop_words)),
            ('n_non_stop_unique_tokens', len(set(non_stop_words))),
            ('average_sentence_length', avg(len(words), n_sentences)),
            ('average_token_length', avg(sum([len(word) for word in words]), len(words))),
            ('average_token_length_syllables', avg(n_syllables, len(words))),
            ('most_common_non_stop_words', self.most_common_words(non_stop_words)),
        )
