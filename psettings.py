__author__ = 'MA573RWARR10R'
import sys
from termcolor import colored

QUIET = False

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

DEFAULT_DYNAMIC_TYPE = "comments_count"
DYNAMIC_TYPES = ["likes_count", "dislikes_count", "reposts_count", "comments_count"]
# DYNAMIC_TYPES_INDEXES = {
#     k: v for k, v in zip(DYNAMIC_TYPES, range(6, 11))
# }

LEAD_ELEMENT_TYPE = 1
TITLE_ELEMENT_TYPE = 2
CONTENT_ELEMENT_TYPE = 3

DATABASE_TYPE = 'mysql_local'
# DATABASE_TYPE = 'mysql_remote'

VALID_TEXT_LENGTH = 15
VALID_TEXT_WORDS_COUNT = 3
WORDS_PER_LINK = 5
WORDS_PER_LIST_MARKER = 5

KEYWORDS_COUNT = 8

LDA_CHUNKSIZE = 500
LDA_PASSES = 50
LDA_TOPIC_WORD_COUNT = 5
LDA_TOPICS_COUNT = 15

MINI_LDA_TOPIC_WORD_COUNT = 3
MINI_LDA_TOPICS_COUNT = 5
MINI_LDA_RESULT_TOPIC_WORD_COUNT = 3

R_PLUMBER_SERVICE_ADDRESS = "http://127.0.0.1:8000/"

R_EMOTION_COLORS_VALUES_COUNT = 20

WARNING_FLAG = colored(" [!  WARNING   !] ", 'yellow', None, ['bold'])
SUCCESS_FLAG = colored(" [!  SUCCESS   !] ", 'green', None, ['bold'])
ERROR_FLAG = colored(" [!  ERROR     !] ", 'red', None, ['bold'])
EXCEPTION_FLAG = colored(" [!  EXCEPTION !] ", 'magenta', None, ['bold'])
INFO_FLAG = colored(" [   INFO       ] ", 'white', None, ['bold'])

COMPARE_WITH_TOP_COUNT = 5
DISPLAYED_TOP_COUNT = 5

SUPPORTED_LANGUAGES = ['en', 'us', 'de']

TRENDS_SUPPORTED_GEOS = ['US', 'DE', 'GB']
TRENDS_TOP_PER_CATEGORY_LIMIT = 50
TRENDS_INITIAL_COUNT = 5
TRENDS_INITIAL_CATEGORY_NAME = 'all'

import os

DEFAULT_PROJECT_PATH = os.environ['EXPONENTA_PROJECT_PATH']

RT_TFIDF_PATH = DEFAULT_PROJECT_PATH + "relevant_trends/models_data/"
RT_W2V_PATH = DEFAULT_PROJECT_PATH + "relevant_trends/models_data/"

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
# RT_REDIS_EDITION_TRENDS_NOW_URL = 'http://127.0.0.1:6378/trends_now/'
# RT_REDIS_EDITION_TRENDS_NOW_RESULT_URL = 'http://127.0.0.1:6378/trends_result/'


CATEGORIES_SHORT = {
    'business': 'b',
    'entertainment': 'e',
    'sports': 's',
    'sci/tech': 't',
    'health': 'm',
}

SSL_CERT_PATH = "/home/ec2-user/keys/privkey.pem"
SSL_KEY_PATH = "/home/ec2-user/keys/fullchain.pem"
# SSL_CERT_PATH = "ssl_keys/cert.pem"
# SSL_KEY_PATH = "ssl_keys/key.pem"

EXPONENTA_DASHBOARD_NOISY_DEBUG = True
ONLY_CONTENT_RESOURCES = ['facebook.com']

if DEFAULT_PROJECT_PATH not in sys.path:
    sys.path.append(DEFAULT_PROJECT_PATH)
