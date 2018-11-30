from re import findall
import re
import numpy as np
from psettings import *


def _extract_list_markers(s):
    list_markers = list(findall(r'[0-9.]+.', s))
    return len(list_markers)


def _extract_links_count(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls)


def _check(text):
    stuff_describe_message = "Text is NULL."

    if text is None or type(text) == type(np.nan) or type(text) == type(np.float):
        return False, stuff_describe_message

    if len(text) < VALID_TEXT_LENGTH:
        stuff_describe_message = "Text is too short."
        return False, stuff_describe_message

    words_count = len(re.findall('[a-zA-Z]+', text))

    if words_count < VALID_TEXT_WORDS_COUNT:
        stuff_describe_message = "Insufficient words for analysis."
        return False, stuff_describe_message

    lmarkers_count = _extract_list_markers(text)
    links_count = _extract_links_count(text)

    if lmarkers_count != 0:
        if words_count / lmarkers_count < WORDS_PER_LIST_MARKER:
            stuff_describe_message = "Too much list markers in your text."
            return False, stuff_describe_message

    if links_count != 0:
        if words_count / links_count < WORDS_PER_LINK:
            stuff_describe_message = "Too much links in your text."
            return False, stuff_describe_message

    return True, ""


def is_text(publication, simple_text=None, as_tuple=False, as_tuple_content_index=None):
    if simple_text is None:
        if not as_tuple:
            to_check = publication.content
        else:
            to_check = publication[as_tuple_content_index]
    else:
        to_check = simple_text

    return _check(to_check)
