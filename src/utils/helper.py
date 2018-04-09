from bz2 import BZ2File
from gzip import GzipFile

import consts
from logger import log


def get_class_name(class_type):
    return '{}.{}'.format(class_type.__module__, class_type.__name__)


def check_type(variable, class_type):
    assert isinstance(variable, class_type), \
        'expecting an instance of {}, {} found'.format(
            get_class_name(class_type), type(variable))


def escape(text, char_set=consts.escape_char_set):
    for char in char_set:
        if char in consts.escape_char_map:
            text = text.replace(char, consts.escape_char_map[char])
        else:
            log.warning('escape rule for {} undefined'.format(char))
    return text


def unescape(text, char_set=consts.escape_char_set):
    for char in char_set:
        if char in consts.escape_char_map:
            text = text.replace(consts.escape_char_map[char], char)
        else:
            log.warning('unescape rule for {} undefined'.format(char))
    return text


def convert_ontonotes_ner_tag(tag, to_corenlp=False):
    if to_corenlp:
        return consts.ontonotes_to_corenlp_mapping.get(tag, '')
    else:
        return consts.ontonotes_to_valid_mapping.get(tag, '')


def convert_corenlp_ner_tag(tag):
    return consts.corenlp_to_valid_mapping.get(tag, '')


def smart_file_handler(filename, mod='r'):
    if filename.endswith('bz2'):
        f = BZ2File(filename, mod)
    elif filename.endswith('gz'):
        f = GzipFile(filename, mod)
    else:
        f = open(filename, mod)
    return f
