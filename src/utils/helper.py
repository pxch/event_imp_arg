import os
from bz2 import BZ2File
from collections import Counter
from gzip import GzipFile
from itertools import dropwhile

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


def read_vocab_count(vocab_count_file):
    counter = Counter()
    with smart_file_handler(vocab_count_file, 'r') as fin:
        for line in fin.readlines():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word = parts[0]
                count = int(parts[1])
                counter[word] = count
    return counter


def write_vocab_count(counter, vocab_count_file, n_top=None):
    with smart_file_handler(vocab_count_file, 'w') as fout:
        for word, count in counter.most_common(n_top):
            fout.write('{}\t{}\n'.format(word, count))


def prune_counter(counter, thres=1):
    for word, count in dropwhile(
            lambda word_count: word_count[1] >= thres, counter.most_common()):
        del counter[word]


def read_vocab_list(vocab_list_file):
    vocab_list = []
    with smart_file_handler(vocab_list_file, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if line:
                vocab_list.append(line)
    return vocab_list


def write_vocab_list(counter, vocab_list_file, n_top=None):
    with smart_file_handler(vocab_list_file, 'w') as fout:
        for word, _ in counter.most_common(n_top):
            fout.write('{}\n'.format(word))


def suppress_fd(fd_number):
    assert fd_number in [1, 2]
    # open a null file descriptor
    null_fd = os.open(os.devnull, os.O_RDWR)
    # save the current stdout(1) / stderr(2) file descriptor
    save_fd = os.dup(fd_number)
    # put /dev/null fd on stdout(1) / stderr(2)
    os.dup2(null_fd, fd_number)

    return null_fd, save_fd


def restore_fd(fd_number, null_fd, save_fd):
    # restore stdout(1) / stderr(2) to the saved file descriptor
    os.dup2(save_fd, fd_number)
    # close the null file descriptor
    os.close(null_fd)
