import os
from copy import deepcopy

import numpy as np
from gensim.models import KeyedVectors


class Word2VecModel(object):
    def __init__(self, name, word2vec):
        self.name = name
        self.word2vec = word2vec
        self.vocab_size, self.vector_size = word2vec.syn0.shape

    @classmethod
    def load_model(cls, fname, fvocab=None, binary=True, name=None):
        if name is None:
            name = os.path.splitext(os.path.basename(fname))[0]
        word2vec = KeyedVectors.load_word2vec_format(
            fname, fvocab=fvocab, binary=binary)
        # normalize word2vec vectors
        word2vec.init_sims(replace=True)
        return cls(name=name, word2vec=word2vec)

    def save_model(self, directory, prefix='', save_vocab=True, binary=True):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if prefix == '':
            prefix = self.name
        fname = os.path.join(directory, '{}.bin'.format(prefix))
        fvocab = None
        if save_vocab:
            fvocab = os.path.join(directory, '{}.vocab'.format(prefix))
        self.word2vec.save_word2vec_format(fname, fvocab=fvocab, binary=binary)

    def get_vocab(self):
        return self.word2vec.vocab

    def get_id2word(self):
        return [word for (id, word) in sorted(
            [(v.index, word) for (word, v) in self.word2vec.vocab.items()])]

    def get_vector_matrix(self):
        return self.word2vec.syn0

    def set_vector_matrix(self, vectors):
        assert isinstance(vectors, np.ndarray), \
            'vectors must be a {} instance'.format(get_class_name(np.ndarray))
        assert vectors.shape == (self.vocab_size, self.vector_size), \
            'dimension of vectors {} mismatch with ({}, {})'.format(
                vectors.shape, self.vocab_size, self.vector_size)
        self.word2vec.syn0 = deepcopy(vectors)
        self.word2vec.syn0norm = self.word2vec.syn0

    def get_word_index(self, word):
        if word == '':
            return -1
        word_vocab = self.word2vec.vocab.get(word)
        if word_vocab is not None:
            return word_vocab.index
        else:
            return -1

    def get_word_vec(self, word):
        if word == '':
            return None
        try:
            return self.word2vec.word_vec(word)
        except KeyError:
            return None

    def get_index_vec(self, index):
        if index < 0 or index >= self.vocab_size:
            return None
        return self.word2vec.syn0[index]
