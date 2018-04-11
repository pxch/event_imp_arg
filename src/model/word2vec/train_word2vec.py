import argparse
import bz2
import os

import gensim

from utils import log


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in bz2.BZ2File(os.path.join(self.dirname, fname)):
                yield line.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train', required=True,
        help='Directory containing training text data for the model')
    parser.add_argument(
        '--output', required=True,
        help='File path to save the resulting word vectors')
    parser.add_argument(
        '--save_vocab',
        help='File path to save the vocabulary')
    parser.add_argument(
        '--sg', type=int, default=0, choices=[0, 1],
        help='Use skip-gram model; default is 0 '
             '(use continous bag of words model)')
    parser.add_argument(
        '--size', type=int, default=100,
        help='Size of word vectors; default is 100')
    parser.add_argument(
        '--window', type=int, default=5,
        help='Max skip length between words; default is 5')
    parser.add_argument(
        '--sample', type=float, default=1e-3,
        help='Rate with which higher frequency words would be randomly '
             'down-sampled; default is 1e-3, useful range is (0, 1e-5)')
    parser.add_argument(
        '--hs', type=int, default=0, choices=[0, 1],
        help='Use hierarchical Softmax; default is 0 (not used)')
    parser.add_argument(
        '--negative', type=int, default=5,
        help='Number of negative examples; default is 5, '
             'common values are 3 - 10 (0 = not used)')
    parser.add_argument(
        '--workers', type=int, default=3,
        help='Number of workers; default is 3')
    parser.add_argument(
        '--iter', type=int, default=5,
        help='Number of iterations; default is 5')
    parser.add_argument(
        '--min_count', type=int, default=5,
        help='Discard words that appear less than min-count times; '
             'default is 5')
    parser.add_argument(
        '--max_vocab_size', type=int, default=None,
        help='Maximum size of vocabulary; default is None (no limit)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.025,
        help='Starting learning rate; default is 0.025 for skip-gram '
             'and 0.05 for CBOW')
    parser.add_argument(
        '--binary', type=int, default=0, choices=[0, 1],
        help='Save the resulting vectors in binary mode; default is 0 (off)')

    opts = parser.parse_args()

    assert opts.size > 0, '--size must be a positive integer'
    assert opts.window > 0, '--window must be a positive integer'
    assert opts.sample > 0, '--sample must be a positive number'
    assert opts.hs == 1 or opts.negative > 0, \
        '--negative must be a positive integer when hs = 0'
    assert opts.workers > 0, '--workers must be a positive integer'
    assert opts.iter > 0, '--iter must be a positive integer'
    assert opts.min_count >= 0, '--min_count must be a non-negative integer'
    assert opts.max_vocab_size is None or opts.max_vocab_size > 0, \
        '--max_vocab_size must be None or a positive integer'
    assert opts.alpha > 0, '--alpha must be a positive number'

    log.info('Training word2vec model with parameters {}'.format(opts))

    log.info('Reading training data from {}'.format(opts.train))
    sentences = MySentences(opts.train)

    model = gensim.models.Word2Vec(
        sentences,
        sg=opts.sg,
        size=opts.size,
        window=opts.window,
        sample=opts.sample,
        hs=opts.hs,
        negative=opts.negative,
        workers=opts.workers,
        iter=opts.iter,
        min_count=opts.min_count,
        max_vocab_size=opts.max_vocab_size,
        alpha=opts.alpha
    )

    log.info('Outputting resulting word vectors to {}'.format(opts.output))
    if opts.save_vocab is not None:
        log.info('Outputting resulting word vocabulary to {}'.format(
            opts.save_vocab))

    model.wv.save_word2vec_format(
        opts.output, fvocab=opts.save_vocab, binary=opts.binary)
