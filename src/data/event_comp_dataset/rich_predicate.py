import math
import random

from common.event_script import Predicate
from utils import Word2VecModel, check_type, consts


class RichPredicate(object):
    def __init__(self, word, neg=False, prt=''):
        self.word = word
        self.neg = neg
        self.prt = prt
        # list of candidates to lookup the Word2Vec vocabulary
        self.candidates = []
        # word2vec index of the predicate
        self.wv = -1

    @classmethod
    def build(cls, pred, use_lemma=True):
        check_type(pred, Predicate)
        word = pred.get_representation(use_lemma=use_lemma)
        return cls(word, pred.neg, pred.prt)

    def get_candidates(self):
        if not self.candidates:
            # add just the verb to the list
            self.candidates = [self.word]
            # add verb_prt to the list if the particle exists
            if self.prt:
                self.candidates.append(self.word + '_' + self.prt)
            # add not_verb and not_verb_prt to the list if there exists
            # a negation relation to the predicate
            if self.neg:
                for idx in range(len(self.candidates)):
                    self.candidates.append('not_' + self.candidates[idx])
            # reverse the list, so now the order of candidates becomes:
            # 1) not_verb_prt (if both negation and particle exists)
            # 2) not_verb (if negation exists)
            # 3) verb_prt (if particle exists)
            # 4) verb
            self.candidates.reverse()
            # append the UNK token the list of candidates in case none of
            # the above can be found in the vocabulary
            # candidates.append('UNK')
        return self.candidates

    def get_index(self, model, include_type=True, use_unk=True,
                  pred_count_dict=None):
        # TODO: add logic to process stop predicates
        check_type(model, Word2VecModel)
        candidates = self.get_candidates()
        # add UNK to the candidates if use_unk is set to True
        if use_unk:
            candidates.append('UNK')

        # drop the predicate (return index -1) if its frequency is too high
        # use the threshold of count as consts.PRED_COUNT_THRES (100,000)
        if candidates and pred_count_dict:
            pred_count = pred_count_dict.get(candidates[0], 0)
            if pred_count > consts.pred_count_thres:
                if random.random() < 1.0 - math.sqrt(
                                float(consts.pred_count_thres) / pred_count):
                    self.wv = -1
                    return

        if include_type:
            candidates = [candidate + '-PRED' for candidate in candidates]
        index = -1
        for text in candidates:
            index = model.get_word_index(text)
            if index != -1:
                break
        self.wv = index

    def get_text(self, pred_vocab_list=None, include_type=False):
        text = 'UNK'
        for candidate in self.get_candidates():
            if (not pred_vocab_list) or candidate in pred_vocab_list:
                text = candidate
                break
        if include_type:
            text += '-PRED'
        return text

    def get_wv(self):
        return self.wv
