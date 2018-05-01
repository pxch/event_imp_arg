import pickle as pkl
from collections import defaultdict

from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer

from helper import core_arg_list
from helper import shorten_wsj_fileid, convert_nombank_label
from rich_tree_pointer import RichTreePointer
from utils import log


class CandidateDict(object):
    def __init__(self, propbank_reader=None, nombank_reader=None,
                 corenlp_reader=None, max_dist=2):
        self.propbank_reader = propbank_reader
        self.nombank_reader = nombank_reader
        self.corenlp_reader = corenlp_reader
        self.max_dist = max_dist
        self.candidate_dict = defaultdict(list)
        if self.propbank_reader and self.nombank_reader and self.corenlp_reader:
            self.read_only = False
        else:
            self.read_only = True

    def __iter__(self):
        for key, candidates in self.candidate_dict.items():
            yield key, candidates

    def get_candidates(self, pred_pointer):
        candidates = []

        fileid = pred_pointer.fileid

        for sentnum in range(max(0, pred_pointer.sentnum - self.max_dist),
                             pred_pointer.sentnum):
            key = '{}:{}'.format(fileid, sentnum)
            assert key in self.candidate_dict
            candidates.extend(self.candidate_dict[key])

        key = '{}:{}'.format(fileid, pred_pointer.sentnum)
        assert key in self.candidate_dict
        for candidate in self.candidate_dict[key]:
            if pred_pointer not in candidate.get_pred_pointer_list():
                candidates.append(candidate)

        return candidates

    def add_candidates(self, pred_pointer, include_non_head_entity=True):
        assert not self.read_only

        fileid = pred_pointer.fileid
        instances = []
        instances.extend(self.propbank_reader.search_by_fileid(fileid))
        instances.extend(self.nombank_reader.search_by_fileid(fileid))

        for sentnum in range(max(0, pred_pointer.sentnum - self.max_dist),
                             pred_pointer.sentnum + 1):
            key = '{}:{}'.format(fileid, sentnum)
            if key not in self.candidate_dict:
                self.add_key(key, instances,
                             include_non_head_entity=include_non_head_entity)

    def add_key(self, key, instances, include_non_head_entity=True):
        assert not self.read_only

        assert key not in self.candidate_dict

        candidate_list = []
        arg_pointer_list = []

        for instance in instances:
            assert shorten_wsj_fileid(instance.fileid) == key.split(':')[0]

            if instance.sentnum == int(key.split(':')[1]):

                for candidate in Candidate.from_instance(instance):
                    if candidate.arg_pointer not in arg_pointer_list:

                        candidate.arg_pointer.parse_treebank()
                        candidate.arg_pointer.parse_corenlp(
                            self.corenlp_reader,
                            include_non_head_entity=include_non_head_entity)

                        if candidate.arg_pointer.corenlp_word_surface != '':
                            arg_pointer_list.append(candidate.arg_pointer)
                            candidate_list.append(candidate)

                    else:
                        index = arg_pointer_list.index(candidate.arg_pointer)
                        candidate_list[index].merge(candidate)

        self.candidate_dict[key] = candidate_list

    def print_all_candidates(self, file_path):
        fout = open(file_path, 'w')
        for key, candidates in self.candidate_dict:
            fout.write(key + '\n')
            for candidate in candidates:
                fout.write('\t{}\t{}\n'.format(
                    candidate.arg_label,
                    candidate.arg_pointer.pretty_print(self.corenlp_reader)))
        fout.close()

    @classmethod
    def load(cls, candidate_dict_path, propbank_reader=None,
             nombank_reader=None, corenlp_reader=None, max_dist=2):
        log.info('Loading candidate dict from {}'.format(candidate_dict_path))

        candidate_dict = pkl.load(open(candidate_dict_path, 'r'))
        result = cls(
            propbank_reader=propbank_reader, nombank_reader=nombank_reader,
            corenlp_reader=corenlp_reader, max_dist=max_dist)
        result.candidate_dict = candidate_dict

        log.info('Done')

        return result

    def save(self, candidate_dict_path):
        log.info('Saving candidate dict to {}'.format(candidate_dict_path))
        pkl.dump(self.candidate_dict, open(candidate_dict_path, 'w'))


class Candidate(object):
    def __init__(self, fileid, sentnum, pred, arg, arg_label, tree):
        self.fileid = fileid
        self.sentnum = sentnum

        self.arg_pointer = RichTreePointer(fileid, sentnum, arg, tree=tree)

        pred_pointer = RichTreePointer(fileid, sentnum, pred, tree=tree)

        self.pred_list = [(pred_pointer, arg_label)]

    def merge(self, candidate):
        assert isinstance(candidate, Candidate)
        assert self.arg_pointer == candidate.arg_pointer
        self.pred_list.extend(candidate.pred_list)

    def get_pred_pointer_list(self):
        return [pred_pointer for pred_pointer, _ in self.pred_list]

    @staticmethod
    def from_instance(instance):
        candidate_list = []

        fileid = shorten_wsj_fileid(instance.fileid)
        sentnum = instance.sentnum
        pred = instance.predicate
        tree = instance.tree

        for arg_pointer, label in instance.arguments:
            cvt_label = convert_nombank_label(label)
            if cvt_label in core_arg_list:
                if isinstance(arg_pointer, NombankChainTreePointer) or \
                        isinstance(arg_pointer, PropbankChainTreePointer):
                    for p in arg_pointer.pieces:
                        candidate_list.append(Candidate(
                            fileid, sentnum, pred, p, cvt_label, tree))
                else:
                    candidate_list.append(Candidate(
                        fileid, sentnum, pred, arg_pointer, cvt_label, tree))

        return candidate_list

    def is_oracle(self, imp_args):
        assert self.arg_pointer.has_corenlp_info()
        assert all(arg.has_corenlp_info() for arg in imp_args)

        # if candidate has the same pointer
        if self.arg_pointer in imp_args:
            return True

        # if candidate has the same lemmas from CoreNLP document
        if self.arg_pointer.corenlp_lemma_surface in \
                [arg.corenlp_lemma_surface for arg in imp_args]:
            return True

        # if candidate has the pointer consisting of one imp_arg pointer
        # plus one preceding preposition
        if not self.arg_pointer.is_split_pointer:
            for arg in imp_args:
                if not arg.is_split_pointer:
                    if self.arg_pointer.sentnum == arg.sentnum:
                        cand_tb = self.arg_pointer.treebank_info_list[0]
                        arg_tb = arg.treebank_info_list[0]
                        if cand_tb.eq_with_preceding_prep(arg_tb):
                            return True

        return False

    def dice_score(self, imp_args, use_corenlp_tokens=True):
        dice_score = 0.0

        if len(imp_args) > 0:
            dice_score_list = []
            for arg in imp_args:
                dice_score_list.append(
                    self.arg_pointer.dice_score(
                        arg, use_corenlp_tokens=use_corenlp_tokens))

            dice_score = max(dice_score_list)

        return dice_score
