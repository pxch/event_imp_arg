from collections import defaultdict
from copy import deepcopy

from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.nombank import NombankSplitTreePointer

from candidate import Candidate
from corenlp_reader import CoreNLPReader
from data.nltk import PTBReader
from helper import convert_nombank_label
from helper import core_arg_list, nominal_predicate_mapping
from imp_arg_instance import ImpArgInstance
from rich_tree_pointer import RichTreePointer
from utils import check_type, log


class Predicate(object):
    def __init__(self, pred_pointer, imp_args, exp_args):
        self.pred_pointer = pred_pointer
        self.fileid = pred_pointer.fileid
        self.sentnum = pred_pointer.sentnum
        self.n_pred = ''
        self.v_pred = ''
        self.imp_args = imp_args
        self.exp_args = exp_args
        self.candidates = []

    def set_pred(self, n_pred):
        self.n_pred = n_pred
        self.v_pred = nominal_predicate_mapping[n_pred]

    def has_imp_arg(self, label, max_dist=-1):
        if label in self.imp_args:
            if max_dist == -1:
                return True
            else:
                for arg_pointer in self.imp_args[label]:
                    if 0 <= self.sentnum - arg_pointer.sentnum <= max_dist:
                        return True
        return False

    def num_imp_arg(self, max_dist=-1):
        return sum([1 for label in self.imp_args
                    if self.has_imp_arg(label, max_dist)])

    def has_oracle(self, label):
        for candidate in self.candidates:
            if candidate.is_oracle(self.imp_args[label]):
                return True
        return False

    def num_oracle(self):
        return sum([1 for label in self.imp_args if self.has_oracle(label)])

    def parse_args(self, treebank_reader, corenlp_reader,
                   include_non_head_entity=True):
        check_type(treebank_reader, PTBReader)

        check_type(corenlp_reader, CoreNLPReader)

        for label in self.imp_args:
            for arg in self.imp_args[label]:
                arg.get_treebank(treebank_reader)
                arg.parse_treebank()
                arg.parse_corenlp(
                    corenlp_reader,
                    include_non_head_entity=include_non_head_entity)

        for label, fillers in self.exp_args.items():
            for arg in fillers:
                arg.get_treebank(treebank_reader)
                arg.parse_treebank()
                arg.parse_corenlp(
                    corenlp_reader,
                    include_non_head_entity=include_non_head_entity)
            if label in core_arg_list and len(fillers) > 1:
                assert len(fillers) == 2
                new_fillers = []
                for arg in fillers:
                    # remove pointer pointing to WH-determiner
                    if arg.tree.pos()[arg.tree_pointer.wordnum][1] != 'WDT':
                        new_fillers.append(arg)
                # should only exists one non-WH-determiner pointer
                assert len(new_fillers) == 1
                self.exp_args[label] = new_fillers

    def get_candidate_keys(self, max_dist=2):
        key_list = []
        for sentnum in range(max(0, self.sentnum - max_dist), self.sentnum + 1):
            key_list.append('{}:{}'.format(self.fileid, sentnum))
        return key_list

    def add_candidates(self, instances, max_dist=2):
        for instance in instances:
            if 0 <= self.sentnum - instance.sentnum <= max_dist:
                candidate_list = Candidate.from_instance(instance)
                self.candidates.extend(candidate_list)

    def check_exp_args(self, instance, add_missing_args=False,
                       remove_conflict_imp_args=False, verbose=False):

        unmatched_labels = deepcopy(self.exp_args.keys())

        if instance is not None:
            nombank_arg_dict = defaultdict(list)
            for arg_pointer, label in instance.arguments:
                cvt_label = convert_nombank_label(label)
                if cvt_label:
                    nombank_arg_dict[cvt_label].append(arg_pointer)

            for label in nombank_arg_dict:
                nombank_args = nombank_arg_dict[label]

                if label not in self.exp_args:
                    message = \
                        '{} has {} in Nombank but not found in explicit ' \
                        'arguments.'.format(self.pred_pointer, label)
                    if add_missing_args:
                        message += \
                            '\n\tAdding missing explicit {}: {}.'.format(
                                label, nombank_args)
                        self.exp_args[label] = \
                            [RichTreePointer(self.fileid, self.sentnum, arg)
                             for arg in nombank_args]
                        if remove_conflict_imp_args and label in self.imp_args:
                            message += '\n\tRemoving implicit {}.'.format(label)
                            self.imp_args.pop(label, None)
                    else:
                        message += '\n\tIgnored...'
                    if verbose:
                        log.info(message)
                    continue

                exp_args = [p.tree_pointer for p in self.exp_args[label]]
                unmatched_labels.remove(label)

                if exp_args != nombank_args:
                    message = '{} has mismatch in {}: {} --> {}'.format(
                        self.pred_pointer, label, exp_args, nombank_args)
                    if len(nombank_args) == 1:
                        nombank_arg = nombank_args[0]
                        if isinstance(nombank_arg, NombankSplitTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                self.exp_args[label] = [RichTreePointer(
                                    self.fileid, self.sentnum, nombank_arg)]
                                if verbose:
                                    log.info(message + '\n\tReplaced...')
                                continue
                        if isinstance(nombank_arg, NombankChainTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                if verbose:
                                    log.info(message + '\n\tIgnored...')
                                continue

                    raise AssertionError(message)

        if unmatched_labels:
            message = '{} has {} in explicit arguments but not found in ' \
                      'Nombank.'.format(self.pred_pointer, unmatched_labels)
            raise AssertionError(message)

    @classmethod
    def build(cls, instance):
        check_type(instance, ImpArgInstance)

        pred_pointer = instance.pred_pointer

        tmp_imp_args = defaultdict(list)
        exp_args = defaultdict(list)

        for argument in instance.arguments:

            label = argument[0].lower()
            arg_pointer = argument[1]
            attribute = argument[2]

            # remove arguments located in sentences following the predicate
            if arg_pointer.fileid != pred_pointer.fileid or \
                    arg_pointer.sentnum > pred_pointer.sentnum:
                continue

            # add explicit arguments to exp_args
            if attribute == 'Explicit':
                exp_args[label].append(arg_pointer)
                # remove the label from tmp_imp_args, as we do not process
                # an implicit argument if some explicit arguments with
                # the same label exist
                tmp_imp_args.pop(label, None)

            # add non-explicit arguments to tmp_imp_args
            else:
                # do not add the argument when some explicit arguments with
                # the same label exist
                if label not in exp_args:
                    tmp_imp_args[label].append((arg_pointer, attribute))

        # process implicit arguments
        imp_args = {}
        for label, fillers in tmp_imp_args.items():

            # remove incorporated arguments from tmp_imp_args
            # incorporated argument: argument with the same node as
            # the predicate itself
            if pred_pointer in [pointer for pointer, _ in fillers]:
                continue

            # add non-split arguments to imp_args
            imp_args[label] = [pointer for pointer, attribute in fillers
                               if attribute == '']
            split_pointers = [pointer for pointer, attribute in fillers
                              if attribute == 'Split']

            sentnum_set = set([pointer.sentnum for pointer in split_pointers])

            # group split arguments by their sentnum,
            # and sort pieces by nombank_pointer.wordnum within each group
            grouped_split_pointers = []
            for sentnum in sentnum_set:
                grouped_split_pointers.append(sorted(
                    [pointer for pointer in split_pointers
                     if pointer.sentnum == sentnum],
                    key=lambda p: p.tree_pointer.wordnum))

            # add each split pointer to imp_args
            for split_pointers in grouped_split_pointers:
                imp_args[label].append(RichTreePointer.merge(split_pointers))

        return cls(pred_pointer, imp_args, exp_args)

    def pretty_print(self, verbose=False, include_candidates=False,
                     include_dice_score=False, corenlp_reader=None):
        result = '{}\t{}\n'.format(self.pred_pointer, self.n_pred)

        for label, fillers in self.imp_args.items():
            result += '\tImplicit {}:\n'.format(label)
            for filler in fillers:
                if verbose:
                    result += '\t\t{}\n'.format(
                        filler.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}\n'.format(filler)

        for label, fillers in self.exp_args.items():
            result += '\tExplicit {}:\n'.format(label)
            for filler in fillers:
                if verbose:
                    result += '\t\t{}\n'.format(
                        filler.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}\n'.format(filler)

        if include_candidates:
            result += '\tCandidates:\n'
            for candidate in self.candidates:
                if verbose:
                    result += '\t\t{}'.format(
                        candidate.arg_pointer.pretty_print(corenlp_reader))
                else:
                    result += '\t\t{}'.format(candidate.arg_pointer)

                if include_dice_score:
                    dice_list = {}
                    for label, fillers in self.imp_args.items():
                        dice_list[label] = candidate.dice_score(fillers)
                        result += '\t{}\n'.format(
                            ', '.join(['{}: {:.2f}'.format(label, dice)
                                       for label, dice in dice_list.items()]))
                else:
                    result += '\n'

        return result
