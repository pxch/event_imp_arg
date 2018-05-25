from copy import deepcopy

import numpy as np

from data.event_comp_dataset import CoreArgument
from data.event_comp_dataset.rich_entity import EntitySalience
from rich_tree_pointer import RichTreePointer
from utils import Word2VecModel, check_type


class RichCandidate(object):
    def __init__(self, arg_pointer, dice_score, core, entity_salience):
        check_type(arg_pointer, RichTreePointer)
        self.arg_pointer = arg_pointer

        self.dice_score = dice_score

        check_type(core, CoreArgument)
        self.core = deepcopy(core)

        if entity_salience is not None:
            check_type(entity_salience, EntitySalience)
        self.entity_salience = entity_salience

    def get_index(self, model, arg_type='', use_unk=True):
        return self.core.get_index(model, arg_type, use_unk=use_unk)

    @classmethod
    def build(cls, candidate, imp_args, corenlp_reader, use_lemma=True,
              use_entity=True, use_corenlp_tokens=True):

        arg_pointer = candidate.arg_pointer.copy(
            include_treebank=(not use_corenlp_tokens),
            include_corenlp=use_corenlp_tokens)

        dice_score = candidate.dice_score(
            imp_args, use_corenlp_tokens=use_corenlp_tokens)

        core = candidate.arg_pointer.get_core_argument(
            corenlp_reader, use_lemma=use_lemma, use_entity=use_entity)

        entity_salience = candidate.arg_pointer.get_entity_salience(
            corenlp_reader, use_entity=use_entity)

        return cls(arg_pointer, dice_score, core, entity_salience)


class RichImplicitArgument(object):
    def __init__(self, label, arg_type, fillers, rich_candidate_list,
                 best_candidate_idx):
        self.label = label
        self.arg_type = arg_type
        self.fillers = fillers

        if len(self.fillers) > 0:
            self.exist = True
        else:
            self.exist = False

        self.rich_candidate_list = rich_candidate_list
        self.candidate_wv_list = []
        self.valid_candidate_idx_list = []

        self.has_coherence_score = False
        self.coherence_score_list = []
        self.max_coherence_score = 0.0
        self.max_coherence_score_idx = -1
        self.coherence_score_wo_arg = 0.0

        self.best_candidate_idx = best_candidate_idx

    def get_arg_idx(self):
        if self.arg_type == 'SUBJ':
            return 1
        elif self.arg_type == 'OBJ':
            return 2
        else:
            return 3

    def get_pos_candidate(self):
        return self.rich_candidate_list[self.best_candidate_idx]

    def get_pos_wv(self):
        return self.candidate_wv_list[self.best_candidate_idx]

    def get_neg_candidate_list(self):
        return self.rich_candidate_list[:self.best_candidate_idx] + \
               self.rich_candidate_list[self.best_candidate_idx+1:]

    def get_neg_wv_list(self):
        return self.candidate_wv_list[:self.best_candidate_idx] + \
               self.candidate_wv_list[self.best_candidate_idx+1:]

    def get_index(self, model, include_type=True, use_unk=True):
        check_type(model, Word2VecModel)

        self.candidate_wv_list = []

        for rich_candidate in self.rich_candidate_list:
            candidate_wv = rich_candidate.get_index(
                model, self.arg_type if include_type else '', use_unk=use_unk)
            self.candidate_wv_list.append(candidate_wv)

        self.valid_candidate_idx_list = \
            [candidate_idx for candidate_idx, candidate_wv
             in enumerate(self.candidate_wv_list) if candidate_wv != -1]

    def set_coherence_score_list(self, coherence_score_list):
        assert len(coherence_score_list) == len(self.rich_candidate_list) + 1
        self.coherence_score_wo_arg = coherence_score_list[0]
        self.coherence_score_list = coherence_score_list[1:]
        if len(self.coherence_score_list) > 0:
            self.has_coherence_score = True
            self.max_coherence_score = self.coherence_score_list.max()

            max_indices = \
                [idx for idx, score in enumerate(self.coherence_score_list)
                 if score == self.max_coherence_score]

            max_dice_list = np.asarray(
                [self.rich_candidate_list[idx].dice_score
                 for idx in max_indices])
            self.max_coherence_score_idx = max_indices[max_dice_list.argmax()]

    def reset_coherence_score_list(self):
        self.coherence_score_list = []
        self.max_coherence_score = 0.0
        self.max_coherence_score_idx = -1
        self.coherence_score_wo_arg = 0.0

        self.has_coherence_score = False

    def get_eval_dice_score(self):
        assert self.max_coherence_score_idx != -1
        return self.rich_candidate_list[self.max_coherence_score_idx].dice_score

    @classmethod
    def build(cls, label, arg_type, fillers, candidates, corenlp_reader,
              use_lemma=True, use_entity=True, use_corenlp_tokens=True):
        rich_candidate_list = [
            RichCandidate.build(
                candidate, fillers, corenlp_reader,
                use_lemma=use_lemma, use_entity=use_entity,
                use_corenlp_tokens=use_corenlp_tokens)
            for candidate in candidates]

        best_candidate_idx = -1
        best_dice_score = -1
        for idx, candidate in enumerate(rich_candidate_list):
            if candidate.dice_score > best_dice_score:
                best_dice_score = candidate.dice_score
                best_candidate_idx = idx

        return cls(label, arg_type, fillers, rich_candidate_list,
                   best_candidate_idx)
