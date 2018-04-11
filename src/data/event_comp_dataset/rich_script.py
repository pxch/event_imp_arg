import random
from collections import defaultdict
from copy import deepcopy
from itertools import permutations

from common.event_script import Script
from indexed_event import IndexedEventTriple
from rich_entity import EntitySalience, RichEntity
from rich_event import RichEvent
from utils import Word2VecModel, check_type, consts


class RichScript(object):
    def __init__(self, doc_name, rich_events, rich_entities):
        self.doc_name = doc_name
        self.rich_events = rich_events
        self.rich_entities = rich_entities
        self.num_events = len(self.rich_events)
        self.num_entities = len(self.rich_entities)

    def get_index(self, model, include_type=True, use_unk=True,
                  pred_count_dict=None):
        check_type(model, Word2VecModel)
        for rich_event in self.rich_events:
            rich_event.get_index(
                model, include_type=include_type, use_unk=use_unk,
                pred_count_dict=pred_count_dict)

    # return list of events with indexed predicate (rich_pred.wv != -1)
    def get_indexed_events(self):
        return [rich_event for rich_event in self.rich_events
                if rich_event.rich_pred.get_wv() != -1]

    def get_word2vec_training_seq(
            self, pred_vocab_list, arg_vocab_list, ner_vocab_list,
            include_type=True, include_all_pobj=True):
        sequence = []
        for rich_event in self.rich_events:
            sequence.extend(
                rich_event.get_word2vec_training_seq(
                    pred_vocab_list=pred_vocab_list,
                    arg_vocab_list=arg_vocab_list,
                    ner_vocab_list=ner_vocab_list,
                    include_type=include_type,
                    include_all_pobj=include_all_pobj))
        return sequence

    def get_pretraining_input_list(self):
        pretraining_input_list = []
        for rich_event in self.get_indexed_events():
            pos_input = rich_event.get_pos_input(include_all_pobj=False)
            if pos_input is not None:
                pretraining_input_list.append(pos_input)
        return pretraining_input_list

    def get_pair_input_list(self, pair_type_list, left_sample_type, **kwargs):
        indexed_event_list = self.get_indexed_events()
        if len(indexed_event_list) <= 1:
            return []

        assert 'tf_arg' in pair_type_list
        assert left_sample_type in ['one', 'all']

        left_input_list = [rich_event.get_pos_input(include_all_pobj=False)
                           for rich_event in indexed_event_list]

        pair_input_dict = defaultdict(list)

        for event_idx, rich_event in enumerate(indexed_event_list):
            left_input_idx_list = \
                range(0, event_idx) + range(event_idx, len(indexed_event_list))

            for pair_type in pair_type_list:
                pair_input_list = \
                    rich_event.get_pair_input_list(pair_type, **kwargs)
                for pair_input in pair_input_list:
                    if left_sample_type == 'one':
                        left_input = left_input_list[
                            random.choice(left_input_idx_list)]
                        pair_input_dict[pair_type].append(
                            IndexedEventTriple(left_input, *pair_input))
                    else:
                        for left_input_idx in left_input_idx_list:
                            left_input = left_input_list[left_input_idx]
                            pair_input_dict[pair_type].append(
                                IndexedEventTriple(left_input, *pair_input))

        results = []

        tf_arg_list = pair_input_dict['tf_arg']
        results.extend(tf_arg_list)
        num_tf_arg = len(tf_arg_list)

        if 'wo_arg' in pair_input_dict:
            wo_arg_list = pair_input_dict['wo_arg']
            if len(wo_arg_list) > num_tf_arg:
                results.extend(
                    random.sample(wo_arg_list, int(0.8 * num_tf_arg)))
            else:
                results.extend(wo_arg_list)
        if 'two_args' in pair_input_dict:
            two_args_list = pair_input_dict['two_args']
            if len(two_args_list) > num_tf_arg:
                results.extend(
                    random.sample(two_args_list, int(0.8 * num_tf_arg)))
            else:
                results.extend(two_args_list)

        random.shuffle(results)

        return results

    def get_pair_tuning_input_list(self, neg_sample_type):
        # TODO: remove old function
        # return empty list when number of entities is less than or equal to 1,
        # since there exists no negative inputs
        if self.num_entities <= 1:
            return []
        # return empty list when number of events with indexed predicate is
        # less than of equal to 1, since there exists no left inputs
        indexed_event_list = self.get_indexed_events()
        if len(indexed_event_list) <= 1:
            return []
        assert neg_sample_type in ['one', 'neg', 'all'], \
            'neg_sample_type can only be ' \
            'one (one random negative event and one random left event), ' \
            'neg (one random left event for every negative event), or ' \
            'all (every left event for every negative event)'
        results = []
        pos_input_list = [rich_event.get_pos_input(include_all_pobj=False)
                          for rich_event in indexed_event_list]
        for pos_idx, pos_event in enumerate(indexed_event_list):
            pos_input = pos_input_list[pos_idx]
            if pos_input is None:
                continue
            left_input_idx_list = \
                range(0, pos_idx) + range(pos_idx, len(indexed_event_list))
            for arg_idx in [1, 2, 3]:
                if pos_event.has_neg(arg_idx):
                    pos_salience = \
                        pos_event.get_argument(arg_idx).get_pos_salience()
                    if neg_sample_type == 'one':
                        neg_input, neg_salience = random.choice(
                            pos_event.get_neg_input_list(
                                arg_idx, include_salience=True))
                        left_input = pos_input_list[
                            random.choice(left_input_idx_list)]
                        results.append(IndexedEventTriple(
                            left_input, pos_input, neg_input, arg_idx, arg_idx,
                            pos_salience, neg_salience))
                    else:
                        neg_input_list = pos_event.get_neg_input_list(
                            arg_idx, include_salience=True)
                        for neg_input, neg_salience in neg_input_list:
                            if neg_sample_type == 'neg':
                                left_input = pos_input_list[
                                    random.choice(left_input_idx_list)]
                                results.append(IndexedEventTriple(
                                    left_input, pos_input, neg_input, arg_idx,
                                    arg_idx, pos_salience, neg_salience))
                            else:
                                for left_input_idx in left_input_idx_list:
                                    left_input = pos_input_list[left_input_idx]
                                    results.append(IndexedEventTriple(
                                        left_input, pos_input, neg_input,
                                        arg_idx, arg_idx, pos_salience,
                                        neg_salience))
        return results

    def get_pair_tuning_input_list_wo_arg(self, sample_type, model,
                                          include_type=True, use_unk=True):
        # TODO: remove old function
        # return empty list when number of events with indexed predicate is
        # less than of equal to 1, since there exists no left inputs
        indexed_event_list = self.get_indexed_events()
        if len(indexed_event_list) <= 1:
            return []
        assert sample_type in ['one', 'all'], \
            'sample_type can only be ' \
            'one (one random left event for every negative sample), or' \
            'all (every left event for every negative sample)'
        results = []
        pos_input_list = [rich_event.get_pos_input(include_all_pobj=False)
                          for rich_event in indexed_event_list]

        arg_type_map = {1: 'SUBJ', 2: 'OBJ', 3: 'PREP'}

        for pos_idx, pos_event in enumerate(indexed_event_list):
            pos_input = pos_input_list[pos_idx]
            if pos_input is None:
                continue
            left_input_idx_list = \
                range(0, pos_idx) + range(pos_idx, len(indexed_event_list))
            for arg_idx in [1, 2, 3]:
                if pos_event.has_neg(arg_idx):
                    pos_salience = \
                        pos_event.get_argument(arg_idx).get_pos_salience()
                    neg_input = deepcopy(pos_input)
                    neg_input.set_argument(arg_idx, -1)
                    neg_salience = EntitySalience(**{})
                elif pos_event.get_argument(arg_idx) is None:
                    # do not add pair when there is no entity in the script
                    if len(self.rich_entities) == 0:
                        continue
                    pos_salience = EntitySalience(**{})
                    neg_input = deepcopy(pos_input)
                    random_entity = random.choice(self.rich_entities)
                    arg_type = arg_type_map[arg_idx] if include_type else ''
                    neg_input.set_argument(
                        arg_idx,
                        random_entity.get_index(
                            model, arg_type=arg_type, use_unk=use_unk))
                    neg_salience = random_entity.get_salience()
                else:
                    continue
                if sample_type == 'one':
                    left_input = pos_input_list[
                        random.choice(left_input_idx_list)]
                    results.append(IndexedEventTriple(
                        left_input, pos_input, neg_input, arg_idx, arg_idx,
                        pos_salience, neg_salience))
                else:
                    for left_input_idx in left_input_idx_list:
                        left_input = pos_input_list[left_input_idx]
                        results.append(IndexedEventTriple(
                            left_input, pos_input, neg_input, arg_idx, arg_idx,
                            pos_salience, neg_salience))
        return results

    def get_pair_tuning_input_list_two_args(self, sample_type):
        # TODO: remove old function
        # return empty list when number of events with indexed predicate is
        # less than of equal to 1, since there exists no left inputs
        indexed_event_list = self.get_indexed_events()
        if len(indexed_event_list) <= 1:
            return []
        assert sample_type in ['one', 'all'], \
            'sample_type can only be ' \
            'one (one random left event for every negative sample), or' \
            'all (every left event for every negative sample)'
        results = []
        pos_input_list = [rich_event.get_pos_input(include_all_pobj=False)
                          for rich_event in indexed_event_list]

        for pos_idx, pos_event in enumerate(indexed_event_list):
            if pos_input_list[pos_idx] is None:
                continue
            left_input_idx_list = \
                range(0, pos_idx) + range(pos_idx, len(indexed_event_list))

            arg_idx_with_entity = [idx for idx in [1, 2, 3]
                                   if pos_event.has_neg(idx)]
            if len(arg_idx_with_entity) < 2:
                continue

            for pos_arg_idx, neg_arg_idx in permutations(
                    arg_idx_with_entity, 2):
                pos_input = deepcopy(pos_input_list[pos_idx])
                pos_input.set_argument(neg_arg_idx, -1)
                neg_input = deepcopy(pos_input_list[pos_idx])
                neg_input.set_argument(
                    neg_arg_idx, neg_input.get_argument(pos_arg_idx))
                pos_salience = \
                    pos_event.get_argument(pos_arg_idx).get_pos_salience()
                neg_salience = pos_salience
                neg_input.set_argument(pos_arg_idx, -1)
                if sample_type == 'one':
                    left_input = pos_input_list[
                        random.choice(left_input_idx_list)]
                    results.append(IndexedEventTriple(
                        left_input, pos_input, neg_input, pos_arg_idx,
                        neg_arg_idx, pos_salience, neg_salience))
                else:
                    for left_input_idx in left_input_idx_list:
                        left_input = pos_input_list[left_input_idx]
                        results.append(IndexedEventTriple(
                            left_input, pos_input, neg_input, pos_arg_idx,
                            neg_arg_idx, pos_salience, neg_salience))

        return results

    @classmethod
    def build(cls, script, prep_vocab_list, use_lemma=True,
              filter_stop_events=False):
        check_type(script, Script)
        # FIXME: should use the token count of original document
        token_count_dict = script.get_token_count(use_lemma=use_lemma)
        rich_entity_list = []
        for entity in script.entities:
            rich_entity = RichEntity.build(entity, token_count_dict,
                                           use_lemma=use_lemma)
            rich_entity_list.append(rich_entity)
        rich_events = []
        for event in script.events:
            rich_event = RichEvent.build(
                event,
                rich_entity_list=rich_entity_list,
                prep_vocab_list=prep_vocab_list,
                use_lemma=use_lemma
            )
            if (not filter_stop_events) or \
                    (rich_event.rich_pred.get_text(include_type=False)
                     not in consts.stop_preds):
                rich_events.append(rich_event)
        return cls(script.doc_name, rich_events, rich_entity_list)
