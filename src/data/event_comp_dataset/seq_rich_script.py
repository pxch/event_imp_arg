import random
from collections import defaultdict
from copy import deepcopy

from common.event_script import Script
from indexed_event import IndexedEventTriple
from rich_entity import EntitySalience, RichEntity
from seq_rich_entity import SeqRichEntity
from seq_rich_event import SeqRichEvent
from utils import Word2VecModel, check_type


class SeqRichScript(object):
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

    def get_pair_tuning_input_list(self, neg_sample_type):
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
        pos_input_all_list = [
            rich_event.get_pos_input_all() for rich_event in indexed_event_list]

        for event_idx, pos_event in enumerate(indexed_event_list):
            if event_idx == 0:
                continue

            left_input_list = \
                [pos_input for pos_input_all in pos_input_all_list[:event_idx]
                 for pos_input in pos_input_all]

            for arg_idx, rich_arg in enumerate(pos_event.rich_arg_list):
                if rich_arg.has_neg():
                    if rich_arg.arg_type == 'SUBJ':
                        arg_type_idx = 1
                    elif rich_arg.arg_type == 'OBJ':
                        arg_type_idx = 2
                    else:
                        arg_type_idx = 3
                    pos_input = pos_event.get_pos_input(arg_idx)
                    pos_salience = rich_arg.get_pos_salience()
                    neg_input_list = pos_event.get_neg_input_list(
                        arg_idx, include_salience=True)
                    if neg_sample_type == 'one':
                        neg_input, neg_salience = random.choice(neg_input_list)
                        left_input = random.choice(left_input_list)
                        results.append(IndexedEventTriple(
                            left_input, pos_input, neg_input, arg_type_idx,
                            arg_type_idx, pos_salience, neg_salience))
                    else:

                        for neg_input, neg_salience in neg_input_list:
                            if neg_sample_type == 'neg':
                                left_input = random.choice(left_input_list)
                                results.append(IndexedEventTriple(
                                    left_input, pos_input, neg_input,
                                    arg_type_idx, arg_type_idx, pos_salience,
                                    neg_salience))
                            else:
                                for left_input in left_input_list:
                                    results.append(IndexedEventTriple(
                                        left_input, pos_input, neg_input,
                                        arg_type_idx, arg_type_idx,
                                        pos_salience, neg_salience))
        return results

    @staticmethod
    def create_psuedo_entity(core):
        num_mentions_named = 0
        num_mentions_nominal = 0
        num_mentions_pronominal = 0
        if core.ner != '':
            num_mentions_named += 1
        elif core.pos.startswith('NN'):
            num_mentions_nominal += 1
        elif core.pos.startswith('PRP'):
            num_mentions_pronominal += 1

        salience = EntitySalience(
            num_mentions_named=num_mentions_named,
            num_mentions_nominal=num_mentions_nominal,
            num_mentions_pronominal=num_mentions_pronominal,
            num_mentions_total=1
        )
        return RichEntity(core, salience)

    @classmethod
    def build(cls, script, prep_vocab_list, use_lemma=True,
              filter_repetitive_prep=False):
        check_type(script, Script)

        seq_rich_entity_list = []
        for entity in script.entities:
            seq_rich_entity = SeqRichEntity.build(entity, use_lemma=use_lemma)
            seq_rich_entity_list.append(seq_rich_entity)

        seq_rich_event_list = []
        rich_entity_list = []
        entity_idx_list = []
        entity_mentions_dict = defaultdict(list)

        for event in script.events:
            seq_rich_event = SeqRichEvent.build(
                event,
                rich_entity_list=deepcopy(rich_entity_list),
                entity_idx_list=deepcopy(entity_idx_list),
                prep_vocab_list=prep_vocab_list,
                use_lemma=use_lemma,
                filter_repetitive_prep=filter_repetitive_prep)
            seq_rich_event_list.append(seq_rich_event)

            for rich_arg in seq_rich_event.rich_arg_list:
                entity_idx = rich_arg.entity_idx
                if entity_idx == -1:
                    rich_entity_list.append(
                        SeqRichScript.create_psuedo_entity(rich_arg.core))
                    entity_idx_list.append(-1)
                else:
                    if entity_idx not in entity_idx_list:
                        entity_idx_list.append(entity_idx)
                        rich_entity_list.append(None)
                    entity_mentions_dict[entity_idx].append(
                        rich_arg.mention_idx)
                    updated_rich_entity = \
                        seq_rich_entity_list[entity_idx].get_rich_entity(
                            entity_mentions_dict[entity_idx])
                    mapped_entity_idx = entity_idx_list.index(entity_idx)
                    rich_entity_list[mapped_entity_idx] = updated_rich_entity

        return cls(script.doc_name, seq_rich_event_list, seq_rich_entity_list)
