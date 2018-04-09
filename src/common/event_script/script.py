from collections import defaultdict, Counter
from itertools import product

from common import document
from entity import Entity
from event import Event
from utils import check_type, log


class Script(object):
    def __init__(self, doc_name):
        # name of the document from which the script is created
        self._doc_name = doc_name
        # list of all entities in the script
        self._entities = []
        # number of all entities in the script
        self._num_entities = 0
        # list of all events in the script
        self._events = []
        # number of all events in the script
        self._num_events = 0

    @property
    def doc_name(self):
        return self._doc_name

    @property
    def entities(self):
        return self._entities

    @property
    def num_entities(self):
        return self._num_entities

    @property
    def events(self):
        return self._events

    @property
    def num_events(self):
        return self._num_events

    def add_entity(self, entity):
        check_type(entity, Entity)
        self._entities.append(entity)
        self._num_entities += 1

    def add_event(self, event):
        check_type(event, Event)
        self._events.append(event)
        self._num_events += 1

    def __eq__(self, other):
        if not isinstance(other, Script):
            return False
        else:
            return self.doc_name == other.doc_name \
                   and all(entity == other_entity for entity, other_entity
                           in zip(self.entities, other.entities)) \
                   and all(event == other_event for event, other_event
                           in zip(self.events, other.events))

    def __ne__(self, other):
        return not self.__eq__(other)

    def has_entities(self):
        return self.num_entities > 0

    def has_events(self):
        return self.num_events > 0

    def get_entity(self, idx):
        assert 0 <= idx < self.num_entities, \
            'Entity idx {} out of range'.format(idx)
        result = self.entities[idx]
        check_type(result, Entity)
        return result

    def get_event(self, idx):
        assert 0 <= idx < self.num_events, \
            'Event idx {} out of range'.format(idx)
        result = self.events[idx]
        check_type(result, Event)
        return result

    def check_entity_idx_range(self):
        for event in self.events:
            for arg in event.get_all_args_with_entity(include_arg_type=False):
                assert 0 <= arg.entity_idx < self.num_entities, \
                    '{} in {} has entity_idx {} out of range'.format(
                        arg.to_text(), event.to_text(), arg.entity_idx)
                entity = self.get_entity(arg.entity_idx)
                assert 0 <= arg.mention_idx < entity.num_mentions, \
                    '{} in {} has mention_idx {} out of range'.format(
                        arg.to_text(), event.to_text(), arg.mention_idx)

    def get_token_count(self, use_lemma=True):
        # TODO: deprecate this method as it is not a true count of tokens
        token_count = defaultdict(int)
        for event in self.events:
            for arg in event.get_all_args_without_entity(
                    include_arg_type=False):
                token_count[
                    arg.get_representation(use_lemma=use_lemma)] += 1
        for entity in self.entities:
            for mention in entity.mentions:
                for token in mention.tokens:
                    token_count[
                        token.get_representation(use_lemma=use_lemma)] += 1
        return token_count

    def get_vocab_count(self, use_lemma=True):
        vocab_count = defaultdict(Counter)

        # list of all arguments
        all_args = []
        # iterate through all events to count predicates and prepositions
        for event in self.events:
            # add the predicate of each event
            pred_representation = \
                event.pred.get_full_representation(use_lemma=use_lemma)
            vocab_count['predicate'][pred_representation] += 1

            # add all prepositions of each event
            for prep, _ in event.pobj_list:
                if prep != '':
                    vocab_count['preposition'][prep] += 1

            all_args.extend(event.get_all_args(include_arg_type=False))

        # iterate through all arguments to count arguments and name entities
        for arg in all_args:
            if arg.has_entity():
                mention = self.get_entity(arg.entity_idx).rep_mention
                arg_representation = \
                    mention.get_representation(use_lemma=use_lemma)
                vocab_count['argument'][arg_representation] += 1
                if mention.ner != '':
                    vocab_count['name_entity'][arg_representation] += 1
                    vocab_count['name_entity_tag'][mention.ner] += 1
            else:
                arg_representation = arg.get_representation(use_lemma=use_lemma)
                vocab_count['argument'][arg_representation] += 1
                if arg.ner != '':
                    vocab_count['name_entity'][arg_representation] += 1
                    vocab_count['name_entity_tag'][arg.ner] += 1
        return vocab_count

    def to_text(self):
        entities_text = '\n'.join(['entity-{:0>3d}\t{}'.format(
            entity_idx, entity.to_text()) for entity_idx, entity
            in enumerate(self.entities)])
        events_text = '\n'.join(['event-{:0>4d}\t{}'.format(
            event_idx, event.to_text()) for event_idx, event
            in enumerate(self.events)])
        return '{}\n\nEntities:\n{}\n\nEvents:\n{}\n'.format(
            self.doc_name, entities_text, events_text)

    @classmethod
    def from_text(cls, text):
        input_lines = [l.strip() for l in text.splitlines() if l.strip() != '']

        parse_entity = False
        parse_event = False

        script = cls(input_lines[0])
        for line in input_lines[1:]:
            if line == 'Entities:':
                assert (not parse_entity) and (not parse_event)
                parse_entity = True
                continue
            elif line == 'Events:':
                assert parse_entity and (not parse_event)
                parse_event = True
                parse_entity = False
                continue
            if parse_entity:
                entity = Entity.from_text(line.partition('\t')[2])
                script.add_entity(entity)
            elif parse_event:
                event = Event.from_text(line.partition('\t')[2])
                script.add_event(event)
            else:
                raise RuntimeError(
                    'cannot parse EventScript from: {}'.format(text))

        return script

    @classmethod
    def from_doc(cls, doc):
        check_type(doc, document.Document)
        script = cls(doc.doc_name)

        # add all entities from document
        for coref in doc.corefs:
            entity = Entity.from_coref(coref)
            script.add_entity(entity)

        if not script.has_entities():
            log.warning('script {} has no entities'.format(doc.doc_name))

        # add all events from document
        for sent in doc.sents:
            # iterate through all tokens
            for pred_token in sent.tokens:
                if pred_token.pos.startswith('VB'):
                    # exclude "be" verbs
                    if pred_token.lemma == 'be':
                        continue
                    # exclude modifying verbs
                    if sent.dep_graph.lookup_label(
                            'head', pred_token.token_idx, 'xcomp'):
                        continue
                    # TODO: exclude verbs in quotes
                    # NOBUG: do not exclude stop verbs now
                    # both negation and particle need to be counted in
                    # detecting a stop verb, we will remove stop verbs
                    # in constructing RichScript

                    # find whether the verb has negation
                    neg = False
                    if sent.dep_graph.lookup_label(
                            'head', pred_token.token_idx, 'neg'):
                        neg = True

                    # find whether the verb has particle
                    prt = ''
                    prt_tokens = sent.lookup_label(
                        'head', pred_token.token_idx, 'compound:prt')
                    if prt_tokens:
                        if len(prt_tokens) > 1:
                            log.warning(
                                'Predicate {} contains {} particles'.format(
                                    pred_token.pretty_print(),
                                    len(prt_tokens)))
                        prt = prt_tokens[0].lemma

                    subj_list = sent.get_subj_list(pred_token.token_idx)
                    dobj_list = sent.get_dobj_list(pred_token.token_idx)
                    pobj_list = sent.get_pobj_list(pred_token.token_idx)

                    if (not subj_list) and (not dobj_list):
                        continue
                    if not subj_list:
                        subj_list.append(None)
                    if not dobj_list:
                        dobj_list.append(None)

                    for arg_tuple in product(subj_list, dobj_list):
                        event = Event.from_tokens(
                            pred_token,
                            arg_tuple[0],
                            arg_tuple[1],
                            pobj_list,
                            neg=neg,
                            prt=prt)
                        script.add_event(event)

        if not script.has_events():
            log.warning('script {} has no events'.format(doc.doc_name))

        return script


class ScriptCorpus(object):
    def __init__(self):
        # list of all scripts in the corpus
        self._scripts = []
        # number of all scripts in the corpus
        self._num_scripts = 0

    @property
    def scripts(self):
        return self._scripts

    @property
    def num_scripts(self):
        return self._num_scripts

    def __eq__(self, other):
        return all(script == other_script for script, other_script
                   in zip(self.scripts, other.scripts))

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_script(self, script):
        check_type(script, Script)
        self._scripts.append(script)
        self._num_scripts += 1

    def to_text(self):
        return '\n###DOC###\n\n'.join(
            script.to_text() for script in self.scripts)

    @classmethod
    def from_text(cls, text):
        script_corpus = cls()
        for script_text in text.split('\n###DOC###\n\n'):
            script_corpus.add_script(Script.from_text(script_text))
        return script_corpus
