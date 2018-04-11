import re

from common import document
from entity import Entity
from token import Token
from utils import check_type, consts, unescape


class Argument(Token):
    def __init__(self, word, lemma, pos, ner='', entity_idx=-1, mention_idx=-1):
        super(Argument, self).__init__(word, lemma, pos)

        # name entity tag of the argument, default is empty
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

        # index of the entity where the argument belongs to
        # default is -1 (no entity)
        assert type(entity_idx) == int and entity_idx >= -1, \
            'entity_idx must be a non-negative integer, or -1 (no entity)'
        self._entity_idx = entity_idx

        # index of the mention where the argument belongs to
        # default is -1 (no mention)
        assert type(mention_idx) == int and mention_idx >= -1, \
            'mention_idx must be a non-negative integer, or -1 (no mention)'
        assert (not self.has_entity()) or mention_idx >= 0, \
            'mention_idx cannot be -1 when entity_idx is not -1'
        self._mention_idx = mention_idx

    @property
    def ner(self):
        return self._ner

    @property
    def entity_idx(self):
        return self._entity_idx

    @property
    def mention_idx(self):
        return self._mention_idx

    def has_entity(self):
        return self.entity_idx >= 0

    def has_mention(self):
        return self.mention_idx >= 0

    def reset_entity_info(self):
        self._entity_idx = -1
        self._mention_idx = -1

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False
        else:
            return self.word == other.word and self.lemma == other.lemma \
                   and self.pos == other.pos and self.ner == other.ner \
                   and self.entity_idx == other.entity_idx \
                   and self.mention_idx == other.mention_idx

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_entity(self, entity_list):
        if self.has_entity():
            assert self.entity_idx < len(entity_list), \
                'Entity index {} out of range'.format(self.entity_idx)
            entity = entity_list[self.entity_idx]
            check_type(entity, Entity)
            return entity
        return None

    def get_mention(self, entity_list):
        if self.has_entity():
            entity = self.get_entity(entity_list)
            check_type(entity, Entity)
            assert self.has_mention()
            assert self.mention_idx < entity.num_mentions, \
                'Mention index {} out of range'.format(self.mention_idx)
            return entity.get_mention(self.mention_idx)
        return None

    def to_text(self):
        text = super(Argument, self).to_text()
        text += '/{}'.format(self.ner if self.ner != '' else 'NONE')
        if self.has_entity():
            text += '//entity-{}-{}'.format(self.entity_idx, self.mention_idx)
        return text

    arg_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)/(?P<ner>[^/]*)'
        r'((?://entity-)(?P<entity_idx>\d+)(?:-)(?P<mention_idx>\d+))?$')

    @classmethod
    def from_text(cls, text):
        match = cls.arg_re.match(text)
        assert match, 'cannot parse Argument from {}'.format(text)
        groups = match.groupdict()

        word = unescape(groups['word'])
        lemma = unescape(groups['lemma'])
        pos = unescape(groups['pos'])
        ner = groups['ner'] if groups['ner'] != 'NONE' else ''
        entity_idx = int(groups['entity_idx']) if groups['entity_idx'] else -1
        mention_idx = \
            int(groups['mention_idx']) if groups['mention_idx'] else -1

        return cls(word, lemma, pos, ner, entity_idx, mention_idx)

    @classmethod
    def from_token(cls, token):
        check_type(token, document.Token)

        word = token.word
        lemma = token.lemma
        pos = token.pos
        ner = token.ner
        entity_idx = token.coref_idx()
        mention_idx = token.mention_idx()

        return cls(word, lemma, pos, ner, entity_idx, mention_idx)
