from collections import Counter

from common import document
from mention import Mention
from utils import consts, check_type


class Entity(object):
    def __init__(self, mentions):
        # list of all mentions in the entity
        assert mentions, 'must provide at least one mention'
        for mention in mentions:
            check_type(mention, Mention)
        self._mentions = mentions
        # number of all mentions in the entity
        self._num_mentions = len(self.mentions)
        # the most representative mention in the entity
        self._rep_mention = None
        for mention in self.mentions:
            if mention.rep:
                if self._rep_mention is None:
                    self._rep_mention = mention
                else:
                    raise RuntimeError(
                        'cannot have more than one representative mentions')
        if self._rep_mention is None:
            raise RuntimeError('no representative mention provided')

        # NOBUG: set self.ner to be the most frequent ner of all mentions
        # might be different than the ner of rep_mention
        ner_counter = Counter()
        for mention in self.mentions:
            if mention.ner != '':
                ner_counter[mention.ner] += 1
        if len(ner_counter):
            ner = ner_counter.most_common(1)[0][0]
            assert ner in consts.valid_ner_tags, 'unrecognized NER tag: ' + ner
            self._ner = ner
        else:
            self._ner = ''

    @property
    def mentions(self):
        return self._mentions

    @property
    def num_mentions(self):
        return self._num_mentions

    @property
    def rep_mention(self):
        return self._rep_mention

    @property
    def ner(self):
        return self._ner

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        else:
            return all(mention == other_mention for mention, other_mention
                       in zip(self.mentions, other.mentions))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_mention(self, idx):
        assert 0 <= idx < self.num_mentions, \
            'Mention idx {} out of range'.format(idx)
        result = self.mentions[idx]
        check_type(result, Mention)
        return result

    def get_representation(self, use_lemma=True):
        return self.rep_mention.get_representation(use_lemma=use_lemma)

    def to_text(self):
        return ' :: '.join([mention.to_text() for mention in self.mentions])

    @classmethod
    def from_text(cls, text):
        mentions = [Mention.from_text(mention_text.strip())
                    for mention_text in text.split(' :: ')]
        return cls(mentions)

    @classmethod
    def from_coref(cls, coref):
        check_type(coref, document.Coreference)

        mentions = [Mention.from_mention(mention) for mention in coref.mentions]
        return cls(mentions)
