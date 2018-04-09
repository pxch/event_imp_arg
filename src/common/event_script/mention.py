from common import document
from core_argument import CoreArgument
from token import Token
from utils import check_type, consts


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx, head_token_idx,
                 rep, ner, tokens):
        # index of the sentence where the mention is located
        self._sent_idx = sent_idx

        # the start, end, and head token index of the mention, starts with 0
        assert 0 <= start_token_idx <= head_token_idx < end_token_idx, \
            'head_token_idx {} must be between start_token_idx {} ' \
            'and end_token_idx {}'.format(
                head_token_idx, start_token_idx, end_token_idx)
        self._start_token_idx = start_token_idx
        self._end_token_idx = end_token_idx
        self._head_token_idx = head_token_idx

        # whether the mention is the most representative mention in the entity
        assert type(rep) == bool, 'rep must be a boolean value'
        self._rep = rep

        # name entity tag of the mention
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

        # list of all tokens in the mention
        assert len(tokens) == end_token_idx - start_token_idx, \
            'number of tokens {} does not match start_token_idx {} ' \
            'and end_token_idx {}'.format(
                len(tokens), start_token_idx, end_token_idx)
        for token in tokens:
            check_type(token, Token)
        self._tokens = tokens

        # the head token of the mention
        self._head_token = \
            self.tokens[self.head_token_idx - self.start_token_idx]

    @property
    def sent_idx(self):
        return self._sent_idx

    @property
    def start_token_idx(self):
        return self._start_token_idx

    @property
    def end_token_idx(self):
        return self._end_token_idx

    @property
    def head_token_idx(self):
        return self._head_token_idx

    @property
    def rep(self):
        return self._rep

    @property
    def tokens(self):
        return self._tokens

    @property
    def head_token(self):
        return self._head_token

    @property
    def ner(self):
        return self._ner

    def __eq__(self, other):
        if not isinstance(other, Mention):
            return False
        else:
            return self.sent_idx == other.sent_idx and \
                   self.start_token_idx == other.start_token_idx and \
                   self.end_token_idx == other.end_token_idx and \
                   self.head_token_idx == other.head_token_idx and \
                   self.rep == other.rep and self.ner == other.ner and \
                   all(token == other_token for token, other_token
                       in zip(self.tokens, other.tokens)) and \
                   self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_core_argument(self, use_lemma=True):
        word = self.head_token.get_representation(use_lemma=use_lemma)
        pos = self.head_token.pos
        ner = self.ner
        return CoreArgument(word, pos, ner)

    def get_representation(self, use_lemma=True):
        return self.head_token.get_representation(use_lemma=use_lemma)

    def to_text(self):
        return '{}:{}:{}:{}:{}:{}:{}'.format(
            self.sent_idx,
            self.start_token_idx,
            self.end_token_idx,
            self.head_token_idx,
            1 if self.rep else 0,
            self.ner if self.ner != '' else 'NONE',
            ':'.join([token.to_text() for token in self.tokens])
        )

    @classmethod
    def from_text(cls, text):
        parts = [p.strip() for p in text.split(':')]
        assert len(parts) >= 7, \
            'expected at least 7 parts separated by colon, found {}'.format(
                len(parts))

        sent_idx = int(parts[0])
        start_token_idx = int(parts[1])
        end_token_idx = int(parts[2])
        head_token_idx = int(parts[3])
        rep = True if int(parts[4]) == 1 else False
        ner = parts[5] if parts[5] != 'NONE' else ''
        tokens = [Token.from_text(token_text.strip()) for token_text
                  in parts[6:]]

        return cls(sent_idx, start_token_idx, end_token_idx, head_token_idx,
                   rep, ner, tokens)

    @classmethod
    def from_mention(cls, mention):
        check_type(mention, document.Mention)

        sent_idx = mention.sent_idx
        start_token_idx = mention.start_token_idx
        end_token_idx = mention.end_token_idx
        head_token_idx = mention.head_token_idx
        rep = mention.rep
        # NOBUG: just use ner of the head token, should be correct
        ner = mention.head_token.ner
        tokens = [Token.from_token(token) for token in mention.tokens]

        return cls(sent_idx, start_token_idx, end_token_idx, head_token_idx,
                   rep, ner, tokens)
