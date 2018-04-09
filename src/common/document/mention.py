from dependency import DependencyGraph
from utils import check_type, log


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx, **kwargs):
        # index of the sentence where the mention is located\
        self._sent_idx = sent_idx

        assert 0 <= start_token_idx < end_token_idx, \
            'invalid: start_token_idx = {}, end_token_idx = {}'.format(
                start_token_idx, end_token_idx)
        # inclusive index of the first token of the mention in the sentence
        self._start_token_idx = start_token_idx
        # non-inclusive index of the last token of the mention in the sentence
        self._end_token_idx = end_token_idx

        # index of the head token of the mention in the sentence
        self._head_token_idx = -1
        if 'head_token_idx' in kwargs:
            self.head_token_idx = kwargs['head_token_idx']
        # boolean indicator of whether the mention is a representative mention
        self._rep = False
        if 'rep' in kwargs:
            self.rep = kwargs['rep']
        # text representation of the mention
        self._text = ''
        if 'text' in kwargs:
            self.text = kwargs['text']

        # index of the coreference chain which the mention belongs to
        self._coref_idx = -1
        # index of the mention in the coreference chain
        self._mention_idx = -1

        # list of all tokens in the mention
        # set in self.add_token_info() method
        self._tokens = []
        # pointer to the head token of the mention
        # set in self.add_token_info() method
        self._head_token = None
        # word form of the head token of the mention
        # set in self.add_token_info() method
        self._head_text = ''

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

    @head_token_idx.setter
    def head_token_idx(self, head_token_idx):
        assert type(head_token_idx) == int, 'head_token_idx must be an integer'
        assert self.start_token_idx <= head_token_idx < self.end_token_idx, \
            'head_token_idx must be between start_token_idx and end_token_idx'
        self._head_token_idx = head_token_idx

    @property
    def rep(self):
        return self._rep

    @rep.setter
    def rep(self, rep):
        assert type(rep) == bool, 'rep must be a boolean value'
        self._rep = rep

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    @property
    def coref_idx(self):
        return self._coref_idx

    @coref_idx.setter
    def coref_idx(self, coref_idx):
        assert type(coref_idx) == int and coref_idx >= 0, \
            'coref_idx must be a non-negative integer'
        self._coref_idx = coref_idx

    @property
    def mention_idx(self):
        return self._mention_idx

    @mention_idx.setter
    def mention_idx(self, mention_idx):
        assert type(mention_idx) == int and mention_idx >= 0, \
            'mention_idx must be a non-negative integer'
        self._mention_idx = mention_idx

    @property
    def tokens(self):
        return self._tokens

    @property
    def head_token(self):
        return self._head_token

    @property
    def head_text(self):
        return self._head_text

    def __str__(self):
        return '{}-{}-{}-{}-{}-{}'.format(
            self._sent_idx, self._start_token_idx, self._end_token_idx,
            self._head_token_idx, self._rep, self._text)

    def has_same_span(self, other):
        if not isinstance(other, Mention):
            return False
        else:
            return self.sent_idx == other.sent_idx and \
                   self.start_token_idx == other.start_token_idx and \
                   self.end_token_idx == other.end_token_idx

    def set_head_token_idx(self, dep_graph):
        check_type(dep_graph, DependencyGraph)
        if self.head_token_idx != -1:
            log.warning('Overriding existing head_token_idx {}'.format(
                self.head_token_idx))
        self.head_token_idx = dep_graph.get_head_token_idx(
            self.start_token_idx, self.end_token_idx)

    def add_token_info(self, token_list):
        # set self.tokens
        if not self._tokens:
            self._tokens = token_list[self.start_token_idx:self.end_token_idx]
        # if self.text is not set, set it to the concatenation of the word
        # forms of all tokens
        if self.text == '':
            self.text = ' '.join([token.word for token in self.tokens])
        if self.head_token_idx != -1 and self.head_token is None:
            self._head_token = token_list[self.head_token_idx]
            self._head_text = self.head_token.word
