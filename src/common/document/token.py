from coreference import Coreference
from mention import Mention
from utils import check_type, consts, log


class Token(object):
    def __init__(self, word, lemma, pos, **kwargs):
        # word form of the token
        self._word = word.encode('ascii', 'ignore')
        # lemma form of the token
        self._lemma = lemma.encode('ascii', 'ignore')
        # part-of-speech tag of the token
        self._pos = pos

        # name entity tag of the token
        self._ner = ''
        if 'ner' in kwargs:
            self.ner = kwargs['ner']

        # index of the sentence where the token is located, starts with 0
        self._sent_idx = -1
        # index of the token in the sentence, starts with 0
        self._token_idx = -1

        # the coreference chain which the token belongs to
        self._coref = None
        # the mention where the token belongs to
        self._mention = None

    @property
    def word(self):
        return self._word

    @property
    def lemma(self):
        return self._lemma

    @property
    def pos(self):
        return self._pos

    @property
    def ner(self):
        return self._ner

    @ner.setter
    def ner(self, ner):
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

    @property
    def sent_idx(self):
        return self._sent_idx

    @sent_idx.setter
    def sent_idx(self, sent_idx):
        assert type(sent_idx) == int and sent_idx >= 0, \
            'sent_idx must be a non-negative integer'
        self._sent_idx = sent_idx

    @property
    def token_idx(self):
        return self._token_idx

    @token_idx.setter
    def token_idx(self, token_idx):
        assert type(token_idx) == int and token_idx >= 0, \
            'token_idx must be a non-negative integer'
        self._token_idx = token_idx

    @property
    def coref(self):
        return self._coref

    @property
    def mention(self):
        return self._mention

    def coref_idx(self):
        return self.coref.idx if self.coref else -1

    def mention_idx(self):
        return self.mention.mention_idx if self.mention else -1

    def add_coref_info(self, coref, mention):
        check_type(coref, Coreference)
        check_type(mention, Mention)
        if self.mention is not None:
            log.warning(
                'Token ({}) has existing mention ({})'.format(
                    self.pretty_print(), self._mention))
            # TODO: maybe keep the shorter one in nested mentions?
            if self._mention.start_token_idx <= mention.start_token_idx and \
                    self._mention.end_token_idx >= mention.end_token_idx:
                log.warning(
                    'The new mention ({}) is nested in the existing mention, '
                    'ignore the new mention'.format(mention))
                return
            else:
                log.warning(
                    'Thew new mention ({}) has longer span than the existing '
                    'mention, override existing mention'.format(mention))
        self._coref = coref
        self._mention = mention

    def __str__(self):
        return '{}/{}/{}'.format(self.word, self.lemma, self.pos)

    def pretty_print(self):
        return '{}-{}'.format(self.token_idx, self.__str__())
