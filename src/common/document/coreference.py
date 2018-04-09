from mention import Mention
from utils import check_type, log


class Coreference(object):
    def __init__(self, idx):
        # index of the coreference chain in the document
        self._idx = idx
        # list of all mentions in the coreference chain
        self._mentions = []
        # number of all mentions in the coreference chain
        self._num_mentions = 0
        # pointer to the representative mention
        self._rep_mention = None

    @property
    def idx(self):
        return self._idx

    @property
    def mentions(self):
        return self._mentions

    @property
    def num_mentions(self):
        return self._num_mentions

    @property
    def rep_mention(self):
        assert self._rep_mention is None or self._rep_mention.rep
        return self._rep_mention

    @rep_mention.setter
    def rep_mention(self, rep_mention):
        check_type(rep_mention, Mention)
        if self._rep_mention is not None:
            if self._rep_mention.has_same_span(rep_mention):
                return
            else:
                log.warning('Overriding existing rep_mention ({})'.format(
                    self._rep_mention))
                self._rep_mention.rep = False
        self._rep_mention = rep_mention

    def add_mention(self, mention):
        check_type(mention, Mention)
        # set the coref_idx attrib of the mention
        mention.coref_idx = self.idx
        # set the mention_idx attrib of the mention
        mention.mention_idx = self.num_mentions
        self._mentions.append(mention)
        if mention.rep:
            self.rep_mention = mention
        self._num_mentions += 1

    def get_mention(self, idx):
        assert 0 <= idx < self.num_mentions, \
            'Mention index {} out of range'.format(idx)
        result = self._mentions[idx]
        check_type(result, Mention)
        return result

    def __str__(self):
        return ' '.join([str(mention) for mention in self._mentions])

    def pretty_print(self):
        return 'entity#{:0>3d}    {}'.format(self._idx, self.rep_mention.text)

    def find_rep_mention(self):
        for mention in self._mentions:
            assert mention.head_token is not None, \
                'Cannot find the representative mention unless ' \
                'all mentions have head_token set'
        # select mentions headed by proper nouns
        candidates = [mention for mention in self._mentions
                      if mention.head_token.pos.startswith('NNP')]
        # if no mentions are headed by proper nouns, select mentions containing
        # proper nouns
        if not candidates:
            candidates = [mention for mention in self._mentions
                          if any(token.pos.startswith('NNP')
                                 for token in mention.tokens)]
        # if no mentions are headed by proper nouns, select mentions headed
        # by common nouns
        if not candidates:
            candidates = [mention for mention in self._mentions
                          if mention.head_token.pos.startswith('NN')]
        # if no mentions are headed by either proper nouns or common noun,
        # use all mentions as candidates
        if not candidates:
            candidates = self._mentions

        # select from candidate mentions the one with longest text
        cand_length = [len(candidate.text) for candidate in candidates]
        rep_mention = candidates[cand_length.index(max(cand_length))]
        rep_mention.rep = True
        self.rep_mention = rep_mention
