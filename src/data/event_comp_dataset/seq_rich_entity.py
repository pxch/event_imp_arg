from collections import defaultdict

from common.event_script import Entity
from core_argument import CoreArgument
from rich_entity import EntitySalience, RichEntity
from utils import check_type


class SeqRichEntity(object):
    def __init__(self, rep_idx, core_list, mention_type_list):
        assert 0 <= rep_idx < len(core_list), \
            'rep_idx {} out of range [0, {}]'.format(rep_idx, len(core_list))
        self.rep_idx = rep_idx

        for core in core_list:
            check_type(core, CoreArgument)
        self.core_list = core_list

        # assert len(sent_idx_list) == len(core_list), 'dimension mismatch!'
        # self.sent_idx_list = sent_idx_list
        #
        # assert len(head_count_list) == len(core_list), 'dimension mismatch!'
        # self.head_count_list = head_count_list

        assert len(mention_type_list) == len(core_list), 'dimension mismatch!'
        for mention_type in mention_type_list:
            assert mention_type in ['named', 'nominal', 'pronominal', 'other']
        self.mention_type_list = mention_type_list

    @classmethod
    def build(cls, entity, use_lemma=True):
        check_type(entity, Entity)

        rep_idx = -1
        core_list = []
        # send_idx_list = []
        # head_count_list = []
        mention_type_list = []

        for mention_idx, mention in enumerate(entity.mentions):
            if mention.rep:
                assert rep_idx == -1, 'found multiple representative mentions!'
                rep_idx = mention_idx

            word = mention.head_token.get_representation(use_lemma=use_lemma)
            pos = mention.head_token.pos
            ner = mention.ner
            core_list.append(CoreArgument(word, pos, ner))

            # send_idx_list.append(mention.sent_idx)
            # head_count_list.append(token_count_dict.get(word, 0))

            if ner != '':
                mention_type_list.append('named')
            elif pos.startswith('NN'):
                mention_type_list.append('nominal')
            elif pos.startswith('PRP'):
                mention_type_list.append('pronominal')
            else:
                mention_type_list.append('other')

        assert rep_idx != -1, 'found no representative mention!'

        return cls(rep_idx, core_list, mention_type_list)

    def get_rich_entity(self, mention_idx_list=None):
        if not mention_idx_list:
            mention_idx_list = range(0, len(self.core_list))

        # if rep_idx not in mention_idx_list, use the first mention
        # in mention_idx_list as rep_mention
        if self.rep_idx in mention_idx_list:
            rep_idx = self.rep_idx
        else:
            rep_idx = mention_idx_list[0]

        # get the core and head_count from rep_mention
        core = self.core_list[rep_idx]
        # head_count = self.head_count_list[rep_idx]

        # use sent_idx of the first mention in mention_idx_list as first_loc
        # first_loc = self.sent_idx_list[mention_idx_list[0]]

        num_mentions = defaultdict(int)

        for mention_idx in mention_idx_list:
            num_mentions['total'] += 1
            num_mentions[self.mention_type_list[mention_idx]] += 1

        salience = EntitySalience(
            num_mentions_named=num_mentions['named'],
            num_mentions_nominal=num_mentions['nominal'],
            num_mentions_pronominal=num_mentions['pronominal'],
            num_mentions_total=num_mentions['total']
        )

        return RichEntity(core, salience)
