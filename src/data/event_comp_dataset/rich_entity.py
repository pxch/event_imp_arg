from common.event_script import Entity
from core_argument import CoreArgument
from utils import check_type, consts


class EntitySalience(object):
    def __init__(self, **kwargs):
        for feature in consts.salience_features:
            if feature in kwargs:
                self.__dict__[feature] = kwargs[feature]
            else:
                self.__dict__[feature] = 0

    def get_feature(self, feature):
        assert feature in consts.salience_features
        return self.__dict__[feature]

    def get_feature_list(self, feature_list=None):
        if feature_list is None:
            feature_list = consts.salience_features
        result = []
        for feature in feature_list:
            result.append(self.get_feature(feature))
        return result

    def to_text(self):
        return ','.join(map(str, self.get_feature_list()))

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) == consts.num_salience_features, \
            'expecting {} parts separated by ",", found {}'.format(
                consts.num_salience_features, len(parts))
        kwargs = {}
        for feature, value in zip(consts.salience_features, parts):
            kwargs[feature] = int(value)
        return cls(**kwargs)


class RichEntity(object):
    def __init__(self, core, salience):
        check_type(core, CoreArgument)
        self.core = core
        check_type(salience, EntitySalience)
        self.salience = salience

    @classmethod
    def build(cls, entity, token_count_dict, use_lemma=True):
        check_type(entity, Entity)
        # get the representation and ner of the entity
        # core = entity.get_core_argument(use_lemma=use_lemma)

        word = entity.rep_mention.head_token.get_representation(
            use_lemma=use_lemma)
        pos = entity.rep_mention.head_token.pos
        # TODO: should I use entity.ner rather than entity.rep_mention.ner?
        ner = entity.rep_mention.ner
        core = CoreArgument(word, pos, ner)

        # get the sentence index of which the first mention is located
        first_loc = entity.mentions[0].sent_idx
        # get the count of the head word in the document
        head_count = token_count_dict.get(core.word, 0)
        # initialize number of named mentions / nominal mentions /
        # pronominal mentions / total mentions to be 0
        num_mentions_named = 0
        num_mentions_nominal = 0
        num_mentions_pronominal = 0
        num_mentions_total = 0
        # count different types of mentions
        for mention in entity.mentions:
            num_mentions_total += 1
            # add num_mentions_named if mention.ner is not empty
            if mention.ner != '':
                num_mentions_named += 1
            # add num_mentions_nominal if mention.pos starts with NN
            elif mention.head_token.pos.startswith('NN'):
                num_mentions_nominal += 1
            # add num_mentions_pronominal if mention.pos starts with PRP
            elif mention.head_token.pos.startswith('PRP'):
                num_mentions_pronominal += 1
        salience = EntitySalience(
            first_loc=first_loc,
            head_count=head_count,
            num_mentions_named=num_mentions_named,
            num_mentions_nominal=num_mentions_nominal,
            num_mentions_pronominal=num_mentions_pronominal,
            num_mentions_total=num_mentions_total
        )
        return cls(core, salience)

    def get_index(self, model, arg_type='', use_unk=True):
        return self.core.get_index(model, arg_type, use_unk=use_unk)

    def get_text_with_vocab_list(self, arg_vocab_list=None,
                                 ner_vocab_list=None):
        return self.core.get_text_with_vocab_list(
            arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)

    def get_salience(self):
        return self.salience
