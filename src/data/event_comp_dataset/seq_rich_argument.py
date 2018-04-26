from common.event_script import Argument
from core_argument import CoreArgument
from rich_entity import RichEntity
from utils import Word2VecModel, check_type


class SeqRichArgument(object):
    def __init__(self, arg_type, core, rich_entity_list, entity_idx_list,
                 entity_idx, mention_idx):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        # type of argument
        self.arg_type = arg_type
        check_type(core, CoreArgument)
        # word and ner of the true argument
        self.core = core
        # word2vec index of the true argument
        self.core_wv = -1

        for rich_entity in rich_entity_list:
            check_type(rich_entity, RichEntity)
        self.rich_entity_list = rich_entity_list

        assert len(entity_idx_list) == len(rich_entity_list), \
            'dimension mismatch!'
        self.entity_idx_list = entity_idx_list

        self.entity_idx = entity_idx
        self.mention_idx = mention_idx

        if entity_idx_list:
            assert entity_idx != -1 and entity_idx in entity_idx_list, \
                'entity_idx_list {} provided but does not ' \
                'contain entity_idx {}'.format(entity_idx_list, entity_idx)
            self.target_idx = entity_idx_list.index(entity_idx)
        else:
            self.target_idx = -1

        self.entity_wv_list = []

    def get_index(self, model, include_type=True, use_unk=True):
        check_type(model, Word2VecModel)
        self.core_wv = self.core.get_index(
            model, self.arg_type if include_type else '', use_unk=use_unk)
        for rich_entity in self.rich_entity_list:
            self.entity_wv_list.append(rich_entity.get_index(
                model, self.arg_type if include_type else '', use_unk=use_unk))
        assert all(entity_wv != -1 for entity_wv in self.entity_wv_list)

    def has_neg(self):
        return len(self.rich_entity_list) > 1

    def is_first_mention(self):
        return self.mention_idx == 0

    def get_entity(self):
        return self.rich_entity_list[self.target_idx]

    def get_pos_text(self, arg_vocab_list=None, ner_vocab_list=None,
                     include_type=True):
        if self.rich_entity_list:
            pos_core = self.rich_entity_list[self.target_idx].core
        else:
            pos_core = self.core
        pos_text = pos_core.get_text_with_vocab_list(
            arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)
        if include_type:
            pos_text += '-' + self.arg_type
        return pos_text

    def get_pos_wv(self):
        if self.rich_entity_list:
            return self.entity_wv_list[self.target_idx]
        else:
            return self.core_wv

    def get_pos_salience(self):
        if self.rich_entity_list:
            return self.rich_entity_list[self.target_idx].get_salience()
        else:
            return None

    def get_neg_text_list(self, arg_vocab_list=None, ner_vocab_list=None,
                          include_type=True):
        neg_text_list = []
        if self.has_neg():
            for neg_entity in self.rich_entity_list[0:self.target_idx] + \
                              self.rich_entity_list[self.target_idx+1:]:
                neg_text = neg_entity.get_text_with_vocab_list(
                    arg_vocab_list=arg_vocab_list,
                    ner_vocab_list=ner_vocab_list)
                if include_type:
                    neg_text += '-' + self.arg_type
                neg_text_list.append(neg_text)
        return neg_text_list

    def get_neg_wv_list(self):
        neg_wv_list = []
        if self.has_neg():
            for neg_wv in self.entity_wv_list[0:self.target_idx] + \
                          self.entity_wv_list[self.target_idx+1:]:
                neg_wv_list.append(neg_wv)
        return neg_wv_list

    def get_neg_salience_list(self):
        neg_salience_list = []
        if self.has_neg():
            for neg_entity in self.rich_entity_list[0:self.target_idx] + \
                              self.rich_entity_list[self.target_idx + 1:]:
                neg_salience_list.append(neg_entity.get_salience())
        return neg_salience_list

    def get_all_wv_list(self):
        all_wv_list = []
        if self.has_neg():
            all_wv_list.extend(self.entity_wv_list)
        return all_wv_list

    def get_all_salience_list(self):
        all_salience_list = []
        if self.has_neg():
            for rich_entity in self.rich_entity_list:
                all_salience_list.append(rich_entity.get_salience())
        return all_salience_list

    def get_target_idx(self):
        return self.target_idx

    @classmethod
    def build(cls, arg_type, arg, rich_entity_list, entity_idx_list,
              use_lemma=True):
        check_type(arg, Argument)

        word = arg.get_representation(use_lemma=use_lemma)
        pos = arg.pos
        ner = arg.ner
        core = CoreArgument(word, pos, ner)

        if arg.entity_idx == -1 or arg.entity_idx not in entity_idx_list:
            return cls(
                arg_type, core,
                rich_entity_list=[],
                entity_idx_list=[],
                entity_idx=arg.entity_idx,
                mention_idx=arg.mention_idx)
        else:
            return cls(
                arg_type, core,
                rich_entity_list=rich_entity_list,
                entity_idx_list=entity_idx_list,
                entity_idx=arg.entity_idx,
                mention_idx=arg.mention_idx)
