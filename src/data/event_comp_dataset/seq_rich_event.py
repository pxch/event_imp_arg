from utils import Word2VecModel, check_type
from rich_predicate import RichPredicate
from seq_rich_argument import SeqRichArgument
from indexed_event import IndexedEvent
from copy import deepcopy
from common.event_script import Event


class SeqRichEvent(object):
    def __init__(self, rich_pred, rich_arg_list):
        check_type(rich_pred, RichPredicate)
        self.rich_pred = rich_pred

        for rich_arg in rich_arg_list:
            check_type(rich_arg, SeqRichArgument)
        self.rich_arg_list = rich_arg_list

    def get_index(self, model, include_type=True, use_unk=True,
                  pred_count_dict=None):
        check_type(model, Word2VecModel)
        self.rich_pred.get_index(
            model, include_type=include_type, use_unk=use_unk,
            pred_count_dict=pred_count_dict)
        for rich_arg in self.rich_arg_list:
            rich_arg.get_index(
                model, include_type=include_type, use_unk=use_unk)

    def get_argument(self, arg_idx):
        assert 0 <= arg_idx < len(self.rich_arg_list)
        return self.rich_arg_list[arg_idx]

    def has_neg(self, arg_idx):
        argument = self.get_argument(arg_idx)
        return argument.has_neg()

    def get_word2vec_training_seq(
        self, pred_vocab_list, arg_vocab_list, ner_vocab_list,
            include_type=True):
        sequence = [self.rich_pred.get_text(
            pred_vocab_list=pred_vocab_list, include_type=include_type)]
        for rich_arg in self.rich_arg_list:
            sequence.append(rich_arg.get_pos_text(
                arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list,
                include_type=include_type))
        return sequence

    def get_pos_input(self, arg_idx):
        pred_input = self.rich_pred.get_wv()
        if pred_input == -1:
            return None
        subj_input = -1
        obj_input = -1
        pobj_input_list = []
        for rich_arg in self.rich_arg_list:
            if rich_arg.arg_type == 'SUBJ':
                subj_input = rich_arg.get_pos_wv()
            elif rich_arg.arg_type == 'OBJ':
                obj_input = rich_arg.get_pos_wv()
            else:
                pobj_input_list.append(rich_arg.get_pos_wv())
        target_arg = self.get_argument(arg_idx)
        # TODO: should I just use the first pobj, or a random one, or all?
        if target_arg.arg_type in ['SUBJ', 'OBJ']:
            pobj_input = pobj_input_list[0] if pobj_input_list else -1
        else:
            pobj_input = target_arg.get_pos_wv()

        return IndexedEvent(pred_input, subj_input, obj_input, pobj_input)

    def get_pos_input_all(self):
        pos_input_all = []
        pred_input = self.rich_pred.get_wv()
        if pred_input == -1:
            return pos_input_all
        subj_input = -1
        obj_input = -1
        pobj_input_list = []
        for rich_arg in self.rich_arg_list:
            if rich_arg.arg_type == 'SUBJ':
                subj_input = rich_arg.get_pos_wv()
            elif rich_arg.arg_type == 'OBJ':
                obj_input = rich_arg.get_pos_wv()
            else:
                pobj_input_list.append(rich_arg.get_pos_wv())
        if pobj_input_list:
            for pobj_input in pobj_input_list:
                pos_input_all.append(
                    IndexedEvent(pred_input, subj_input, obj_input, pobj_input))
        else:
            pos_input_all.append(
                IndexedEvent(pred_input, subj_input, obj_input, -1))
        return pos_input_all

    def get_neg_input_list(self, arg_idx, include_salience=True):
        pos_input = self.get_pos_input(arg_idx)
        if not pos_input:
            return []
        neg_input_list = []
        if self.has_neg(arg_idx):
            target_arg = self.get_argument(arg_idx)
            if target_arg.arg_type == 'SUBJ':
                target_arg_idx = 1
            elif target_arg.arg_type == 'OBJ':
                target_arg_idx = 2
            else:
                target_arg_idx = 3
            neg_wv_list = target_arg.get_neg_wv_list()
            neg_salience_list = target_arg.get_neg_salience_list()
            for arg_wv, arg_salience in zip(neg_wv_list, neg_salience_list):
                neg_input = deepcopy(pos_input)

                neg_input.set_argument(target_arg_idx, arg_wv)
                if include_salience:
                    neg_input_list.append((neg_input, arg_salience))
                else:
                    neg_input_list.append(neg_input)
        return neg_input_list

    def get_eval_input_list_all(self, include_salience=True):
        if self.rich_pred.get_wv() == -1:
            return []
        eval_input_list_all = []
        for arg_idx, target_arg in enumerate(self.rich_arg_list):
            if self.has_neg(arg_idx):
                pos_input = self.get_pos_input(arg_idx)
                if pos_input is None:
                    continue
                eval_input_list = []
                if target_arg.arg_type == 'SUBJ':
                    target_arg_idx = 1
                elif target_arg.arg_type == 'OBJ':
                    target_arg_idx = 2
                else:
                    target_arg_idx = 3
                arg_wv_list = target_arg.get_all_wv_list()
                arg_salience_list = target_arg.get_all_salience_list()
                for arg_wv, arg_salience in zip(arg_wv_list, arg_salience_list):
                    eval_input = deepcopy(pos_input)
                    eval_input.set_argument(target_arg_idx, arg_wv)
                    if include_salience:
                        eval_input_list.append((eval_input, arg_salience))
                    else:
                        eval_input_list.append(eval_input)
                eval_input_list_all.append((target_arg, eval_input_list))
        return eval_input_list_all

    @classmethod
    def build(cls, event, rich_entity_list, entity_idx_list, prep_vocab_list,
              use_lemma=True, filter_repetitive_prep=False):
        check_type(event, Event)

        rich_pred = RichPredicate.build(event.pred, use_lemma=use_lemma)

        rich_arg_list = []

        if event.subj is not None:
            rich_arg_list.append(SeqRichArgument.build(
                'SUBJ', event.subj, rich_entity_list, entity_idx_list,
                use_lemma=use_lemma))

        if event.obj is not None:
            rich_arg_list.append(SeqRichArgument.build(
                'OBJ', event.obj, rich_entity_list, entity_idx_list,
                use_lemma=use_lemma))

        prep_list = []
        for prep, pobj in event.pobj_list:
            prep = prep if prep in prep_vocab_list else ''
            arg_type = 'PREP_' + prep if prep != '' else 'PREP'
            if (not filter_repetitive_prep) or prep not in prep_list:
                rich_arg_list.append(SeqRichArgument.build(
                    arg_type, pobj, rich_entity_list, entity_idx_list,
                    use_lemma=use_lemma))
            prep_list.append(prep)

        return cls(rich_pred, rich_arg_list)
