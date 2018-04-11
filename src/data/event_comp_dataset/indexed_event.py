import abc
from copy import deepcopy

from rich_entity import EntitySalience
from utils import check_type


class BaseIndexedEvent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, pred_input, subj_input, obj_input):
        # FIXME
        # assert pred_input != -1, 'predicate input cannot be -1 (non indexed)'
        self.pred_input = pred_input
        self.subj_input = subj_input
        self.obj_input = obj_input

    def __str__(self):
        return self.to_text()

    @abc.abstractmethod
    def __repr__(self):
        return

    @abc.abstractmethod
    def to_text(self):
        return

    def get_predicate(self):
        return self.pred_input

    @abc.abstractmethod
    def get_all_argument(self):
        return

    @abc.abstractmethod
    def get_argument(self, arg_idx):
        return

    @abc.abstractmethod
    def set_argument(self, arg_idx, arg_input):
        return


class IndexedEvent(BaseIndexedEvent):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input):
        super(IndexedEvent, self).__init__(pred_input, subj_input, obj_input)
        self.pobj_input = pobj_input

    def __repr__(self):
        return 'Indexed Event: ' + self.to_text()

    def to_text(self):
        return '{},{},{},{}'.format(
            self.pred_input, self.subj_input, self.obj_input, self.pobj_input)

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) == 4, \
            'expecting 4 parts separated by ",", found {}'.format(len(parts))
        pred_input = int(parts[0])
        subj_input = int(parts[1])
        obj_input = int(parts[2])
        pobj_input = int(parts[3])
        return cls(pred_input, subj_input, obj_input, pobj_input)

    def get_all_argument(self):
        return [self.subj_input, self.obj_input, self.pobj_input]

    def get_argument(self, arg_idx):
        assert arg_idx in [1, 2, 3], \
            'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
            'or 3 (for pobj_input)'
        if arg_idx == 1:
            return self.subj_input
        elif arg_idx == 2:
            return self.obj_input
        else:
            return self.pobj_input

    def set_argument(self, arg_idx, arg_input):
        assert arg_idx in [1, 2, 3], \
            'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
            'or 3 (for pobj_input)'
        if arg_idx == 1:
            self.subj_input = arg_input
        elif arg_idx == 2:
            self.obj_input = arg_input
        else:
            self.pobj_input = arg_input


class IndexedEventMultiPobj(BaseIndexedEvent):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input_list):
        super(IndexedEventMultiPobj, self).__init__(
            pred_input, subj_input, obj_input)
        self.pobj_input_list = pobj_input_list

    def __repr__(self):
        return 'Indexed Event (Multiple Pobj): ' + self.to_text()

    def to_text(self):
        return '{},{},{}{}'.format(
            self.pred_input, self.subj_input, self.obj_input,
            ',{}'.format(','.join(
                [str(pobj_input) for pobj_input in self.pobj_input_list]))
            if self.pobj_input_list else '')

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) >= 3, \
            'expecting at least 3 parts separated by ",", found {}'.format(
                len(parts))
        pred_input = int(parts[0])
        subj_input = int(parts[1])
        obj_input = int(parts[2])
        pobj_input_list = []
        for p in parts[3:]:
            pobj_input_list.append(int(p))
        return cls(pred_input, subj_input, obj_input, pobj_input_list)

    def get_all_argument(self):
        return [self.subj_input, self.obj_input] + self.pobj_input_list

    def get_argument(self, arg_idx):
        # skip arg_idx 3 to avoid confusion with IndexedEvent
        assert arg_idx in [1, 2] or \
               (arg_idx - 4) in range(len(self.pobj_input_list)), \
               'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
               'or 4, 5, ... (for all arguments in pobj_input_list)'
        if arg_idx == 1:
            return self.subj_input
        elif arg_idx == 2:
            return self.obj_input
        else:
            return self.pobj_input_list[arg_idx - 4]

    def set_argument(self, arg_idx, arg_input):
        # skip arg_idx 3 to avoid confusion with IndexedEvent
        assert arg_idx in [1, 2] or \
               (arg_idx - 4) in range(len(self.pobj_input_list)), \
               'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
               'or 4, 5, ... (for all arguments in pobj_input_list)'
        if arg_idx == 1:
            self.subj_input = arg_input
        elif arg_idx == 2:
            self.obj_input = arg_input
        else:
            self.pobj_input_list[arg_idx - 4] = arg_input


class IndexedEventTriple(object):
    def __init__(self, left_event, pos_event, neg_event, pos_arg_idx,
                 neg_arg_idx, pos_salience, neg_salience):
        check_type(left_event, IndexedEvent)
        self.left_event = deepcopy(left_event)
        check_type(pos_event, IndexedEvent)
        self.pos_event = deepcopy(pos_event)
        check_type(neg_event, IndexedEvent)
        self.neg_event = deepcopy(neg_event)

        assert pos_arg_idx in [1, 2, 3], \
            'pos_arg_type must be 1 (for subj), 2 (for obj), or 3 (for pobj)'
        self.pos_arg_idx = pos_arg_idx
        assert neg_arg_idx in [1, 2, 3], \
            'neg_arg_type must be 1 (for subj), 2 (for obj), or 3 (for pobj)'
        self.neg_arg_idx = neg_arg_idx

        # extra features for entity salience
        check_type(pos_salience, EntitySalience)
        self.pos_salience = pos_salience
        check_type(neg_salience, EntitySalience)
        self.neg_salience = neg_salience

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'Indexed Event Triple: ' + self.to_text()

    def to_text(self):
        return ' / '.join([self.left_event.to_text(), self.pos_event.to_text(),
                           self.neg_event.to_text(), str(self.pos_arg_idx),
                           str(self.neg_arg_idx), self.pos_salience.to_text(),
                           self.neg_salience.to_text()])

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(' / ')
        assert len(parts) in [6, 7], \
            'expecting 6 (old style) or 7 parts separated by " / ", ' \
            'found {}'.format(len(parts))
        left_event = IndexedEvent.from_text(parts[0])
        pos_event = IndexedEvent.from_text(parts[1])
        neg_event = IndexedEvent.from_text(parts[2])
        # TODO: remove support of old-style indexed event triple corpus
        if len(parts) == 6:
            pos_arg_idx = int(parts[3])
            neg_arg_idx = pos_arg_idx
            pos_salience = EntitySalience.from_text(parts[4])
            neg_salience = EntitySalience.from_text(parts[5])
        else:
            pos_arg_idx = int(parts[3])
            neg_arg_idx = int(parts[4])
            pos_salience = EntitySalience.from_text(parts[5])
            neg_salience = EntitySalience.from_text(parts[6])
        return cls(left_event, pos_event, neg_event, pos_arg_idx, neg_arg_idx,
                   pos_salience, neg_salience)
