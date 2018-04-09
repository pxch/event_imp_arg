import re

from argument import Argument
from predicate import Predicate
from utils import check_type, log


class Event(object):
    def __init__(self, pred, subj, dobj, pobj_list):
        # TODO: add sent_idx attribute
        check_type(pred, Predicate)
        self._pred = pred

        if subj is not None:
            check_type(subj, Argument)
        self._subj = subj

        if dobj is not None:
            check_type(dobj, Argument)
        self._dobj = dobj

        if not all(prep != '' for prep, _ in pobj_list):
            log.warning('some of prepositions in pobj_list are empty')
        for idx, (prep, pobj) in enumerate(pobj_list):
            if prep == '':
                log.warning(
                    'Empty preposition found in #{} of pobj_list'.format(idx))
            check_type(pobj, Argument)
        self._pobj_list = pobj_list

    @property
    def pred(self):
        return self._pred

    @property
    def subj(self):
        return self._subj

    @property
    def dobj(self):
        return self._dobj

    @property
    def pobj_list(self):
        return self._pobj_list

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        else:
            return self.pred == other.pred and \
                   self.subj == other.subj and \
                   self.dobj == other.dobj and \
                   all(prep == other_prep and pobj == other_pobj
                       for (prep, pobj), (other_prep, other_pobj)
                       in zip(self.pobj_list, other.pobj_list))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_all_args(self, include_arg_type=False):
        # TODO: change OBJ to DOBJ to avoid confusion
        all_args = [('SUBJ', self.subj), ('OBJ', self.dobj)] + \
                   [('PREP_' + prep, pobj) for prep, pobj in self.pobj_list]
        all_args = [arg for arg in all_args if arg[1] is not None]
        if include_arg_type:
            return all_args
        else:
            return [arg[1] for arg in all_args]

    def get_all_args_with_entity(self, include_arg_type=False):
        all_args = self.get_all_args(include_arg_type=True)
        all_args = [arg for arg in all_args if arg[1].has_entity()]
        if include_arg_type:
            return all_args
        else:
            return [arg[1] for arg in all_args]

    def get_all_args_without_entity(self, include_arg_type=False):
        all_args = self.get_all_args(include_arg_type=True)
        all_args = [arg for arg in all_args if not arg[1].has_entity()]
        if include_arg_type:
            return all_args
        else:
            return [arg[1] for arg in all_args]

    def to_text(self):
        # TODO: change OBJ to DOBJ to avoid confusion
        return '{} :SUBJ: {} :OBJ: {}{}'.format(
            self.pred.to_text(),
            self.subj.to_text() if self.subj is not None else 'NONE',
            self.dobj.to_text() if self.dobj is not None else 'NONE',
            ''.join([' :POBJ: {} : {}'.format(prep, pobj.to_text())
                     for prep, pobj in self.pobj_list])
        )

    @classmethod
    def from_text(cls, text):
        # TODO: change OBJ to DOBJ to avoid confusion
        parts = [p for p in re.split(' :(?:SUBJ|OBJ|POBJ): ', text)]
        assert len(parts) >= 3, \
            'expected at least 3 parts, separated by :(SUBJ|OBJ|POBJ):, ' \
            'found {}'.format(len(parts))

        pred = Predicate.from_text(parts[0])
        subj = None
        if parts[1] != 'NONE':
            subj = Argument.from_text(parts[1])
        obj = None
        if parts[2] != 'NONE':
            obj = Argument.from_text(parts[2])
        pobj_list = []
        if len(parts) > 3:
            for part in parts[3:]:
                prep, pobj = part.split(' : ')
                if prep != '':
                    pobj_list.append((prep, Argument.from_text(pobj)))

        return cls(pred, subj, obj, pobj_list)

    @classmethod
    def from_tokens(cls, pred_token, subj_token, obj_token,
                    pobj_token_list, neg=False, prt=''):
        pred = Predicate.from_token(pred_token, neg=neg, prt=prt)
        subj = None
        if subj_token is not None:
            subj = Argument.from_token(subj_token)
        obj = None
        if obj_token is not None:
            obj = Argument.from_token(obj_token)
        pobj_list = [(prep, Argument.from_token(pobj_token))
                     for prep, pobj_token in pobj_token_list]

        return cls(pred, subj, obj, pobj_list)
