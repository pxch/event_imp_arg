from nltk.tree import Tree

from dependency import Dependency, DependencyGraph
from token import Token
from utils import check_type, log


class Sentence(object):
    def __init__(self, idx):
        # index of the sentence in the document
        self._idx = idx
        # list of all tokens in the sentence
        self._tokens = []
        # number of tokens in the sentence
        self._num_tokens = 0
        # list of all dependencies in the sentence
        # excluding the root dependency
        self._deps = []
        # dependency graph built upon all dependencies
        self._dep_graph = None
        # constituency tree, could be empty
        self._tree = None

    @property
    def idx(self):
        return self._idx

    @property
    def tokens(self):
        return self._tokens

    @property
    def num_tokens(self):
        return self._num_tokens

    def add_token(self, token):
        check_type(token, Token)
        # set the sent_idx attrib of the token
        token.sent_idx = self.idx
        # set the token_idx attrib of the token
        token.token_idx = self.num_tokens
        self._tokens.append(token)
        self._num_tokens += 1

    def add_dep(self, dep):
        check_type(dep, Dependency)
        self._deps.append(dep)

    def get_token(self, idx):
        assert 0 <= idx < self.num_tokens, \
            'Token idx {} out of range'.format(idx)
        result = self._tokens[idx]
        check_type(result, Token)
        return result

    def build_dep_graph(self):
        self._dep_graph = DependencyGraph(self.idx, self.num_tokens)
        self._dep_graph.build(self._deps)

    @property
    def dep_graph(self):
        return self._dep_graph

    def lookup_label(self, direction, token_idx, dep_label):
        return [self.get_token(idx) for idx in
                self._dep_graph.lookup_label(direction, token_idx, dep_label)]

    # get list of all subjective token indices for a predicate
    def get_subj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('head', pred_idx, 'nsubj'))
        # agent of passive verb
        results.extend(self.lookup_label('head', pred_idx, 'nmod:agent'))
        # controlling subject
        results.extend(self.lookup_label('head', pred_idx, 'nsubj:xsubj'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all objective token indices for a predicate
    def get_dobj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('head', pred_idx, 'dobj'))
        # nsubjpass of passive verb
        results.extend(self.lookup_label('head', pred_idx, 'nsubjpass'))
        # TODO: include acl relation?
        # results.extend(self.lookup_label('mod', pred_idx, 'acl'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all prepositional objective token indices for a predicate
    def get_pobj_list(self, pred_idx):
        results = []
        # look for all nmod dependencies
        for label, indices in self._dep_graph.lookup('head', pred_idx).items():
            if label.startswith('nmod') and ':' in label:
                prep_label = label.split(':')[1]
                # exclude nmod:agent (subject)
                if prep_label != 'agent':
                    results.extend([(prep_label, self.get_token(idx))
                                    for idx in indices])
        return sorted(results, key=lambda pair: pair[1].token_idx)

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        check_type(tree, Tree)
        if self._tree is not None:
            log.warning('Overriding existing constituency tree')
        self._tree = tree

    def __str__(self):
        return ' '.join([str(token) for token in self._tokens]) + \
               '\t#DEP#\t' + ' '.join([str(dep) for dep in self._deps])

    def pretty_print(self):
        return ' '.join([token.pretty_print() for token in self._tokens]) + \
               '\n\t' + ' '.join([str(dep) for dep in self._deps])

    def plain_text(self):
        return ' '.join([token.word for token in self._tokens])
