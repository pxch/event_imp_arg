from collections import defaultdict

from utils import check_type, log


class Dependency(object):
    def __init__(self, label, head_idx, mod_idx, extra=False):
        self._label = label.encode('ascii', 'ignore')
        self._head_idx = head_idx
        self._mod_idx = mod_idx
        self._extra = extra
        # for dependencies converted from Ontonotes corpora
        if self._label == 'nsubjpass:xsubj':
            self._extra = True

    @property
    def label(self):
        return self._label

    @property
    def head_idx(self):
        return self._head_idx

    @property
    def mod_idx(self):
        return self._mod_idx

    @property
    def extra(self):
        return self._extra

    def __str__(self):
        return '{}-{}-{}'.format(self._label, self._head_idx, self._mod_idx)


class DependencyGraph(object):
    def __init__(self, sent_idx, num_tokens):
        self._sent_idx = sent_idx
        self._num_tokens = num_tokens
        # edges {label: [list of modifier token indices]} of every token
        # in which the token is the head
        self._head_edges = None
        # edges {label: [list of head token indices]} of every token
        # in which the token is the modifier
        self._mod_edges = None

    def build(self, deps):
        self._head_edges = [defaultdict(list) for _ in range(self._num_tokens)]
        self._mod_edges = [defaultdict(list) for _ in range(self._num_tokens)]

        for dep in deps:
            check_type(dep, Dependency)
            self._head_edges[dep.head_idx][dep.label].append(
                (dep.mod_idx, dep.extra))
            self._mod_edges[dep.mod_idx][dep.label].append(
                (dep.head_idx, dep.extra))

    def pretty_print(self):
        result = 'Number of tokens: {}'.format(self._num_tokens)
        result += '\nEdges from head to modifier'
        for idx in range(self._num_tokens):
            result += '\n\tToken #{}\t'.format(idx)
            result += '\t'.join(['{} = {}'.format(label, edges)
                                 for label, edges in
                                 self._head_edges[idx].items()])
        result += '\nEdges from modifier to head'
        for idx in range(self._num_tokens):
            result += '\n\tToken #{}\t'.format(idx)
            result += '\t'.join(['{} = {}'.format(label, edges)
                                 for label, edges in
                                 self._mod_edges[idx].items()])
        return result

    def check_token_idx(self, token_idx):
        assert 0 <= token_idx < self._num_tokens, \
            'In sentence #{}, token idx {} out of range'.format(
                self._sent_idx, token_idx)

    def lookup_label(self, direction, token_idx, dep_label, include_extra=True):
        assert direction in ['head', 'mod'], 'direction can only be head/mod'
        self.check_token_idx(token_idx)
        if direction == 'head':
            edges = self._head_edges[token_idx].get(dep_label, [])
        else:
            edges = self._mod_edges[token_idx].get(dep_label, [])
        if include_extra:
            results = [idx for idx, extra in edges]
        else:
            results = [idx for idx, extra in edges if not extra]
        return results

    def lookup(self, direction, token_idx, include_extra=True):
        assert direction in ['head', 'mod'], 'direction can only be head/mod'
        self.check_token_idx(token_idx)
        if direction == 'head':
            all_edges = self._head_edges[token_idx]
        else:
            all_edges = self._mod_edges[token_idx]
        results = {}
        for label, edges in all_edges.items():
            if include_extra:
                indices = [idx for idx, extra in edges]
            else:
                indices = [idx for idx, extra in edges if not extra]
            if indices:
                results[label] = indices
        return results

    # get the parent token index and corresponding dependency label
    # of the input token, return -1 if the input token is root
    def get_parent(self, token_idx, msg_prefix=''):
        parent = self.lookup('mod', token_idx, include_extra=False)
        if len(parent) == 0:
            return 'root', -1
        if len(parent) > 1 or len(parent.items()[0][1]) > 1:
            log.warning(
                '{}: In sentence #{}, token #{} has more than 1 non-extra '
                'head token: {}'.format(
                    msg_prefix, self._sent_idx, token_idx, parent))
        return parent.items()[0][0], parent.items()[0][1][0]

    # get the path [(label, parent_idx)] from the input token to the root
    def get_root_path(self, token_idx, msg_prefix=''):
        root_path = []
        current_idx = token_idx
        while current_idx != -1:
            label, parent_idx = self.get_parent(current_idx, msg_prefix)
            root_path.append((label, parent_idx))
            current_idx = parent_idx
        return root_path

    # get the head token index from a range of tokens
    def get_head_token_idx(self, start_token_idx, end_token_idx, msg_prefix=''):
        self.check_token_idx(start_token_idx)
        self.check_token_idx(end_token_idx - 1)
        assert start_token_idx < end_token_idx, \
            'start_token_idx:{} >= end_token_idx:{}'.format(
                start_token_idx, end_token_idx)
        head_idx_map = []
        for token_idx in range(start_token_idx, end_token_idx):
            head_trace = [token_idx]
            while start_token_idx <= head_trace[-1] < end_token_idx:
                _, head_idx = self.get_parent(head_trace[-1], msg_prefix)
                # warn if there is a loop in finding one token's head token
                if head_idx in head_trace:
                    log.warning(
                        '{}: In sentence #{}, token #{} has loop in its head '
                        'trace list.'.format(
                            msg_prefix, self._sent_idx, token_idx))
                    break
                head_trace.append(head_idx)
            head_idx_map.append((token_idx, head_trace[-2]))
        head_idx_list = [head_idx for _, head_idx in head_idx_map]
        # warn if the tokens in the range don't have the same head token
        if min(head_idx_list) != max(head_idx_list):
            log.warning(
                '{}: In sentence #{}, tokens within the range [{}, {}] do not '
                'have the same head token'.format(
                    msg_prefix, self._sent_idx, start_token_idx, end_token_idx))
        return min(head_idx_list)
