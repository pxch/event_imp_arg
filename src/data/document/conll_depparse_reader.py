from copy import deepcopy

from common.document import Dependency
from common.document import Sentence
from common.document import Token
from utils import log, smart_file_handler


def read_conll_depparse(filename):
    fin = smart_file_handler(filename, 'r')

    all_sents = []
    sent_idx = 0
    sent = Sentence(sent_idx)

    for line_idx, line in enumerate(fin.readlines()):
        if line == '\n':
            all_sents.append(deepcopy(sent))
            sent_idx += 1
            sent = Sentence(sent_idx)
        else:
            items = line.strip().split('\t')
            try:
                token_idx = int(items[0])
            except ValueError:
                continue
            if token_idx == sent.num_tokens:
                log.warning(
                    'line #{} ({}) has duplicated token index, ignored.'.format(
                        line_idx, line.strip().replace('\t', ' ')))
                continue
            word = items[1]
            lemma = items[2]
            pos = items[4]
            sent.add_token(Token(word, lemma, pos))
            try:
                head_idx = int(items[6])
            except ValueError:
                continue
            dep_label = items[7]
            if dep_label != 'root':
                sent.add_dep(Dependency(
                    label=dep_label,
                    head_idx=head_idx - 1,
                    mod_idx=token_idx - 1,
                    extra=False))
            if items[8] != '_':
                for e_dep in items[8].strip().split('|'):
                    try:
                        e_dep_head_idx = int(e_dep.split(':')[0])
                    except ValueError:
                        continue
                    e_dep_label = ':'.join(e_dep.split(':')[1:])
                    sent.add_dep(Dependency(
                        label=e_dep_label,
                        head_idx=e_dep_head_idx - 1,
                        mod_idx=token_idx - 1,
                        extra=True))

    return all_sents
