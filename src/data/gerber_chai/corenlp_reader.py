import pickle as pkl
from os.path import join

from common.event_script import Script
from config import cfg
from data.document_reader import read_corenlp_doc
from data.event_comp_dataset import RichScript
from helper import expand_wsj_fileid
from utils import log, read_vocab_list


class CoreNLPReader(object):
    def __init__(self, corenlp_dict):
        self.corenlp_dict = corenlp_dict

    def get_all(self, fileid):
        return self.corenlp_dict[fileid]

    def get_idx_mapping(self, fileid):
        return self.corenlp_dict[fileid][0]

    def get_doc(self, fileid):
        return self.corenlp_dict[fileid][1]

    def get_script(self, fileid):
        return self.corenlp_dict[fileid][2]

    def get_rich_script(self, fileid):
        return self.corenlp_dict[fileid][3]

    @classmethod
    def build(cls, instances, corenlp_root, verbose=False):
        prep_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.prep_vocab_list_file))

        log.info('Building CoreNLP Reader from {}'.format(corenlp_root))
        corenlp_dict = {}

        for instance in instances:
            pred_pointer = instance.pred_pointer
            if pred_pointer.fileid not in corenlp_dict:

                path = join(corenlp_root, 'idx',
                            expand_wsj_fileid(pred_pointer.fileid))
                idx_mapping = []
                with open(path, 'r') as fin:
                    for line in fin:
                        idx_mapping.append([int(i) for i in line.split()])

                path = join(corenlp_root, 'parsed',
                            expand_wsj_fileid(pred_pointer.fileid, '.xml.bz2'))
                doc = read_corenlp_doc(path, verbose=verbose)

                script = Script.from_doc(doc)

                rich_script = RichScript.build(
                    script,
                    prep_vocab_list=prep_vocab_list,
                    use_lemma=True,
                    filter_stop_events=False
                )

                corenlp_dict[pred_pointer.fileid] = \
                    (idx_mapping, doc, script, rich_script)

        log.info('Done')
        return cls(corenlp_dict)

    @classmethod
    def load(cls, corenlp_dict_path):
        log.info('Loading CoreNLP Reader from {}'.format(corenlp_dict_path))
        corenlp_dict = pkl.load(open(corenlp_dict_path, 'r'))
        log.info('Done')

        return cls(corenlp_dict)

    def save(self, corenlp_dict_path):
        log.info('Saving CoreNLP dict to {}'.format(corenlp_dict_path))
        pkl.dump(self.corenlp_dict, open(corenlp_dict_path, 'w'))
