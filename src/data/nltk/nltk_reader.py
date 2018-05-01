from collections import defaultdict

from config import cfg
from nltk_corpus import wsj_treebank, propbank, nombank
from utils import log


class PTBReader(object):
    def __init__(self):
        log.info('Building PTBReader from {}'.format(cfg.wsj_root))
        self.treebank = wsj_treebank
        log.info('Found {} files'.format(len(self.treebank.fileids())))

        self.all_sents = []
        self.all_tagged_sents = []
        self.all_parsed_sents = []
        self.treebank_fileid = ''

    def read_file(self, treebank_fileid):
        if treebank_fileid != self.treebank_fileid:
            self.all_sents = self.treebank.sents(fileids=treebank_fileid)
            self.all_tagged_sents = \
                self.treebank.tagged_sents(fileids=treebank_fileid)
            self.all_parsed_sents = \
                self.treebank.parsed_sents(fileids=treebank_fileid)
            self.treebank_fileid = treebank_fileid


class SemanticCorpusReader(object):
    def __init__(self, instances, indexing=False):
        self.instances = instances
        self.num_instances = len(self.instances)
        log.info('Found {} instances'.format(self.num_instances))

        self.instances_by_fileid = defaultdict(list)
        if indexing:
            self.build_index()

    def build_index(self):
        log.info('Building index by fileid for {}'.format(
            self.__class__.__name__))
        for instance in self.instances:
            fileid = self.convert_fileid(instance.fileid)
            self.instances_by_fileid[fileid].append(instance)

    @staticmethod
    def convert_fileid(fileid):
        # result = re.sub(r'^\d\d/', '', fileid)
        # result = re.sub(r'\.mrg$', '', result)
        # return result
        return fileid[3:11]

    def search_by_fileid(self, fileid):
        return self.instances_by_fileid.get(fileid, [])

    def search_by_pointer(self, pointer):
        for instance in self.search_by_fileid(pointer.fileid):
            if instance.sentnum == pointer.sentnum \
                    and instance.wordnum == pointer.tree_pointer.wordnum:
                return instance
        return None


class PropbankReader(SemanticCorpusReader):
    def __init__(self, indexing=False):
        log.info('Building PropbankReader from {}/{}'.format(
            cfg.propbank_root, cfg.propbank_file))
        super(PropbankReader, self).__init__(
            propbank.instances(), indexing=indexing)


class NombankReader(SemanticCorpusReader):
    def __init__(self, indexing=False):
        log.info('Building NombankReader from {}/{}'.format(
            cfg.nombank_root, cfg.nombank_file))
        super(NombankReader, self).__init__(
            nombank.instances(), indexing=indexing)
