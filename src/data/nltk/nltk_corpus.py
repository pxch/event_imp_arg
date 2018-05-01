from nltk.corpus import BracketParseCorpusReader
from nltk.corpus import NombankCorpusReader
from nltk.corpus import PropbankCorpusReader
from nltk.data import FileSystemPathPointer

from config import cfg

wsj_treebank = BracketParseCorpusReader(
    root=cfg.wsj_root,
    fileids=cfg.wsj_file_pattern,
    tagset='wsj',
    encoding='ascii'
)


def fileid_xform_function(fileid):
    # result = re.sub(r'^wsj/', '', fileid)
    # return result
    return fileid[4:]


propbank = PropbankCorpusReader(
    root=FileSystemPathPointer(cfg.propbank_root),
    propfile=cfg.propbank_file,
    framefiles=cfg.frame_file_pattern,
    verbsfile=cfg.propbank_verbs_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=wsj_treebank
)

nombank = NombankCorpusReader(
    root=FileSystemPathPointer(cfg.nombank_root),
    nomfile=cfg.nombank_file,
    framefiles=cfg.frame_file_pattern,
    nounsfile=cfg.nombank_nouns_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=wsj_treebank
)
