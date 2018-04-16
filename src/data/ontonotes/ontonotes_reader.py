from copy import deepcopy
from os.path import join

from on.corpora import coreference, name, subcorpus

from common.document import Mention, Coreference, Document
from config import cfg
from data.document_reader import read_conll_depparse
from utils import check_type, convert_ontonotes_ner_tag, log

ontonotes_annotations_source = join(cfg.ontonotes_root, 'english/annotations')


def read_coref_link(coref_link):
    check_type(coref_link, coreference.coreference_link)

    mention = Mention(
        coref_link.sentence_index,
        coref_link.start_word_index,
        coref_link.end_word_index + 1)
    if coref_link.subtree is not None:
        mention.text = coref_link.subtree.get_trace_adjusted_word_string()
    return mention


def read_coref_chain(coref_idx, coref_chain):
    check_type(coref_chain, coreference.coreference_chain)

    coref = Coreference(coref_idx)
    for coref_link in coref_chain:
        if coref_link.end_word_index >= coref_link.start_word_index:
            coref.add_mention(read_coref_link(coref_link))
    return coref


def read_coref_doc(coref_doc):
    check_type(coref_doc, coreference.coreference_document)

    all_corefs = []
    coref_idx = 0
    for coref_chain in coref_doc:
        if coref_chain.type == 'IDENT':
            coref = read_coref_chain(coref_idx, coref_chain)
            if len(coref.mentions) > 0:
                all_corefs.append(deepcopy(coref))
                coref_idx += 1
    return all_corefs


def read_name_doc(name_doc):
    check_type(name_doc, name.name_tagged_document)

    all_name_entities = []
    for name_entity_set in name_doc:
        for name_entity_hash in name_entity_set:
            for name_entity in name_entity_hash:
                all_name_entities.append(name_entity)
    return all_name_entities


def add_name_entity_to_doc(doc, name_entity):
    check_type(doc, Document)
    check_type(name_entity, name.name_entity)

    sent = doc.get_sent(name_entity.sentence_index)
    for token_idx in range(
            name_entity.start_word_index, name_entity.end_word_index + 1):
        token = sent.get_token(token_idx)
        # map ontonotes ner tags to coarse grained ner tags
        token.ner = convert_ontonotes_ner_tag(name_entity.type)


def read_doc_from_ontonotes(coref_doc, name_doc, verbose=True):
    doc_id = coref_doc.document_id.split('@')[0]
    assert doc_id == name_doc.document_id.split('@')[0], \
        '{} and {} do not have the same document_id'.format(coref_doc, name_doc)

    if verbose:
        log.info('Reading ontonotes document {}'.format(doc_id))

    conll_file_path = join(ontonotes_annotations_source, doc_id + '.depparse')

    all_sents = read_conll_depparse(conll_file_path)

    all_corefs = read_coref_doc(coref_doc)

    doc_name = doc_id.split('/')[-1]
    doc = Document.construct(doc_name, all_sents, all_corefs)

    for name_entity in read_name_doc(name_doc):
        add_name_entity_to_doc(doc, name_entity)

    return doc


def read_all_docs_from_ontonotes(corpora, verbose=True):
    check_type(corpora, subcorpus)
    assert 'coref' in corpora, 'corpora must contains coref bank'
    assert 'name' in corpora, 'corpora must contains name bank'

    all_docs = []
    for coref_doc, name_doc in zip(corpora['coref'], corpora['name']):
        doc = read_doc_from_ontonotes(coref_doc, name_doc, verbose=verbose)
        all_docs.append(doc)

    return all_docs
