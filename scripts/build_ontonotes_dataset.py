import argparse
from os import makedirs
from os.path import exists, join

from common.event_script import Script, ScriptCorpus
from data.ontonotes import load_ontonotes, read_all_docs_from_ontonotes
from utils import log

on_short_corpus_ids = ['english-bn-cnn', 'english-bn-voa', 'english-nw-xinhua']
on_long_corpus_ids = ['english-nw-wsj']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='path to store OntoNotes dataset')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print all document names')
    parser.add_argument('--suppress_warning', action='store_true',
                        help='suppress all warning messages from logger')

    args = parser.parse_args()

    if args.suppress_warning:
        log.warning = log.debug

    if not exists(args.output_path):
        makedirs(args.output_path)

    on_short_scripts = ScriptCorpus()

    for corpus_id in on_short_corpus_ids:
        corpus = load_ontonotes(corpus_id)
        log.info('Reading documents from {}'.format(corpus_id))
        docs = read_all_docs_from_ontonotes(corpus, verbose=args.verbose)
        log.info('Extracting event scripts from {}'.format(corpus_id))
        for doc in docs:
            on_short_scripts.add_script(Script.from_doc(doc))

    on_short_path = join(args.output_path, 'on_short_scripts.txt')
    log.info('Writing OnShort event scripts to {}'.format(on_short_path))
    with open(on_short_path, 'w') as fout:
        fout.write(on_short_scripts.to_text())

    on_long_scripts = ScriptCorpus()

    for corpus_id in on_long_corpus_ids:
        corpus = load_ontonotes(corpus_id)
        log.info('Reading documents from {}'.format(corpus_id))
        docs = read_all_docs_from_ontonotes(corpus, verbose=args.verbose)
        log.info('Extracting event scripts from {}'.format(corpus_id))
        for doc in docs:
            on_long_scripts.add_script(Script.from_doc(doc))

    on_long_path = join(args.output_path, 'on_long_scripts.txt')
    log.info('Writing OnLong event scripts to {}'.format(on_long_path))
    with open(on_long_path, 'w') as fout:
        fout.write(on_long_scripts.to_text())
