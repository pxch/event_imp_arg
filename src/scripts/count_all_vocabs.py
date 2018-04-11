import argparse
from collections import defaultdict, Counter
from os import listdir
from os.path import isfile, join

from common.event_script import ScriptCorpus
from utils import log, smart_file_handler, write_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='directory for ScriptCorpus files')
    parser.add_argument('output_path',
                        help='directory to write vocabulary counts')
    parser.add_argument('--use_lemma', action='store_true',
                        help='if turned on, use the lemma form of a token,'
                             'otherwise use the word form')
    parser.add_argument('-v', '--verbose',
                        help='print all document names', action='store_true')

    args = parser.parse_args()

    input_files = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isfile(join(args.input_path, f)) and f.endswith('.bz2')])

    all_vocab_count = defaultdict(Counter)

    for input_f in input_files:
        if args.verbose:
            log.info('Processing file {}'.format(input_f))
        with smart_file_handler(input_f, 'r') as fin:
            script_corpus = ScriptCorpus.from_text(fin.read())
            for script in script_corpus.scripts:
                if args.verbose:
                    log.info('Reading script {}'.format(script.doc_name))
                vocab_count = script.get_vocab_count(use_lemma=args.use_lemma)
                for key in vocab_count:
                    all_vocab_count[key] += vocab_count[key]

    for key in all_vocab_count:
        fout = smart_file_handler(join(args.output_path, key + '.bz2'), 'w')
        write_counter(all_vocab_count[key], fout)
        fout.close()
