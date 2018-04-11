import argparse
from os import listdir
from os.path import isfile, join, dirname, realpath

from common.event_script import ScriptCorpus
from config import cfg
from data.event_comp_dataset import RichScript
from utils import log, read_vocab_list, smart_file_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='directory to read ScriptCorpus files')
    parser.add_argument('output_path', help='path to write training sequence')
    parser.add_argument('--pred_vocab', help='path to predicate vocab file')
    parser.add_argument('--arg_vocab', help='path to argument vocab file')
    parser.add_argument('--ner_vocab', help='path to name entity vocab file')
    parser.add_argument('--prep_vocab', help='path to preposition vocab file')
    parser.add_argument(
        '-v', '--verbose', help='print all document names', action='store_true')

    args = parser.parse_args()

    fout = smart_file_handler(args.output_path, 'w')

    input_files = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isfile(join(args.input_path, f)) and f.endswith('.bz2')])

    cur_dir_path = dirname(realpath(__file__))

    if args.pred_vocab:
        pred_vocab_list = read_vocab_list(args.pred_vocab)
    else:
        pred_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.pred_vocab_list_file))
    if args.arg_vocab:
        arg_vocab_list = read_vocab_list(args.arg_vocab)
    else:
        arg_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.arg_vocab_list_file))
    if args.ner_vocab:
        ner_vocab_list = read_vocab_list(args.ner_vocab)
    else:
        ner_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.ner_vocab_list_file))
    if args.prep_vocab:
        prep_vocab_list = read_vocab_list(args.prep_vocab)
    else:
        prep_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.prep_vocab_list_file))

    for input_f in input_files:
        if args.verbose:
            log.info('Processing file {}'.format(input_f))
        with smart_file_handler(input_f, 'r') as fin:
            script_corpus = ScriptCorpus.from_text(fin.read())
            for script in script_corpus.scripts:
                if args.verbose:
                    log.info('Reading script {}'.format(script.doc_name))
                rich_script = RichScript.build(
                    script,
                    prep_vocab_list=prep_vocab_list,
                    use_lemma=True,
                    filter_stop_events=False
                )
                sequence = rich_script.get_word2vec_training_seq(
                    pred_vocab_list=pred_vocab_list,
                    arg_vocab_list=arg_vocab_list,
                    ner_vocab_list=ner_vocab_list,
                    include_type=True,
                    include_all_pobj=True
                )
                if sequence:
                    fout.write(' '.join(sequence) + '\n')

    fout.close()
