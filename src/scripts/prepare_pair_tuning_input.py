import argparse
from os import listdir
from os.path import isfile, join, dirname, realpath

from common.event_script import ScriptCorpus
from config import cfg
from data.event_comp_dataset import RichScript
from utils import Word2VecModel
from utils import log, read_vocab_count, read_vocab_list, smart_file_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='directory for ScriptCorpus files')
    parser.add_argument('output_path', help='path to write training sequence')
    parser.add_argument('word2vec', help='path to word2vec vector file')
    parser.add_argument('word2vec_vocab', help='path to word2vec vocab file')
    parser.add_argument('--prep_vocab', help='path to preposition vocab file')
    parser.add_argument('--use_lemma', action='store_true',
                        help='if turned on, use the lemma form of a token,'
                             'otherwise use the word form')
    parser.add_argument('--subsampling', action='store_true',
                        help='if turned on, most frequent predicates would be '
                             'randomly subsampled according to their frequency')
    parser.add_argument('--pair_type_list', default='tf_arg',
                        help='comma separated list of pair types, options: '
                             'tf_arg (true arg vs fake arg) '
                             'wo_arg (w/ arg vs w/o arg) '
                             'two_args (cand_a as arg_i vs cand_a as arg_j)')
    parser.add_argument('--left_sample_type', default='one',
                        help='how to sample left event: '
                             'one (one left event for one pos/neg pair), '
                             'all (all left events for one pos/neg pair)')
    parser.add_argument('--neg_sample_type', default='one',
                        help='how to sample negative event (only in tf_arg): '
                             'one (one negative event for one positive event) '
                             'all (all negative events for one positive event)')
    parser.add_argument('-v', '--verbose',
                        help='print all document names', action='store_true')

    args = parser.parse_args()

    fout = smart_file_handler(args.output_path, 'w')

    input_files = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isfile(join(args.input_path, f)) and f.endswith('.bz2')])

    model = Word2VecModel.load_model(
        args.word2vec, fvocab=args.word2vec_vocab, binary=True)

    cur_dir_path = dirname(realpath(__file__))

    if args.prep_vocab:
        prep_vocab_list = read_vocab_list(args.prep_vocab)
    else:
        prep_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.prep_vocab_list_file))

    pred_count_dict = None
    if args.subsampling:
        pred_count_dict = read_vocab_count(
            join(cfg.vocab_path, cfg.pred_vocab_count_file))

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
                    use_lemma=args.use_lemma,
                    filter_stop_events=False
                )
                rich_script.get_index(model, include_type=True, use_unk=True,
                                      pred_count_dict=pred_count_dict)
                pair_tuning_inputs = rich_script.get_pair_input_list(
                    args.pair_type_list.split(','),
                    args.left_sample_type,
                    neg_sample_type=args.neg_sample_type,
                    rich_entities=rich_script.rich_entities,
                    model=model,
                    include_type=True,
                    use_unk=True
                )
                if len(pair_tuning_inputs) > 0:
                    fout.write('\n'.join(map(str, pair_tuning_inputs)) + '\n')

    fout.close()
