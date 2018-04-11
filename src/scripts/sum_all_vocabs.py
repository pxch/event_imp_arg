import argparse
from collections import defaultdict, Counter
from os import listdir
from os.path import isdir, join

from utils import log, smart_file_handler
from utils import prune_counter, read_counter, write_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='directory to read vocabulary counts')
    parser.add_argument('output_path',
                        help='directory to write vocabulary counts')
    parser.add_argument('--min_count', type=int, default=5,
                        help='minimum count to keep the word')

    args = parser.parse_args()

    input_dirs = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isdir(join(args.input_path, f))])

    all_vocab = defaultdict(Counter)

    for input_dir in input_dirs:
        log.info('Reading vocabulary count from {}'.format(input_dir))
        with smart_file_handler(join(input_dir, 'argument.bz2'), 'r') as fin:
            all_vocab['argument'] += read_counter(fin)
            prune_counter(all_vocab['argument'], args.min_count)
        with smart_file_handler(join(input_dir, 'name_entity.bz2'), 'r') as fin:
            all_vocab['name_entity'] += read_counter(fin)
            prune_counter(all_vocab['name_entity'], args.min_count)
        with smart_file_handler(
                join(input_dir, 'name_entity_tag.bz2'), 'r') as fin:
            all_vocab['name_entity_tag'] += read_counter(fin)
            # prune_counter(all_vocab['name_entity_tag'], args.min_count)
        with smart_file_handler(join(input_dir, 'predicate.bz2'), 'r') as fin:
            all_vocab['predicate'] += read_counter(fin)
            prune_counter(all_vocab['predicate'], args.min_count)
        with smart_file_handler(join(input_dir, 'preposition.bz2'), 'r') as fin:
            all_vocab['preposition'] += read_counter(fin)
            prune_counter(all_vocab['preposition'], args.min_count)

    for key in all_vocab:
        fout = smart_file_handler(join(args.output_path, key + '.bz2'), 'w')
        write_counter(all_vocab[key], fout)
        fout.close()
