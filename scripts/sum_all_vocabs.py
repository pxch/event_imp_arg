import argparse
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, join

from utils import log, prune_counter
from utils import read_vocab_count, write_vocab_count, write_vocab_list

non_preposition = ['tmod', 'npmod', '\'s']


def build_vocab(name, counter, min_count, output_path, save_count=False,
                save_list=False):
    log.info('Found {} {}(s)'.format(len(counter), name))
    prune_counter(counter, min_count)
    log.info('Keep {} {}(s) with counts over {}'.format(
        len(counter), name, min_count))

    count_path = join(output_path, '{}_min_{}_count'.format(name, min_count))
    if save_count:
        log.info('Save {} count to {}'.format(name, count_path))
        write_vocab_count(counter, count_path)

    list_path = join(output_path, '{}_min_{}'.format(name, min_count))
    if save_list:
        log.info('Save {} list to {}'.format(name, list_path))
        write_vocab_list(counter, list_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='directory to read vocabulary counts')
    parser.add_argument('output_path',
                        help='directory to write vocabulary counts')
    parser.add_argument('--keep_min', type=int, default=5,
                        help='minimum count to keep the word during summing')
    parser.add_argument('--predicate_min', type=int, default=100,
                        help='minimum count to keep predicates at last')
    parser.add_argument('--argument_min', type=int, default=500,
                        help='minimum count to keep arguments at last')
    parser.add_argument('--top_preposition', type=int, default=50,
                        help='number of top frequent prepositions to keep')

    args = parser.parse_args()

    input_dirs = sorted([
        join(args.input_path, f) for f in listdir(args.input_path)
        if isdir(join(args.input_path, f))])

    predicate_counter = Counter()
    argument_counter = Counter()
    name_entity_counter = Counter()
    preposition_counter = Counter()

    for input_dir in input_dirs:
        log.info('Reading vocabulary count from {}'.format(input_dir))
        predicate_counter += \
            read_vocab_count(join(input_dir, 'predicate.bz2'))
        prune_counter(predicate_counter, args.keep_min)

        argument_counter += \
            read_vocab_count(join(input_dir, 'argument.bz2'))
        prune_counter(argument_counter, args.keep_min)

        name_entity_counter += \
            read_vocab_count(join(input_dir, 'name_entity.bz2'))
        prune_counter(name_entity_counter, args.keep_min)

        preposition_counter += \
            read_vocab_count(join(input_dir, 'preposition.bz2'))
        prune_counter(preposition_counter, args.keep_min)

    if not isdir(args.output_path):
        makedirs(args.output_path)

    build_vocab(
        name='predicate',
        counter=predicate_counter,
        min_count=args.predicate_min,
        output_path=args.output_path,
        save_count=True,
        save_list=True
    )

    build_vocab(
        name='argument',
        counter=argument_counter,
        min_count=args.argument_min,
        output_path=args.output_path,
        save_count=True,
        save_list=True
    )

    build_vocab(
        name='name_entity',
        counter=name_entity_counter,
        min_count=args.argument_min,
        output_path=args.output_path,
        save_count=True,
        save_list=True
    )

    log.info('Filter non preposition words: {}'.format(non_preposition))
    for word in non_preposition:
        del preposition_counter[word]

    preposition_list_path = join(args.output_path, 'preposition')
    log.info('Save top {} preposition list to {}'.format(
        args.top_preposition, preposition_list_path))
    write_vocab_list(preposition_counter, preposition_list_path,
                     args.top_preposition)
