from bz2 import BZ2File
from math import ceil
from os import listdir

import numpy
from os.path import isdir, isfile, join

from indexed_event import IndexedEvent, IndexedEventTriple


class IndexedCorpusReader(object):
    def __init__(self, corpus_type, corpus_dir):
        assert corpus_type in ['pretraining', 'pair_tuning'], \
            'corpus_type can only be pretraining on pair_tuning'
        self.corpus_type = corpus_type
        if corpus_type == 'pretraining':
            self.from_text_fn = IndexedEvent.from_text
        else:
            self.from_text_fn = IndexedEventTriple.from_text
        assert isdir(corpus_dir), '{} is not a directory'.format(corpus_dir)
        self.corpus_dir = corpus_dir
        try:
            self.length = int(
                open(join(corpus_dir, 'line_count'), 'r').readline().strip())
        except:
            raise IOError('File {}/line_count not found!'.format(corpus_dir))
        self.filenames = sorted(
            [join(corpus_dir, f) for f in listdir(corpus_dir)
             if isfile(join(corpus_dir, f)) and not f.endswith('line_count')])

    def __len__(self):
        return self.length

    def __iter__(self):
        for filename in self.filenames:
            if filename.endswith('bz2'):
                index_file = BZ2File(filename, 'r')
            else:
                index_file = open(filename, 'r')
            for line in index_file.readlines():
                line = line.strip()
                if line:
                    yield self.from_text_fn(line)


class PretrainingCorpusIterator(object):
    def __init__(self, corpus_dir, model, layer_input=-1, batch_size=1):
        self.corpus_dir = corpus_dir
        self.reader = IndexedCorpusReader('pretraining', self.corpus_dir)
        self.model = model
        self.layer_input = layer_input
        self.batch_size = batch_size
        self.num_batch = int(ceil(float(len(self.reader)) / batch_size))
        if layer_input == -1:
            # Compile the expression for the deepest hidden layer
            self.projection_fn = model.project
        else:
            # Compile the theano expression for this layer's input
            self.projection_fn = model.get_layer_input_function(layer_input)

    def restart(self):
        self.reader = IndexedCorpusReader('pretraining', self.corpus_dir)

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        pred_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        subj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        obj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pobj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)

        data_point_index = 0

        for single_input in self.reader:
            pred_inputs[data_point_index] = single_input.pred_input
            subj_inputs[data_point_index] = single_input.subj_input
            obj_inputs[data_point_index] = single_input.obj_input
            pobj_inputs[data_point_index] = single_input.pobj_input
            data_point_index += 1

            # If we've filled up the batch, yield it
            if data_point_index == self.batch_size:
                yield self.projection_fn(
                    pred_inputs, subj_inputs, obj_inputs, pobj_inputs)
                data_point_index = 0

        if data_point_index > 0:
            # We've partially filled a batch: yield this as the last item
            yield self.projection_fn(
                pred_inputs, subj_inputs, obj_inputs, pobj_inputs)


class PairTuningCorpusIterator(object):
    def __init__(self, corpus_dir, batch_size=1, use_salience=True,
                 salience_features=None):
        self.corpus_dir = corpus_dir
        self.reader = IndexedCorpusReader('pair_tuning', self.corpus_dir)
        self.batch_size = batch_size
        self.num_batch = int(ceil(float(len(self.reader)) / batch_size))
        self.use_salience = use_salience

        self.num_salience_features = 0
        self.salience_features = []
        if self.use_salience:
            assert salience_features is not None
            self.salience_features = salience_features
            self.num_salience_features = len(self.salience_features)

    def restart(self):
        self.reader = IndexedCorpusReader('pair_tuning', self.corpus_dir)

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        left_pred_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        left_subj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        left_obj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        left_pobj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pos_pred_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pos_subj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pos_obj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pos_pobj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        neg_pred_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        neg_subj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        neg_obj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        neg_pobj_input = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pos_arg_idx_input = numpy.zeros(self.batch_size, dtype=numpy.float32)
        neg_arg_idx_input = numpy.zeros(self.batch_size, dtype=numpy.float32)
        if self.use_salience:
            pos_entity_salience_input = numpy.zeros(
                [self.batch_size, self.num_salience_features],
                dtype=numpy.float32)
            neg_entity_salience_input = numpy.zeros(
                [self.batch_size, self.num_salience_features],
                dtype=numpy.float32)

        data_point_index = 0

        for pair_input in self.reader:
            left_pred_input[data_point_index] = pair_input.left_event.pred_input
            left_subj_input[data_point_index] = pair_input.left_event.subj_input
            left_obj_input[data_point_index] = pair_input.left_event.obj_input
            left_pobj_input[data_point_index] = pair_input.left_event.pobj_input
            pos_pred_input[data_point_index] = pair_input.pos_event.pred_input
            pos_subj_input[data_point_index] = pair_input.pos_event.subj_input
            pos_obj_input[data_point_index] = pair_input.pos_event.obj_input
            pos_pobj_input[data_point_index] = pair_input.pos_event.pobj_input
            neg_pred_input[data_point_index] = pair_input.neg_event.pred_input
            neg_subj_input[data_point_index] = pair_input.neg_event.subj_input
            neg_obj_input[data_point_index] = pair_input.neg_event.obj_input
            neg_pobj_input[data_point_index] = pair_input.neg_event.pobj_input
            pos_arg_idx_input[data_point_index] = float(pair_input.pos_arg_idx)
            neg_arg_idx_input[data_point_index] = float(pair_input.neg_arg_idx)

            if self.use_salience:
                pos_entity_salience_input[data_point_index] = \
                    numpy.asarray(pair_input.pos_salience.get_feature_list(
                        self.salience_features)).astype(numpy.float32)
                neg_entity_salience_input[data_point_index] = \
                    numpy.asarray(pair_input.neg_salience.get_feature_list(
                        self.salience_features)).astype(numpy.float32)
            data_point_index += 1

            # If we've filled up the batch, yield it
            if data_point_index == self.batch_size:
                if self.use_salience:
                    yield left_pred_input, \
                          left_subj_input, \
                          left_obj_input, \
                          left_pobj_input, \
                          pos_pred_input, \
                          pos_subj_input, \
                          pos_obj_input, \
                          pos_pobj_input, \
                          neg_pred_input, \
                          neg_subj_input, \
                          neg_obj_input, \
                          neg_pobj_input, \
                          pos_arg_idx_input, \
                          neg_arg_idx_input, \
                          pos_entity_salience_input, \
                          neg_entity_salience_input
                else:
                    yield left_pred_input, \
                          left_subj_input, \
                          left_obj_input, \
                          left_pobj_input, \
                          pos_pred_input, \
                          pos_subj_input, \
                          pos_obj_input, \
                          pos_pobj_input, \
                          neg_pred_input, \
                          neg_subj_input, \
                          neg_obj_input, \
                          neg_pobj_input, \
                          pos_arg_idx_input, \
                          neg_arg_idx_input
                data_point_index = 0

        # FIXME: Should return this last partial batch,
        # but having a smaller batch is currently messing up training
        # If you update this, allow for the triple option as well
        if False and data_point_index > 0:
            # We've partially filled a batch: yield this as the last item
            if self.use_salience:
                yield left_pred_input[:data_point_index], \
                      left_subj_input[:data_point_index], \
                      left_obj_input[:data_point_index], \
                      left_pobj_input[:data_point_index], \
                      pos_pred_input[:data_point_index], \
                      pos_subj_input[:data_point_index], \
                      pos_obj_input[:data_point_index], \
                      pos_pobj_input[:data_point_index], \
                      neg_pred_input[:data_point_index], \
                      neg_subj_input[:data_point_index], \
                      neg_obj_input[:data_point_index], \
                      neg_pobj_input[:data_point_index], \
                      pos_arg_idx_input[:data_point_index], \
                      neg_arg_idx_input[:data_point_index], \
                      pos_entity_salience_input[:data_point_index], \
                      neg_entity_salience_input[:data_point_index]
            else:
                yield left_pred_input[:data_point_index], \
                      left_subj_input[:data_point_index], \
                      left_obj_input[:data_point_index], \
                      left_pobj_input[:data_point_index], \
                      pos_pred_input[:data_point_index], \
                      pos_subj_input[:data_point_index], \
                      pos_obj_input[:data_point_index], \
                      pos_pobj_input[:data_point_index], \
                      neg_pred_input[:data_point_index], \
                      neg_subj_input[:data_point_index], \
                      neg_obj_input[:data_point_index], \
                      neg_pobj_input[:data_point_index], \
                      pos_arg_idx_input[:data_point_index], \
                      neg_arg_idx_input[:data_point_index]
