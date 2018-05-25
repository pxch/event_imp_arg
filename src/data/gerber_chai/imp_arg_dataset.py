from collections import defaultdict
from copy import deepcopy

import numpy as np
from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer
from nltk.stem import WordNetLemmatizer
from os.path import join
from sklearn.model_selection import KFold

from candidate import CandidateDict
from common import event_script
from config import cfg
from corenlp_reader import CoreNLPReader
from data.event_comp_dataset import IndexedEvent
from data.event_comp_dataset import RichScript
from data.nltk import NombankReader, PropbankReader, PTBReader
from helper import compute_f1, pred_list
from helper import convert_nombank_label
from helper import expand_wsj_fileid, shorten_wsj_fileid
from helper import nominal_predicate_mapping, core_arg_list
from imp_arg_instance import ImpArgInstance
from predicate import Predicate
from rich_predicate import RichPredicate
from rich_tree_pointer import RichTreePointer
from stats import print_stats, print_eval_stats
from utils import Word2VecModel, check_type, log
from utils import read_vocab_list


class ImpArgDataset(object):
    def __init__(self, corenlp_root, n_splits=10, split_by_original=False,
                 max_candidate_dist=2, include_non_head_entity=True,
                 suppress_warning=False):
        # root path to the CoreNLP parsed wsj corpus
        self.corenlp_root = corenlp_root

        # number of splits in n-fold cross validation
        self.n_splits = n_splits
        # if True, split the dataset by the original order,
        # otherwise by the sorted order
        self.split_by_original = split_by_original

        # maximum sentence distance between a candidate and the predicate
        self.max_candidate_dist = max_candidate_dist

        # if True, count the entity indices of non-head words in a candidate
        # when determining the entity idx of the candidate, otherwise only
        # use the entity idx of the head word
        self.include_non_head_entity = include_non_head_entity

        # if True, do not print warning message to stderr
        if suppress_warning:
            log.warning = log.debug

        self.all_instances = []
        self.instance_order_list = []
        self.train_test_folds = []

        self.all_predicates = []
        self.all_rich_predicates = []

        self._treebank_reader = None
        self._nombank_reader = None
        self._propbank_reader = None
        self._predicate_mapping = None
        self._corenlp_reader = None
        self._candidate_dict = None

    def read_dataset(self, file_path):
        log.info('Reading implicit argument dataset from {}'.format(file_path))
        input_xml = open(file_path, 'r')

        all_instances = []
        for line in input_xml.readlines()[1:-1]:
            instance = ImpArgInstance.parse(line.strip())
            all_instances.append(instance)

        log.info('Found {} instances'.format(len(all_instances)))

        self.all_instances = sorted(
            all_instances, key=lambda ins: str(ins.pred_pointer))

        self.instance_order_list = [self.all_instances.index(instance)
                                    for instance in all_instances]

        instance_order_list = np.asarray(self.instance_order_list)

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        if self.split_by_original:
            for train, test in kf.split(self.instance_order_list):
                train_indices = instance_order_list[train]
                test_indices = instance_order_list[test]
                self.train_test_folds.append((train_indices, test_indices))
        else:
            self.train_test_folds = list(kf.split(self.instance_order_list))

    def print_dataset(self, file_path):
        fout = open(file_path, 'w')
        fout.write('<annotations>\n')

        for instance in self.all_instances:
            fout.write(str(instance) + '\n')

        fout.write('</annotations>\n')
        fout.close()

    def print_dataset_by_pred(self, dir_path):
        all_instances_by_pred = defaultdict(list)
        for instance in self.all_instances:
            n_pred = self.predicate_mapping[str(instance.pred_pointer)]
            all_instances_by_pred[n_pred].append(instance)

        for n_pred in all_instances_by_pred:
            fout = open(join(dir_path, n_pred), 'w')
            for instance in all_instances_by_pred[n_pred]:
                fout.write(str(instance) + '\n')
            fout.close()

    @property
    def treebank_reader(self):
        if self._treebank_reader is None:
            self._treebank_reader = PTBReader()
        return self._treebank_reader

    @property
    def nombank_reader(self):
        if self._nombank_reader is None:
            self._nombank_reader = NombankReader()
            self._nombank_reader.build_index()
        return self._nombank_reader

    @property
    def propbank_reader(self):
        if self._propbank_reader is None:
            self._propbank_reader = PropbankReader()
            self._propbank_reader.build_index()
        return self._propbank_reader

    @property
    def predicate_mapping(self):
        if self._predicate_mapping is None:
            assert self.treebank_reader is not None

            log.info('Building predicate mapping')
            self._predicate_mapping = {}

            lemmatizer = WordNetLemmatizer()

            for instance in self.all_instances:
                pred_pointer = instance.pred_pointer

                self.treebank_reader.read_file(
                    expand_wsj_fileid(pred_pointer.fileid, '.mrg'))

                word = self.treebank_reader.all_sents[
                    pred_pointer.sentnum][pred_pointer.tree_pointer.wordnum]

                n_pred = lemmatizer.lemmatize(word.lower(), pos='n')

                if n_pred not in nominal_predicate_mapping:
                    for subword in n_pred.split('-'):
                        if subword in nominal_predicate_mapping:
                            n_pred = subword
                            break

                assert n_pred in nominal_predicate_mapping, \
                    'unexpected nominal predicate: {}'.format(n_pred)
                assert str(pred_pointer) not in self._predicate_mapping, \
                    'pred_node {} already found'.format(pred_pointer)
                self._predicate_mapping[str(pred_pointer)] = n_pred

        return self._predicate_mapping

    @property
    def corenlp_reader(self):
        if self._corenlp_reader is None:
            self._corenlp_reader = CoreNLPReader.build(
                self.all_instances, self.corenlp_root)
        return self._corenlp_reader

    def build_predicates(self):
        assert len(self.all_instances) > 0
        assert self.treebank_reader is not None
        assert self.nombank_reader is not None
        assert self.predicate_mapping is not None
        assert self.corenlp_reader is not None

        if len(self.all_predicates) > 0:
            log.warning('Overriding existing predicates')
            self.all_predicates = []

        log.info('Building predicates')
        for instance in self.all_instances:
            predicate = Predicate.build(instance)
            predicate.set_pred(
                self.predicate_mapping[str(predicate.pred_pointer)])
            self.all_predicates.append(predicate)

        log.info('Checking explicit arguments with Nombank instances')
        for predicate in self.all_predicates:
            nombank_instance = self.nombank_reader.search_by_pointer(
                predicate.pred_pointer)
            predicate.check_exp_args(
                nombank_instance, add_missing_args=False,
                remove_conflict_imp_args=False, verbose=False)

        log.info('Parsing all implicit and explicit arguments')
        for predicate in self.all_predicates:
            predicate.parse_args(
                self.treebank_reader, self.corenlp_reader,
                include_non_head_entity=self.include_non_head_entity)
        log.info('Done')

    @property
    def candidate_dict(self):
        if self._candidate_dict is None:
            assert len(self.all_predicates) > 0
            assert self.propbank_reader is not None
            assert self.nombank_reader is not None
            assert self.corenlp_reader is not None
            log.info('Building candidate dict from Propbank and Nombank')
            self._candidate_dict = CandidateDict(
                propbank_reader=self.propbank_reader,
                nombank_reader=self.nombank_reader,
                corenlp_reader=self.corenlp_reader,
                max_dist=self.max_candidate_dist)

            for predicate in self.all_predicates:
                self._candidate_dict.add_candidates(
                    predicate.pred_pointer,
                    include_non_head_entity=self.include_non_head_entity)
            log.info('Done')

        return self._candidate_dict

    def add_candidates(self):
        assert len(self.all_predicates) > 0
        assert self.candidate_dict is not None
        log.info('Adding candidates to predicates')
        for predicate in self.all_predicates:
            for candidate in self.candidate_dict.get_candidates(
                    predicate.pred_pointer):
                predicate.candidates.append(candidate)

    def print_stats(self, verbose=0):
        print_stats(self.all_predicates, verbose=verbose)

    def build_extra_event(self, instance, pred, sent):
        subj = None
        obj = None
        pobj_list = []

        fileid = shorten_wsj_fileid(instance.fileid)
        sentnum = instance.sentnum

        for tree_pointer, label in instance.arguments:
            cvt_label = convert_nombank_label(label)
            if cvt_label not in core_arg_list:
                continue
            arg_pointer_list = []
            if isinstance(tree_pointer, NombankChainTreePointer) or \
                    isinstance(tree_pointer, PropbankChainTreePointer):
                for p in tree_pointer.pieces:
                    arg_pointer_list.append(RichTreePointer(
                        fileid, sentnum, p, tree=instance.tree))
            else:
                arg_pointer_list.append(RichTreePointer(
                    fileid, sentnum, tree_pointer, tree=instance.tree))

            argument_list = []
            for arg_pointer in arg_pointer_list:
                arg_pointer.parse_treebank()
                arg_pointer.parse_corenlp(self.corenlp_reader,
                                          include_non_head_entity=True)

                arg_token_idx = arg_pointer.head_idx()
                if arg_token_idx != -1 and arg_token_idx != pred.wordnum:
                    arg_token = sent.get_token(arg_token_idx)
                    argument = event_script.Argument.from_token(arg_token)
                    argument_list.append(argument)

            if cvt_label == 'arg0' and argument_list:
                subj = argument_list[0]
            elif cvt_label == 'arg1' and argument_list:
                obj = argument_list[0]
            else:
                pobj_list.extend(
                    [('', argument) for argument in argument_list])

        event = event_script.Event(pred, subj, obj, pobj_list)

        return event

    def get_extra_events(self, fileid, idx_mapping, doc, verbification_dict,
                         use_nombank=True, use_propbank=True):
        extra_events = []

        if use_nombank:
            for instance in self.nombank_reader.search_by_fileid(fileid):
                sentnum = instance.sentnum
                sent = doc.get_sent(sentnum)

                try:
                    pred_token_idx = \
                        idx_mapping[sentnum].index(instance.wordnum)
                except ValueError:
                    continue

                nom_pred_token = sent.get_token(pred_token_idx)
                if nom_pred_token.lemma not in verbification_dict:
                    continue
                pred_lemma = verbification_dict[nom_pred_token.lemma]
                pred = event_script.Predicate(
                    word=pred_lemma,
                    lemma=pred_lemma,
                    pos='VB',
                    sentnum=sentnum,
                    wordnum=pred_token_idx)

                event = self.build_extra_event(
                    instance, pred, sent)
                extra_events.append(event)

        if use_propbank:
            for instance in self.propbank_reader.search_by_fileid(fileid):
                sentnum = instance.sentnum
                sent = doc.get_sent(sentnum)

                try:
                    pred_token_idx = \
                        idx_mapping[sentnum].index(instance.wordnum)
                except ValueError:
                    continue

                pred_token = sent.get_token(pred_token_idx)

                pred = event_script.Predicate(
                    pred_token.word,
                    pred_token.lemma,
                    'VB',
                    sentnum=sentnum,
                    wordnum=pred_token_idx)

                event = self.build_extra_event(
                    instance, pred, sent)
                extra_events.append(event)

        return extra_events

    def add_extra_events(self, verbification_dict, use_nombank=True,
                         use_propbank=True):
        if use_nombank:
            log.info('Adding extra events from NomBank to CoreNLP scripts')
        if use_propbank:
            log.info('Adding extra events from PropBank to CoreNLP scripts')

        prep_vocab_list = read_vocab_list(
            join(cfg.vocab_path, cfg.prep_vocab_list_file))

        for fileid in self.corenlp_reader.corenlp_dict.keys():
            log.info('Process file {}'.format(fileid))

            idx_mapping = self.corenlp_reader.get_idx_mapping(fileid)
            doc = self.corenlp_reader.get_doc(fileid)

            extra_events = self.get_extra_events(
                fileid, idx_mapping, doc, verbification_dict,
                use_nombank=use_nombank, use_propbank=use_propbank)

            script = self.corenlp_reader.get_script(fileid)
            for extra_event in extra_events:
                script.add_extra_event(extra_event)

            rich_script = RichScript.build(
                script,
                prep_vocab_list=prep_vocab_list,
                use_lemma=True,
                filter_stop_events=False
            )

            self.corenlp_reader.corenlp_dict[fileid] = (
                idx_mapping, doc, deepcopy(script), deepcopy(rich_script))

    def build_rich_predicates(
            self, use_corenlp_token=True, labeled_arg_only=False,
            avail_candidates_dict=None):
        assert len(self.all_predicates) > 0
        if len(self.all_rich_predicates) > 0:
            log.warning('Overriding existing rich predicates')
            self.all_rich_predicates = []

        log.info('Building rich predicates with {}'.format(
            'CoreNLP tokens' if use_corenlp_token else 'PTB tokens'))
        for predicate in self.all_predicates:
            avail_candidates = None
            if avail_candidates_dict is not None:
                avail_candidates = \
                    avail_candidates_dict[str(predicate.pred_pointer)]

            rich_predicate = RichPredicate.build(
                predicate,
                corenlp_reader=self.corenlp_reader,
                use_lemma=True,
                use_entity=True,
                use_corenlp_tokens=use_corenlp_token,
                labeled_arg_only=labeled_arg_only,
                avail_candidates=avail_candidates)
            self.all_rich_predicates.append(rich_predicate)
        log.info('Done')

    def get_index(self, word2vec_model):
        check_type(word2vec_model, Word2VecModel)

        pred_wv_mapping = {}
        for pred in pred_list:
            index = word2vec_model.get_word_index(pred + '-PRED')
            assert index != -1
            pred_wv_mapping[pred] = index

        for rich_predicate in self.all_rich_predicates:
            rich_predicate.set_pred_wv(pred_wv_mapping)
            rich_predicate.get_index(
                word2vec_model, include_type=True, use_unk=True)

    def get_context_input_list_mapping(self, word2vec_model):
        context_input_list_mapping = {}

        for rich_predicate in self.all_rich_predicates:
            if rich_predicate.fileid not in context_input_list_mapping:
                rich_script = self.corenlp_reader.get_rich_script(
                    rich_predicate.fileid)
                rich_script.get_index(
                    word2vec_model, include_type=True, use_unk=True)

                rich_event_list = rich_script.get_indexed_events()

                context_input_list = \
                    [rich_event.get_pos_input(include_all_pobj=False)
                     for rich_event in rich_event_list]

                context_input_list_mapping[rich_predicate.fileid] = \
                    context_input_list

        return context_input_list_mapping

    def compute_coherence_score(
            self, event_comp_model, use_max_score=True,
            missing_labels_mapping=None):
        assert len(self.all_rich_predicates) > 0

        if type(event_comp_model) == list:
            assert len(event_comp_model) == self.n_splits
            word2vec_model = event_comp_model[0].word2vec
        else:
            word2vec_model = event_comp_model.word2vec

        self.get_index(word2vec_model)
        context_input_list_mapping = \
            self.get_context_input_list_mapping(word2vec_model)

        exclude_pred_idx_list = []

        # pbar = tqdm(total=len(self.all_rich_predicates), desc='Processed',
        #             ncols=100)

        for fold_idx in range(self.n_splits):
            for pred_idx in self.train_test_folds[fold_idx][1]:
                # pbar.update(1)
                rich_predicate = self.all_rich_predicates[pred_idx]
                if len(rich_predicate.imp_args) == 0:
                    continue

                for imp_arg in rich_predicate.imp_args:
                    imp_arg.reset_coherence_score_list()

                if missing_labels_mapping is not None:
                    missing_labels = missing_labels_mapping[
                        str(self.all_predicates[pred_idx].pred_pointer)]
                else:
                    missing_labels = None

                if missing_labels is not None and len(missing_labels) == 0:
                    continue

                context_input_list = \
                    context_input_list_mapping[rich_predicate.fileid]
                num_context = len(context_input_list)

                if num_context == 0:
                    exclude_pred_idx_list.append(pred_idx)
                    continue

                if type(event_comp_model) == list:
                    pair_composition_network = \
                        event_comp_model[fold_idx].pair_composition_network
                else:
                    pair_composition_network = \
                        event_comp_model.pair_composition_network

                coherence_fn = pair_composition_network.coherence_fn
                use_salience = pair_composition_network.use_salience
                salience_features = pair_composition_network.salience_features

                pred_input_a = np.zeros(num_context, dtype=np.int32)
                subj_input_a = np.zeros(num_context, dtype=np.int32)
                obj_input_a = np.zeros(num_context, dtype=np.int32)
                pobj_input_a = np.zeros(num_context, dtype=np.int32)
                for context_idx, context_input in enumerate(context_input_list):
                    check_type(context_input, IndexedEvent)
                    pred_input_a[context_idx] = context_input.pred_input
                    subj_input_a[context_idx] = context_input.subj_input
                    obj_input_a[context_idx] = context_input.obj_input
                    pobj_input_a[context_idx] = context_input.pobj_input

                eval_input_list_all = \
                    rich_predicate.get_eval_input_list_all(
                        include_salience=True, missing_labels=missing_labels)

                num_candidates = rich_predicate.num_candidates

                coherence_score_list_all = []

                for label, arg_idx, eval_input_list in eval_input_list_all:
                    coherence_score_list = []

                    arg_idx_input = \
                        np.asarray([float(arg_idx)] * num_context).astype(
                            np.float32)

                    for eval_input, arg_salience in eval_input_list:
                        check_type(eval_input, IndexedEvent)
                        pred_input_b = np.asarray(
                            [eval_input.pred_input] * num_context).astype(
                            np.int32)
                        subj_input_b = np.asarray(
                            [eval_input.subj_input] * num_context).astype(
                            np.int32)
                        obj_input_b = np.asarray(
                            [eval_input.obj_input] * num_context).astype(
                            np.int32)
                        pobj_input_b = np.asarray(
                            [eval_input.pobj_input] * num_context).astype(
                            np.int32)

                        if use_salience:
                            if arg_salience is not None:
                                salience_feature = \
                                    arg_salience.get_feature_list(
                                        salience_features)
                            else:
                                # NOBUG: this should never happen
                                log.warning(
                                    'salience feature = None, filled with 0')
                                salience_feature = [0.0] * len(
                                    salience_features)

                            saliance_input = np.tile(
                                salience_feature, [num_context, 1]).astype(
                                np.float32)

                            coherence_output = coherence_fn(
                                pred_input_a, subj_input_a, obj_input_a,
                                pobj_input_a,
                                pred_input_b, subj_input_b, obj_input_b,
                                pobj_input_b,
                                arg_idx_input, saliance_input)
                        else:
                            coherence_output = coherence_fn(
                                pred_input_a, subj_input_a, obj_input_a,
                                pobj_input_a,
                                pred_input_b, subj_input_b, obj_input_b,
                                pobj_input_b,
                                arg_idx_input)

                        if use_max_score:
                            coherence_score_list.append(coherence_output.max())
                        else:
                            coherence_score_list.append(coherence_output.sum())

                    assert len(coherence_score_list) == num_candidates + 1
                    coherence_score_list_all.append(
                        (label, coherence_score_list))

                num_label = len(eval_input_list_all)
                coherence_score_matrix = np.ndarray(
                    shape=(num_label, num_candidates + 1))
                row_idx = 0
                for label, coherence_score_list in coherence_score_list_all:
                    coherence_score_matrix[row_idx, :] = np.array(
                        coherence_score_list)
                    row_idx += 1

                for column_idx in range(1, num_candidates):
                    max_coherence_score_idx = \
                        coherence_score_matrix[:, column_idx].argmax()
                    for row_idx in range(num_label):
                        if row_idx != max_coherence_score_idx:
                            coherence_score_matrix[row_idx, column_idx] = -1.0
                '''
                max_coherence_score_idx_list = []
                for row_idx in range(num_label):
                    max_coherence_score_idx_list.append(
                        coherence_score_matrix[row_idx, 1:].argmax())
                '''

                label_list = [label for label, _ in coherence_score_list_all]
                for imp_arg in rich_predicate.imp_args:
                    if imp_arg.label in label_list:
                        row_idx = label_list.index(imp_arg.label)
                        imp_arg.set_coherence_score_list(
                            coherence_score_matrix[row_idx, :])
                '''
                for row_idx in range(num_label):
                    assert coherence_score_list_all[row_idx][0] == \
                           rich_predicate.imp_args[row_idx].label
                    rich_predicate.imp_args[row_idx].set_coherence_score_list(
                        coherence_score_matrix[row_idx, :])
                '''
        # pbar.close()

        log.info('Predicates with no context events:')
        for pred_idx in exclude_pred_idx_list:
            rich_predicate = self.all_rich_predicates[pred_idx]
            log.info(
                'Predicate #{}: {}, missing_imp_args = {}, '
                'imp_args = {}'.format(
                    pred_idx,
                    rich_predicate.n_pred,
                    len(rich_predicate.imp_args),
                    len([imp_arg for imp_arg in rich_predicate.imp_args
                         if imp_arg.exist])))

    def cross_val(self, comp_wo_arg=True):
        assert len(self.all_rich_predicates) > 0

        optimized_thres = []

        for train, test in self.train_test_folds:
            thres_list = [float(x) / 100 for x in range(0, 100)]

            precision_list = []
            recall_list = []
            f1_list = []

            for thres in thres_list:
                total_dice = 0.0
                total_gt = 0.0
                total_model = 0.0

                for idx in train:
                    rich_predicate = self.all_rich_predicates[idx]
                    rich_predicate.eval(thres, comp_wo_arg=comp_wo_arg)

                    total_dice += rich_predicate.sum_dice
                    total_gt += rich_predicate.num_gt
                    total_model += rich_predicate.num_model

                precision, recall, f1 = \
                    compute_f1(total_dice, total_gt, total_model)

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            max_f1 = max(f1_list)
            max_thres = thres_list[f1_list.index(max_f1)]
            optimized_thres.append(max_thres)

            for idx in test:
                rich_predicate = self.all_rich_predicates[idx]
                rich_predicate.thres = max_thres

        for rich_predicate in self.all_rich_predicates:
            rich_predicate.eval(rich_predicate.thres, comp_wo_arg=comp_wo_arg)

    def print_eval_stats(self):
        print_eval_stats(self.all_rich_predicates)
