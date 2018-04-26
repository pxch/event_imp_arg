import numpy as np

from base_evaluator import BaseEvaluator
from model.event_comp import EventCompositionModel
from data.event_comp_dataset import IndexedEvent
from utils import check_type


class EventCompositionEvaluator(BaseEvaluator):
    def __init__(self, logger=None, use_lemma=True, include_type=True,
                 ignore_first_mention=False, filter_stop_events=True,
                 use_max_score=True):
        super(EventCompositionEvaluator, self).__init__(
            logger=logger,
            use_lemma=use_lemma,
            include_type=include_type,
            include_all_pobj=False,
            ignore_first_mention=ignore_first_mention,
            filter_stop_events=filter_stop_events
        )
        self.use_max_score = use_max_score
        self.model_name = 'event_composition'

    def set_model(self, model):
        check_type(model, EventCompositionModel)
        self.model = model
        self.set_embedding_model(model.word2vec)

    def log_evaluator_info(self):
        super(EventCompositionEvaluator, self).log_evaluator_info()
        self.logger.info(
            'evaluator specific configs: use_max_score = {}'.format(
                self.use_max_score))

    def get_most_coherent(self, arg_type, eval_input_list, context_input_list,
                          use_max_score=True):
        coherence_fn = self.model.pair_composition_network.coherence_fn
        use_salience = self.model.pair_composition_network.use_salience
        salience_features = \
            self.model.pair_composition_network.salience_features
        coherence_score_list = []
        num_context = len(context_input_list)

        if arg_type == 'SUBJ':
            arg_idx_input = np.asarray([1.] * num_context).astype(np.float32)
        elif arg_type == 'OBJ':
            arg_idx_input = np.asarray([2.] * num_context).astype(np.float32)
        elif arg_type.startswith('PREP'):
            arg_idx_input = np.asarray([3.] * num_context).astype(np.float32)
        else:
            raise ValueError(
                'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(
                    arg_type))

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

        for eval_input, arg_salience in eval_input_list:
            check_type(eval_input, IndexedEvent)
            pred_input_b = np.asarray(
                [eval_input.pred_input] * num_context).astype(np.int32)
            subj_input_b = np.asarray(
                [eval_input.subj_input] * num_context).astype(np.int32)
            obj_input_b = np.asarray(
                [eval_input.obj_input] * num_context).astype(np.int32)
            pobj_input_b = np.asarray(
                [eval_input.pobj_input] * num_context).astype(np.int32)
            if use_salience:
                saliance_input = np.tile(
                    arg_salience.get_feature_list(salience_features),
                    [num_context, 1]).astype(np.float32)
                coherence_output = coherence_fn(
                    pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
                    pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
                    arg_idx_input, saliance_input)
            else:
                coherence_output = coherence_fn(
                    pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
                    pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
                    arg_idx_input)
            if use_max_score:
                coherence_score_list.append(coherence_output.max())
            else:
                coherence_score_list.append(coherence_output.sum())

        most_coherent_idx = coherence_score_list.index(
            max(coherence_score_list))
        return most_coherent_idx

    def evaluate_event_list(self, rich_event_list):
        pos_input_list = \
            [rich_event.get_pos_input(include_all_pobj=self.include_all_pobj)
                for rich_event in rich_event_list]
        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            context_input_list = \
                pos_input_list[:event_idx] + pos_input_list[event_idx + 1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=self.include_all_pobj, include_salience=True)
            for rich_arg, eval_input_list in eval_input_list_all:
                if not self.ignore_argument(rich_arg):
                    most_coherent_idx = self.get_most_coherent(
                        rich_arg.arg_type, eval_input_list, context_input_list,
                        self.use_max_score)
                    correct = (most_coherent_idx == rich_arg.get_target_idx())
                    num_choices = len(eval_input_list)

                    kwargs = BaseEvaluator.get_arg_group_info(rich_arg)

                    self.eval_stats.add_eval_result(
                        correct,
                        num_choices,
                        **kwargs
                    )
                    self.logger.debug(
                        'Processing {}, correct = {}, num_choices = {}'.format(
                            rich_arg.arg_type, correct, num_choices))
