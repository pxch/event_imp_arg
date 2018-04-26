import numpy as np

from base_evaluator import BaseEvaluator
from data.event_comp_dataset import IndexedEvent, IndexedEventMultiPobj
from utils import check_type


def cos_sim(vec1, vec2):
    if np.count_nonzero(vec1) == 0 or np.count_nonzero(vec2) == 0:
        return 0.0
    return vec1.dot(vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)


def get_coherence_scores(target_vector, context_vector_list):
    return [cos_sim(target_vector, context_vector)
            for context_vector in context_vector_list]


def get_most_coherent(eval_vector_list, context_vector_list,
                      use_max_score=True):
    coherence_score_list = []
    for eval_vector in eval_vector_list:
        if use_max_score:
            coherence_score_list.append(max(
                get_coherence_scores(eval_vector, context_vector_list)))
        else:
            coherence_score_list.append(sum(
                get_coherence_scores(eval_vector, context_vector_list)))

    most_coherent_idx = coherence_score_list.index(
        max(coherence_score_list))
    return most_coherent_idx


class Word2VecEvaluator(BaseEvaluator):
    def __init__(self, logger=None, use_lemma=True, include_type=True,
                 include_all_pobj=True, ignore_first_mention=False,
                 filter_stop_events=True, use_max_score=True):
        super(Word2VecEvaluator, self).__init__(
            logger=logger,
            use_lemma=use_lemma,
            include_type=include_type,
            include_all_pobj=include_all_pobj,
            ignore_first_mention=ignore_first_mention,
            filter_stop_events=filter_stop_events
        )
        self.use_max_score = use_max_score
        self.model_name = 'word2vec'

    def set_model(self, model):
        self.set_embedding_model(model)

    def log_evaluator_info(self):
        super(Word2VecEvaluator, self).log_evaluator_info()
        self.logger.info(
            'evaluator specific configs: use_max_score = {}'.format(
                self.use_max_score))

    def get_event_vector(self, event_input, include_all_pobj=True):
        if include_all_pobj:
            check_type(event_input, IndexedEventMultiPobj)
        else:
            check_type(event_input, IndexedEvent)
        # initialize event vector to be all zero
        vector = np.zeros(self.embedding_model.vector_size)
        # add vector for predicate
        pred_vector = self.embedding_model.get_index_vec(
            event_input.get_predicate())
        if pred_vector is not None:
            vector += pred_vector
        else:
            return None
        # add vectors for all arguments
        for arg_input in event_input.get_all_argument():
            arg_vector = self.embedding_model.get_index_vec(arg_input)
            if arg_vector is not None:
                vector += arg_vector
        return vector

    def evaluate_event_list(self, rich_event_list):
        pos_input_list = \
            [rich_event.get_pos_input(include_all_pobj=self.include_all_pobj)
                for rich_event in rich_event_list]
        pos_vector_list = \
            [self.get_event_vector(
                pos_input, include_all_pobj=self.include_all_pobj)
                for pos_input in pos_input_list]

        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            context_vector_list = \
                pos_vector_list[:event_idx] + pos_vector_list[event_idx+1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=self.include_all_pobj, include_salience=False)

            for rich_arg, eval_input_list in eval_input_list_all:
                if not self.ignore_argument(rich_arg):
                    eval_vector_list = \
                        [self.get_event_vector(
                            eval_input, include_all_pobj=self.include_all_pobj)
                            for eval_input in eval_input_list]
                    most_coherent_idx = get_most_coherent(
                        eval_vector_list,
                        context_vector_list,
                        self.use_max_score
                    )
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
