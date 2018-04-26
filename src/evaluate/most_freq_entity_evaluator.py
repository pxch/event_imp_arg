from base_evaluator import BaseEvaluator


def is_most_freq_entity(entity_idx, entity_freqs):
    entity_freqs[entity_idx] -= 1
    most_freq_entity_idx = entity_freqs.index(max(entity_freqs))
    entity_freqs[entity_idx] += 1
    return entity_idx == most_freq_entity_idx


class MostFreqEntityEvaluator(BaseEvaluator):
    def __init__(self, logger=None, use_lemma=True, include_type=True,
                 include_all_pobj=True, ignore_first_mention=False,
                 filter_stop_events=True):
        super(MostFreqEntityEvaluator, self).__init__(
            logger=logger,
            use_lemma=use_lemma,
            include_type=include_type,
            include_all_pobj=include_all_pobj,
            ignore_first_mention=ignore_first_mention,
            filter_stop_events=filter_stop_events
        )
        self.model_name = 'most_freq_entity'

    def set_model(self, model):
        self.set_embedding_model(model)

    def evaluate_event_list(self, rich_event_list):
        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            for arg_idx in rich_event.get_arg_idx_list(
                    include_all_pobj=self.include_all_pobj):
                if rich_event.has_neg(arg_idx):
                    rich_arg = rich_event.get_argument(arg_idx)
                    if not self.ignore_argument(rich_arg):
                        entity_freqs = [rich_entity.salience.num_mentions_total
                                        for rich_entity in
                                        rich_arg.rich_entity_list]
                        correct = is_most_freq_entity(
                            rich_arg.entity_idx, entity_freqs)
                        num_choices = len(entity_freqs)

                        kwargs = BaseEvaluator.get_arg_group_info(rich_arg)

                        self.eval_stats.add_eval_result(
                            correct,
                            num_choices,
                            **kwargs
                        )
