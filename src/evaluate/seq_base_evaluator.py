from base_evaluator import BaseEvaluator
import abc
from data.event_comp_dataset.seq_rich_argument import SeqRichArgument
from utils import check_type, consts, read_vocab_list
from data.event_comp_dataset.seq_rich_script import SeqRichScript
from os.path import join
from tqdm import tqdm
from config import cfg
from common.event_script import Script


class SeqBaseEvaluator(BaseEvaluator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger, use_lemma=True, include_type=True,
                 ignore_first_mention=False, filter_stop_events=True,
                 filter_repetitive_prep=True):
        super(SeqBaseEvaluator, self).__init__(
            logger=logger,
            use_lemma=use_lemma,
            include_type=include_type,
            ignore_first_mention=ignore_first_mention,
            filter_stop_events=filter_stop_events
        )
        self.include_all_pobj = False
        self.filter_repetitive_prep = filter_repetitive_prep
        self.stop_pred_ids = None

    def log_evaluator_info(self):
        super(SeqBaseEvaluator, self).log_evaluator_info()
        self.logger.info(
            'general configs: filter_repetitive_prep = {}'.format(
                self.filter_repetitive_prep))

    def evaluate(self, all_scripts, **kwargs):
        self.set_config(**kwargs)
        self.log_evaluator_info()
        self.eval_stats.reset()

        if self.filter_stop_events:
            self.stop_pred_ids = [
                self.embedding_model.get_word_index(stop_pred + '-PRED')
                for stop_pred in consts.stop_preds]

        for script in tqdm(all_scripts, desc='Processed', ncols=100):
            check_type(script, Script)

            self.logger.debug('Processing script {}'.format(script.doc_name))

            # load prep_vocab_list
            prep_vocab_list = read_vocab_list(
                join(cfg.vocab_path, cfg.prep_vocab_list_file))

            # build the rich_script from script
            rich_script = SeqRichScript.build(
                script,
                prep_vocab_list=prep_vocab_list,
                use_lemma=self.use_lemma,
                filter_repetitive_prep=self.filter_repetitive_prep
            )
            # index the rich_script with the embedding model
            rich_script.get_index(
                self.embedding_model,
                include_type=self.include_type,
                use_unk=True
            )

            # get the list of indexed events in the script
            rich_event_list = rich_script.get_indexed_events()

            self.evaluate_event_list(rich_event_list)

        self.print_stats()

    @staticmethod
    def get_arg_group_info(rich_arg):
        assert isinstance(rich_arg, SeqRichArgument)
        kwargs = {}

        if rich_arg.arg_type.startswith('PREP'):
            kwargs['arg_type'] = 'POBJ'
        else:
            kwargs['arg_type'] = rich_arg.arg_type

        if rich_arg.core.pos.startswith('NN'):
            kwargs['pos'] = 'Noun'
        elif rich_arg.core.pos.startswith('PRP'):
            kwargs['pos'] = 'Pronoun'
        else:
            kwargs['pos'] = 'Other'

        rich_entity = rich_arg.get_entity()
        kwargs['ner'] = rich_entity.core.ner
        if kwargs['ner'] == '':
            kwargs['ner'] = 'NONE'

        entity_len = rich_entity.salience.num_mentions_total
        if entity_len < 10:
            kwargs['entity_len'] = str(entity_len)
        else:
            kwargs['entity_len'] = '10+'

        mention_idx = rich_arg.mention_idx + 1
        if mention_idx < 10:
            kwargs['mention_idx'] = str(mention_idx)
        else:
            kwargs['mention_idx'] = '10+'

        return kwargs
