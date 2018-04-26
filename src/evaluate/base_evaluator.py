import abc
import logging
from os.path import join

from tqdm import tqdm

from common.event_script import Script
from config import cfg
from data.event_comp_dataset import RichScript
from data.event_comp_dataset.rich_argument import RichArgumentWithEntity
from eval_stats import AccuracyStatsGroup, EvalStats
from utils import Word2VecModel, check_type, consts, read_vocab_list


class BaseEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger, use_lemma=True, include_type=True,
                 include_all_pobj=False, ignore_first_mention=False,
                 filter_stop_events=True):
        self.eval_stats = EvalStats()
        self.add_default_accuracy_groups()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.use_lemma = use_lemma
        self.include_type = include_type
        self.include_all_pobj = include_all_pobj
        self.ignore_first_mention = ignore_first_mention
        self.filter_stop_events = filter_stop_events
        self.model = None
        self.model_name = ''
        self.embedding_model = None
        self.embedding_model_name = ''

    @abc.abstractmethod
    def set_model(self, model):
        return

    def add_accuracy_group(self, name, accuracy_group):
        self.eval_stats.add_accuracy_group(name, accuracy_group)

    def add_default_accuracy_groups(self):
        arg_type_accuracy_group = AccuracyStatsGroup(
            'Arg Type', ['SUBJ', 'OBJ', 'POBJ'])
        self.add_accuracy_group('arg_type', arg_type_accuracy_group)

        pos_accuracy_group = AccuracyStatsGroup(
            'POS', ['Noun', 'Pronoun', 'Other'])
        self.add_accuracy_group('pos', pos_accuracy_group)

        # ner_accuracy_group = AccuracyStatsGroup(
        #     'NER', consts.valid_ner_tags + ['NONE'])
        # self.add_accuracy_group('ner', ner_accuracy_group)

        entity_len_accuracy_group = AccuracyStatsGroup(
            'Entity Frequency', map(str, range(1, 10)) + ['10+'])
        self.add_accuracy_group('entity_len', entity_len_accuracy_group)

        # mention_idx_accuracy_group = AccuracyStatsGroup(
        #     'Mention Index', map(str, range(1, 10)) + ['10+'])
        # self.add_accuracy_group('mention_idx', mention_idx_accuracy_group)

    def set_embedding_model(self, embedding_model):
        check_type(embedding_model, Word2VecModel)
        self.logger.info('set embedding model: {}'.format(embedding_model.name))
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model.name

    def set_config(self, **kwargs):
        for key in kwargs:
            if key in self.__dict__:
                self.logger.debug('set {} = {}'.format(key, kwargs[key]))
                self.__dict__[key] = kwargs[key]
            else:
                self.logger.warning(
                    '{} is not a valid configuration keyword'.format(key))

    def log_evaluator_info(self):
        self.logger.info(
            'evaluation based on {}, with embedding model {}'.format(
                self.model_name, self.embedding_model_name))
        self.logger.info(
            'embedding configs: use_lemma = {}, include_type = {}'.format(
                self.use_lemma, self.include_type))
        self.logger.info(
            'general configs: include_all_pobj = {}, '
            'ignore_first_mention = {}, filter_stop_events = {}'.format(
                self.include_all_pobj, self.ignore_first_mention,
                self.filter_stop_events))

    # an argument should be ignored, if both self.ignore_first_mention is True,
    # and rich_argument.is_first_mention() returns True
    def ignore_argument(self, rich_argument):
        return self.ignore_first_mention and rich_argument.is_first_mention()

    @abc.abstractmethod
    def evaluate_event_list(self, rich_event_list):
        return

    def evaluate(self, all_scripts, **kwargs):
        self.set_config(**kwargs)
        self.log_evaluator_info()
        self.eval_stats.reset()
        for script in tqdm(all_scripts, desc='Processed', ncols=100):
            check_type(script, Script)

            # ignore script where there is less than 2 events
            # (i.e., no context events to be compared to)
            if len(script.events) < 2:
                continue
            # ignore script where there is less than 2 entities
            # (i.e., only one candidate to select from)
            if len(script.entities) < 2:
                continue

            self.logger.debug('Processing script {}'.format(script.doc_name))

            # load prep_vocab_list
            prep_vocab_list = read_vocab_list(
                join(cfg.vocab_path, cfg.prep_vocab_list_file))

            # build the rich_script from script
            rich_script = RichScript.build(
                script,
                prep_vocab_list=prep_vocab_list,
                use_lemma=self.use_lemma,
                filter_stop_events=self.filter_stop_events
            )
            # index the rich_script with the embedding model
            rich_script.get_index(
                self.embedding_model,
                include_type=self.include_type,
                use_unk=True
            )

            # get the list of indexed events in the script
            rich_event_list = rich_script.get_indexed_events()
            # ignore rich_script where there is less than 2 indexed events
            # (i.e., no context events to be compared to)
            if len(rich_event_list) < 2:
                continue

            self.evaluate_event_list(rich_event_list)

        self.print_stats()

    def print_stats(self):
        self.eval_stats.print_table()

    @staticmethod
    def get_arg_group_info(rich_arg):
        assert isinstance(rich_arg, RichArgumentWithEntity)
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
