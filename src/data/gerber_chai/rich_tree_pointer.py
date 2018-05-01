from collections import Counter
from copy import deepcopy

from nltk.corpus.reader.nombank import NombankSplitTreePointer
from nltk.corpus.reader.nombank import NombankTreePointer
from nltk.corpus.reader.propbank import PropbankSplitTreePointer
from nltk.corpus.reader.propbank import PropbankTreePointer
from nltk.stem import WordNetLemmatizer

from common.event_script import Argument
from corenlp_reader import CoreNLPReader
from data.event_comp_dataset import CoreArgument
from data.event_comp_dataset.rich_entity import EntitySalience
from data.nltk import PTBReader
from helper import expand_wsj_fileid
from utils import check_type

lemmatizer = WordNetLemmatizer()


class TreebankInfo(object):
    def __init__(self, treepos, idx_list, word_list):
        self.treepos = deepcopy(treepos)
        self.idx_list = deepcopy(idx_list)
        self.word_list = deepcopy(word_list)
        self._surface = ''

    @classmethod
    def build(cls, tree_pointer, tree):
        assert isinstance(tree_pointer, NombankTreePointer) or \
               isinstance(tree_pointer, PropbankTreePointer)
        treepos = tree_pointer.treepos(tree)
        idx_list = []
        word_list = []
        for idx in range(len(tree.leaves())):
            if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
                idx_list.append(idx)
                word_list.append(tree.leaves()[idx])

        return cls(treepos, idx_list, word_list)

    @property
    def surface(self):
        if self._surface == '':
            self._surface = ' '.join(self.word_list)
        return self._surface

    def eq_with_preceding_prep(self, other):
        assert isinstance(other, type(self))
        if self.idx_list[1:] == other.idx_list and \
                self.treepos == other.treepos[:-1]:
            return True
        return False


class CoreNLPInfo(object):
    def __init__(self, idx_list, head_idx, word_list, lemma_list,
                 entity_idx=-1, mention_idx=-1):
        self.idx_list = deepcopy(idx_list)
        self.head_idx = head_idx
        self.word_list = deepcopy(word_list)
        self.lemma_list = deepcopy(lemma_list)

        self.entity_idx = entity_idx
        self.mention_idx = mention_idx

        self._word_surface = ''
        self._lemma_surface = ''

        self._head_word = ''
        self._head_lemma = ''

    @classmethod
    def build(cls, treebank_info, corenlp_sent, idx_mapping,
              include_non_head_entity=True, msg_prefix=''):
        idx_list = [idx_mapping.index(idx) for idx
                    in treebank_info.idx_list if idx in idx_mapping]
        head_idx = -1
        head_token = None
        if idx_list:
            head_idx = corenlp_sent.dep_graph.get_head_token_idx(
                idx_list[0], idx_list[-1] + 1, msg_prefix)
            head_token = corenlp_sent.get_token(head_idx)

        token_list = [corenlp_sent.get_token(idx) for idx in idx_list]

        word_list = [token.word for token in token_list]
        lemma_list = [token.lemma for token in token_list]

        entity_idx = -1
        mention_idx = -1

        if idx_list:
            if head_token.coref:
                entity_idx = head_token.coref.idx
                mention_idx = head_token.mention.mention_idx

            elif include_non_head_entity:
                entity_idx_counter = Counter()
                for token in token_list:
                    if token.coref:
                        entity_idx_counter[token.coref.idx] += 1

                if entity_idx_counter:
                    entity_idx = entity_idx_counter.most_common(1)[0][0]

                    mention_idx_counter = Counter()
                    for token in token_list:
                        token_entity_idx = \
                            token.coref.idx if token.coref else -1
                        if token_entity_idx == entity_idx:
                            mention_idx = token.mention.mention_idx
                            mention_idx_counter[mention_idx] += 1

                    mention_idx = \
                        mention_idx_counter.most_common(1)[0][0]

        return cls(idx_list, head_idx, word_list, lemma_list,
                   entity_idx, mention_idx)

    @property
    def word_surface(self):
        if self._word_surface == '':
            self._word_surface = ' '.join(self.word_list)
        return self._word_surface

    @property
    def lemma_surface(self):
        if self._lemma_surface == '':
            self._lemma_surface = ' '.join(self.lemma_list)
        return self._lemma_surface

    @property
    def head_word(self):
        if self._head_word == '' and self.head_idx != -1:
            self._head_word = \
                self.word_list[self.idx_list.index(self.head_idx)]
        return self._head_word

    @property
    def head_lemma(self):
        if self._head_lemma == '' and self.head_idx != -1:
            self._head_lemma = \
                self.lemma_list[self.idx_list.index(self.head_idx)]
        return self._head_lemma


class RichTreePointer(object):
    def __init__(self, fileid, sentnum, tree_pointer, tree=None):
        # treebank file name, format: wsj_0000
        assert fileid[:4] == 'wsj_' and fileid[4:].isdigit()
        self.fileid = fileid
        # sentence number, starting from 0
        self.sentnum = sentnum
        # is_split_pointer = False if Propbank/NombankTreePointer
        # is_split_pointer = True if Propbank/NombankSplitTreePointer
        # raise AssertionError if Propbank/NombankChainTreePointer
        if isinstance(tree_pointer, NombankTreePointer) or \
                isinstance(tree_pointer, PropbankTreePointer):
            self.is_split_pointer = False
        elif isinstance(tree_pointer, NombankSplitTreePointer) or \
                isinstance(tree_pointer, PropbankSplitTreePointer):
            self.is_split_pointer = True
        else:
            raise AssertionError(
                'Unsupported tree_pointer type: {}'.format(type(tree_pointer)))
        # tree pointer
        self.tree_pointer = tree_pointer

        # parse tree of the sentence from PTB
        self.tree = tree
        # treebank related information, including treepos, idx_list, word_list
        self.treebank_info_list = []
        # surface form from treebank, space separated string of all words
        self._treebank_surface = ''

        # CoreNLP related information
        self.corenlp_info_list = []
        # surface form from CoreNLP words
        self._corenlp_word_surface = ''
        # surface form from CoreNLP lemmas
        self._corenlp_lemma_surface = ''
        # index of the head piece if SplitTreePointer, or 0 otherwise
        self.head_piece = 0
        # entity_idx of head_piece if exists, or the most common entity_idx
        # among all pieces
        self.entity_idx = -1
        # mention_idx of head_piece if exists, or the most common mention_idx
        # among pieces that has the corresponding entity_idx
        self.mention_idx = -1

    def get_treebank(self, treebank_reader):
        if not self.has_treebank():
            check_type(treebank_reader, PTBReader)

            treebank_fileid = expand_wsj_fileid(self.fileid, '.mrg')

            treebank_reader.read_file(treebank_fileid)

            self.tree = treebank_reader.all_parsed_sents[self.sentnum]

        return self.tree

    def has_treebank(self):
        return self.tree is not None

    def parse_treebank(self):
        assert self.has_treebank()

        if not self.treebank_info_list:
            if self.is_split_pointer:
                for piece in self.tree_pointer.pieces:
                    self.treebank_info_list.append(
                        TreebankInfo.build(piece, self.tree))

            else:
                self.treebank_info_list.append(
                    TreebankInfo.build(self.tree_pointer, self.tree))

    def has_treebank_info(self):
        return len(self.treebank_info_list) > 0

    @property
    def treebank_surface(self):
        if not self._treebank_surface:
            assert self.has_treebank_info()
            self._treebank_surface = \
                ' '.join([tb.surface for tb in self.treebank_info_list])
        return self._treebank_surface

    def parse_corenlp(self, corenlp_reader, include_non_head_entity=True):
        check_type(corenlp_reader, CoreNLPReader)

        assert self.has_treebank_info()

        doc = corenlp_reader.get_doc(self.fileid)
        corenlp_sent = doc.get_sent(self.sentnum)
        idx_mapping = corenlp_reader.get_idx_mapping(self.fileid)[self.sentnum]

        if not self.corenlp_info_list:
            for treebank_info in self.treebank_info_list:
                self.corenlp_info_list.append(
                    CoreNLPInfo.build(
                        treebank_info, corenlp_sent, idx_mapping,
                        include_non_head_entity=include_non_head_entity,
                        msg_prefix=self.fileid))

        self.head_piece = 0
        min_root_path_length = 999

        for piece_idx, corenlp_info in enumerate(self.corenlp_info_list):
            if corenlp_info.head_idx != -1:
                root_path_length = len(corenlp_sent.dep_graph.get_root_path(
                    corenlp_info.head_idx))
                # with same root_path_length, take the latter token
                if root_path_length <= min_root_path_length:
                    min_root_path_length = root_path_length
                    self.head_piece = piece_idx

        self.entity_idx = -1
        self.mention_idx = -1

        head_corenlp_info = self.corenlp_info_list[self.head_piece]
        if head_corenlp_info.entity_idx != -1:
            self.entity_idx = head_corenlp_info.entity_idx
            self.mention_idx = head_corenlp_info.mention_idx
        else:
            entity_idx_counter = Counter()
            for corenlp_info in self.corenlp_info_list:
                if corenlp_info.entity_idx != -1:
                    entity_idx_counter[corenlp_info.entity_idx] += 1

            if entity_idx_counter:
                self.entity_idx = entity_idx_counter.most_common(1)[0][0]

                mention_idx_counter = Counter()
                for corenlp_info in self.corenlp_info_list:
                    if corenlp_info.entity_idx == self.entity_idx:
                        mention_idx_counter[corenlp_info.mention_idx] += 1

                self.mention_idx = mention_idx_counter.most_common(1)[0][0]

    def has_corenlp_info(self):
        return len(self.corenlp_info_list) > 0

    @property
    def corenlp_word_surface(self):
        if not self._corenlp_word_surface:
            self._corenlp_word_surface = \
                ' '.join([cn.word_surface for cn in self.corenlp_info_list])
        return self._corenlp_word_surface

    @property
    def corenlp_lemma_surface(self):
        if not self._corenlp_lemma_surface:
            self._corenlp_lemma_surface = \
                ' '.join([cn.lemma_surface for cn in self.corenlp_info_list])
        return self._corenlp_lemma_surface

    def get_core_argument(
            self, corenlp_reader, use_lemma=True, use_entity=True):
        if use_entity and self.entity_idx != -1:
            script = corenlp_reader.get_script(self.fileid)
            entity = script.entities[self.entity_idx]
            word = entity.rep_mention.head_token.get_representation(
                use_lemma=use_lemma)
            pos = entity.rep_mention.head_token.pos
            # TODO: should I use entity.ner rather than entity.rep_mention.ner?
            ner = entity.rep_mention.ner
            core_argument = CoreArgument(word, pos, ner)
        else:
            head_idx = self.corenlp_info_list[self.head_piece].head_idx
            assert head_idx != -1
            doc = corenlp_reader.get_doc(self.fileid)
            token = doc.get_token(self.sentnum, head_idx)
            argument = Argument.from_token(token)
            word = argument.get_representation(use_lemma=use_lemma)
            pos = argument.pos
            ner = argument.ner
            core_argument = CoreArgument(word, pos, ner)
        return core_argument

    def get_entity_salience(self, corenlp_reader, use_entity=True):
        if use_entity and self.entity_idx != -1:
            rich_script = corenlp_reader.get_rich_script(self.fileid)
            rich_entity = rich_script.rich_entities[self.entity_idx]
            return rich_entity.get_salience()
        else:
            first_loc = self.sentnum
            head_count = 1
            num_mentions_named = 0
            num_mentions_nominal = 0
            num_mentions_pronominal = 0
            num_mentions_total = 1

            core = self.get_core_argument(
                corenlp_reader, use_lemma=True, use_entity=use_entity)
            if core.ner != '':
                num_mentions_named = 1
            elif core.pos.startswith('NN'):
                num_mentions_nominal = 1
            elif core.pos.startswith('PRP'):
                num_mentions_pronominal = 1

            salience = EntitySalience(
                first_loc=first_loc,
                head_count=head_count,
                num_mentions_named=num_mentions_named,
                num_mentions_nominal=num_mentions_nominal,
                num_mentions_pronominal=num_mentions_pronominal,
                num_mentions_total=num_mentions_total)
            return salience

    def token_list(self, use_corenlp_tokens=True):
        if use_corenlp_tokens:
            assert self.has_corenlp_info()
            token_list = self.corenlp_lemma_surface.split()
        else:
            assert self.has_treebank_info()
            token_list = [lemmatizer.lemmatize(token) for token
                          in self.treebank_surface.split()]
        return token_list

    def dice_score(self, other, use_corenlp_tokens=True):
        token_set = set(self.token_list(use_corenlp_tokens))
        other_token_set = set(other.token_list(use_corenlp_tokens))

        return 2.0 * len(token_set.intersection(other_token_set)) / (
            len(token_set) + len(other_token_set))

    def __str__(self):
        return '{}:{}:{}'.format(
            self.fileid, self.sentnum, self.tree_pointer)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self, include_treebank=False, include_corenlp=False):
        new_rich_tree_pointer = RichTreePointer(
            self.fileid, self.sentnum, deepcopy(self.tree_pointer))

        if include_treebank:
            new_rich_tree_pointer.tree = deepcopy(self.tree)
            new_rich_tree_pointer.treebank_info_list = \
                [deepcopy(tb) for tb in self.treebank_info_list]

        if include_corenlp:
            new_rich_tree_pointer.corenlp_info_list = \
                [deepcopy(cn) for cn in self.corenlp_info_list]
            new_rich_tree_pointer.head_piece = self.head_piece
            new_rich_tree_pointer.entity_idx = self.entity_idx
            new_rich_tree_pointer.mention_idx = self.mention_idx

        return new_rich_tree_pointer

    def pretty_print(self, corenlp_reader=None):
        result = str(self)
        result += '\t\t' + self.corenlp_word_surface
        result += '\tHEAD = ' + \
                  self.corenlp_info_list[self.head_piece].head_word
        if self.entity_idx != -1:
            result += '\t\tentity#{:0>3d}'
            if corenlp_reader:
                script = corenlp_reader.get_script(self.fileid)
                entity = script.entities[self.entity_idx]
                result += ' HEAD = {}'.format(
                    self.entity_idx, entity.get_core_argument())
        return result

    @staticmethod
    def from_text(text):
        items = text.split(':')
        if len(items) != 4:
            raise ValueError('bad pointer.')
        fileid = items[0]
        sentnum = int(items[1])
        nombank_pointer = NombankTreePointer(int(items[2]), int(items[3]))
        return RichTreePointer(fileid, sentnum, nombank_pointer)

    @staticmethod
    def merge(pieces):
        assert len(pieces) > 1, 'SplitPointer must contain more than one piece'
        fileid = pieces[0].fileid
        sentnum = pieces[0].sentnum
        assert all(fileid == piece.fileid for piece in pieces[1:]), \
            'inconsistent fileid'
        assert all(sentnum == piece.sentnum for piece in pieces[1:]), \
            'inconsistent sentnum'
        nombank_pointer = NombankSplitTreePointer(
            [piece.tree_pointer for piece in pieces])
        return RichTreePointer(fileid, sentnum, nombank_pointer)
