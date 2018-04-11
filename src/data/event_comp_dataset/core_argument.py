from utils import Word2VecModel, check_type, consts


class CoreArgument(object):
    def __init__(self, word, pos, ner):
        self._word = word
        self._pos = pos
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

    @property
    def word(self):
        return self._word

    @property
    def pos(self):
        return self._pos

    @property
    def ner(self):
        return self._ner

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos \
               and self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{} // {} // {}'.format(self.word, self.pos, self.ner)

    def get_text_with_vocab_list(self, arg_vocab_list, ner_vocab_list):
        text = self.word
        if self.ner != '':
            if ner_vocab_list and text not in ner_vocab_list:
                text = self.ner
        else:
            if arg_vocab_list and text not in arg_vocab_list:
                text = 'UNK'
        return text

    @staticmethod
    def get_candidates_by_arg_type(text, arg_type):
        candidates = []
        if arg_type != '':
            candidates.append(text + '-' + arg_type)
            # back off to remove preposition
            if arg_type.startswith('PREP'):
                candidates.append(text + '-PREP')
        else:
            candidates.append(text)
        return candidates

    def get_index(self, model, arg_type='', use_unk=True):
        check_type(model, Word2VecModel)

        # add candidates from self.word
        candidates = CoreArgument.get_candidates_by_arg_type(
            self.word, arg_type)
        # add candidates from self.ner if self.ner is not empty string
        if self.ner != '':
            candidates.extend(
                CoreArgument.get_candidates_by_arg_type(self.ner, arg_type))
        # add UNK to the candidates if use_unk is set to True
        if use_unk:
            candidates.extend(
                CoreArgument.get_candidates_by_arg_type('UNK', arg_type))
        index = -1
        # iterate through all candidates, and return the first one that exists
        # in the vocabulary of the Word2Vec model, otherwise return -1
        for text in candidates:
            index = model.get_word_index(text)
            if index != -1:
                break
        return index
