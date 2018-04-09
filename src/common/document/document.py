from coreference import Coreference
from sentence import Sentence
from utils import check_type


class Document(object):
    def __init__(self, doc_name):
        # name of the document
        self._doc_name = doc_name
        # list of all sentences in the document
        self._sents = []
        # number of all sentences in the document
        self._num_sents = 0
        # list of all coreference chains in the document
        self._corefs = []
        # number of all coreference chains in the document
        self._num_corefs = 0

    @property
    def doc_name(self):
        return self._doc_name

    @property
    def sents(self):
        return self._sents

    @property
    def num_sents(self):
        return self._num_sents

    @property
    def corefs(self):
        return self._corefs

    @property
    def num_corefs(self):
        return self._num_corefs

    def add_sent(self, sent):
        check_type(sent, Sentence)
        if sent.dep_graph is None:
            sent.build_dep_graph()
        self._sents.append(sent)
        self._num_sents += 1

    def add_coref(self, coref):
        check_type(coref, Coreference)
        self._corefs.append(coref)
        self._num_corefs += 1

    def get_sent(self, idx):
        assert 0 <= idx < self.num_sents, \
            'Sentence index {} out of range'.format(idx)
        result = self._sents[idx]
        check_type(result, Sentence)
        return result

    def get_coref(self, idx):
        assert 0 <= idx < self.num_corefs, \
            'Coreference index {} out of range'.format(idx)
        result = self._corefs[idx]
        check_type(result, Coreference)
        return result

    def get_token(self, sent_idx, token_idx):
        return self.get_sent(sent_idx).get_token(token_idx)

    def get_mention(self, coref_idx, mention_idx):
        return self.get_coref(coref_idx).get_mention(mention_idx)

    def __str__(self):
        result = '\t\t#SENT#\t\t'.join([str(sent) for sent in self._sents])
        result += '\t\t\t#COREFERENCE#\t\t\t'
        result += '\t\t#COREF#\t\t'.join([str(coref) for coref in self._corefs])
        return result

    def pretty_print(self):
        result = '\n'.join(
            ['Sent #{}\n\t'.format(sent_idx) + sent.pretty_print()
             for sent_idx, sent in enumerate(self._sents)])
        result += '\nEntities:\n'
        result += '\n'.join(['\t' + coref.pretty_print()
                             for coref in self._corefs])
        return result

    def plain_text(self):
        return '\n'.join([sent.to_plain_text() for sent in self._sents])

    def preprocessing(self):
        for coref in self._corefs:
            for mention in coref.mentions:
                sent = self.get_sent(mention.sent_idx)
                # set mention.head_token_idx if it is -1 (unset)
                if mention.head_token_idx == -1:
                    mention.set_head_token_idx(sent.dep_graph)
                # add token_info to mention
                mention.add_token_info(sent.tokens)
                # set coref_info to the head token
                # TODO: or add to all tokens of the mention?
                mention.head_token.add_coref_info(coref, mention)
            # set rep_mention if it is None
            if coref.rep_mention is None:
                coref.find_rep_mention()

    @classmethod
    def construct(cls, doc_name, sents, corefs):
        doc = cls(doc_name)
        for sent in sents:
            doc.add_sent(sent)
        for coref in corefs:
            doc.add_coref(coref)
        doc.preprocessing()
        return doc
