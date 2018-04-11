import pickle as pkl
from os import makedirs
from os.path import exists, join

from event_vector_network import EventVectorNetwork
from pair_composition_network import PairCompositionNetwork
from utils import Word2VecModel, check_type, consts


class EventCompositionModel(object):
    def __init__(self, word2vec, event_vector_layer_sizes=None,
                 pair_composition_layer_sizes=None, use_salience=True,
                 salience_features=None):
        check_type(word2vec, Word2VecModel)
        self.word2vec = word2vec
        if event_vector_layer_sizes:
            self.event_vector_network = EventVectorNetwork(
                word_vectors=self.word2vec.get_vector_matrix(),
                vector_size=self.word2vec.vector_size,
                layer_sizes=event_vector_layer_sizes
            )
        else:
            self.event_vector_network = None
        if pair_composition_layer_sizes:
            self.pair_composition_network = PairCompositionNetwork(
                event_vector_network=self.event_vector_network,
                layer_sizes=pair_composition_layer_sizes,
                use_salience=use_salience,
                salience_features=salience_features
            )
        else:
            self.pair_composition_network = None

    def add_event_vector_network(self, layer_sizes):
        assert self.event_vector_network is None, \
            'cannot add EventVectorNetwork when one already exists'
        self.event_vector_network = EventVectorNetwork(
            word_vectors=self.word2vec.get_vector_matrix(),
            vector_size=self.word2vec.vector_size,
            layer_sizes=layer_sizes
        )

    def add_pair_projection_network(self, layer_sizes, use_salience=True,
                                    salience_features=None):
        assert self.pair_composition_network is None, \
            'cannot add PairCompositionNetwork when one already exists'
        assert self.event_vector_network is not None, \
            'cannot add PairCompositionNetwork when EventVectorNetwork ' \
            'does not exists'
        self.pair_composition_network = PairCompositionNetwork(
            event_vector_network=self.event_vector_network,
            layer_sizes=layer_sizes,
            use_salience=use_salience,
            salience_features=salience_features
        )

    def save_model(self, directory, save_word2vec=True,
                   save_event_vector=True, save_pair_composition=False):
        if not exists(directory):
            makedirs(directory)

        if save_word2vec:
            self.word2vec.set_vector_matrix(
                self.event_vector_network.get_word_vectors())
            self.word2vec.save_model(directory, 'word2vec')

        if save_event_vector and self.event_vector_network:
            with open(join(directory, 'ev_weights'), 'w') as f:
                pkl.dump(self.event_vector_network.get_weights(), f)
            with open(join(directory, "ev_layer_sizes"), 'w') as f:
                pkl.dump(self.event_vector_network.layer_sizes, f)
        if save_pair_composition and self.pair_composition_network:
            with open(join(directory, 'pc_weights'), 'w') as f:
                pkl.dump(self.pair_composition_network.get_weights(), f)
            with open(join(directory, 'pc_layer_sizes'), 'w') as f:
                pkl.dump(self.pair_composition_network.layer_sizes, f)
            if self.pair_composition_network.use_salience:
                open(join(directory, 'use_salience'), 'w').close()
                with open(join(directory, 'salience_features'), 'w') as f:
                    pkl.dump(self.pair_composition_network.salience_features, f)

    @classmethod
    def load_model(cls, directory):
        if not exists(directory):
            raise RuntimeError('{} does not exist, abort'.format(directory))

        # load word2vec
        word2vec_vector_file = join(directory, 'word2vec.bin')
        word2vec_vocab_file = join(directory, 'word2vec.vocab')
        if not exists(word2vec_vector_file) or not exists(word2vec_vocab_file):
            raise RuntimeError(
                '{} does not contain word2vec vectors and vocabs, '
                '{} and {} expected'.format(
                    directory, word2vec_vector_file, word2vec_vocab_file))
        word2vec = Word2VecModel.load_model(
            word2vec_vector_file, fvocab=word2vec_vocab_file)

        # load event vector network layer sizes, if exists
        event_vector_layer_sizes_file = join(directory, 'ev_layer_sizes')
        if exists(event_vector_layer_sizes_file):
            with open(event_vector_layer_sizes_file, 'r') as f:
                event_vector_layer_sizes = pkl.load(f)
        else:
            event_vector_layer_sizes = None

        # load pair composition network layer sizes, if exists
        pair_composition_layer_sizes_file = join(directory, 'pc_layer_sizes')
        if exists(pair_composition_layer_sizes_file):
            with open(pair_composition_layer_sizes_file, 'r') as f:
                pair_composition_layer_sizes = pkl.load(f)
        else:
            pair_composition_layer_sizes = None

        # set use_salience to True if there exists a file called use_salience
        if exists(join(directory, 'use_salience')):
            use_salience = True
            salience_features_file = join(directory, 'salience_features')
            if exists(salience_features_file):
                with open(salience_features_file, 'r') as f:
                    salience_features = pkl.load(f)
            else:
                salience_features = consts.salience_features
        else:
            use_salience = False
            salience_features = None

        # initialize the event composition model
        model = cls(
            word2vec=word2vec,
            event_vector_layer_sizes=event_vector_layer_sizes,
            pair_composition_layer_sizes=pair_composition_layer_sizes,
            use_salience=use_salience, salience_features=salience_features)

        # load event vector network weights, if exists
        event_vector_weights_file = join(directory, 'ev_weights')
        if exists(event_vector_weights_file):
            assert model.event_vector_network is not None, \
                'cannot load event vector network weights ' \
                'when the network is not initialized'
            with open(event_vector_weights_file, 'r') as f:
                event_vector_weights = pkl.load(f)
            model.event_vector_network.set_weights(event_vector_weights)

        # load pair composition network weights, if exists
        pair_composition_weights_file = join(directory, 'pc_weights')
        if exists(pair_composition_weights_file):
            assert model.pair_composition_network is not None, \
                'cannot load pair composition network weights ' \
                'when the network is not initialized'
            with open(pair_composition_weights_file, 'r') as f:
                pair_composition_weights = pkl.load(f)
            model.pair_composition_network.set_weights(pair_composition_weights)

        return model
