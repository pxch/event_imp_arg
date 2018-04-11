import numpy
import theano
import theano.tensor as T

from autoencoder import DenoisingAutoencoder


class PairCompositionNetwork(object):
    def __init__(self, event_vector_network, layer_sizes, use_salience=True,
                 salience_features=None):
        self.event_vector_network = event_vector_network
        self.layer_sizes = layer_sizes

        self.input_a, self.input_b = \
            self.event_vector_network.get_projection_pair()

        self.arg_idx_input = T.vector('arg_type')
        self.neg_arg_idx_input = T.vector('neg_arg_type')

        self.input_vector = T.concatenate(
            (self.input_a, self.input_b,
             self.arg_idx_input.dimshuffle(0, 'x')), axis=1)

        self.use_salience = use_salience
        if self.use_salience:
            self.salience_input = T.matrix('salience')
            # variables for negative entity salience
            self.neg_salience_input = T.matrix('neg_salience')

            self.input_vector = T.concatenate(
                (self.input_vector, self.salience_input), axis=1)

        # Initialize each layer as an autoencoder,
        # allowing us to initialize it by pretraining
        self.layers = []
        self.layer_outputs = []
        input_size = \
            self.event_vector_network.layer_sizes[-1] * 2 + 1

        self.num_salience_features = 0
        self.salience_features = None
        if self.use_salience:
            assert salience_features is not None
            self.salience_features = salience_features
            self.num_salience_features = len(self.salience_features)
            input_size += self.num_salience_features

        layer_input = self.input_vector
        for layer_size in layer_sizes:
            self.layers.append(
                DenoisingAutoencoder(
                    x=layer_input,
                    n_visible=input_size,
                    n_hidden=layer_size,
                    non_linearity='tanh'
                )
            )
            input_size = layer_size
            layer_input = self.layers[-1].hidden_layer
            self.layer_outputs.append(layer_input)
        self.final_projection = layer_input

        # Add a final layer, which will only ever be trained with
        # a supervised objective
        # This is simply a logistic regression layer to predict
        # a coherence score for the input pair
        self.prediction_weights = theano.shared(
            # Just initialize to zeros, so we start off predicting 0.5
            # for every input
            numpy.asarray(
                numpy.random.uniform(
                    low=2. * -numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    high=2. * numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    size=(layer_sizes[-1], 1),
                ),
                dtype=theano.config.floatX
            ),
            name='prediction_w',
            borrow=True
        )
        self.prediction_bias = theano.shared(
            value=numpy.zeros(1, dtype=theano.config.floatX),
            name='prediction_b',
            borrow=True
        )
        self.prediction = T.nnet.sigmoid(
            T.dot(self.final_projection, self.prediction_weights) +
            self.prediction_bias
        )

        self.pair_inputs = [
            self.event_vector_network.pred_input_a,
            self.event_vector_network.subj_input_a,
            self.event_vector_network.obj_input_a,
            self.event_vector_network.pobj_input_a,
            self.event_vector_network.pred_input_b,
            self.event_vector_network.subj_input_b,
            self.event_vector_network.obj_input_b,
            self.event_vector_network.pobj_input_b,
            self.arg_idx_input
        ]
        if self.use_salience:
            self.pair_inputs.append(self.salience_input)

        self.triple_inputs = [
            self.event_vector_network.pred_input_a,
            self.event_vector_network.subj_input_a,
            self.event_vector_network.obj_input_a,
            self.event_vector_network.pobj_input_a,
            self.event_vector_network.pred_input_b,
            self.event_vector_network.subj_input_b,
            self.event_vector_network.obj_input_b,
            self.event_vector_network.pobj_input_b,
            self.event_vector_network.pred_input_c,
            self.event_vector_network.subj_input_c,
            self.event_vector_network.obj_input_c,
            self.event_vector_network.pobj_input_c,
            self.arg_idx_input,
            self.neg_arg_idx_input
        ]
        if self.use_salience:
            self.triple_inputs.append(self.salience_input)
            self.triple_inputs.append(self.neg_salience_input)

        self._coherence_fn = None

    @property
    def coherence_fn(self):
        if self._coherence_fn is None:
            self._coherence_fn = theano.function(
                inputs=self.pair_inputs,
                outputs=self.prediction,
                name='pair_coherence',
            )
        return self._coherence_fn

    def get_coherence_pair(self):
        # Clone prediction function so we can perform two predictions
        # in the same step
        coherence_a = self.prediction

        # Replace b inputs with c inputs
        input_replacements = dict(zip(self.triple_inputs[4:8],
                                      self.triple_inputs[8:12]))
        # Replace arg_idx_input with neg_arg_idx_input
        input_replacements[self.triple_inputs[12]] = self.triple_inputs[13]
        # Replace salience_input with neg_salience_input
        if self.use_salience:
            input_replacements[self.triple_inputs[14]] = self.triple_inputs[15]

        coherence_b = theano.clone(self.prediction, replace=input_replacements)

        return coherence_a, coherence_b

    def get_layer_input_function(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError(
                'cannot get input function for layer {} '
                'in a {}-layer network'.format(layer_num, len(self.layers)))
        elif layer_num == 0:
            # The input to the first layer is just the concatenated input vector
            output_eq = self.input_vector
        else:
            # Otherwise it's the output from the previous layer
            output_eq = self.layer_outputs[layer_num-1]

        return theano.function(
            inputs=self.pair_inputs,
            outputs=output_eq,
            name='layer-{}-input'.format(layer_num)
        )

    def copy_coherence_function(self, input_a=None, input_b=None,
                                arg_idx_input=None, salience_input=None):
        """
        Build a new coherence function, copying all weights and such from
        this network, replacing components given as kwargs. Note that this
        uses the same shared variables and any other non-replaced components
        as the network's original expression graph: bear in mind if you use
        it to update weights or combine with other graphs.

        """
        input_a = input_a or self.input_a
        input_b = input_b or self.input_b
        arg_idx_input = arg_idx_input or self.arg_idx_input
        salience_input = salience_input or self.salience_input

        # Build a new coherence function, combining these two projections
        input_vector = T.concatenate(
            [input_a, input_b, arg_idx_input.dimshuffle(0, 'x'),
             salience_input], axis=input_a.ndim-1)

        # Initialize each layer as an autoencoder.
        # We'll then set its weights and never use it as an autoencoder
        layers = []
        layer_outputs = []
        input_size = \
            self.event_vector_network.layer_sizes[-1] * 2 + \
            1 + self.num_salience_features
        layer_input = input_vector
        for layer_size in self.layer_sizes:
            layers.append(
                DenoisingAutoencoder(
                    x=layer_input,
                    n_visible=input_size,
                    n_hidden=layer_size,
                    non_linearity='tanh'
                )
            )
            input_size = layer_size
            layer_input = layers[-1].hidden_layer
            layer_outputs.append(layer_input)
        final_projection = layer_input

        # Set the weights of all layers to the ones trained in the base network
        for layer, layer_weights in zip(layers, self.get_weights()):
            layer.set_weights(layer_weights)

        # Add a final layer
        # This is simply a logistic regression layer to predict
        # a coherence score for the input pair
        activation = \
            T.dot(final_projection, self.prediction_weights) + \
            self.prediction_bias
        # Remove the last dimension, which should now just be of size 1
        activation = activation.reshape(activation.shape[:-1],
                                        ndim=activation.ndim-1)
        prediction = T.nnet.sigmoid(activation)

        return prediction, input_vector, layers, layer_outputs, activation

    def get_weights(self):
        return [ae.get_weights() for ae in self.layers] + \
               [self.prediction_weights.get_value(),
                self.prediction_bias.get_value()]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
        self.prediction_weights.set_value(weights[len(self.layers)])
        self.prediction_bias.set_value(weights[len(self.layers) + 1])
