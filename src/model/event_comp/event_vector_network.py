import numpy
import theano
import theano.tensor as T

from autoencoder import DenoisingAutoencoder


class EventVectorNetwork(object):
    def __init__(self, word_vectors, vector_size, layer_sizes, pred_input=None,
                 subj_input=None, obj_input=None, pobj_input=None,
                 inputs_a=None, inputs_b=None, inputs_c=None):
        self.vector_size = vector_size
        self.layer_sizes = layer_sizes
        self.projection_size = self.layer_sizes[-1]

        # Very first inputs are integers to select the input vectors
        if pred_input is not None:
            self.pred_input = pred_input
        else:
            self.pred_input = T.vector('pred', dtype='int32')
        if subj_input is not None:
            self.subj_input = subj_input
        else:
            self.subj_input = T.vector('subj', dtype='int32')
        if obj_input is not None:
            self.obj_input = obj_input
        else:
            self.obj_input = T.vector('obj', dtype='int32')
        if pobj_input is not None:
            self.pobj_input = pobj_input
        else:
            self.pobj_input = T.vector('pobj', dtype='int32')

        # Wrap the input vector matrices in a Theano variable
        self.vectors = theano.shared(
            numpy.asarray(word_vectors, dtype=theano.config.floatX),
            name='vectors',
            borrow=False
        )

        # In order to stop the projections being thrown off by empty arguments,
        # we need to learn an empty argument vector.
        # This is initialized to zero.
        self.empty_subj_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name='empty_subj',
            borrow=True,
            broadcastable=(True, False),
        )
        self.empty_obj_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name='empty_obj',
            borrow=True,
            broadcastable=(True, False),
        )
        self.empty_pobj_vector = theano.shared(
            numpy.asarray(
                numpy.zeros(self.vector_size)[None, :],
                dtype=theano.config.floatX
            ),
            name='empty_pobj',
            borrow=True,
            broadcastable=(True, False),
        )

        self.input_size = 4 * self.vector_size
        # Build the theano expression for this network
        self.input_vector, self.layers, self.layer_outputs, \
            self.projection_layer = \
            EventVectorNetwork.build_projection_layer(
                self.pred_input, self.subj_input, self.obj_input,
                self.pobj_input, self.vectors, self.empty_subj_vector,
                self.empty_obj_vector, self.empty_pobj_vector,
                self.input_size, self.layer_sizes)

        self.norm_projection_layer = \
            self.projection_layer / T.sqrt(
                (self.projection_layer ** 2.).sum(axis=1)).reshape(
                (self.projection_layer.shape[0], 1))

        # Composition of two projections
        if inputs_a is not None:
            self.pred_input_a, self.subj_input_a, self.obj_input_a, \
                self.pobj_input_a = inputs_a
        else:
            self.pred_input_a = T.vector('pred_a', dtype='int32')
            self.subj_input_a = T.vector('subj_a', dtype='int32')
            self.obj_input_a = T.vector('obj_a', dtype='int32')
            self.pobj_input_a = T.vector('pobj_a', dtype='int32')

        if inputs_b is not None:
            self.pred_input_b, self.subj_input_b, self.obj_input_b, \
                self.pobj_input_b = inputs_b
        else:
            self.pred_input_b = T.vector('pred_b', dtype='int32')
            self.subj_input_b = T.vector('subj_b', dtype='int32')
            self.obj_input_b = T.vector('obj_b', dtype='int32')
            self.pobj_input_b = T.vector('pobj_b', dtype='int32')

        # Or three
        if inputs_c is not None:
            self.pred_input_c, self.subj_input_c, self.obj_input_c, \
                self.pobj_input_c = inputs_c
        else:
            self.pred_input_c = T.vector('pred_c', dtype='int32')
            self.subj_input_c = T.vector('subj_c', dtype='int32')
            self.obj_input_c = T.vector('obj_c', dtype='int32')
            self.pobj_input_c = T.vector('pobj_c', dtype='int32')

        # Compile the Theano functions
        # This projects all the way from the input to the output,
        # once each layer's been trained
        self.project = theano.function(
            inputs=[self.pred_input, self.subj_input, self.obj_input,
                    self.pobj_input],
            outputs=self.projection_layer,
            name='project'
        )

    def copy_projection_function(self, pred_input=None, subj_input=None,
                                 obj_input=None, pobj_input=None):
        """
        Build a new projection function, copying all weights and such from this
        network, replacing components given as kwargs. Note that this uses the
        same shared variables and any other non-replaced components as the 
        network's original expression graph: bear in mind if you use it to 
        update weights or combine with other graphs.

        """
        pred_input = pred_input or self.pred_input
        subj_input = subj_input or self.subj_input
        obj_input = obj_input or self.obj_input
        pobj_input = pobj_input or self.pobj_input

        # Build a new projection function
        input_vector, layers, layer_outputs, projection_layer = \
            EventVectorNetwork.build_projection_layer(
                pred_input, subj_input, obj_input, pobj_input,
                self.vectors, self.empty_subj_vector, self.empty_obj_vector,
                self.empty_pobj_vector, self.input_size, self.layer_sizes)

        # Set all the layers' weights to the ones in the base network
        for layer, layer_weights in zip(layers, self.get_weights()):
            # get_weights gives some more things too,
            # but the first weights correspond to layers
            layer.set_weights(layer_weights)

        return projection_layer, input_vector, layers, layer_outputs

    @staticmethod
    def build_projection_layer(pred_input, subj_input, obj_input, pobj_input,
                               vectors, empty_subj_vector, empty_obj_vector,
                               empty_pobj_vector, input_size, layer_sizes):
        # Rearrange these so we can test for -1 indices
        # In the standard case, this does dimshuffle((0, "x")), which changes
        # a 1D vector into a column vector
        shuffled_dims = tuple(list(range(subj_input.ndim)) + ['x'])
        subj_input_col = subj_input.dimshuffle(shuffled_dims)
        obj_input_col = obj_input.dimshuffle(shuffled_dims)
        pobj_input_col = pobj_input.dimshuffle(shuffled_dims)

        # Make the input to the first autoencoder by selecting the appropriate
        # vectors from the given matrices
        input_vector = T.concatenate(
            [
                vectors[pred_input],
                T.switch(T.neq(subj_input_col, -1), vectors[subj_input],
                         empty_subj_vector),
                T.switch(T.neq(obj_input_col, -1), vectors[obj_input],
                         empty_obj_vector),
                T.switch(T.neq(pobj_input_col, -1), vectors[pobj_input],
                         empty_pobj_vector),
            ],
            axis=pred_input.ndim
        )

        # Build and initialize each layer of the autoencoder
        previous_output = input_vector
        layers = []
        layer_outputs = []
        for layer_size in layer_sizes:
            layers.append(
                DenoisingAutoencoder(
                    x=previous_output,
                    n_hidden=layer_size,
                    n_visible=input_size,
                    non_linearity='tanh',
                )
            )
            input_size = layer_size
            previous_output = layers[-1].hidden_layer
            layer_outputs.append(previous_output)
        projection_layer = previous_output

        return input_vector, layers, layer_outputs, projection_layer

    def get_projection_pair(self, normalize=False):
        if normalize:
            projection = self.norm_projection_layer
        else:
            projection = self.projection_layer

        # Use the deepest projection layer as our target representation
        # Clone this so that we can perform two projections in the same step
        projection_a = theano.clone(projection, replace={
            self.pred_input: self.pred_input_a,
            self.subj_input: self.subj_input_a,
            self.obj_input: self.obj_input_a,
            self.pobj_input: self.pobj_input_a,
        })
        projection_b = theano.clone(projection, replace={
            self.pred_input: self.pred_input_b,
            self.subj_input: self.subj_input_b,
            self.obj_input: self.obj_input_b,
            self.pobj_input: self.pobj_input_b,
        })
        return projection_a, projection_b

    def get_projection_triple(self, normalize=False):
        if normalize:
            projection = self.norm_projection_layer
        else:
            projection = self.projection_layer

        # Use the deepest projection layer as our target representation
        # Clone this so that we can perform three projections in the same step
        projection_a = theano.clone(projection, replace={
            self.pred_input: self.pred_input_a,
            self.subj_input: self.subj_input_a,
            self.obj_input: self.obj_input_a,
            self.pobj_input: self.pobj_input_a,
        })
        projection_b = theano.clone(projection, replace={
            self.pred_input: self.pred_input_b,
            self.subj_input: self.subj_input_b,
            self.obj_input: self.obj_input_b,
            self.pobj_input: self.pobj_input_b,
        })
        projection_c = theano.clone(projection, replace={
            self.pred_input: self.pred_input_c,
            self.subj_input: self.subj_input_c,
            self.obj_input: self.obj_input_c,
            self.pobj_input: self.pobj_input_c,
        })
        return projection_a, projection_b, projection_c

    def get_layer_input_function(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError(
                'cannot get input function for layer {} '
                'in a {}-layer network'.format(layer_num, len(self.layers)))
        elif layer_num == 0:
            # The input to the first layer is just the concatenated
            # input vectors
            output_eq = self.input_vector
        else:
            # Otherwise it's the output from the previous layer
            output_eq = self.layer_outputs[layer_num-1]

        return theano.function(
            inputs=[self.pred_input, self.subj_input, self.obj_input,
                    self.pobj_input],
            outputs=output_eq,
            name='layer-{}-input'.format(layer_num),
        )

    def get_layer_projection(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError(
                'cannot get input function for layer {} '
                'in a {}-layer network'.format(layer_num, len(self.layers)))
        elif layer_num == -1:
            # Special value to get the input to the first layer:
            # i.e. just the concatenated input vectors
            output_eq = self.input_vector
        else:
            output_eq = self.layer_outputs[layer_num]

        return theano.function(
            inputs=[self.pred_input, self.subj_input, self.obj_input,
                    self.pobj_input],
            outputs=output_eq,
            name='layer-{}-projection'.format(layer_num),
        )

    def get_word_vectors(self):
        return self.vectors.get_value()

    def get_weights(self):
        return [ae.get_weights() for ae in self.layers] + \
               [self.empty_subj_vector.get_value(),
                self.empty_obj_vector.get_value(),
                self.empty_pobj_vector.get_value()]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
        self.empty_subj_vector.set_value(weights[len(self.layers)])
        self.empty_obj_vector.set_value(weights[len(self.layers)+1])
        self.empty_pobj_vector.set_value(weights[len(self.layers)+2])
