import time

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import get_console_logger


class DenoisingAutoencoderIterableTrainer(object):
    """
    Train without loading all the data into memory at once.
    """

    def __init__(self, network):
        self.network = network
        self.learning_rate = T.scalar('learning_rate')
        self.regularization = T.scalar('regularization')

    def train(self, batch_iterator, iterations=10000, log=None,
              training_cost_prop_change_threshold=0.0005, learning_rate=0.1,
              regularization=0., corruption_level=0., loss='xent',
              log_every_batch=1000):
        """
        Train on data stored in Theano tensors. Uses minibatch training.

        batch_iterator should be a repeatable iterator producing batches.

        iteration_callback is called after each iteration with args (
        iteration, error array).

        The algorithm will assume it has converged and stop early if the 
        proportional change between successive
        training costs drops below training_cost_prop_change_threshold for 
        five iterations in a row.

        Uses L2 regularization.

        """
        if log is None:
            log = get_console_logger()

        log.info(
            'Training params: learning rate={}, noise ratio={:.1f}%, '
            'regularization={}'.format(
                learning_rate, corruption_level * 100.0, regularization))
        log.info('Training with SGD')

        # Compile functions
        # Prepare cost/update functions for training
        cost, updates = self.network.get_cost_updates(
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            corruption_level=corruption_level,
            loss=loss)
        # Prepare training functions
        train_fn = theano.function(
            inputs=[
                self.network.x,
                theano.In(self.learning_rate, value=0.1),
                theano.In(self.regularization, value=0.0)
            ],
            outputs=cost,
            updates=updates,
        )

        # Keep a record of costs, so we can plot them
        training_costs = []

        # Keep a copy of the best weights so far
        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            for batch_num, batch in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should
                # with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch.shape[0])
                batch[:] = batch[shuffle]

                # Update the model with this batch's data
                err += train_fn(batch,
                                learning_rate=learning_rate,
                                regularization=regularization)

                if (batch_num + 1) % log_every_batch == 0:
                    log.info(
                        'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                            i, batch_num + 1, batch_iterator.num_batch))

            log.info(
                'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                    i, batch_iterator.num_batch, batch_iterator.num_batch))

            training_costs.append(err / batch_num)

            log.info(
                'COMPLETED ITERATION {:d}: training cost={:.5f}'.format(
                    i, training_costs[-1]))

            # Check the proportional change between this iteration's training
            # cost and the last
            if len(training_costs) > 2:
                training_cost_prop_change = abs(
                    (training_costs[-2] - training_costs[-1]) /
                    training_costs[-2])
                if training_cost_prop_change < \
                        training_cost_prop_change_threshold:
                    # Very small change in training cost - maybe we've converged
                    below_threshold_its += 1
                    if below_threshold_its >= 5:
                        # We've had enough iterations with very small changes:
                        # we've converged
                        log.info(
                            'Proportional change in training cost ({}) below '
                            '{} for 5 successive iterations: converged'.format(
                                training_cost_prop_change,
                                training_cost_prop_change_threshold))
                        break
                    else:
                        log.info(
                            'Proportional change in training cost ({}) below '
                            '{} for {} successive iterations: waiting until '
                            'it\'s been low for 5 iterations'.format(
                                training_cost_prop_change,
                                training_cost_prop_change_threshold,
                                below_threshold_its))
                else:
                    # Reset the below threshold counter
                    below_threshold_its = 0


class DenoisingAutoencoder(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, x, n_visible=784, n_hidden=500, weight=None,
                 bias_hidden=None, bias_visible=None, non_linearity='sigmoid'):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type x: theano.tensor.TensorType
        :param x: a symbolic description of the input or None for standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type weight: theano.tensor.TensorType
        :param weight: Theano variable pointing to a set of weights that should
                be shared belong the dA and another architecture; if dA
                should be standalone set this to None

        :type bias_hidden: theano.tensor.TensorType
        :param bias_hidden: Theano variable pointing to a set of biases values
                (for hidden units) that should be shared belong dA and another
                architecture; if dA should be standalone set this to None

        :type bias_visible: theano.tensor.TensorType
        :param bias_visible: Theano variable pointing to a set of biases values
                (for visible units) that should be shared belong dA and another
                architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # TODO: determine whether or not to use a fixed seed in rng
        numpy_rng = numpy.random.RandomState(int(time.time()))
        # create a Theano random generator that gives symbolic random values
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : weight' was written as `weight_prime` and bias' as `bias_prime`
        if not weight:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_weight = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            weight = theano.shared(value=initial_weight, name='W', borrow=True)

        if not bias_visible:
            bias_visible = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bias_hidden:
            bias_hidden = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.non_linearity = non_linearity
        if non_linearity == 'sigmoid':
            self.activation_fn = T.nnet.sigmoid
            # self.inverse_activation_fn = lambda x: T.log(x / (1-x))
        elif non_linearity == 'tanh':
            self.activation_fn = T.tanh
            # self.inverse_activation_fn = T.arctanh
        else:
            raise ValueError('Unknown non-linearity "{}". Must be '
                             '"sigmoid" or "tanh"'.format(non_linearity))

        self.weight = weight
        # bias corresponds to the bias of the hidden
        self.bias = bias_hidden
        # bias_prime corresponds to the bias of the visible
        self.bias_prime = bias_visible
        # tied weights, therefore W_prime is W transpose
        self.weight_prime = self.weight.T
        # if no input_x is given, generate a variable representing the input
        if x is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.matrix(name='input')
        else:
            self.x = x

        self.params = [self.weight, self.bias, self.bias_prime]
        self.hidden_layer = self.get_hidden_values(self.x)
        self._hidden_fn = theano.function(
            inputs=[self.x],
            outputs=self.hidden_layer,
        )

    def projection(self, xs):
        return self._hidden_fn(xs)

    def get_weights(self):
        """
        Return a copy of all the weight arrays in a tuple.
        """
        return (self.weight.get_value().copy(),
                self.bias.get_value().copy(),
                self.bias_prime.get_value().copy())

    def set_weights(self, weights):
        """
        Set all weights from a tuple, like that returned by get_weights().
        """
        self.weight.set_value(weights[0])
        self.bias.set_value(weights[1])
        self.bias_prime.set_value(weights[2])

    def get_corrupted_input(self, x, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        if corruption_level > 0:
            return self.theano_rng.binomial(
                size=x.shape, n=1, p=1 - corruption_level,
                dtype=theano.config.floatX) * x
        else:
            return x

    def get_hidden_values(self, x):
        """ Computes the values of the hidden layer """
        return self.activation_fn(T.dot(x, self.weight) + self.bias)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return self.activation_fn(
            T.dot(hidden, self.weight_prime) + self.bias_prime)

    def get_reconstruction(self, corruption_level):
        tilde_x = self.get_corrupted_input(
            self.x, corruption_level=corruption_level)
        y = self.get_hidden_values(tilde_x)
        return self.get_reconstructed_input(y)

    def get_cost(self, regularization=0.0, corruption_level=0., loss='xent'):
        z = self.get_reconstruction(corruption_level=corruption_level)

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if loss == 'xent':
            # Cross-entropy loss function
            if self.non_linearity == 'tanh':
                # If we're using tanh activation, we need to shift & scale
                # the output into the (0, 1) range
                z = 0.5 * (z + 1.)
            cost = - T.sum(
                (self.x * T.log(z) + (1 - self.x) * T.log(1 - z)), axis=1)
        elif loss == 'l2':
            # Squared error loss function
            cost = T.sum(0.5 * ((self.x - z) ** 2), axis=1)
        else:
            raise ValueError('Unknown loss function "{}". '
                             'Expected one of: "xent", "l2"'.format(loss))
        # L is now a vector, where each element is the cross-entropy cost of
        # the reconstruction of the corresponding example of the minibatch
        cost = T.mean(cost) + regularization * T.mean(self.weight ** 2.)
        return cost

    def get_cost_updates(self, learning_rate, regularization,
                         corruption_level=0., loss='xent'):
        """ This function computes the cost and the updates for one training
        step of the dA

        """
        cost = self.get_cost(regularization=regularization,
                             corruption_level=corruption_level, loss=loss)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = [T.grad(cost, param) for param in self.params]
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return cost, updates
