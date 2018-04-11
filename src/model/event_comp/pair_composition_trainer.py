import numpy
import theano
import theano.tensor as T

from pair_composition_network import PairCompositionNetwork
from utils import check_type, get_console_logger


class PairCompositionTrainer(object):
    def __init__(self, model, learning_rate=0.025, min_learning_rate=0.0001,
                 regularization=0.01, update_event_vectors=False,
                 update_input_vectors=False, update_empty_vectors=False):
        check_type(model, PairCompositionNetwork)
        self.model = model

        self.min_learning_rate = min_learning_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.learning_rate_var = T.scalar(
            'learning_rate', dtype=theano.config.floatX)

        # Collect parameters to be tuned from all layers
        self.params = []
        self.regularized_params = []
        # Always add our own between-event composition function weights
        self.params.extend(
            sum([[layer.W, layer.b] for layer in model.layers], [])
        )
        self.regularized_params.extend([layer.W for layer in model.layers])

        self.params.append(self.model.prediction_weights)
        self.params.append(self.model.prediction_bias)
        self.regularized_params.append(self.model.prediction_weights)

        if update_event_vectors:
            # Add the event-internal composition weights
            self.params.extend(
                sum([[layer.W, layer.b] for layer
                     in self.model.event_vector_network.layers], []))
            self.regularized_params.extend(
                [layer.W for layer in self.model.event_vector_network.layers])

        if update_input_vectors:
            self.params.append(self.model.event_vector_network.vectors)

        if update_empty_vectors:
            self.params.extend([
                self.model.event_vector_network.empty_subj_vector,
                self.model.event_vector_network.empty_obj_vector,
                self.model.event_vector_network.empty_pobj_vector
            ])

        self.update_event_vectors = update_event_vectors
        self.update_input_vectors = update_input_vectors
        self.update_empty_vectors = update_empty_vectors

    def get_triple_cost_updates(self, regularization=None, compute_update=True):
        if regularization is None:
            regularization = self.regularization

        # Compute the two similarities predicted by our current composition
        pos_coherences, neg_coherences = self.model.get_coherence_pair()
        # We want pos coherences to be higher than neg
        # Try to make pos - neg as high as possible (it ranges between -1 and 1)
        cost_per_input = -T.log(pos_coherences) - T.log(1. - neg_coherences)
        cost = T.mean(cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            reg_term = regularization * T.sum(
                [T.sum(w ** 2) for w in self.regularized_params])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.cast(
                T.sum([T.prod(T.shape(w)) for w in self.regularized_params]),
                theano.config.floatX)
            cost += reg_term

        if compute_update:
            # Now differentiate to get the updates
            gparams = [T.grad(cost, param) for param in self.params]
            updates = [(param, param - self.learning_rate_var * gparam)
                       for param, gparam in zip(self.params, gparams)]

            return cost, updates
        else:
            return cost

    @staticmethod
    def compute_val_cost(cost_fn, val_batch_iterator):
        cost = 0.0
        batch_num = 0
        for batch_num, batch_inputs in enumerate(val_batch_iterator):
            cost += cost_fn(*batch_inputs)
        return cost / (batch_num + 1)

    def train(self, batch_iterator, iterations=10000, iteration_callback=None,
              log=None, training_cost_prop_change_threshold=0.0005,
              val_batch_iterator=None, stopping_iterations=10,
              log_every_batch=1000):
        # TODO: add logic for validation set and stopping_iterations parameter
        if log is None:
            log = get_console_logger()

        log.info(
            'Tuning params: learning rate={} (->{}), regularization={}'.format(
                self.learning_rate, self.min_learning_rate,
                self.regularization))
        if self.update_event_vectors:
            log.info('Updating event vector network')
        if self.update_input_vectors:
            log.info('Updating word2vec word representations')
        if self.update_empty_vectors:
            log.info('Training empty argument vectors')

        # Compile functions
        # Prepare cost/update functions for training
        cost, updates = self.get_triple_cost_updates(compute_update=True)
        # Prepare training functions
        train_fn = theano.function(
            inputs=self.model.triple_inputs + [
                # Allow the learning rate to be set per update
                theano.In(self.learning_rate_var, value=self.learning_rate)
            ],
            outputs=cost,
            updates=updates,
        )
        # Prepare cost functions without regularization for validation
        cost_without_reg = self.get_triple_cost_updates(
            regularization=0., compute_update=False)
        cost_fn = theano.function(
            inputs=self.model.triple_inputs,
            outputs=cost_without_reg,
        )

        # Keep a record of costs, so we can plot them
        training_costs = []
        val_costs = []

        # Keep a copy of the best weights so far
        best_weights = best_iter = best_val_cost = None
        if val_batch_iterator is not None:
            best_weights = self.model.get_weights()
            best_iter = -1
            best_val_cost = PairCompositionTrainer.compute_val_cost(
                cost_fn, val_batch_iterator)

        below_threshold_its = 0

        learning_rate = self.learning_rate
        last_update_lr_iter = 0

        if val_batch_iterator is not None:
            # Compute the initial cost on the validation set
            val_cost = PairCompositionTrainer.compute_val_cost(
                cost_fn, val_batch_iterator)
            log.info('Initial validation cost: {:.4f}'.format(val_cost))

        for i in range(iterations):
            err = 0.0
            batch_num = 0

            for batch_num, batch_inputs in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should
                # with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch_inputs[0].shape[0])
                for batch_data in batch_inputs:
                    batch_data[:] = batch_data[shuffle]

                # Update the model with this batch's data
                err += train_fn(*batch_inputs, learning_rate=learning_rate)

                if (batch_num + 1) % log_every_batch == 0:
                    log.info(
                        'Iteration {}: Processed {:>8d}/{:>8d} batches, '
                        'learning rate = {:g}'.format(
                            i, batch_num + 1, batch_iterator.num_batch,
                            learning_rate))

            log.info(
                'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                    i, batch_iterator.num_batch, batch_iterator.num_batch))

            training_costs.append(err / (batch_num+1))

            if val_batch_iterator is not None:
                # Compute the cost function on the validation set
                val_cost = PairCompositionTrainer.compute_val_cost(
                    cost_fn, val_batch_iterator)
                val_costs.append(val_cost)
                if val_cost <= best_val_cost:
                    # We assume that, if the validation error remains the same,
                    # it's better to use the new set of
                    # weights (with, presumably, a better training error)
                    if val_cost == best_val_cost:
                        log.info(
                            'Same validation cost: {:.4f}, '
                            'using new weights'.format(val_cost))
                    else:
                        log.info(
                            'New best validation cost: {:.4f}'.format(val_cost))
                    # Update our best estimate
                    best_weights = self.model.get_weights()
                    best_iter = i
                    best_val_cost = val_cost
                if val_cost >= best_val_cost \
                        and i - best_iter >= stopping_iterations:
                    # We've gone on long enough without improving validation
                    # error, time to call a halt and use the best validation
                    # error we got
                    log.info('Stopping after {} iterations of increasing '
                             'validation cost'.format(stopping_iterations))
                    break

            log.info(
                'COMPLETED ITERATION {}: training cost={:.5g}, '
                'validation cost={:.5g}'.format(
                    i, training_costs[-1], val_costs[-1]))

            if val_costs[-1] >= best_val_cost and i - best_iter >= 2 \
                    and i - last_update_lr_iter >= 2 \
                    and learning_rate > self.min_learning_rate:
                # We've gone on 2 iterations without improving validation
                # error, time to reduce the learning rate
                learning_rate /= 2
                if learning_rate < self.min_learning_rate:
                    learning_rate = self.min_learning_rate
                last_update_lr_iter = i
                log.info(
                    'Halving learning rate to {} after 2 iterations of '
                    'increasing validation cost'.format(learning_rate))

            if iteration_callback is not None:
                # Not computing training error at the moment
                iteration_callback(i)

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
                            'Proportional change in training cost ({:g}) below '
                            '{:g} for five successive iterations: '
                            'converged'.format(
                                training_cost_prop_change,
                                training_cost_prop_change_threshold))
                        break
                    else:
                        log.info(
                            'Proportional change in training cost ({:g}) below '
                            '{:g} for {} successive iterations: waiting until '
                            'it is been low for five iterations'.format(
                                training_cost_prop_change,
                                training_cost_prop_change_threshold,
                                below_threshold_its))
                else:
                    # Reset the below threshold counter
                    below_threshold_its = 0

        if best_weights is not None:
            # Use the weights that gave us the best error on the validation set
            self.model.set_weights(best_weights)
