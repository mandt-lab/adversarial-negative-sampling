import sys
import numpy as np
import tensorflow as tf
import h5py

from aux_model.abstract_aux_model import AbstractAuxModel


class UniformAuxModel(AbstractAuxModel):
    def __init__(self, dat, log_file=sys.stdout, supervised=False):
        log_file.write(
            '\n# Drawing negative samples with uniform probabilities.\n')
        log_file.flush()

        self.padded_range_e = dat.range_e
        # self._pll = -np.log(dat.range_e)
        self._pll = 0.0
        self.avg_lls = tf.fill(
            (dat.range_e,), tf.constant(self._pll, dtype=tf.float32))
        # tf.zeros((dat.range_e - 1,), dtype=tf.float32)

        self.expanded_means_e = {}
        self.expanded_means_r = {}
        AbstractAuxModel.__init__(self, supervised=supervised)

    def unnormalized_score(self, head_or_tail, args, subset):
        return tf.fill((1, 1), tf.constant(self._pll, dtype=tf.float32))
        # return tf.zeros((1, 1), tf.float32)

    def create_sampler(self, head_or_tail, num_samples):
        # Only implemented for `num_samples == 1`
        return tf.random.uniform(
            (self._minibatch_size, num_samples),
            minval=0,
            maxval=self.padded_range_e,
            dtype=tf.int32), tf.fill((1, 1), tf.constant(self._pll, dtype=tf.float32))

    def training_samples_ll(self, labels):
        return tf.fill((1, 1), tf.constant(self._pll, dtype=tf.float32))


class FrequencyAuxModel(AbstractAuxModel):
    def __init__(self, dat, log_file=sys.stdout, supervised=False, exponent=1.0):
        log_file.write(
            '\n# Drawing negative samples according to their frequencies^%g in the training set.\n'
            % exponent)
        log_file.flush()

        self.padded_range_e = dat.range_e

        # Initialize counts with 1 so that no entity has zero probability
        # (this corresponds to a prior over the frequencies).
        counts = np.ones((dat.range_e,), dtype=np.int32)
        if supervised:
            all_labels = dat.labels['train']
        else:
            all_labels = dat.dat['train'][:, :2].flatten()

        for i in all_labels:
            counts[i] += 1
        self._log_counts = exponent * tf.constant(
            np.log(counts).reshape((1, -1)), dtype=tf.float32)

        self.expanded_means_e = {}
        self.expanded_means_r = {}
        AbstractAuxModel.__init__(self, supervised=supervised)

    def unnormalized_score(self, head_or_tail, args, subset):
        return self._log_counts

    def create_sampler(self, head_or_tail, num_samples):
        samples = tf.reshape(
            tf.multinomial(self._log_counts,
                           self._minibatch_size * num_samples),
            (self._minibatch_size, num_samples))
        log_likelihoods = tf.zeros((1, 1), dtype=tf.float32)
        return (samples, log_likelihoods)

    def training_samples_ll(self, labels):
        return tf.zeros((1, 1), dtype=tf.float32)
