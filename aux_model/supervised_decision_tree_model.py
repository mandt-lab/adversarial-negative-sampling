import sys
import numpy as np
import tensorflow as tf
import h5py
from dataset import SupervisedDataset

from aux_model.abstract_aux_model import AbstractAuxModel


class SupervisedDecisionTreeModel(AbstractAuxModel):
    def __init__(self, path, dat, log_file=sys.stdout):
        AbstractAuxModel.__init__(self, supervised=True)

        log_file.write('\n# Loading auxiliary model ... ')
        log_file.flush()

        model_path, dat_dir_path = path.split(':')

        with h5py.File(model_path, 'r') as f:
            with tf.variable_scope('aux_model_params'):
                m = f['model']
                self._num_bits = m['num_bits']
                self._num_leaves = 2**self._num_bits
                self._bits = tf.constant(
                    [[2**i for i in range(self._num_bits-1, -1, -1)]], dtype=tf.int32)
                self._bitshifts = tf.constant(
                    [[i for i in range(self._num_bits, 0, -1)]], dtype=tf.int32)
                self._category_to_leaf = tf.constant(
                    f[m['category_to_leaf']][()].reshape((-1, 1)) - 1, dtype=tf.int32)
                self._leaf_to_category = tf.constant(
                    np.minimum(f[m['leaf_to_category']][()] - 1, dat.range_e), dtype=tf.int32)
                self._weights = tf.constant(
                    f[m['weights']][()], dtype=tf.float32)
                self._biases = tf.constant(
                    f[m['biases']][()], dtype=tf.float32)
                self.avg_lls = tf.pad(tf.constant(
                    f[m['avg_lls']][()], dtype=tf.float32), [[0, 1]])

                k = self._weights.shape.dims[1]

                assert m['num_categories'] == dat.range_e
                assert dat.range_e <= 2**self._num_bits
                assert self._category_to_leaf.shape.dims == [dat.range_e, 1]
                assert self._leaf_to_category.shape.dims == [
                    2**self._num_bits, ]
                assert self._weights.shape.dims == [
                    2**self._num_bits - 1, k]
                assert self._biases.shape.dims == [
                    2**self._num_bits - 1, ]

        self._dat = SupervisedDataset(
            dat_dir_path, log_file=log_file, emb_dim=k)

        assert self._dat.range_e == dat.range_e

        self.padded_range_e = dat.range_e + 1

    def unnormalized_score(self, head_or_tail, args, subset):
        '''It's actually a normalized score (if we include the padding)'''
        valid_features = tf.gather(  # (B, d)
            self._dat.features[subset], self.minibatch_htr)

        weight = (tf.matmul(  # shape (num_leaves-1, minibatch_size)
            self._weights, valid_features, transpose_b=True, name='weight')
            + tf.reshape(self._biases, (-1, 1)))
        # shape (num_leaves-1, minibatch_size)
        bias = tf.nn.softplus(weight)
        decisions = tf.not_equal(  # shape (range_e, depth)
            tf.bitwise.bitwise_and(
                self._category_to_leaf, self._bits),
            0)
        nodes = tf.bitwise.right_shift(  # shape (range_e, depth)
            tf.bitwise.bitwise_or(
                self._category_to_leaf, self._num_leaves),
            self._bitshifts) - 1

        selected_weight = tf.transpose(  # shape (minibatch_size, range_e, depth)
            tf.gather(weight, nodes),
            (2, 0, 1))
        selected_bias = tf.transpose(  # shape (minibatch_size, range_e, depth)
            tf.gather(bias, nodes),
            (2, 0, 1))

        conditional_weight = tf.multiply(  # shape (minibatch_size, range_e, depth)
            tf.cast(tf.expand_dims(decisions, 0), tf.float32),
            selected_weight)
        return tf.reduce_sum(conditional_weight - selected_bias, axis=2)

    def predictive_ll(self):
        # FIXME: this is currently hard-coded to use the validation set
        valid_features = tf.gather(  # (B, d)
            self._dat.features['valid'], self.minibatch_htr)
        valid_labels = tf.gather(  # (B,)
            self._dat.labels['valid'], self.minibatch_htr)
        valid_leaves = tf.gather(  # (B,)
            self._category_to_leaf, valid_labels)

        decisions = tf.not_equal(  # shape (B, depth)
            tf.bitwise.bitwise_and(
                valid_leaves, self._bits),
            0)
        nodes = tf.bitwise.right_shift(  # shape (B, depth)
            tf.bitwise.bitwise_or(
                valid_leaves, self._num_leaves),
            self._bitshifts) - 1

        w = tf.gather(self._weights, nodes)  # shape (B, depth, d)
        b = tf.gather(self._biases, nodes)  # shape (B, depth)

        unsigned_scores = (tf.squeeze(  # shape (B, depth)
            tf.matmul(w, tf.expand_dims(valid_features, 1), transpose_b=True), 2)
            + b)

        neg_signed_scores = tf.multiply(  # shape (B, depth)
            1 - 2 * tf.cast(tf.expand_dims(decisions, 0), tf.float32),
            unsigned_scores)

        # shape (B,)
        return -tf.reduce_sum(tf.nn.softplus(neg_signed_scores), axis=2)

    def training_samples_ll(self, labels):
        features = tf.gather(  # (B, d)
            self._dat.features['train'], self.minibatch_htr)
        leaves = tf.gather(  # (B,)
            self._category_to_leaf, labels)

        decisions = tf.not_equal(  # shape (B, depth)
            tf.bitwise.bitwise_and(
                leaves, self._bits),
            0, name='training_samples_ll_decisions')
        nodes = tf.bitwise.right_shift(  # shape (B, depth)
            tf.bitwise.bitwise_or(
                leaves, self._num_leaves),
            self._bitshifts) - 1

        w = tf.gather(self._weights, nodes)  # shape (B, depth, d)
        b = tf.gather(self._biases, nodes)  # shape (B, depth)

        unsigned_scores = (tf.squeeze(  # shape (B, depth)
            tf.matmul(w, tf.expand_dims(features, 1), transpose_b=True), 2, name='unsigned_scores_without_bias')
            + b)

        neg_signed_scores = tf.multiply(  # shape (B, depth)
            1 - 2 * tf.cast(tf.expand_dims(decisions, 0), tf.float32),
            unsigned_scores, name='neg_signed_scores')

        # shape (B,)
        return -tf.reduce_sum(tf.nn.softplus(neg_signed_scores), axis=2)

    def create_sampler(self, head_or_tail, num_samples):
        train_features = tf.gather(  # (B, d)
            self._dat.features['train'], self.minibatch_htr)

        leaves, log_likelihoods = tf.contrib.eager.py_func(
            draw_eagerly,
            [train_features, self._weights, self._biases, num_samples,
                self._num_leaves, self._num_bits],
            [tf.int32, tf.float32])
        # `leaves` has shape (minibatch_size, num_samples).

        leaves = tf.stop_gradient(leaves)
        log_likelihoods = tf.stop_gradient(log_likelihoods)
        return tf.gather(self._leaf_to_category, leaves), log_likelihoods


def draw_eagerly(x, w, b, num_samples, num_leaves, depth):
    minibatch_size = x.get_shape().as_list()[0]
    x = tf.expand_dims(x, axis=1)  # shape (minibatch_size, 1, embedding_dim)
    node = tf.ones((minibatch_size, num_samples), tf.int32)
    lls = tf.zeros((minibatch_size, num_samples), tf.float32, name='lls')

    for _ in range(depth):
        current_w = tf.gather(w, node - 1)
        current_b = tf.gather(b, node - 1)
        # `current_w` has shape (minibatch_size, num_samples, embedding_dim).
        # `current_b` has shape (minibatch_size, num_samples).
        logits = (tf.squeeze(  # shape (minibatch_size, num_samples)
            tf.matmul(current_w, x, transpose_b=True), axis=2)
            + current_b)
        probability = tf.sigmoid(logits)
        rnd = tf.random_uniform(node.get_shape(), dtype=tf.float32)
        decisions = tf.greater(probability, rnd)
        lls += tf.log(tf.where(decisions, probability, 1 - probability))
        node = tf.bitwise.bitwise_or(
            tf.bitwise.left_shift(node, 1),
            tf.cast(decisions, tf.int32))

    return tf.bitwise.bitwise_and(node, tf.cast(num_leaves - 1, tf.int32)), lls
