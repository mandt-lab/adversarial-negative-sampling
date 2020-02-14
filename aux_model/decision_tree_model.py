import sys
import numpy as np
import tensorflow as tf
import h5py

from aux_model.abstract_aux_model import AbstractAuxModel


class DecisionTreeModel(AbstractAuxModel):
    def __init__(self, path, dat, tree_index=0, log_file=sys.stdout):
        AbstractAuxModel.__init__(self)

        log_file.write('\n# Loading auxiliary model ... ')
        log_file.flush()

        self.padded_range_e = dat.range_e + 1

        self._emb_br = {}
        self._emb_e = {}
        self._emb_r = {}
        self._ent_to_leaf = {}
        self._leaf_to_ent = {}
        self._minibatch_emb_e = {}
        self._minibatch_emb_r = {}
        self._idx_in = {'t': self.idx_h, 'h': self.idx_t}
        self._idx_predict = {'t': self.idx_t, 'h': self.idx_h}

        with tf.device('/cpu:0'):
            self._bits, self._bitshifts, self._depth, self._num_leaves = self._load_part(
                path, 't', tree_index, dat.range_e, dat.range_r)
            bits, bitshifts, depth, num_leaves = self._load_part(
                path, 'h', tree_index, dat.range_e, dat.range_r)

            assert np.all(bits == self._bits)
            assert np.all(bitshifts == self._bitshifts)
            assert depth == self._depth
            assert num_leaves == self._num_leaves
            assert 2**self._depth == self._num_leaves

            self._bits = tf.constant(bits, dtype=tf.int32, name='bits')
            self._bitshifts = tf.constant(
                bitshifts, dtype=tf.int32, name='bitshifts')

        log_file.write('done.\n')
        log_file.write('aux_k = %d\n' %
                       self._emb_e['t'].get_shape().as_list()[1])
        log_file.write('aux_num_leaves = %d\n' % self._num_leaves)
        log_file.write('aux_depth = %d\n' % self._depth)
        log_file.flush()

    def _load_part(self, path, predict, tree_index, range_e, range_r):
        with h5py.File(path % predict, 'r') as f:
            with tf.variable_scope('bookkeeping'):
                m = f['model']
                assert tree_index >= 0 and tree_index < m['population']

                embedding_dim = m['k']

                num_leaves = m['num_leaves']
                assert num_leaves >= range_e
                assert num_leaves < 2 * range_e

                bits = f[m['bits']].value
                assert len(bits.shape) == 1
                depth = bits.shape[0]
                assert 2 ** depth == num_leaves

                bitshifts = f[m['bitshifts']].value
                assert bitshifts.shape == bits.shape

                self._ent_to_leaf[predict] = tf.constant(
                    f[m['ent_to_leaf']].value[:range_e,
                                              tree_index, np.newaxis] - 1,
                    dtype=tf.int32, name='ent_to_leaf')
                assert self._ent_to_leaf[predict].shape == (range_e, 1)

                self._leaf_to_ent[predict] = tf.constant(
                    np.minimum(
                        f[m['leaf_to_ent']].value[:, tree_index] - 1,
                        range_e),
                    dtype=tf.int32, name='leaf_to_ent')
                assert self._leaf_to_ent[predict].shape == (num_leaves,)

            with tf.variable_scope('emb'):
                self._emb_e[predict] = tf.constant(
                    f[m['emb_e'][0]].value, dtype=tf.float32, name='emb_e')
                assert self._emb_e[predict].shape == (range_e, embedding_dim)

                self._emb_r[predict] = tf.constant(
                    f[m['emb_r'][0]].value, dtype=tf.float32, name='emb_r')
                assert self._emb_r[predict].shape == (range_r, embedding_dim)

                br_offset = (num_leaves - 1) * tree_index
                self._emb_br[predict] = tf.constant(
                    f[m['emb_br'][0]].value[
                        br_offset: br_offset + num_leaves - 1, :].copy(),
                    dtype=tf.float32, name='emb_br')
                assert self._emb_br[predict].shape == (
                    num_leaves - 1, embedding_dim)

        self._minibatch_emb_e[predict] = tf.gather(
            self._emb_e[predict], self._idx_in[predict])
        self._minibatch_emb_r[predict] = tf.gather(
            self._emb_r[predict], self.idx_r)

        return bits, bitshifts, depth, num_leaves

    def unnormalized_score(self, head_or_tail, args):
        '''Define a tensorflow op that calculates the prediction scores (logits).

        This is also sometimes called `logits`.

        Returns:
        A tensor of shape `(minibatch_size, range_e)` that holds the
        unnormalized represents the negative log likelihood of the data.
        Should *not* be normalized to the batch size or sample size.
        '''
        with tf.device('/cpu:0'):
            x = (self._minibatch_emb_e[head_or_tail] *
                 self._minibatch_emb_r[head_or_tail])

            weight = tf.matmul(  # shape (num_leaves-1, minibatch_size)
                self._emb_br[head_or_tail], x, transpose_b=True, name='weight')
            # shape (num_leaves-1, minibatch_size)
            bias = tf.nn.softplus(weight)
            decisions = tf.not_equal(  # shape (range_e, depth)
                tf.bitwise.bitwise_and(
                    self._ent_to_leaf[head_or_tail], self._bits),
                0)
            nodes = tf.bitwise.right_shift(  # shape (range_e, depth)
                tf.bitwise.bitwise_or(
                    self._ent_to_leaf[head_or_tail], self._num_leaves),
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

    def training_samples_ll(self, head_or_tail):
        x = (self._minibatch_emb_e[head_or_tail] *
             self._minibatch_emb_r[head_or_tail])  # shape (B, d)
        leaves = tf.gather(  # (B,)
            self._ent_to_leaf[head_or_tail], self._idx_predict[head_or_tail])

        decisions = tf.not_equal(  # shape (B, depth)
            tf.bitwise.bitwise_and(
                leaves, self._bits),
            0)
        nodes = tf.bitwise.right_shift(  # shape (B, depth)
            tf.bitwise.bitwise_or(
                leaves, self._num_leaves),
            self._bitshifts) - 1

        w = tf.gather(self._emb_br[head_or_tail], nodes)  # shape (B, depth, d)
        # b = tf.gather(self._biases, nodes)  # shape (B, depth)

        unsigned_scores = (tf.squeeze(  # shape (B, depth)
            tf.matmul(w, tf.expand_dims(x, 1), transpose_b=True), 2))
        # + b)

        neg_signed_scores = tf.multiply(  # shape (B, depth)
            1 - 2 * tf.cast(decisions, tf.float32),
            unsigned_scores)

        # shape (B,)
        return -tf.reduce_sum(tf.nn.softplus(neg_signed_scores), axis=1)

    def create_sampler(self, head_or_tail, num_samples):
        x = self._minibatch_emb_e[head_or_tail] * \
            self._minibatch_emb_r[head_or_tail]

        leaves, log_likelihoods = tf.contrib.eager.py_func(
            draw_eagerly,
            [x, self._emb_br[head_or_tail], num_samples,
                self._num_leaves, self._depth],
            [tf.int32, tf.float32])
        # `leaves` has shape (minibatch_size, num_samples).

        leaves = tf.stop_gradient(leaves)
        log_likelihoods = tf.stop_gradient(log_likelihoods)
        return tf.gather(self._leaf_to_ent[head_or_tail], leaves), log_likelihoods


def draw_eagerly(x, br, num_samples, num_leaves, depth):
    # sys.stdout.write('begin draw_eagerly\n')
    # sys.stdout.flush()

    minibatch_size = x.get_shape().as_list()[0]
    x = tf.expand_dims(x, axis=1)  # shape (minibatch_size, 1, embedding_dim)
    br_idx = tf.ones((minibatch_size, num_samples), tf.int32)
    lls = tf.zeros((minibatch_size, num_samples), tf.float32, name='lls')

    for _ in range(depth):
        # sys.stdout.write('loop draw_eagerly\n')
        # sys.stdout.flush()

        current_br = tf.gather(br, br_idx - 1)
        # `current_br` has shape (minibatch_size, num_samples, embedding_dim).
        logits = tf.squeeze(  # shape (minibatch_size, num_samples)
            tf.matmul(current_br, x, transpose_b=True), axis=2)
        probability = tf.sigmoid(logits)
        rnd = tf.random_uniform(br_idx.get_shape(), dtype=tf.float32)
        decisions = tf.greater(probability, rnd)
        lls += tf.log(tf.where(decisions, probability, 1 - probability))
        br_idx = tf.bitwise.bitwise_or(
            tf.bitwise.left_shift(br_idx, 1),
            tf.cast(decisions, tf.int32))

    # sys.stdout.write('end draw_eagerly\n')
    # sys.stdout.flush()
    return tf.bitwise.bitwise_and(br_idx, tf.cast(num_leaves - 1, tf.int32)), lls
