import sys
import tensorflow as tf

from main_model.abstract_main_model import AbstractMainModel


class DistMultModel(AbstractMainModel):
    def __init__(self, args, dat, rng, aux_model=None, log_file=sys.stdout):
        AbstractMainModel.__init__(
            self, args, dat, rng, aux_model=aux_model, log_file=log_file)

    def define_emb(self, args, range_e, range_r):
        '''Defines variables of shape (range_e, emb_dim) and (2*range_r, emb_dim)'''
        initializer = tf.contrib.layers.xavier_initializer()
        emb_e = tf.Variable(initializer(
            (range_e, args.embedding_dim)), name='emb_e')
        emb_r = tf.Variable(initializer(
            (2 * range_r, args.embedding_dim)), name='emb_r')
        return {'emb': emb_e}, {'emb': emb_r}

    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, idx_target, args):
        x = emb_in_e['emb'] * emb_r['emb']
        # `x` has shape (minibatch_size, num_samples, embedding_dimension)

        logits = tf.matmul(tf.transpose(x, [1, 0, 2]),
                           tf.transpose(emb_all_e['emb'], [1, 0, 2]),
                           transpose_b=True)
        # `logits` has shape (num_samples, minibatch_size, range_e)
        return logits

    def single_log_prior(self, log_lambda, emb):
        # Nuclear 3-norm prior, see TODO
        # TODO: This currently uses black box updates although the expected log prior can
        #       actually be calculated analytically.
        emb_shape = emb['emb'].get_shape().as_list()
        total_dimensions = emb_shape[1] * emb_shape[2]
        log_numerator = ((-1.0 / 3.0) * tf.exp(log_lambda)
                         * tf.reduce_sum(tf.abs(emb['emb'])**3, axis=(1, 2)))
        log_denominator = (-total_dimensions / 3.0) * log_lambda
        return log_numerator - log_denominator
