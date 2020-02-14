import sys
import tensorflow as tf

from abstract_model import AbstractModel
import optimizer


class SupervisedModel(AbstractModel):
    '''Simple model for supervised classification.'''

    def __init__(self, args, dat, rng, aux_model=None, log_file=sys.stdout):
        log_file.write('\n# Creating model.\n')
        log_file.flush()

        self.aux_model = aux_model
        if aux_model is not None:
            # Add padding entity for unassigned leaves of auxiliary model.
            padded_range_e = aux_model.padded_range_e
        else:
            padded_range_e = dat.range_e

        self._summaries = []
        initializer = tf.contrib.layers.xavier_initializer()

        self.emb = tf.Variable(initializer(
            (padded_range_e, dat.embedding_dim)), name='emb')

        if aux_model is None:
            self.bias = tf.Variable(initializer(
                (padded_range_e,)), name='bias')
        else:
            self.bias = tf.Variable(
                tf.pad(-aux_model.avg_lls, [[0, 1]]), name='bias')

        self.log_normalizer = tf.Variable(
            tf.zeros((), dtype=tf.float32), name='log_normalizer')
        self._summaries.append(tf.summary.scalar(
            'log_normalizer', self.log_normalizer))

        with tf.variable_scope('minibatch'):
            if aux_model is None:
                self.minibatch_htr = tf.placeholder(
                    tf.int32, shape=(None,), name='minibatch')
            else:
                self.minibatch_htr = aux_model.minibatch_htr

            minibatch_size = tf.shape(self.minibatch_htr)[0]
            minibatch_size_float = tf.cast(minibatch_size, tf.float32)

            self.feed_train_features = tf.placeholder(
                tf.float32, shape=dat.features['train'].shape)
            all_train_features = tf.Variable(
                self.feed_train_features, dtype=tf.float32, trainable=False)

            features_minibatch = tf.gather(  # (B, d)
                all_train_features, self.minibatch_htr)
            labels_pos = tf.gather(  # (B,)
                tf.constant(dat.labels['train'], dtype=tf.int32), self.minibatch_htr)

        with tf.variable_scope('evaluation'):
            valid_features = tf.gather(  # (B, d)
                dat.features['valid'], self.minibatch_htr)
            valid_labels = tf.gather(  # (B,)
                dat.labels['valid'], self.minibatch_htr)

            emb_eval = self.emb
            bias_eval = self.bias
            if padded_range_e != dat.range_e:
                emb_eval = emb_eval[:-1, :]
                bias_eval = bias_eval[:-1]
            valid_scores_main = (  # shape (batch, categories)
                tf.matmul(valid_features, emb_eval,
                          transpose_b=True, name='valid_scores')
                + bias_eval)
            valid_scores_aux = aux_model.unnormalized_score(None, args)
            valid_scores = valid_scores_main + valid_scores_aux
            self.valid_likelihood = -tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=valid_labels, logits=valid_scores))

        with tf.variable_scope('log_likelihood'):
            emb_pos = tf.gather(  # (B, d)
                self.emb, labels_pos, name='emb_pos')
            bias_pos = tf.gather(  # (B,)
                self.bias, labels_pos, name='bias_pos')

            if aux_model is None:
                self.scores = (  # shape (batch, categories)
                    tf.matmul(
                        features_minibatch, self.emb, transpose_b=True, name='scores')
                    + tf.expand_dims(self.bias, 0))
                # The documentation for `tf.nn.sparse_softmax_cross_entropy_with_logits` is unclear
                # about signs and normalization. It turns out that the function does the following,
                # assuming that `labels.shape = (m,)` and `logits.shape = (m, n)`:
                #   tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                #   = - [[logits[i, labels[i]] for i in range(m)] for j in range(n)]
                #     + log(sum(exp(logits), axis=1))
                neg_log_likelihood = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_pos, logits=self.scores))
            else:
                labels_neg, lls_neg = aux_model.create_sampler(
                    None, args.neg_samples)
                # `labels_neg` has shape (minibatch_size, neg_samples)
                emb_neg = tf.gather(  # (B, n, d)
                    self.emb, labels_neg, name='emb_neg')
                bias_neg = tf.gather(  # (B, n)
                    self.bias, labels_neg, name='bias_neg')

                scores_pos = (  # (B, 1, 1)
                    tf.matmul(
                        tf.expand_dims(features_minibatch, 1),
                        tf.expand_dims(emb_pos, 1),
                        transpose_b=True, name='scores_pos')
                    + tf.expand_dims(tf.expand_dims(bias_pos + self.log_normalizer, 1), 2))
                scores_neg = (  # (B, 1, n)
                    tf.matmul(
                        tf.expand_dims(features_minibatch, 1),
                        emb_neg,
                        transpose_b=True, name='scores_neg')
                    + tf.expand_dims(bias_neg + self.log_normalizer, 1))

                self._summaries += [
                    tf.summary.histogram(
                        'acceptance_pos_t', tf.nn.sigmoid(scores_pos)),
                    tf.summary.histogram('acceptance_neg_t', tf.nn.sigmoid(scores_neg))]
                neg_log_likelihood = (
                    (1.0 / args.neg_samples) * tf.reduce_sum(
                        tf.nn.softplus(scores_neg))
                    + tf.reduce_sum(tf.nn.softplus(-scores_pos)))

        with tf.variable_scope('regularizer'):
            regularizer = args.initial_reg_strength * tf.reduce_sum(emb_pos**2)
            if aux_model is not None:
                lls_pos = aux_model.training_samples_ll(labels_pos)
                # TODO: maybe we want to use `aux_model.avg_lls` instead of `lls_neg`
                reg_bias_neg = tf.reduce_sum((bias_neg + lls_neg)**2)
                # or, regularize complete score_neg/pos towards -lls_neg/pos
                reg_bias_pos = tf.reduce_sum((bias_pos + lls_pos)**2)
                regularizer += (
                    (args.initial_reg_strength / args.neg_samples) *
                    tf.reduce_sum(emb_neg**2)
                    + (args.bias_reg / args.neg_samples) * reg_bias_neg
                    + args.bias_reg * reg_bias_pos)

        self.loss = tf.add_n([neg_log_likelihood, regularizer], name='loss')

        with tf.variable_scope('loss_parts'):
            normalizer_per_embedding = (
                len(dat.labels['train']) /
                (dat.embedding_dim * padded_range_e) * minibatch_size_float)
            normalizer_per_datapoint = 1.0 / minibatch_size_float

            self._summaries.append(tf.summary.scalar(
                'regularizer_per_embedding_and_dimension', normalizer_per_embedding * regularizer))
            self._summaries.append(tf.summary.scalar(
                'neg_log_likelihood_per_datapoint', normalizer_per_datapoint * neg_log_likelihood))
            self._summaries.append(tf.summary.scalar(
                'loss_per_datapoint', normalizer_per_datapoint * self.loss))

        global_step, lr, lr_summary = optimizer.define_learning_rate(args)
        self._summaries.append(lr_summary)
        opt = optimizer.define_optimizer(args, lr)

        with tf.variable_scope('opt'):
            self._e_step = opt.minimize(
                self.loss, var_list=[self.emb, self.bias, self.log_normalizer], global_step=global_step)

        self._summary_op = tf.summary.merge(self._summaries)

    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, args):
        assert emb_r is None
        assert False
        # x = self._pointwise_complex_product(emb_in_e, emb_r)
        # # `x` has shape (minibatch_size, num_samples, embedding_dimension)
        # logits = self._re_complex_batch1_matmul_adjoint(x, emb_all_e)
        # # `logits` has shape (num_samples, minibatch_size, range_e)
        # return logits

    @property
    def e_step(self):
        '''Tensorflow op for a gradient step with fixed hyperparameters.'''
        return self._e_step

    @property
    def summary_op(self):
        '''A tensorflow op that evaluates some summary statistics for Tensorboard.

        Run this op in a session and use a `tf.summary.FileWriter` to write the result
        to a file. Visualize the summaries with Tensorboard.'''
        return self._summary_op
