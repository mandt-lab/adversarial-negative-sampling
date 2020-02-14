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

        if aux_model is None or not args.initialize_to_inverse_aux:
            self.bias = tf.Variable(initializer(
                (padded_range_e,)), name='bias')
        else:
            self.bias = tf.Variable(
                -aux_model.avg_lls, name='bias')

        if aux_model is not None:
            if args.use_log_norm_weight:
                self.log_normalizer_weight = tf.Variable(initializer(
                    (dat.embedding_dim,)), name='log_normalizer_weight')
            self.log_normalizer_bias_var = tf.Variable(
                tf.zeros((), dtype=tf.float32), name='log_normalizer_bias')
            self.log_normalizer_bias = 0 * self.log_normalizer_bias_var
            self._summaries.append(tf.summary.scalar(
                'log_normalizer_bias', self.log_normalizer_bias))

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
            evaluation_features = {
                subset: tf.gather(
                    dat.features[subset], self.minibatch_htr)  # (B, d)
                for subset in ['valid', 'test']}

            self.evaluation_labels = {
                subset: tf.gather(
                    dat.labels[subset], self.minibatch_htr)  # (B,)
                for subset in ['valid', 'test']}

            emb_eval = self.emb
            bias_eval = self.bias
            if padded_range_e != dat.range_e:
                assert padded_range_e == dat.range_e + 1
                emb_eval = emb_eval[:-1, :]
                bias_eval = bias_eval[:-1]

            evaluation_scores_main = {
                subset: bias_eval + tf.matmul(evaluation_features[subset], emb_eval,
                                              transpose_b=True, name='%s_scores' % subset)  # (B, dat.range_e)
                for subset in ['valid', 'test']}

            evaluation_scores_aux = {
                subset: aux_model.unnormalized_score(None, args, subset)
                for subset in ['valid', 'test']}
            evaluation_scores = {
                subset: (evaluation_scores_main[subset] +
                         evaluation_scores_aux[subset])
                for subset in ['valid', 'test']}

            target_indices = {
                subset: self.evaluation_labels[subset] + dat.range_e * tf.range(
                    tf.shape(self.minibatch_htr)[0])
                for subset in ['valid', 'test']}

            target_scores = {
                subset: tf.expand_dims(
                    tf.gather(tf.reshape(evaluation_scores[subset], (-1,)), target_indices[subset]), 1)
                for subset in ['valid', 'test']}
            target_scores_main = {
                subset: tf.expand_dims(
                    tf.gather(tf.reshape(evaluation_scores_main[subset], (-1,)), target_indices[subset]), 1)
                for subset in ['valid', 'test']}

            # Make sure to get the corner cases right:
            # * Count scores that are worse than target score and subtract them from `dat.range_e`
            #   to ensure that NaN values are always punished.
            # * Use `dat.range_e`, which is the first dimension of `evaluation_scores`, not
            #   `padded_range_e`.
            # * Use strict comparison `<` and not `<=` to punish models that set all scores to the
            #   same value (e.g., zero).
            self.evaluation_ranks = {
                subset: dat.range_e - tf.reduce_sum(tf.cast(
                    evaluation_scores[subset] < target_scores[subset], tf.int32),
                    axis=1)
                for subset in ['valid', 'test']}
            self.evaluation_ranks_main = {
                subset: dat.range_e - tf.reduce_sum(tf.cast(
                    evaluation_scores_main[subset] < target_scores_main[subset], tf.int32),
                    axis=1)
                for subset in ['valid', 'test']}

            self.evaluation_log_likelihood = {
                subset: -tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.evaluation_labels[subset],
                    logits=evaluation_scores[subset]))
                for subset in ['valid', 'test']}

            self.evaluation_log_likelihood_main = {
                subset: -tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.evaluation_labels[subset],
                    logits=evaluation_scores_main[subset]))
                for subset in ['valid', 'test']}

        with tf.variable_scope('log_likelihood'):
            emb_pos = tf.gather(  # (B, d)
                self.emb, labels_pos, name='emb_pos')
            bias_pos = tf.gather(  # (B,)
                self.bias, labels_pos, name='bias_pos')

            scores_pos = (  # (B, 1, 1)
                tf.matmul(
                    tf.expand_dims(features_minibatch, 1),
                    tf.expand_dims(emb_pos, 1),
                    transpose_b=True, name='scores_pos')
                + tf.expand_dims(tf.expand_dims(bias_pos, 1), 2))

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
                lls_pos = tf.reshape(
                    aux_model.training_samples_ll(labels_pos), (-1, 1, 1))

                scores_neg = (  # (B, 1, n)
                    tf.matmul(
                        tf.expand_dims(features_minibatch, 1),
                        emb_neg,
                        transpose_b=True, name='scores_neg')
                    + tf.expand_dims(bias_neg, 1))

                # self._summaries += [
                #     tf.summary.histogram('eta_pos', scores_pos),
                #     tf.summary.histogram('exp_eta_neg', tf.exp(scores_neg))]
                # neg_log_likelihood = (
                #     (1.0 / args.neg_samples) * tf.reduce_sum(
                #         tf.exp(scores_neg))
                #     - tf.reduce_sum(scores_pos))
                self._summaries += [
                    tf.summary.histogram(
                        'acceptance_pos_t', tf.nn.sigmoid(scores_pos)),
                    tf.summary.histogram('acceptance_neg_t', tf.nn.sigmoid(scores_neg))]
                if args.model == 'supervised':
                    neg_log_likelihood = (
                        (1.0 / args.neg_samples) * tf.reduce_sum(
                            tf.nn.softplus(scores_neg))
                        + tf.reduce_sum(tf.nn.softplus(-scores_pos)))
                elif args.model == 'supervised_nce':
                    neg_log_likelihood = (
                        (1.0 / args.neg_samples) * tf.reduce_sum(
                            tf.nn.softplus(scores_neg - lls_neg))
                        + tf.reduce_sum(tf.nn.softplus(lls_pos - scores_pos)))

        with tf.variable_scope('regularizer'):
            if args.reg_separate:
                to_regularize_pos = tf.expand_dims(
                    tf.expand_dims(bias_pos, 1), 2)
                to_regularize_neg = tf.expand_dims(bias_neg, 1)
                regularizer = (
                    args.initial_reg_strength * tf.reduce_sum(emb_pos**2)
                    + (args.initial_reg_strength / args.neg_samples) * tf.reduce_sum(emb_neg**2))
            else:
                to_regularize_pos = scores_pos
                to_regularize_neg = scores_neg
                regularizer = 0

            if aux_model is None:
                regularizer += (
                    args.initial_reg_strength * tf.reduce_sum(to_regularize_pos**2))
            else:
                if args.model == 'supervised':
                    log_normalizer = self.log_normalizer_bias
                    if args.use_log_norm_weight:
                        log_normalizer += tf.reshape(
                            tf.matmul(
                                features_minibatch,
                                tf.expand_dims(self.log_normalizer_weight, 0),
                                transpose_b=True, name='log_normalizer'),
                            (-1, 1, 1))
                    regularizer += (
                        args.initial_reg_strength * tf.reduce_sum(
                            (to_regularize_pos + lls_pos - log_normalizer)**2)
                        + (args.initial_reg_strength / args.neg_samples) * tf.reduce_sum(
                            (to_regularize_neg + tf.expand_dims(lls_neg, 1) - log_normalizer)**2))
                elif args.model == 'supervised_nce':
                    regularizer += (
                        args.initial_reg_strength * tf.reduce_sum(
                            to_regularize_pos**2)
                        + (args.initial_reg_strength / args.neg_samples) * tf.reduce_sum(
                            to_regularize_neg**2))

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
            var_list = [self.emb, self.bias]
            if aux_model is not None:
                var_list.append(self.log_normalizer_bias_var)
                if args.use_log_norm_weight:
                    var_list.append(self.log_normalizer_weight)
            self._e_step = opt.minimize(
                self.loss, var_list=var_list, global_step=global_step)

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
