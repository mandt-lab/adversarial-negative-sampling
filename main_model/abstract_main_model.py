import sys
import abc

import tensorflow as tf
import numpy as np

import optimizer
from abstract_model import AbstractModel


class AbstractMainModel(AbstractModel):
    '''Abstract base class for knowledge graph embedding models.

    To define a new model, derive from this class and implement the method
    `define_score()`.
    '''

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
        with tf.device('/cpu:0'):
            with tf.variable_scope('means'):
                self.means_e, self.means_r = self.define_emb(
                    args, padded_range_e, dat.range_r)
            with tf.variable_scope('samples_e'):
                self.samples_e, self.expanded_means_e, self.log_std_e = self.create_all_samplers(
                    self.means_e, args)
            with tf.variable_scope('samples_r'):
                self.samples_r, self.expanded_means_r, self.log_std_r = self.create_all_samplers(
                    self.means_r, args)

        with tf.variable_scope('minibatch'):
            if aux_model is None:
                self.minibatch_htr = tf.placeholder(
                    tf.int32, shape=(None, 3), name='minibatch_htr')
                self.idx_h = self.minibatch_htr[:, 0]
                self.idx_t = self.minibatch_htr[:, 1]
                idx_r_predict_t = self.minibatch_htr[:, 2]
            else:
                self.minibatch_htr = aux_model.minibatch_htr
                self.idx_h = aux_model.idx_h
                self.idx_t = aux_model.idx_t
                idx_r_predict_t = aux_model.idx_r

            idx_r_predict_h = idx_r_predict_t + dat.range_r

            minibatch_size = tf.shape(self.minibatch_htr)[0]
            minibatch_size_float = tf.cast(minibatch_size, tf.float32)

            emb_h = {label: tf.gather(samples, self.idx_h, name='gather_mb_h')
                     for label, samples in self.samples_e.items()}
            emb_t = {label: tf.gather(samples, self.idx_t, name='gather_mb_t')
                     for label, samples in self.samples_e.items()}
            emb_r_predict_t = {label: tf.gather(samples, idx_r_predict_t, name='gather_mb_r_predict_t')
                               for label, samples in self.samples_r.items()}
            emb_r_predict_h = {label: tf.gather(samples, idx_r_predict_h, name='gather_mb_r_predict_h')
                               for label, samples in self.samples_r.items()}

            self.minibatch_mean_h = {
                label: tf.gather(means, self.idx_h)
                for label, means in self.expanded_means_e.items()}
            self.minibatch_mean_t = {
                label: tf.gather(means, self.idx_t)
                for label, means in self.expanded_means_e.items()}
            self.minibatch_mean_r_predict_t = {
                label: tf.gather(means, idx_r_predict_t)
                for label, means in self.expanded_means_r.items()}
            self.minibatch_mean_r_predict_h = {
                label: tf.gather(means, idx_r_predict_h)
                for label, means in self.expanded_means_r.items()}

            # Prefactor for normalization per training data point.
            # normalizer = 1.0 / tf.cast(args.num_samples, tf.float32)

        with tf.variable_scope('log_likelihood'):
            # TODO: factor out duplication of code for head / tail prediction
            if aux_model is None:
                with tf.variable_scope('tail_prediction'):
                    self.scores_predict_t = self.unnormalized_score(
                        emb_h, emb_r_predict_t, self.samples_e, args)
                    ll_predict_t = self._log_likelihood(
                        self.scores_predict_t, self.idx_t, args)
                with tf.variable_scope('head_prediction'):
                    self.scores_predict_h = self.unnormalized_score(
                        emb_t, emb_r_predict_h, self.samples_e, args)
                    ll_predict_h = self._log_likelihood(
                        self.scores_predict_h, self.idx_h, args)
            else:
                with tf.variable_scope('tail_prediction'):
                    idx_neg, lls_neg_t = aux_model.create_sampler(
                        't', args.neg_samples)
                    # `idx_neg` has shape (minibatch_size, neg_samples)
                    emb_neg = {label: tf.squeeze(tf.gather(samples, idx_neg), axis=2)
                               for label, samples in self.samples_e.items()}
                    scores_pos_t, scores_neg_t = self.batch_unnormalized_scores(
                        emb_h, emb_r_predict_t, (emb_t, emb_neg), args)
                    # `scores_pos_t` has shape (minibatch_size, 1)
                    # `scores_neg_t` has shape (minibatch_size, neg_samples)
                    self._summaries += [
                        tf.summary.histogram(
                            'acceptance_pos_t', tf.nn.sigmoid(scores_pos_t)),
                        tf.summary.histogram('acceptance_neg_t', tf.nn.sigmoid(scores_neg_t))]
                    ll_predict_t = (
                        (-1.0 / args.neg_samples) * tf.reduce_sum(
                            tf.nn.softplus(scores_neg_t))
                        - tf.reduce_sum(tf.nn.softplus(-scores_pos_t)))
                with tf.variable_scope('head_prediction'):
                    idx_neg, lls_neg_h = aux_model.create_sampler(
                        'h', args.neg_samples)
                    # `idx_neg` has shape (minibatch_size, neg_samples)
                    emb_neg = {label: tf.squeeze(tf.gather(samples, idx_neg), axis=2)
                               for label, samples in self.samples_e.items()}
                    scores_pos_h, scores_neg_h = self.batch_unnormalized_scores(
                        emb_t, emb_r_predict_h, (emb_h, emb_neg), args)
                    # `scores_pos_h` has shape (minibatch_size, 1)
                    # `scores_neg_h` has shape (minibatch_size, neg_samples)
                    self._summaries += [
                        tf.summary.histogram(
                            'acceptance_pos_h', tf.nn.sigmoid(scores_pos_h)),
                        tf.summary.histogram('acceptance_neg_h', tf.nn.sigmoid(scores_neg_h))]
                    ll_predict_h = (
                        (-1.0 / args.neg_samples) * tf.reduce_sum(
                            tf.nn.softplus(scores_neg_h))
                        - tf.reduce_sum(tf.nn.softplus(-scores_pos_h)))

            # log_likelihood = normalizer * (ll_predict_t + ll_predict_h)
            log_likelihood = ll_predict_t + ll_predict_h

        # with tf.variable_scope('hyperparameters'):
        #     frequencies_e, counts_e, sort_indices_e = self._get_frequencies(
        #         dat.dat['train'][:, :2].flatten(), padded_range_e, 'e')
        #     frequencies_r, counts_r, sort_indices_r = self._get_frequencies(
        #         dat.dat['train'][:, 2], dat.range_r, 'r')
        #     self.log_lambda_e = self._define_log_lambda(args, counts_e, 'e')
        #     self.log_lambda_r = self._define_log_lambda(args, counts_r, 'r')

        # inverse_counts_e = (1.0 / counts_e).astype(np.float32)
        # inverse_counts_r = (1.0 / counts_r).astype(np.float32)
        # self._lambda_sigma_summary(
        #     self.log_lambda_e, self.log_std_e, inverse_counts_e, sort_indices_e, 'e')
        # self._lambda_sigma_summary(
        #     self.log_lambda_r, self.log_std_r, inverse_counts_r, sort_indices_r, 'r')

        # with tf.variable_scope('log_prior'):
        #     # r-counts are the same for head and tail prediction, so gather them only once.
        #     minibatch_inverse_counts_r = tf.gather(
        #         inverse_counts_r, idx_r_predict_t)
        #     log_prior = normalizer * (
        #         tf.reduce_sum(
        #             tf.gather(inverse_counts_e, self.idx_h) * self.single_log_prior(
        #                 tf.gather(self.log_lambda_e, self.idx_h), emb_h))
        #         + tf.reduce_sum(
        #             tf.gather(inverse_counts_e, self.idx_t) * self.single_log_prior(
        #                 tf.gather(self.log_lambda_e, self.idx_t), emb_t))
        #         + tf.reduce_sum(
        #             minibatch_inverse_counts_r * self.single_log_prior(
        #                 tf.gather(self.log_lambda_r, idx_r_predict_t), emb_r_predict_t))
        #         + tf.reduce_sum(
        #             minibatch_inverse_counts_r * self.single_log_prior(
        #                 tf.gather(self.log_lambda_r, idx_r_predict_h), emb_r_predict_h)))

        with tf.variable_scope('regularizer'):
            if args.reg_separate:
                raise "unimplemented"
                # to_regularize_pos = tf.expand_dims(
                #     tf.expand_dims(bias_pos, 1), 2)
                # to_regularize_neg = tf.expand_dims(bias_neg, 1)
                # regularizer = (
                #     args.initial_reg_strength * tf.reduce_sum(emb_pos**2)
                #     + (args.initial_reg_strength / args.neg_samples) * tf.reduce_sum(emb_neg**2))
            else:
                to_regularize_pos_t = scores_pos_t
                to_regularize_neg_t = scores_neg_t
                to_regularize_pos_h = scores_pos_h
                to_regularize_neg_h = scores_neg_h
                regularizer = 0

            if aux_model is None:
                regularizer += args.initial_reg_strength * (
                    tf.reduce_sum(to_regularize_pos_t**2) + tf.reduce_sum(to_regularize_pos_h**2))
            else:
                lls_pos_t = tf.reshape(
                    aux_model.training_samples_ll('t'), (-1, 1, 1))
                lls_pos_h = tf.reshape(
                    aux_model.training_samples_ll('h'), (-1, 1, 1))
                log_normalizer_t = self.log_normalizer_bias
                log_normalizer_h = self.log_normalizer_bias
                if args.use_log_norm_weight:
                    raise "unimplemented"
                    # log_normalizer += tf.reshape(
                    #     tf.matmul(
                    #         features_minibatch,
                    #         tf.expand_dims(self.log_normalizer_weight, 0),
                    #         transpose_b=True, name='log_normalizer'),
                    #     (-1, 1, 1))
                regularizer += (
                    args.initial_reg_strength * (
                        tf.reduce_sum(
                            (to_regularize_pos_t + lls_pos_t - log_normalizer_t)**2)
                        + tf.reduce_sum(
                            (to_regularize_pos_h + lls_pos_h - log_normalizer_h)**2))
                    + (args.initial_reg_strength / args.neg_samples) * (
                        tf.reduce_sum(
                            (to_regularize_neg_t + tf.expand_dims(lls_neg_t, 1) - log_normalizer_t)**2)
                        + tf.reduce_sum(
                            (to_regularize_neg_h + tf.expand_dims(lls_neg_h, 1) - log_normalizer_h)**2)))

        if args.em:
            raise "unimplemented"
            # # Calculate entropy of entire variational distribution (independent of minibatch).
            # # Normalize per training data point.
            # with tf.variable_scope('entropy'):
            #     entropy = (minibatch_size_float / len(dat.dat['train'])) * tf.add_n(
            #         [tf.reduce_sum(i) for i in
            #          list(self.log_std_e.values()) + list(self.log_std_r.values())],
            #         name='entropy')
            # self.loss = -tf.add_n([log_prior, log_likelihood, entropy],
            #                       name='elbo')
        else:
            self.loss = tf.add_n([regularizer, -log_likelihood],
                                 name='log_joint')

        # with tf.variable_scope('loss_parts'):
        #     normalizer_per_embedding = (
        #         len(dat.dat['train']) /
        #         (args.embedding_dim * (padded_range_e + 2 * dat.range_r) * minibatch_size_float))
        #     normalizer_per_datapoint = 0.5 / minibatch_size_float
        #     if args.em:
        #         self._summaries.append(tf.summary.scalar('entropy_per_embedding_and_dimension',
        #                                                  normalizer_per_embedding * entropy))
        #     self._summaries.append(tf.summary.scalar('log_prior_per_embedding_and_dimension',
        #                                              normalizer_per_embedding * log_prior))
        #     self._summaries.append(tf.summary.scalar('log_likelihood_per_datapoint',
        #                                              normalizer_per_datapoint * log_likelihood))
        #     self._summaries.append(tf.summary.scalar('loss_per_datapoint',
        #                                              normalizer_per_datapoint * self.loss))

        global_step, lr, lr_summary = optimizer.define_learning_rate(args)
        self._summaries.append(lr_summary)
        opt = optimizer.define_optimizer(args, lr)

        with tf.variable_scope('e_step'):
            var_list = (tf.trainable_variables('means/') +
                        tf.trainable_variables('samples_e/') +
                        tf.trainable_variables('samples_r/'))
            if aux_model is not None:
                var_list.append(self.log_normalizer_bias_var)
                if args.use_log_norm_weight:
                    var_list.append(self.log_normalizer_weight)
            log_file.write('# %d variational parameters\n' %
                           len(variational_parameters))
            gvs_e = opt.compute_gradients(
                self.loss, var_list=var_list)
            self._e_step = opt.apply_gradients(gvs_e, global_step)

        if args.em:
            with tf.variable_scope('m_step'):
                hyperparameters = tf.trainable_variables('hyperparameters/')
                log_file.write('# %d hyperparameters\n' % len(hyperparameters))
                gvs_m = opt.compute_gradients(
                    self.loss, var_list=hyperparameters)
                m_step = opt.apply_gradients(gvs_m)
            self._em_step = tf.group(self._e_step, m_step, name='em_step')
        else:
            self._em_step = None

        self._summary_op = tf.summary.merge(self._summaries)

    def create_all_samplers(self, means, args):
        expanded_means = {}
        samples = {}
        log_std = {} if args.em else None

        for label, mean in means.items():
            expanded_mean = tf.expand_dims(mean, axis=1)
            expanded_means[label] = expanded_mean
            if args.em:
                scaled_log_std = tf.Variable(
                    tf.fill(expanded_mean.get_shape(),
                            (np.log(args.initial_std) / args.std_speedup).astype(np.float32)),
                    dtype=tf.float32, name='%s_scaled_log_std' % label)
                if args.std_speedup == 1.0:
                    log_std[label] = scaled_log_std
                else:
                    # Scaling a variable by a scalar `alpha` is a simple way to scale individual
                    # learning rates by `alpha` in Adagrad and Adam optimizers. Note that, for SGD
                    # without adaptive learning rates, this scales the learning rate by `alpha**2`.
                    log_std[label] = args.std_speedup * scaled_log_std

                std = tf.exp(log_std[label], name='%s_std' % label)
                self._summaries.append(
                    tf.summary.histogram('%s_std' % label, std))

                shape = expanded_mean.get_shape().as_list()
                shape[1] = args.num_samples
                samples[label] = tf.random_normal(
                    shape, mean=expanded_mean, stddev=std, name='%s_samples' % label)
            else:
                samples[label] = expanded_mean
        return samples, expanded_means, log_std

    def _get_frequencies(self, entries, range_max, label):
        counts = np.zeros(range_max, np.int32)
        for i in entries:
            counts[i] += 1
        counts = np.maximum(1, counts)  # Avoid division by zero.
        sort_indices = np.argsort(counts)
        counts = counts.astype(np.float32)
        if label == 'e':
            frequencies = ((1.0 / len(entries)) * counts).astype(np.float32)
        elif label == 'r':
            counts = np.concatenate((counts, counts))
            frequencies = ((1.0 / (2 * len(entries)))
                           * counts).astype(np.float32)
            sort_indices = np.array([(i, i + range_max)
                                     for i in sort_indices]).flatten()
        else:
            raise "label must be 'e' or 'r'"
        return frequencies, counts, sort_indices

    def _define_log_lambda(self, args, counts, label):
        if args.initial_reg_uniform:
            # Since frequencies add up to 1, the average frequency is `1 / len(frequencies)`.
            initial_lambda = args.initial_reg_strength * counts.sum() / len(counts)
            initializer = tf.fill(
                (len(counts),),
                (args.reg_strength_slowdown * np.log(initial_lambda)).astype(np.float32))
        else:
            initializer = (args.reg_strength_slowdown *
                           np.log(args.initial_reg_strength * counts.astype(np.float32)))

        log_scaled_lambda = tf.Variable(initializer, dtype=tf.float32,
                                        name='log_scaled_lambda_%s' % label, trainable=args.em)
        if args.reg_strength_slowdown == 1.0:
            return log_scaled_lambda
        else:
            # Scaling a variable by a scalar `alpha` is a simple way to scale individual learning
            # rates by `alpha` in Adagrad and Adam optimizers. Note that, for SGD without adaptive
            # learning rates, this scales the learning rate by `alpha**2`.
            return (1.0 / args.reg_strength_slowdown) * log_scaled_lambda

    def _lambda_sigma_summary(self, log_lambda, log_std, inverse_counts, sort_indices, label):
        lmbda = tf.exp(log_lambda)
        sorted_lambda = tf.gather(lmbda, sort_indices)
        downscaled_sorted_lambda = tf.gather(
            inverse_counts * lmbda, sort_indices)

        self._summaries.append(tf.summary.histogram('lambda_%s' % label, sorted_lambda,
                                                    family='hyperparmeters'))
        self._summaries.append(tf.summary.histogram('downscaled_lambda_%s' % label,
                                                    downscaled_sorted_lambda,
                                                    family='hyperparmeters'))

        one_3rd = len(sort_indices) // 3
        with tf.variable_scope('avg_lambda_%s' % label):
            self._summaries.append(tf.summary.scalar(
                'a_low_frequency', tf.reduce_mean(sorted_lambda[:one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'b_med_frequency', tf.reduce_mean(sorted_lambda[one_3rd: -one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'c_high_frequency', tf.reduce_mean(sorted_lambda[-one_3rd:])))

        with tf.variable_scope('avg_downscaled_lambda_%s' % label):
            self._summaries.append(tf.summary.scalar(
                'a_low_frequency', tf.reduce_mean(downscaled_sorted_lambda[:one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'b_med_frequency', tf.reduce_mean(downscaled_sorted_lambda[one_3rd: -one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'c_high_frequency', tf.reduce_mean(downscaled_sorted_lambda[-one_3rd:])))

        if log_std is not None:
            sorted_log_std = tf.add_n([tf.gather(l, sort_indices)
                                       for l in log_std.values()])
            with tf.variable_scope('avg_log_std_%s' % label):
                normalizer = 1.0 / len(log_std)
                self._summaries.append(tf.summary.scalar(
                    'a_low_frequency', normalizer * tf.reduce_mean(sorted_log_std[:one_3rd])))
                self._summaries.append(tf.summary.scalar(
                    'b_med_frequency', normalizer * tf.reduce_mean(sorted_log_std[one_3rd: -one_3rd])))
                self._summaries.append(tf.summary.scalar(
                    'c_high_frequency', normalizer * tf.reduce_mean(sorted_log_std[-one_3rd:])))

    @abc.abstractmethod
    def define_emb(self, args, range_e, range_r):
        '''Define tensorflow Variables for the latent embedding vectors.

        Arguments:
        args -- Namespace containing command line arguments.
        range_e -- Number of distinct entities (possibly including padding).
        range_r -- Number of distinct relations (before adding reciprocal relations).

        Returns:
        A pair of dicts. The first dict maps labels (strings) to `tf.Variable`s for the
        entity embeddings.
        '''
        pass

    def _log_likelihood(self, scores, idx_target, args):
        labels = tf.expand_dims(idx_target, 0)
        if args.em and args.num_samples != 1:
            # Broadcast explicitly since Tensorflow's cross entropy function does not do it.
            labels = tf.tile(labels, [args.num_samples, 1])
            # `labels` has shape (num_samples, minibatch_size)

        # The documentation for `tf.nn.sparse_softmax_cross_entropy_with_logits` is a bit unclear
        # about signs and normalization. It turns out that the function does the following,
        # assuming that `labels.shape = (m, n)` and `logits.shape = (m, n, o)`:
        #   tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        #   = - [[logits[i, j, labels[i, j]] for j in range(n)] for i in range(m)]
        #     + log(sum(exp(logits), axis=2))
        negative_scores = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=scores)
        # `negative_scores` has shape (num_samples, minibatch_size)
        return -tf.reduce_sum(negative_scores)

    @abc.abstractmethod
    def single_log_prior(self, log_lambda, emb):
        '''
        Calculate the log prior of a single embedding vector.

        Arguments:
        log_lambda -- The log of the regularizer strengths. Tensor of shape
            `(minibatch_size,)`.
        emb -- A single embedding vector. A dict with the same keys as the ones
            returned by `define_emb`. Each tensor has shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.

        Returns:
        A tensor of size `(minibatch_size)` containing the weighted log priors for all samples.
        Should *not* be normalized by the number of samples.
        '''
        pass

    @property
    def e_step(self):
        '''Tensorflow op for a gradient step with fixed hyperparameters.'''
        return self._e_step

    @property
    def em_step(self):
        '''Tensorflow op for a gradient step in model and hyperaparametre space.'''
        return self._em_step

    @property
    def summary_op(self):
        '''A tensorflow op that evaluates some summary statistics for Tensorboard.

        Run this op in a session and use a `tf.summary.FileWriter` to write the result
        to a file. Visualize the summaries with Tensorboard.'''
        return self._summary_op


def add_cli_args(parser):
    '''Add command line arguments for all knowledge graph embedding models.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was just added to the parser.
    '''
    generic_args = parser.add_argument_group(
        'Generic model parameters')
    generic_args.add_argument('--model', required=True,
                              choices=['ComplEx', 'DistMult', 'supervised', 'supervised_nce'], help='''
        Choose the knowledge graph embedding model.''')
    generic_args.add_argument('-k', '--embedding_dim', metavar='N', type=int, default=100, help='''
        Set the embedding dimension.''')
    generic_args.add_argument('--em', action='store_true', help='''
        Turn on variational expectation maximization, i.e., optimize over hyperparmeters.''')
    generic_args.add_argument('-S', '--num_samples', metavar='FLOAT', type=int, default=1, help='''
        Set the number of samples from the variational distribution that is used to estimate
        the ELBO; only used if `--em` is used.''')
    generic_args.add_argument('--initial_std', metavar='FLOAT', type=float, default=1.0, help='''
        Initial standard deviation of the variational distribution; only used if `--em` is
        used.''')
    generic_args.add_argument('--initial_reg_strength', metavar='FLOAT', type=float,
                              default=1.0, help='''
        Set the (initial) regularizer strengths $\\lambda$. To make this flag more portable across
        data sets, the provided value will still be scaled with a frequency: if
        `--initial_reg_uniform` is set, then the scaling factor is the average frequency of
        entities or relations. If `--initial_reg_uniform` is not set, then the scaling factor is
        the individual frequency of each entity or relation. Not that, if `--em` is used, then
        `--initial_reg_strength` only affects the initialization of the regularizer strengths as
        the EM algorithm will optimize over the regularizer strengths. If `--em` is not used, then
        the regularizer strengths controlled by this flag are held constant throughout
        training.''')
    generic_args.add_argument('--use_log_norm_weight', action='store_true', help='''
        Use feature dependent log normalizer.''')
    generic_args.add_argument('--initialize_to_inverse_aux', action='store_true', help='''
        Initialize biases of main model to compensate for aux model as well as possible.''')
    generic_args.add_argument('--reg_separate', action='store_true', help='''
        Regularize weights and biases separately rather than regularizing the logits.''')
    generic_args.add_argument('--initial_reg_uniform', action='store_true', help='''
        Set the (initial) regularizer strengths $\\lambda$ to a uniform value: use the value
        provided with `--initial_reg_strength`, scaled only by the average frequency of entities
        or relations. Default is to scale the (initial) regularizer strengths by the frequency of
        each individual entity or relation. See also `--initial_reg_strength`.''')
    generic_args.add_argument('--reg_strength_slowdown', metavar='FLOAT', type=float,
                              default=1.0, help='''
        Divide the learning rate for the regularizer strengths $\\lambda$ by the provided factor
        compared to the learning rates of the variational means. Only used if `--em` is set.
        Typical values are >= 1, reflecting that, although we optimize variational parametres and
        hyperparameters concurrently, the intuition is that we fit a variational distribution in
        order to estimate the marginal likelihood, and we maximize the marginal likelihood over the
        hyperparameters $\\lambda$.''')
    generic_args.add_argument('--std_speedup', metavar='FLOAT', type=float,
                              default=1.0, help='''
        Multiply the learning rate for the variational standard deviations by the provided factor
        compared to the learning rates of the variational means. Only used if `--em` is set.''')
