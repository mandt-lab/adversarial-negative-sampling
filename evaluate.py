import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf


class Evaluator:
    def __init__(self, model, dat, args, log_file=sys.stdout, hits_at=[1, 3, 10]):
        log_file.write('\n# Creating evaluation harness.\n')
        log_file.flush()

        if args.eval_dat == 'both':
            self._eval_sets = ['valid', 'test']
        elif args.eval_dat == 'all':
            self._eval_sets = ['valid', 'test', 'train']
        elif args.eval_dat == 'none':
            self._eval_sets = []
        else:
            self._eval_sets = [args.eval_dat]

        if args.eval_mode == 'both':
            modes = ['raw', 'filtered']
        elif args.eval_mode == 'none':
            modes = []
        else:
            modes = [args.eval_mode]
        self._report_raw_metrics = 'raw' in modes
        self._report_filtered_metrics = 'filtered' in modes

        self._hits_at = np.expand_dims(np.array(hits_at), 1)  # Shape (n, 1).
        self._minibatch_placeholder = model.minibatch_htr
        self._dat = dat
        if self._report_filtered_metrics:
            self._build_filters()

        if args.eval_minibatch_size is not None:
            self._minibatch_size = args.eval_minibatch_size
        else:
            self._minibatch_size = args.minibatch_size

        self._raw_ranks_t, self._all_scores_t, self._scores_t, self._loglike_t = self._define_ranks(
            model, model.minibatch_mean_h, model.minibatch_mean_r_predict_t,
            model.idx_t, dat.range_e, args, 't')
        self._raw_ranks_h, self._all_scores_h, self._scores_h, self._loglike_h = self._define_ranks(
            model, model.minibatch_mean_t, model.minibatch_mean_r_predict_h,
            model.idx_h, dat.range_e, args, 'h')
        # `all_scores` have shape (minibatch_size, range_e).
        # `scores` have shape (minibatch_size).

        log_file.write('progress_columns = [\n    "training epoch",')
        metric_labels = (['mrr', 'mrr_balanced'] +
                         ['hits_at_%d' % i for i in hits_at])
        self._weights = {}
        self._placeholders_and_summaries = {}
        self._linebreaks = []
        num_scalar_summaries = 0

        for dat_label in self._eval_sets:
            # Create summary ops and placeholders to feed into them.
            placeholders = []
            summary_ops = []

            for prediction_i, prediction in [(1, 't'), (0, 'h')]:
                # Calculate weights for balanced MRR.
                counts = np.zeros(self._dat.range_e, np.int32)
                for i in self._dat.dat[dat_label][:, prediction_i].flatten():
                    counts[i] += 1
                counts = np.maximum(1, counts)  # Avoid division by zero.
                self._weights[(dat_label, prediction)] = (
                    (len(self._dat.dat[dat_label]) / self._dat.range_e)
                    / counts.astype(np.float32))

                self._linebreaks.append(
                    num_scalar_summaries + len(summary_ops))
                log_file.write('\n    "%s log_likelihood_per_datapoint_predict_%s",' %
                               (dat_label, prediction))
                ph = tf.placeholder(
                    tf.float32, shape=(),
                    name='ph_%s_log_likelihood_per_datapoint_predict_%s' % (dat_label, prediction))
                placeholders.append(ph)
                summary_ops.append(
                    tf.summary.scalar('log_likelihood', ph,
                                      family='eval_%s_predict_%s' % (dat_label, prediction)))

                for mode in modes:
                    self._linebreaks.append(
                        num_scalar_summaries + len(summary_ops))
                    log_file.write('\n   ')
                    for metric in metric_labels:
                        log_file.write(' "%s %s %s_predict_%s",' %
                                       (dat_label, mode, metric, prediction))
                        ph = tf.placeholder(
                            tf.float32, shape=(),
                            name='ph_%s_%s_%s_predict_%s' % (dat_label, mode, metric, prediction))
                        placeholders.append(ph)
                        summary_ops.append(
                            tf.summary.scalar(
                                '%s_%s' % (mode, metric), ph,
                                family='eval_%s_predict_%s' % (dat_label, prediction)))

            self._placeholders_and_summaries[dat_label] = (
                placeholders, tf.summary.merge(summary_ops))
            num_scalar_summaries += len(summary_ops)

        log_file.write('\n]\n')
        log_file.flush()

    def _define_ranks(self, model, emb_in_e, emb_r, idx_predict, range_e, args, predict):
        all_scores = tf.squeeze(
            model.unnormalized_score(
                emb_in_e, emb_r, model.expanded_means_e, args),
            axis=0)

        if model.aux_model is not None:
            with tf.variable_scope('aux_score'):
                all_scores = all_scores[:, :range_e] + model.aux_model.unnormalized_score(
                    predict, args)

        scores = tf.batch_gather(all_scores, tf.expand_dims(idx_predict, 1))
        raw_ranks = range_e - tf.reduce_sum(tf.cast(all_scores < scores, tf.int32),
                                            axis=1)
        # This defines 1-based ranks effectively as `tf.reduce_sum(all_scores >= scores, axis=1)`,
        # except that it cannot be tricked into producing good scores by `NaN`s.

        loglike = -tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=idx_predict, logits=all_scores))

        return raw_ranks, all_scores, tf.squeeze(scores, axis=1), loglike

    def run(self, session, summary_writer, step, epoch, log_file=sys.stdout):
        log_file.write('    (%d,' % epoch)
        log_file.flush()
        for dat_label in self._eval_sets:
            results = self._evaluate_dat(session, dat_label, epoch, log_file)
            phs, summary_op = self._placeholders_and_summaries[dat_label]
            summaries = session.run(summary_op,
                                    feed_dict={ph: val for ph, val in zip(phs, results)})
            if summary_writer is not None:
                summary_writer.add_summary(summaries, global_step=step)
            for i, result in enumerate(results):
                if i in self._linebreaks:
                    log_file.write('\n    ')
                log_file.write(' %g,' % result)
            log_file.flush()
        log_file.write('),\n')
        log_file.flush()

    def _evaluate_dat(self, session, dat_label, epoch, log_file=sys.stdout):
        log_file.write('\n# evaluate dat')
        log_file.flush()
        weights_t = self._weights[(dat_label, 't')]
        weights_h = self._weights[(dat_label, 'h')]

        ops = [self._raw_ranks_t, self._raw_ranks_h,
               self._loglike_t, self._loglike_h]
        if self._report_filtered_metrics:
            ops += [self._all_scores_t, self._all_scores_h,
                    self._scores_t, self._scores_h]

        sum_log_like_t, sum_log_like_h = 0.0, 0.0
        if self._report_raw_metrics:
            raw_metrics = np.zeros(
                (2, 2 + len(self._hits_at)), dtype=np.float32)
        if self._report_filtered_metrics:
            filtered_metrics = np.zeros(
                (2, 2 + len(self._hits_at)), dtype=np.float32)

        log_file.write('\n# eval dat loop')
        log_file.flush()
        for minibatch in self._dat.iterate_in_minibatches(dat_label, self._minibatch_size, epoch=epoch):
            # session.run(
            #     self._loglike_t, feed_dict={self._minibatch_placeholder: minibatch})
            log_file.write('.')
            log_file.flush()
            if self._report_filtered_metrics:
                (raw_ranks_t, raw_ranks_h, loglike_t, loglike_h,
                 all_scores_t, all_scores_h, scores_t, scores_h) = session.run(
                    ops, feed_dict={self._minibatch_placeholder: minibatch})
            else:
                (raw_ranks_t, raw_ranks_h, loglike_t, loglike_h) = session.run(
                    ops, feed_dict={self._minibatch_placeholder: minibatch})

            log_file.write('_')
            log_file.flush()
            sum_log_like_t += loglike_t
            sum_log_like_h += loglike_h

            if self._report_raw_metrics:
                raw_metrics[0, :] += self._get_minibatch_metrics(
                    raw_ranks_t, minibatch[:, 1], weights_t)
                raw_metrics[1, :] += self._get_minibatch_metrics(
                    raw_ranks_h, minibatch[:, 0], weights_h)

            if self._report_filtered_metrics:
                filtered_ranks_t = self._filter(raw_ranks_t, scores_t, all_scores_t,
                                                minibatch[:, [0, 2]], self._filter_predict_t)
                filtered_ranks_h = self._filter(raw_ranks_h, scores_h, all_scores_h,
                                                minibatch[:, [1, 2]], self._filter_predict_h)
                filtered_metrics[0, :] += self._get_minibatch_metrics(
                    filtered_ranks_t, minibatch[:, 1], weights_t)
                filtered_metrics[1, :] += self._get_minibatch_metrics(
                    filtered_ranks_h, minibatch[:, 0], weights_h)

        log_file.write(' done\n')
        log_file.flush()

        ret = [sum_log_like_t]
        if self._report_raw_metrics:
            ret += list(raw_metrics[0, :])
        if self._report_filtered_metrics:
            ret += list(filtered_metrics[0, :])

        ret.append(sum_log_like_h)
        if self._report_raw_metrics:
            ret += list(raw_metrics[1, :])
        if self._report_filtered_metrics:
            ret += list(filtered_metrics[1, :])

        log_file.write('# eval dat done\n')
        log_file.flush()

        return np.array(ret, dtype=np.float32) / self._dat.length(dat_label)

    def _get_minibatch_metrics(self, ranks, target_indices, weights):
        inverse_ranks = 1.0 / ranks.astype(np.float32)
        target_weights = weights[target_indices]
        sum_rr = np.sum(inverse_ranks)
        balanced_sum_rr = np.dot(target_weights, inverse_ranks)
        hits_at = np.sum(np.expand_dims(ranks, 0) <= self._hits_at, axis=1)
        return np.array([sum_rr, balanced_sum_rr] + list(hits_at), dtype=np.float32)

    def _build_filters(self):
        filter_predict_t = defaultdict(set)
        filter_predict_h = defaultdict(set)
        for subset in self._dat.dat.values():
            for h, t, r in subset:
                filter_predict_t[(h, r)].add(t)
                filter_predict_h[(t, r)].add(h)

        # Turns sets into numpy arrays for more efficient lookup
        self._filter_predict_t = {key: np.array(list(value))
                                  for key, value in filter_predict_t.items()}
        self._filter_predict_h = {key: np.array(list(value))
                                  for key, value in filter_predict_h.items()}

    def _filter(self, raw_ranks, scores, all_scores, inputs, filter_dict):
        return np.array([raw_rank - np.sum(all_s[filter_dict[(e_in, r)]] > score)
                         for raw_rank, score, all_s, (e_in, r)
                         in zip(raw_ranks, scores, all_scores, inputs)])


class SupervisedEvaluator:
    def __init__(self, model, dat, args, hits_thresholds=[100, 10, 1], log_file=sys.stdout):
        if args.eval_minibatch_size is not None:
            self._minibatch_size = args.eval_minibatch_size
        else:
            self._minibatch_size = args.minibatch_size

        self._dat = dat
        self._hits_thresholds = np.array(hits_thresholds)

        if args.eval_dat == 'both':
            self._eval_sets = ['valid', 'test']
        elif args.eval_dat == 'all':
            self._eval_sets = ['valid', 'test', 'train']
        elif args.eval_dat == 'none':
            self._eval_sets = []
        else:
            self._eval_sets = [args.eval_dat]

        self._ll_placeholder = tf.placeholder(
            tf.float32, shape=(), name='ph_valid_log_likelihood_per_datapoint')
        self._hits_at_n_placeholder = tf.placeholder(
            tf.float32, shape=(len(hits_thresholds),), name='ph_hits_at_n')
        hits_at_n = [tf.gather(self._hits_at_n_placeholder, i)
                     for i, _ in enumerate(hits_thresholds)]

        self._summary_op = {}
        for subset in self._eval_sets:
            ll_summary_op = tf.summary.scalar(
                'log_likelihood', self._ll_placeholder, family='eval_%s' % subset)
            hits_at_n_summary_ops = [
                tf.summary.scalar(
                    'hits_at_%d' % threshold, hits, family='eval_%s' % subset)
                for hits, threshold in zip(hits_at_n, hits_thresholds)]
            self._summary_op[subset] = tf.summary.merge(
                [ll_summary_op, *hits_at_n_summary_ops])

        self._evaluation_log_likelihood_op = model.evaluation_log_likelihood
        self._evaluation_ranks_op = model.evaluation_ranks
        self._evaluation_log_likelihood_main_op = model.evaluation_log_likelihood_main
        self._evaluation_ranks_main_op = model.evaluation_ranks_main

        self._minibatch_htr = model.minibatch_htr

        log_file.write(
            'progress_columns = ["training epoch", "net training time (seconds)"')
        for subset in self._eval_sets:
            log_file.write(',\n    "%s log likelihood"' % subset)
            for threshold in hits_thresholds:
                log_file.write(', "%s Hits@%d"' % (subset, threshold))
            log_file.write(
                ',\n    "%s log likelihood (without bias correction)"' % subset)
            for threshold in hits_thresholds:
                log_file.write(
                    ', "%s Hits@%d" (without bias correction)' % (subset, threshold))
        log_file.write(']\n')
        log_file.flush()

    def run(self, session, summary_writer, step, epoch, net_training_time, log_file=sys.stdout):
        log_file.write('(%d, %g' % (epoch, net_training_time))
        log_file.flush()

        sum_ll = 0.0
        sum_ll_main = 0.0
        count = 0
        hit_counts = np.zeros(self._hits_thresholds.shape, np.int32)
        hit_counts_main = np.zeros(self._hits_thresholds.shape, np.int32)

        for subset in self._eval_sets:
            for minibatch in self._dat.iterate_in_minibatches(subset, self._minibatch_size, epoch=epoch):
                ll, ranks, ll_main, ranks_main = session.run(
                    [self._evaluation_log_likelihood_op[subset],
                     self._evaluation_ranks_op[subset],
                     self._evaluation_log_likelihood_main_op[subset],
                     self._evaluation_ranks_main_op[subset]],
                    feed_dict={self._minibatch_htr: minibatch})
                count += len(minibatch)
                sum_ll += ll
                sum_ll_main += ll_main
                hit_counts += np.sum(
                    ranks[np.newaxis, :] <= self._hits_thresholds[:, np.newaxis], axis=1)
                hit_counts_main += np.sum(
                    ranks_main[np.newaxis, :] <= self._hits_thresholds[:, np.newaxis], axis=1)

            log_likelihood = sum_ll / count
            hits_at_n = hit_counts / count
            log_likelihood_main = sum_ll_main / count
            hits_at_n_main = hit_counts_main / count

            log_file.write(',\n    %g' % log_likelihood)
            for h in hits_at_n:
                log_file.write(', %g' % h)
            log_file.write(',\n    %g' % log_likelihood_main)
            for h in hits_at_n_main:
                log_file.write(', %g' % h)
            log_file.flush()

            if summary_writer is not None:
                summaries = session.run(self._summary_op[subset], feed_dict={
                    self._ll_placeholder: log_likelihood, self._hits_at_n_placeholder: hits_at_n})
                summary_writer.add_summary(summaries, global_step=step)

        log_file.write('),\n')


def add_cli_args(parser):
    '''Add command line arguments to control frequency and type of model evaluations.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was just added to the parser.
    '''
    eval_args = parser.add_argument_group(
        'Evaluation parameters')
    eval_args.add_argument('--eval_dat', choices=['valid', 'test', 'both', 'train', 'all', 'none'],
                           default='valid', help='''
        Choose whether to evaluate on the validation set (`--eval_dat valid`), the test set
        (`--eval_dat test`), or both (`--eval_dat both`). For debugging purposes, we also provide
        the choices `--eval_dat train`, `--eval_dat all` (which means evaluating on the train,
        validation, and test set), and `--eval_dat none`.''')
    eval_args.add_argument('--eval_mode', choices=['none', 'raw', 'filtered', 'both'], default='both',
                           help='''
        Choose which type of predicted ranks to use for evaluation: `raw` to use unfiltered ranks,
        `filtered` to use filtered ranks according to Bordes et al., NIPS 2013, or `both` (default)
        to report evaluation metrics based on both filtered and filtered ranks.''')
    eval_args.add_argument('--eval_minibatch_size', type=int, help='''
        Set the minibatch size for evaluations, if different from the training minibatch size.''')
    eval_args.add_argument('--validation_points', type=int, default=1000, help='''
        Number of data points to use for validation. Validation data points are sampled anew
        each time the evaluation is run. If learning curves are jitterish, increase this 
        number.''')
