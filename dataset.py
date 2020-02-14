import os
import sys
import random

import numpy as np


class Dataset:
    def __init__(self, dir_path, log_file=sys.stdout, subsets=['train', 'valid', 'test'],
                 binary_files=False):
        log_file.write(
            '# Loading data set from directory `%s`.\n' % dir_path)
        log_file.flush()

        self.dat = {}
        for label in subsets:
            if binary_files:
                self.dat[label] = self._load_binary_file(
                    os.path.join(dir_path, '%s.np' % label))
            else:
                self.dat[label] = self._load_text_file(
                    os.path.join(dir_path, '%s2id.txt' % label))

            log_file.write('%s_points = %d\n' % (label, len(self.dat[label])))
            log_file.flush()

        self.range_e = max(dat[:, :2].max() for dat in self.dat.values()) + 1
        self.range_r = max(dat[:, 2].max() for dat in self.dat.values()) + 1
        log_file.write('range_e = %d  # number of entities\n' % self.range_e)
        log_file.write('range_r = %d  # number of relations (before adding reciprocal relations)\n'
                       % self.range_r)

    def _load_text_file(self, path):
        with open(path, 'r') as f:
            count = int(f.readline())
            ret = np.empty((count, 3), dtype=np.int32)
            for i in range(count):
                ret[i, :] = [int(val) for val in f.readline().split(' ')]
            assert len(f.read(1)) == 0

        return ret

    def _load_binary_file(self, path):
        return np.fromfile(path, np.int32, -1).reshape((-1, 3))

    def iterate_in_minibatches(self, subset_label, minibatch_size, epoch=None, rng=None):
        '''Note: if rng is provided, the returned iterator may mutate the data set.'''
        dat = self.dat[subset_label]
        if rng is None and subset_label == 'valid':
            rng = random.Random(1234 + epoch)
        if rng is not None:
            if subset_label == 'valid':
                # Estimate validation set metrics based on 1000 samples. Use different samples
                # every time so that smoothed learning curves provide high quality estimates.
                dat = dat[rng.sample(range(len(dat)), 1000)]
            else:
                rng.shuffle(dat)

        # Iterate over all full sized minibatches
        start = -minibatch_size
        for start in range(0, len(dat) - minibatch_size + 1, minibatch_size):
            yield dat[start: start + minibatch_size]

        # If in deterministic mode, also return a final smaller minibatch if the data set size is
        # not a multiple of the minibatch size (not necessary in shuffle mode).
        if rng is None and start + minibatch_size < len(dat):
            yield dat[start + minibatch_size:]

    def length(self, subset_label):
        return self._valid_size if subset_label == 'valid' else len(self.labels[subset_label])


class SupervisedDataset:
    def __init__(self, dir_path, validation_points=None, log_file=sys.stdout,
                 subsets=['train', 'valid', 'test'], emb_dim=None):
        if emb_dim is None:
            emb_dim = 'full'
        else:
            emb_dim = 'k%d' % emb_dim
        self._valid_size = validation_points

        log_file.write(
            '# Loading data set from directory `%s`.\n' % dir_path)
        log_file.flush()

        self.features = {}
        self.labels = {}
        for label in subsets:
            self.features[label] = self._load_features_file(
                os.path.join(dir_path, '%s-features-%s.np' % (label, emb_dim)))
            self.labels[label], self.range_e = self._load_labels_file(
                os.path.join(dir_path, '%s-labels-first.np' % label))
            self.embedding_dim = self.features[label].shape[1]

            log_file.write('%s_points = %d\n' %
                           (label, len(self.labels[label])))
            log_file.flush()

        self.range_r = 1
        log_file.write('range_e = %d  # number of entities\n' % self.range_e)
        log_file.write('range_r = %d  # number of relations (before adding reciprocal relations)\n'
                       % self.range_r)

    def _load_features_file(self, path):
        with open(path, 'rb') as f:
            n, k = np.fromfile(f, dtype=np.int32, count=2)
            dat = np.fromfile(f, dtype=np.float32, count=n*k).reshape((n, k))
            # Make sure that we've reached EOF
            assert len(f.read(1)) == 0
        return dat

    def _load_labels_file(self, path):
        with open(path, 'rb') as f:
            n, range_e = np.fromfile(f, dtype=np.int32, count=2)
            dat = np.fromfile(f, dtype=np.int32, count=n)
            assert len(dat) == n
            # Make sure that we've reached EOF
            assert len(f.read(1)) == 0
        return dat, range_e

    def iterate_in_minibatches(self, subset_label, minibatch_size, epoch=None, rng=None):
        if rng is None and subset_label == 'valid' and self._valid_size is not None:
            rng = random.Random(1234 + epoch)
        if rng is not None:
            if subset_label == 'valid' and self._valid_size is not None:
                # Estimate validation set metrics based on self._valid_size samples. Use different
                # samples every time so that smoothed learning curves provide good estimates.
                dat = np.array(rng.sample(
                    range(len(self.labels[subset_label])), self._valid_size), dtype=np.int32)
            else:
                dat = np.array(rng.permutation(
                    len(self.labels[subset_label])), dtype=np.int32)
        else:
            dat = np.array(
                range(len(self.labels[subset_label])), dtype=np.int32)

        # Iterate over all full sized minibatches
        start = -minibatch_size
        for start in range(0, len(dat) - minibatch_size + 1, minibatch_size):
            yield dat[start: start + minibatch_size]

        # If in deterministic mode, also return a final smaller minibatch if the data set size is
        # not a multiple of the minibatch size (not necessary in shuffle mode).
        if rng is None and start + minibatch_size < len(dat):
            yield dat[start + minibatch_size:]

    def length(self, subset_label):
        return self._valid_size if subset_label == 'valid' else len(self.labels[subset_label])
