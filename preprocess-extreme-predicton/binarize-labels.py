from tqdm import tqdm
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    description='Binarize labels of extreme prediction data sets.')
parser.add_argument('input_dir', help='Path to input directory')

args = parser.parse_args()


def load_dat(path):
    print('Reading in file "%s" ...' % path)
    with open(path) as in_file:
        num_lines, num_categories = (int(i)
                                     for i in in_file.readline().split(' '))
        print('=> %d multi-label predictions over %d categories.' %
              (num_lines, num_categories))
        dat = np.empty(num_lines, dtype=np.int32)

        for i in tqdm(range(num_lines)):
            labels_values = np.array(
                list(int(j)
                     for i in in_file.readline().split(' ')
                     for j in i.split(':')),
                dtype=np.int32).reshape((-1, 2))
            assert np.all(labels_values[:, 1] == 1)
            dat[i] = labels_values[:, 0].min()

        # Make sure that we've reached EOF
        assert len(in_file.read(1)) == 0

    return dat, num_categories


def save_dat(dat, inv_perm, num_nonzero, path):
    print('Writing to file "%s" ...' % path)
    with open(path, 'wb') as out_file:
        np.array([len(dat), num_nonzero], dtype=np.int32).tofile(out_file)
        np.fromiter((inv_perm[dat_i]
                     for dat_i in dat), np.int32).tofile(out_file)


trn_dat, trn_cat = load_dat(os.path.join(args.input_dir, 'trn_lbl_mat.txt'))

counts = np.zeros((trn_cat,), np.int32)
for i in trn_dat:
    counts[i] += 1
permutation = (-counts).argsort()
num_nonzero = np.sum(counts != 0)

print('Found %d used and %d unused labels' %
      (num_nonzero, len(counts) - num_nonzero))

inv_perm = np.empty_like(permutation)
for i, j in enumerate(permutation):
    inv_perm[j] = i

save_dat(trn_dat, inv_perm, num_nonzero, os.path.join(
    args.input_dir, 'train-labels-first.np'))

tst_dat, tst_cat = load_dat(os.path.join(args.input_dir, 'tst_lbl_mat.txt'))
assert tst_cat == trn_cat
holdout_idxs = np.fromiter((i for i, dat_i in enumerate(tst_dat)
                            if inv_perm[dat_i] < num_nonzero), np.int32)

rng = np.random.RandomState(36528475)
holdout_idxs = holdout_idxs[rng.permutation(len(holdout_idxs))]
num_valid = round(0.2 * len(holdout_idxs))
valid_idxs = holdout_idxs[:num_valid]
test_idxs = holdout_idxs[num_valid:]

sel_path = os.path.join(args.input_dir, 'valid-indices.np')
print('Writing indices of %d validation points to "%s" ...' %
      (len(valid_idxs), sel_path))
with open(sel_path, 'wb') as f:
    np.array([len(valid_idxs)], dtype=np.int32).tofile(f)
    valid_idxs.tofile(f)

sel_path = os.path.join(args.input_dir, 'test-indices.np')
print('Writing indices of %d test points to "%s" ...' %
      (len(test_idxs), sel_path))
with open(sel_path, 'wb') as f:
    np.array([len(test_idxs)], dtype=np.int32).tofile(f)
    test_idxs.tofile(f)

save_dat(tst_dat[valid_idxs], inv_perm, num_nonzero, os.path.join(
    args.input_dir, 'valid-labels-first.np'))

save_dat(tst_dat[test_idxs], inv_perm, num_nonzero, os.path.join(
    args.input_dir, 'test-labels-first.np'))

print('Done.')
