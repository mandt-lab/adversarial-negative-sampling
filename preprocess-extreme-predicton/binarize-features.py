from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Binarize dense extreme prediction data sets.')
parser.add_argument('input', help='Path to input file')
parser.add_argument(
    '-f', '--filter', help='Path to file containing indices to filter by.')
parser.add_argument(
    '-o', '--out', required=True, help='Path to output file.')

args = parser.parse_args()

with open(args.input) as in_file:
    n, d = (int(i) for i in in_file.readline().split(' '))
    dense = np.empty((n, d), dtype=np.float32)
    for i in tqdm(range(n)):
        dense[i, :] = [np.float32(j) for j in in_file.readline().split(' ')]

    # Make sure that we've reached EOF
    assert len(in_file.read(1)) == 0

if args.filter is not None:
    with open(args.filter) as filter_file:
        n, = np.fromfile(filter_file, dtype=np.int32, count=1)
        idxs = np.fromfile(filter_file, dtype=np.int32, count=n)
        # Make sure that we've reached EOF
        assert len(filter_file.read(1)) == 0
    dense = dense[idxs, :]

print('Saving to "%s" ... ' % args.out, end='')
with open(args.out, 'wb') as out_file:
    np.array(dense.shape, dtype=np.int32).tofile(out_file)
    dense.tofile(out_file)

print('done.')
