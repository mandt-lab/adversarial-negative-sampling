import argparse
import tensorflow as tf

import dataset
import aux_model.supervised_decision_tree_model

parser = argparse.ArgumentParser(description='''
    Evaluate the log likelihood of a trained probabilistic decision tree on a validation set.''')

positional_args = parser.add_argument_group(
    'Required positional arguments')
positional_args.add_argument('model_path', help='''
    Path to the pretrained model (in .jld2 format).''')
positional_args.add_argument('dat_dir_path', help='''
    Path to a directory containing the training, validation, and test data sets.''')

args = parser.parse_args()

dat = dataset.SupervisedDataset(args.dat_dir_path, emb_dim=16)
aux_model = aux_model.supervised_decision_tree_model.SupervisedDecisionTreeModel(
    '%s:%s' % (args.model_path, args.dat_dir_path), dat)

aux_scores = aux_model.unnormalized_score(None, None)

log_likelihood = tf.reduce_sum(aux_model.predictive_ll())

session = tf.Session()
session.run(tf.global_variables_initializer())

ll = 0.0
count = 0
for mb in dat.iterate_in_minibatches('valid', 1000):
    ll += session.run(log_likelihood, {aux_model.minibatch_htr: mb})
    count += len(mb)

print('Log likelihood: %g' % (ll / count))
