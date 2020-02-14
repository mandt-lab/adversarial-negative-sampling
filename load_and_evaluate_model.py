import tensorflow as tf
import numpy as np
import dataset
import random

from aux_model.supervised_decision_tree_model import SupervisedDecisionTreeModel
import train
import aux_model.baselines
import main_model.supervised_model
import evaluate

# TODO: this is the wrong run (wrong larning rate and regularizer)
args = {
    'adagrad_init': 1e-08,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-08,
    'aux_model': '../dat/Wikipedia-500K/aux-k16-greedykmeans-rfreq0.1.jld2:../dat/Wikipedia-500K',
    'binary_dataset': False,
    'em': False,
    'embedding_dim': 512,
    'epochs': 1000,
    'epochs_per_checkpoint': 10,
    'epochs_per_eval': 1,
    'eval_dat': 'valid',
    'eval_minibatch_size': 10,
    'eval_mode': 'both',
    'force': False,
    'initial_e_epochs': 0,
    'initial_reg_strength': 0.001,
    'initial_reg_uniform': False,
    'initial_std': 1.0,
    'initial_summaries': 100,
    'initialize_from': '/home/jovyan/varred-nce/out/w500k-constlr/proposed-rfreq0.1-n10-reg1e-3-lr0.01-lrzmul0-initinvaux-longrun/checkpoint-395040',
    'initialize_to_inverse_aux': True,
    'input': 'dat/Wikipedia-500K',
    'lr0': 0.01,
    'lr_exponent': 0.0,
    'lr_offset': 68025,
    'minibatch_size': 1000,
    'model': 'supervised',
    'neg_samples': 10,
    'num_samples': 1,
    'optimizer': 'adagrad',
    # 'output': 'out/w500k-constlr/proposed-rfreq0.1-n10-reg1e-3-lr0.01-lrzmul0-initinvaux-longrun',
    'output': '',
    'reg_separate': False,
    'reg_strength_slowdown': 1.0,
    'rng_seed': 2185259521,
    'std_speedup': 1.0,
    'steps_per_summary': 1000,
    'trace': False,
    'use_log_norm_weight': False
}

# args = {
#     'adagrad_init': 1e-08,
#     'adam_beta1': 0.9,
#     'adam_beta2': 0.999,
#     'adam_epsilon': 1e-08,
#     'aux_model': 'uniform',
#     'binary_dataset': False,
#     'em': False,
#     'embedding_dim': 512,
#     'epochs': 1000,
#     'epochs_per_checkpoint': 10,
#     'epochs_per_eval': 1,
#     'eval_dat': 'valid',
#     'eval_minibatch_size': 10,
#     'eval_mode': 'both',
#     'force': False,
#     'initial_e_epochs': 0,
#     'initial_reg_strength': 0.0001,
#     'initial_reg_uniform': False,
#     'initial_std': 1.0,
#     'initial_summaries': 100,
#     'initialize_from': '/home/jovyan/varred-nce/out/w500k-constlr/uniform-zeroll-n10-reg1e-4-lr0.001-lrzmul0-longrun/checkpoint-1646000',
#     'initialize_to_inverse_aux': False,
#     'input': 'dat/Wikipedia-500K/',
#     'lr0': 0.001,
#     'lr_exponent': 0.0,
#     'lr_offset': 68025,
#     'minibatch_size': 1000,
#     'model': 'supervised',
#     'neg_samples': 10,
#     'num_samples': 1,
#     'optimizer': 'adagrad',
#     # 'output': 'out/w500k-constlr/uniform-zeroll-n10-reg1e-4-lr0.001-lrzmul0-longrun',
#     'output': '',
#     'reg_separate': False,
#     'reg_strength_slowdown': 1.0,
#     'rng_seed': 1907515672,
#     'std_speedup': 1.0,
#     'steps_per_summary': 1000,
#     'trace': False,
#     'use_log_norm_weight': False
# }


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


args = Bunch(args)

dat = dataset.SupervisedDataset('../dat/Wikipedia-500K', emb_dim=None)

rng = random.Random()

if args.aux_model == 'uniform':
    aux_model = UniformAuxModel(dat, supervised=True)
else:
    aux_model = SupervisedDecisionTreeModel(args.aux_model, dat)

model = main_model.supervised_model.SupervisedModel(
    args, dat, rng, aux_model=aux_model)

session = tf.Session()
session.run(tf.initializers.global_variables(),
            feed_dict={model.feed_train_features: dat.features['train']},
            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
del dat.features['train']

train.load_checkpoint(model, session, args.initialize_from)

evaluator = evaluate.SupervisedEvaluator(model, dat, args)

sum_ll = 0.0
count = 0
thresholds = np.array([100, 10, 1])
hit_counts = np.array([0 for _ in thresholds])

print('starting evaluation ...')
for minibatch in evaluator._dat.iterate_in_minibatches('valid', evaluator._minibatch_size, epoch=1):
    ll, scores, labels = session.run(
        [evaluator._valid_likelihood_op, model.valid_scores, model.valid_labels],
        feed_dict={evaluator._minibatch_htr: minibatch})
    count += len(minibatch)
    print(count)
    sum_ll += ll
    target_scores = scores[np.arange(len(scores)), labels]
    ranks = (scores.shape[1] -
             np.sum(scores <= target_scores[:, np.newaxis], axis=1) + 1)
    hit_counts += np.sum(
        ranks[np.newaxis, :] <= thresholds[:, np.newaxis], axis=1)

print('llh: %g' % (sum_ll / count))
print('hits@%s: %s' % (thresholds, ['%g' % (i / count) for i in hit_counts]))
