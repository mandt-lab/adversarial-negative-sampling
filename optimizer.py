'''Module for defining user-controllable Tensorflow optimizers.'''

import tensorflow as tf


def add_cli_args(parser):
    '''Add command line arguments that control the optimization method.

    This function defines command line arguments that control an iterative
    optimization method. Currently, it allows choosing between standard SGD,
    Adam, and Adagrad, and to set the corresponding learning rates and momenta.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was added to the parser.
    '''

    group = parser.add_argument_group('Optimization Parameters')
    group.add_argument('--optimizer', metavar='OPT', default='adagrad',
                       choices=['sgd', 'adam', 'adagrad'],  help='''
        Set the Optimization method.''')
    group.add_argument('--lr0', metavar='FLOAT', type=float, default=0.02, help='''
        Set the initial prefactor for the (possibly adaptive) learning rate. See
        `--lr_exponent`.''')
    group.add_argument('--lr_offset', metavar='FLOAT', type=float, default=25*2721, help='''
        Set the time scale on which the learning rate drops (unless `--lr_exponent` is set to
        zero). See `--lr_exponent`.''')
    # TODO: original implementation used `25 * trainTotal / batch_size`
    group.add_argument('--lr_exponent', metavar='FLOAT', type=float, default=0.5, help='''
        Set the exponent by which the prefactor of the learning rate drops as a function of the
        training step. The learning rate in step $t$ is $\\rho_0 (a/(t+a))^b$, where the initial
        learning rate $\\rho_0$ is controlled by `--lr0`, the offset $a$ is controlled by
        `--lr_offset`, and the exponent $b$ is controlled by `--lr_exponent`. In order to satisfy
        the requirements by Robbins and Monro (1951), we recommended the following values; for
        `--optimizer sgd` and `--optimizer adam`:  $0.5 < b <= 1$; for `--optimizer adagrad`:
        $0 < b <= 0.5$.''')
    group.add_argument('--adam_beta1', metavar='FLOAT', type=float, default=0.9, help='''
        Only used for `--optimizer adam`. Set the decay rate of the first moment estimator in the
        Adam optimizer.''')
    group.add_argument('--adam_beta2', metavar='FLOAT', type=float, default=0.999, help='''
        Only used for `--optimizer adam`. Set the decay rate of the second moment estimator in the
        Adam optimizer.''')
    group.add_argument('--adam_epsilon', metavar='FLOAT', type=float, default=1e-8, help='''
        Only used for `--optimizer adam`. Set a regularizer to prevent division by zero in the
        Adam optimizer in edge cases.''')
    group.add_argument('--adagrad_init', metavar='FLOAT', type=float, default=1e-8, help='''
        Only used for `--optimizer adagrad`. Set initial accumulator of the Adagrad optimizer.''')
    return group


def define_learning_rate(args):
    '''Define global step and learning rate.

    Arguments:
    args -- A python namespace containing the parsed command line arguments.
        Must contain the arguments defined by the function `add_cli_args()` in
        the same module.

    Returns:
    A tuple of three tensorflow ops: the global step, the learning rate, and an op that
    generates a scalar summary of the learning rate.
    '''
    concluded_training_steps = tf.Variable(
        0, dtype=tf.int32, name='training_step')
    with tf.variable_scope('learning_rate'):
        step_float = tf.cast(concluded_training_steps, dtype=tf.float32)
        lr = args.lr0 * (args.lr_offset /
                         (step_float + args.lr_offset)) ** args.lr_exponent
        summary_op = tf.summary.scalar('learning_rate', lr)
    return concluded_training_steps, lr, summary_op


def define_optimizer(args, learning_rate):
    '''Create a tensorflow optimizer according to the provided command line arguments.

    Arguments:
    learning_rate -- A python scalar or a scalar tensorflow op with the learning rate.
        See `define_learning_rate`.

    Returns:
    A Tensorflow optimizer.
    '''
    if args.optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif args.optimizer == 'adam':
        return tf.train.AdamOptimizer(
            learning_rate, args.adam_beta1, args.adam_beta2, args.adam_epsilon)
    elif args.optimizer == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate, args.adagrad_init)
    else:
        raise 'Unknown optimizer `%s`.' % args.optimizer
