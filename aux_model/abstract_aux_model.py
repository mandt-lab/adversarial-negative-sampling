import tensorflow as tf
import abc


class AbstractAuxModel(abc.ABC):
    def __init__(self, supervised=False):
        with tf.variable_scope('minibatch'):
            if supervised:
                self.minibatch_htr = tf.placeholder(
                    tf.int32, shape=(None,), name='minibatch')
                self._minibatch_size = tf.shape(self.minibatch_htr)[0]
            else:
                self.minibatch_htr = tf.placeholder(
                    tf.int32, shape=(None, 3), name='minibatch_htr')
                self._minibatch_size = tf.shape(self.minibatch_htr)[0]

                self.idx_h = self.minibatch_htr[:, 0]
                self.idx_t = self.minibatch_htr[:, 1]
                self.idx_r = self.minibatch_htr[:, 2]

    @abc.abstractmethod
    def unnormalized_score(self, head_or_tail, args, subset):
        pass

    @abc.abstractmethod
    def create_sampler(self, head_or_tail, num_samples):
        pass
