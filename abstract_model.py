import sys
import abc

import tensorflow as tf
import numpy as np

import optimizer


class AbstractModel(abc.ABC):
    '''Abstract base class for knowledge graph embedding models.

    You won't usually want to derive directly from this class. In most cases, you'll
    want to derive from either `AbstractMainModel` or from `AbstractAuxModel`.
    '''

    @abc.abstractmethod
    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, args):
        '''Define a tensorflow op that calculates the prediction scores (logits).

        This is also sometimes called `logits`.

        Arguments:
        emb_in_e -- Embedding vectors of the input entities, i.e., the entities on which
            we condition. A dict that maps labels (strings) to tensors of shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.
        emb_r -- Embedding vectors of the relations. If reciprocal relations are used
            then the caller should pass in different embedding vectors for head or tail
            prediction. A dict that maps labels (strings) to tensors of shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.
        emb_all_e -- Embedding vectors of all entities. A dict that maps labels (strings)
            to tensors of shape `(range_e, num_samples, embedding_dimensions...)`.
        args -- Namespace holding command line arguments.

        Returns:
        A tensor of shape `(num_samples, minibatch_size, range_e)` that holds the
        unnormalized represents the negative log likelihood of the data.
        Should *not* be normalized to the batch size or sample size.
        '''
        pass
