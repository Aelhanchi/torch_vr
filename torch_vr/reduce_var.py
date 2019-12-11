import torch
from .prev_grads import PrevGrads


class ReduceVar():
    """Implements the SAGA and SAG variance reduction schemes.

    References::
        - Aaron Defazio, Francis Bach, and Simon Lacoste-Julien,SAGA: A Fast
          Incremental Gradient Method With Support for Non-Strongly
          Convex Composite Objectives,
          arXiv:1407.0202 [cs.LG], 2014 (available at
          https://arxiv.org/abs/1407.0202).

        - Mark Schmidt and Nicolas Le Roux and Francis Bach,Minimizing
          Finite Sums with the Stochastic Average Gradient,
          arXiv:1309.2388 [math.OC], 2013 (available at
          https://arxiv.org/abs/1309.2388).

    Arguments:
        params (iterable): the parameters of the model.
        N (int): number of data points
        layers (iterable): an iterable of nn.Module objects representing
            the layers of the neural network in order. Only Linear and
            Conv2D layers are supported.
        init_dZ (list, optional): list of initial gradient of the output of each layer.
            Default is 'None' which initializes them to 0.
        init_A (list, optional): list of initial inputs of each layer.
            Default is 'None' which initializes them to 0.
        ranks (list, optional): the ranks of the low-rank approximations for the
            convolutional layers. Default is 'None' and uses the full
            gradients.
        method (string, optional): Either "SAGA" or "SAG".
    """

    def __init__(self, params, N, layers, init_dZ=None, init_A=None, ranks=None, method='SAGA'):
        self.N = N
        if method not in ['SAGA', 'SAG']:
            raise ValueError('Supported methods are: "SAGA" and "SAG" ')
        self.method = method
        self.params = list(params)
        self.prev_grads = PrevGrads(N, layers, init_dZ, init_A, ranks)
        self.sum_prev_grads = list()
        for p in self.params:
            if init_dZ is None or init_A is None:
                self.sum_prev_grads.append(torch.zeros_like(p))
            else:
                self.sum_prev_grads.append(p.grad)

    def sample_indices(self, batch_size):
        """Samples a mini-batch of indices uniformly at random without replacement.

        Arguments:
            batch_size (int): the mini-batch size.

        Returns:
            a torch.LongTensor of indices.
        """
        return torch.multinomial(torch.ones(self.N), batch_size, replacement=False)

    def reduce_variance(self, dZ, A, indices):
        """Reduces the variance of the gradient estimator
        and updates the stored gradients.

        Arguments:
            dZ (list): list of gradients of the output of each layer.
            A (list): list of the input of each layer.
            indices (array-like): array of mini-batch indices.
        """
        # replaces the SGD estimator with the SAGA/SAG estimator.
        prev_grads = self.prev_grads.batch_gradient(indices)
        if self.method == 'SAGA':
            for p, spg, pg in zip(self.params, self.sum_prev_grads, prev_grads):
                spg += p.grad
                p.grad = (self.N/len(indices)) * (p.grad - pg) + (spg - p.grad)
                spg -= pg
        else:  # self.method == 'SAG'
            for p, spg, pg in zip(self.params, self.sum_prev_grads, prev_grads):
                spg = spg - pg + p.grad
                p.grad = spg

        # updates the stored gradients
        self.prev_grads.update(dZ, A, indices)
