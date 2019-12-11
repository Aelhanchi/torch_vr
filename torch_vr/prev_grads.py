from .prev_grads_layers import PrevGradLinearLayer
from .prev_grads_layers import PrevGradConv2dLayer


class PrevGrads:
    """ Class that stores, updates, and answers queries about the last
    seen per-example gradients.

    Arguments:
        N (int): number of data points
        layers (iterable): an iterable of nn.Module objects representing
            the ordered layers of the neural network. Only Linear and
            Conv2D layers are supported.
        init_dZ (list, optional): list of initial gradient of the output of each layer.
            Default is 'None' which initializes the gradients to 0.
        init_A (list, optional): list of initial inputs of each layer.
            Default is 'None' which initializes the gradients to 0.
        ranks (list, optional): the ranks of the low-rank approximations for the
            convolutional layers. Default is 'None' and uses the full
            gradients.
    """

    def __init__(self, N, layers, init_dZ=None, init_A=None, ranks=None):
        self.layers = list()
        for (i, layer) in enumerate(list(layers)):
            if layer.__class__.__name__ == 'Linear':
                if init_dZ is not None and init_A is not None:
                    init_dz = init_dZ[i]
                    init_a = init_A[i]
                else:
                    init_dz = None
                    init_a = None
                self.layers.append(PrevGradLinearLayer(N, layer, init_dz, init_a))
            elif layer.__class__.__name__ == 'Conv2d':
                if ranks is not None:
                    rank = ranks[i]
                else:
                    rank = None
                if init_dZ is not None and init_A is not None:
                    init_dz = init_dZ[i]
                    init_a = init_A[i]
                else:
                    init_dz = None
                    init_a = None
                self.layers.append(PrevGradConv2dLayer(N, layer, init_dz, init_a, rank))
            else:
                raise ValueError('Only Linear and Conv2d layers are supported')

    def batch_gradient(self, indices, weights=None):
        grads = list()
        for layer in self.layers:
            grads += layer.batch_gradient(indices, weights)
        return grads

    def individual_gradients(self, indices, weights=None):
        ind_grads = list()
        for layer in self.layers:
            ind_grads += layer.individual_gradients(indices, weights)
        return ind_grads

    def update(self, dZ, A, indices):
        for (layer, dz, a) in zip(self.layers, dZ, A):
            layer.update(dz, a, indices)
