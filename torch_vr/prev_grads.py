import torch
from .prev_grads_layers import PrevGradLinearLayer
from .prev_grads_layers import PrevGradConv2DLayer


class PrevGrads:
    """ Class that stores, updates, and answers queries about the last
    seen per-example gradients.

    Arguments:
        N (int): number of data points
        layers (iterable): an iterable of nn.Module objects representing
            the layers of the neural network in order. Only Linear and
            Conv2D layers are supported.
        init_dZ (list): list of initial gradient of the output of each layer.
            Default is 'None' which initializes them to 0.
        init_A (list): list of initial inputs of each layer.
            Default is 'None' which initializes them to 0.
        ranks (list): the ranks of the low-rank approximations for the
            convolutional layers. Default is 'None' and uses the full
            gradients.
    """

    def __init__(self, N, layers, init_dZ=None, init_A=None, ranks=None):
        self.layers = list()
        for (i, l) in enumerate(list(layers)):
            if l.__class__.__name__ == 'Linear':
                if init_dZ is not None and init_A is not None:
                    self.layers.append(PrevGradLinearLayer(
                        init_dZ[i], init_A[i], l.bias is not None))
                else:
                    dZ = torch.zeros(N, l.out_features)
                    A = torch.zeros(N, l.in_features)
                    self.layers.append(PrevGradLinearLayer(dZ, A, l.bias is not None))
            elif l.__class__.__name__ == 'Conv2d':
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
                self.layers.append(PrevGradConv2DLayer(N, l.kernel_size,
                                                       l.out_channels, l.in_channels,
                                                       init_dz, init_a, l.dilation,
                                                       l.padding, l.stride, rank,
                                                       bias=l.bias is not None))
            else:
                raise ValueError('Only Linear and Conv2d layers are supported')

    def batch_gradient(self, indices, weights=None):
        grads = list()
        for l in self.layers:
            grads += l.batch_gradient(indices, weights)
        return grads

    def individual_gradients(self, indices, weights=None):
        ind_grads = list()
        for l in self.layers:
            ind_grads += l.individual_gradients(indices, weights)
        return ind_grads

    def update(self, dZ, A, indices):
        for (l, dz, a) in zip(self.layers, dZ, A):
            l.update(dz, a, indices)
