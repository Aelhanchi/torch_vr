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

    def __init__(self, N, layers, init_outputs_grad=None, init_inputs=None, ranks=None):
        if init_outputs_grad is None or init_inputs is None:
            init_outputs_grad = [None for i in range(len(layers))]
            init_inputs = [None for i in range(len(layers))]
        if ranks is None:
            ranks = [None for i in range(len(layers))]

        self.layers = list()
        for (i, layer) in enumerate(list(layers)):
            if layer.__class__.__name__ == 'Linear':
                self.layers.append(PrevGradLinearLayer(N, layer, init_outputs_grad[i], init_inputs[i]))
            elif layer.__class__.__name__ == 'Conv2d':
                self.layers.append(PrevGradConv2dLayer(N, layer, init_outputs_grad[i], init_inputs[i], ranks[i]))
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

    def update(self, outputs_grad, inputs, indices):
        for (layer, output_grad, input) in zip(self.layers, outputs_grad, inputs):
            layer.update(output_grad, input, indices)
