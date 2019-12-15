import torch


class Sequential(torch.nn.Module):
    """An equivalent of the nn.Sequential class that allows the extraction
    of the necessary information to reconstruct the per-example gradients.
    The attributes 'outputs_grad' and 'inputs' contain the required
    information and can be used to construct and maintain a ReduceVar
    object.

    Arguments:
        layers (list): a list of the layers of the neural network, in the order
            of their applications.
        activations (list): a list of the activation functions that should be
            applied after each layers, in the order of their applications.
    """
    def __init__(self, layers, activations):
        super().__init__()
        if len(layers) != len(activations):
            raise ValueError('unequal number of layers and activations')

        self.layers = layers
        self.activations = activations

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

        self.inputs = list()
        self.outputs = list()
        self.outputs_grad = list()

    def hook(self, grad):
        self.outputs_grad = [grad] + self.outputs_grad

    def forward(self, input):
        self.inputs.clear()
        self.outputs_grad.clear()
        self.outputs.clear()

        self.inputs.append(input)
        for (layer, activation) in zip(self.layers, self.activations):
            self.outputs.append(layer(self.inputs[-1]))
            if self.outputs[-1].requires_grad:
                self.outputs[-1].register_hook(self.hook)
            self.inputs.append(activation(self.outputs[-1]))

        output = self.inputs.pop()
        return output
