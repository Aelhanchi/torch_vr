import torch


class PrevGradLayer:
    """Base class for the storage and manipulation of the previous per-example
    gradients of a layer"""

    def __init__(self):
        return

    def batch_gradient(self, indices, weights):
        """Computes the weighted sum of the previous
        individual gradients corresponding to the passed indices.

        Arguments:
            indices (Tensor): indices tensor.
            weights (Tensor, optional): weights tensor. Default is'None'
            and assigns a weight of 1 to all examples.

        Returns:
            weighted sum of the previous individual gradients corresponding
            to the passed indices.
        """
        return NotImplementedError

    def norm_gradients(self, indices):
        """Computes the Euclidean norm of the previous
        individual gradients corresponding to the passed indices.

        Arguments:
            indices (Tensor): indices tensor.

        Returns:
            Euclidean norm of the individual gradients corresponding to the
            passed indices.
        """
        return NotImplementedError

    def individual_gradients(self, indices, weights):
        """Computes the weighted previous individual gradients corresponding to the
        passed indices.

        Arguments:
            indices (Tensor): indices tensor.
            weights (Tensor, optional): weights tensor. Default is'None'
            and assigns a weight of 1 to all examples.

        Returns:
            weighted previous individual gradients corresponding to the
            passed indices.
        """
        return NotImplementedError

    def update(self, output_grad, input, indices):
        """Updates the previous per-example gradients with the newly
        computed per-example gradients.

        Arguments:
            indices (Tensor): indices tensor.
            output_grad: gradient of output of layer.
            input: input to layer.
        """
        return NotImplementedError


class PrevGradLinearLayer(PrevGradLayer):
    """Stores, updates, and answers queries about the previous
    per-example gradients of a linear layer.

    Arguments:
        N (int): number of data points.
        layer (nn.Module.Linear): the linear layer.
        init_output_grad (Tensor, optional): initial gradient of output of layer.
        init_input (Tensor, optional): initial input of layer.
    """

    def __init__(self, N, layer, init_output_grad=None, init_input=None):
        super().__init__()
        self.N = N
        if layer.__class__.__name__ != 'Linear':
            raise ValueError('Layer passed is not Linear.')
        self.layer = layer

        if init_output_grad is not None and init_input is not None:
            self.output_grad = init_output_grad.t_()
            self.input = init_input.t_()
        else:
            self.output_grad = torch.zeros(self.layer.out_features, self.N)
            self.input = torch.zeros(self.layer.in_features, self.N)

    def batch_gradient(self, indices, weights=None):
        if weights is None:
            weights = 1.0
        dW = weights * self.output_grad[:, indices] @ self.input.t()[indices, :]
        if self.layer.bias is not None:
            db = (weights * self.output_grad[:, indices]).sum(dim=1)
            return [dW, db]
        else:
            return [dW, ]

    def norm_gradients(self, indices):
        norm_output_grad = self.output_grad[:, indices].pow(2).sum(dim=0)
        norm_input = self.input[:, indices].pow(2).sum(dim=0)
        if self.layer.bias is not None:
            norm_bias = self.output_grad[:, indices].pow(2).sum(dim=0)
        return torch.sqrt(norm_output_grad * norm_input + norm_bias)

    def individual_gradients(self, indices, weights=None):
        if weights is None:
            weights = 1.0
        dW = self.output_grad[:, indices].unsqueeze(1).expand(
            self.layer.out_features, self.layer.in_features, -1)
        dW = (dW * self.input[:, indices])
        dW = (weights * dW).transpose(2, 1).transpose(1, 0)
        if self.layer.bias is not None:
            db = (weights * self.output_grad[:, indices]).t()
            return [dW, db]
        else:
            return [dW, ]

    def update(self, output_grad, input, indices):
        self.output_grad[:, indices] = output_grad.t_()
        self.input[:, indices] = input.t_()


class PrevGradConv2dLayer(PrevGradLayer):
    """Class that stores, updates, and answers queries about the previous
    per-example gradients of a two dimensional convolutional layer.

    Arguments:
        init_dZ (Tensor): initial gradient of output of layer.
        init_A (Tensor): initial input of layer.
        N (int): number of data points.
        layer (nn.Module.Conv2d): the Conv2d layer.
        rank (boolean, optional): The rank of the stored approximation to the
        true gradient. Default is 'None' and stores the full gradient.
    """

    def __init__(self, N, layer, init_output_grad=None, init_input=None, rank=None):
        super().__init__()
        self.N = N
        if layer.__class__.__name__ != 'Conv2d':
            raise ValueError('Layer passed is not Conv2d.')
        self.layer = layer

        if self.layer.bias is not None:
            if init_output_grad is not None:
                self.ind_grads_biases = init_output_grad.flatten(2).sum(dim=2)
            else:
                self.ind_grads_biases = torch.zeros(self.N, self.layer.out_channels)

        self.rank = rank
        if self.rank is None:
            if init_output_grad is not None and init_input is not None:
                self.ind_grads_weights = self._construct_gradients(init_output_grad, init_input)
            else:
                self.ind_grads_weights = torch.zeros(N, self.layer.out_channels, self.layer.in_channels *
                                                     self.layer.kernel_size[0] * self.layer.kernel_size[1])
        else:
            if init_output_grad is not None and init_input is not None:
                ind_grads = self._construct_gradients(init_output_grad, init_input)
                self.U, self.V_T = self._svd_compress(ind_grads)
            else:
                self.U = torch.zeros(self.N, self.layer.out_channels, self.rank)
                self.V_T = torch.zeros(self.N, self.rank, self.layer.in_channels *
                                       self.layer.kernel_size[0] * self.layer.kernel_size[1])

    def _construct_gradients(self, output_grad, input):
        input = torch.nn.functional.unfold(
            input, self.layer.kernel_size, self.layer.dilation, self.layer.padding, self.layer.stride)
        output_grad = output_grad.flatten(2)
        return output_grad @ input.transpose(1, 2)

    def _svd_compress(self, ind_grads):
        if self.rank is None:
            raise ValueError('rank of approximation is None')
        U, S, V = torch.svd(ind_grads)
        V_T = S.unsqueeze(2) * V.transpose(1, 2)
        return (U[:, :, 0:self.rank], V_T[:, 0:self.rank, :])

    def batch_gradient(self, indices, weights=None):
        if weights is None:
            if self.rank is None:
                dW = self.ind_grads_weights[indices, :, :].sum(dim=0)
                dW = self._reshape(dW)
            else:
                dW = (self.U[indices].transpose(0, 1).transpose(1, 2).flatten(1) @
                      self.V_T[indices].transpose(0, 2).flatten(1).transpose(0, 1))
                dW = self._reshape(dW)

            if self.layer.bias is not None:
                db = self.ind_grads_biases[indices, :].sum(dim=0)
                return [dW, db]
            else:
                return [dW, ]
        else:
            if self.rank is None:
                dW = (weights.unsqueeze(1).unsqueeze(2) *
                      self.ind_grads_weights[indices, :, :]).sum(dim=0)
                dW = self._reshape(dW)
            else:
                dW = ((weights.unsqueeze(1).unsqueeze(2) * self.U[indices]).transpose(0, 1).transpose(1, 2).flatten(1) @
                      self.V_T[indices].transpose(0, 2).flatten(1).transpose(0, 1))
                dW = self._reshape(dW)

            if self.layer.bias is not None:
                db = (weights.unsqueeze(1) * self.ind_grads_biases[indices, :]).sum(dim=0)
                return [dW, db]
            else:
                return [dW, ]

    def individual_gradients(self, indices, weights=None):
        if weights is None:
            if self.rank is None:
                dW = self._batch_reshape(self.ind_grads_weights[indices], len(indices))
            else:
                dW = self._batch_reshape(self.U[indices] @ self.V_T[indices], len(indices))
            if self.layer.bias is not None:
                db = self.ind_grads_biases[indices]
                return [dW, db]
            else:
                return [dW, ]
        else:
            if self.rank is None:
                dW = weights.unsqueeze(1).unsqueeze(2) * self.ind_grads_weights[indices]
                dW = self._batch_reshape(dW, len(indices))
            else:
                dW = (weights.unsqueeze(1).unsqueeze(2) * self.U[indices]) @ self.V_T[indices]
                dW = self._batch_reshape(dW, len(indices))

            if self.layer.bias is not None:
                db = weights.unsqueeze(1) * self.ind_grads_biases[indices]
                return [dW, db]
            else:
                return [dW, ]

    def update(self, output_grad, input, indices):
        if self.layer.bias is not None:
            self.ind_grads_biases[indices, :] = output_grad.flatten(2).sum(dim=2)

        ind_grads_weights = self._construct_gradients(output_grad, input)
        if self.rank is None:
            self.ind_grads_weights[indices] = ind_grads_weights
        else:
            self.U[indices], self.V_T[indices] = self._svd_compress(ind_grads_weights)

    def _batch_reshape(self, tensor, batch_size):
        return tensor.reshape(batch_size, -1, self.layer.in_channels,
                              self.layer.kernel_size[0], self.layer.kernel_size[1])

    def _reshape(self, tensor):
        return tensor.reshape(-1, self.layer.in_channels,
                              self.layer.kernel_size[0], self.layer.kernel_size[1])
