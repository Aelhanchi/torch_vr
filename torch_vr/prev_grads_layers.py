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

    def update(self, dZ, A, indices):
        """Updates the previous per-example gradients with the newly
        computed per-example gradients.

        Arguments:
            indices (Tensor): indices tensor.
            dZ: gradient of output of layer.
            A: input to layer.
        """
        return NotImplementedError


class PrevGradLinearLayer(PrevGradLayer):
    """Stores, updates, and answers queries about the previous
    per-example gradients of a linear layer.

    Arguments:
        init_dZ (Tensor): initial gradient of output of layer.
        init_A (Tensor): initial input of layer.
        bias (boolean): whether the layer has a bias paramter or not.
    """

    def __init__(self, init_dZ, init_A, bias=True):
        super().__init__()
        self.dZ = init_dZ.t()
        self.A = init_A.t()
        self.d_in = self.A.shape[0]
        self.d_out = self.dZ.shape[0]
        self.bias = bias

    def batch_gradient(self, indices, weights=None):
        if weights is None:
            weights = 1.0
        dW = weights * self.dZ[:, indices] @ self.A.t()[indices, :]
        if self.bias:
            db = (weights * self.dZ[:, indices]).sum(dim=1)
            return [dW, db]
        else:
            return [dW, ]

    def individual_gradients(self, indices, weights=None):
        dW = self.dZ[:, indices].unsqueeze(1).expand(self.d_out, self.d_in, -1)
        dW = (dW * self.A[:, indices])
        if weights is None:
            weights = 1.0
        dW = (weights * dW).transpose(2, 1).transpose(1, 0)
        if self.bias:
            db = (weights * self.dZ[:, indices]).t()
            return [dW, db]
        else:
            return [dW, ]

    def update(self, dZ, A, indices):
        self.dZ[:, indices] = dZ.t()
        self.A[:, indices] = A.t()


class PrevGradConv2DLayer(PrevGradLayer):
    """Class that stores, updates, and answers queries about the previous
    per-example gradients of a two dimensional convolutional layer.

    Arguments:
        init_dZ (Tensor): initial gradient of output of layer.
        init_A (Tensor): initial input of layer.
        rank (boolean, optional): The rank of the stored approximation to the
        true gradient. Default is 'None' and stores the full gradient.
    """

    def __init__(self, N, kernel_size, c_out, c_in, init_dZ=None, init_A=None, dilation=1, padding=0, stride=1, rank=None, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.c_in = c_in
        self.c_out = c_out

        self.bias = bias
        if self.bias:
            if init_dZ is not None:
                self.ind_grads_biases = init_dZ.flatten(2).sum(dim=2)
            else:
                self.ind_grads_biases = torch.zeros(N, self.c_out)

        self.rank = rank
        if self.rank is None:
            if init_dZ is not None and init_A is not None:
                ind_grads = self._construct_gradients(init_dZ, init_A)
            else:
                ind_grads = torch.zeros(N, self.c_out, self.c_in,
                                        self.kernel_size[0], self.kernel_size[1])
            self.ind_grads_weights = ind_grads
        else:
            if init_dZ is not None and init_A is not None:
                ind_grads = self._construct_gradients(init_dZ, init_A)
                self.U, self.V_T = self._svd_compress(ind_grads)
            else:
                self.U, self.V_T = (torch.zeros(N, self.c_out, self.c_out), torch.zeros(
                    N, self.c_out, self.c_in * self.kernel_size[0] * self.kernel_size[1]))

    def _construct_gradients(self, dZ, A):
        A = torch.nn.functional.unfold(
            A, self.kernel_size, self.dilation, self.padding, self.stride)
        dZ = dZ.flatten(2)
        return dZ @ A.transpose(1, 2)

    def _svd_compress(self, ind_grads):
        if self.rank is None:
            raise ValueError('rank of approximation is None')
        else:
            U, S, V = torch.svd(ind_grads)
            V_T = S.unsqueeze(2) * V.transpose(1, 2)
            return (U[:, :, 0:self.rank], V_T[:, 0:self.rank, :])

    def batch_gradient(self, indices, weights=None):
        if weights is None:
            if self.rank is None:
                dW = self.ind_grads_weights[indices, :, :].sum(dim=0)
                dW = dW.reshape(-1, self.c_in, self.kernel_size[0], self.kernel_size[1])
            else:
                dW = (self.U[indices].transpose(0, 1).transpose(1, 2).flatten(1) @
                      self.V_T[indices].transpose(0, 2).flatten(1).transpose(0, 1))
                dW = dW.reshape(-1, self.c_in, self.kernel_size[0], self.kernel_size[1])

            if self.bias:
                db = self.ind_grads_biases[indices, :].sum(dim=0)
                return [dW, db]
            else:
                return [dW, ]

        else:
            if self.rank is None:
                dW = (weights.unsqueeze(1).unsqueeze(2) *
                      self.ind_grads_weights[indices, :, :]).sum(dim=0)
                dW = dW.reshape(-1, self.c_in, self.kernel_size[0], self.kernel_size[1])
            else:
                dW = ((weights.unsqueeze(1).unsqueeze(2) * self.U[indices]).transpose(0, 1).transpose(1, 2).flatten(1) @
                      self.V_T[indices].transpose(0, 2).flatten(1).transpose(0, 1))
                dW = dW.reshape(-1, self.c_in, self.kernel_size[0], self.kernel_size[1])
            if self.bias:
                db = (weights.unsqueeze(1) * self.ind_grads_biases[indices, :]).sum(dim=0)
                return [dW, db]
            else:
                return [dW, ]

    def individual_gradients(self, indices, weights=None):
        if weights is None:
            if self.rank is None:
                dW = self.ind_grads_weights[indices].reshape(
                    len(indices), -1, self.c_in, self.kernel_size[0], self.kernel_size[1])
            else:
                dW = (self.U[indices] @ self.V_T[indices]).reshape(
                    len(indices), -1, self.c_in, self.kernel_size[0], self.kernel_size[1])
            if self.bias:
                db = self.ind_grads_biases[indices]
                return [dW, db]
            else:
                return [dW, ]
        else:
            if self.rank is None:
                dW = weights.unsqueeze(1).unsqueeze(2) * self.ind_grads_weights[indices]
                dW = dW.reshape(len(indices), -1, self.c_in,
                                self.kernel_size[0], self.kernel_size[1])
            else:
                dW = (weights.unsqueeze(1).unsqueeze(2) * self.U[indices]) @ self.V_T[indices]
                dW = dW.reshape(len(indices), -1, self.c_in,
                                self.kernel_size[0], self.kernel_size[1])
            if self.bias:
                db = weights.unsqueeze(1) * self.ind_grads_biases[indices]
                return [dW, db]
            else:
                return [dW, ]

    def update(self, dZ, A, indices):
        if self.bias:
            self.ind_grads_biases[indices, :] = dZ.flatten(2).sum(dim=2)

        ind_grads_weights = self._construct_gradients(dZ, A)
        if self.rank is None:
            self.ind_grads_weights[indices] = ind_grads_weights
        else:
            self.U[indices], self.V_T[indices] = self._svd_compress(ind_grads_weights)
