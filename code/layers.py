import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward_pass(self, input_matrices):
        pass

    def backward_pass(self, loss_gradient, input_matrices):
        pass

    def apply_backprop(self, back_prop):
        pass


class BaseMath:

    @staticmethod
    def tanh(input_matrices):
        return 1 - 2 / (1 + np.exp(input_matrices * 2))

    @staticmethod
    def tanh_derivative(input_matrices):
        th = BaseMath.tanh(input_matrices)
        return 1 - th * th

    @staticmethod
    def exp(input_matrices):
        return np.exp(input_matrices)


class HiddenLayer(Layer):

    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.activation = activation

    def forward_pass(self, input_matrices):
        output = input_matrices[0] @ self.weights
        if self.activation == 'tanh':
            return BaseMath.tanh(output)

    def backward_pass(self, loss_gradient, input_matrices):
        """Returns the gradient w.r.t layer parameters  (dx , dw) """
        if self.activation == 'tanh':
            dz = BaseMath.tanh_derivative(input_matrices[0] @ self.weights) * loss_gradient
        return dz @ self.weights.transpose(), input_matrices[0].transpose() @ dz

    def apply_backprop(self, back_prop):
        self.weights += back_prop


class BatchSumLayer(Layer):

    def forward_pass(self, input_matrices):
        return np.sum(input_matrices[0])

    def backward_pass(self, loss_gradient, input_matrices):
        return (np.ones(input_matrices[0].shape) * loss_gradient, [])


class SoftmaxLayer(Layer):

    def forward_pass(self, input_matrices):
        e = BaseMath.exp(input_matrices[0])
        c = np.sum(e, 1)
        b = (1 / c)
        a = np.diag(b)
        return a @ e

    def backward_pass(self, loss_gradient, input_matrices):
        e = BaseMath.exp(input_matrices[0])
        c = np.sum(e, 1)
        b = (1 / c)
        a = np.diag(b)
        da = loss_gradient @ e.transpose()
        d2e = a.transpose() @ loss_gradient
        db = np.diag(da)
        dc = db * (-b * b)
        dc = dc.reshape((e.shape[0], 1))
        d1e = dc @ np.ones((1, input_matrices[0].shape[1]))
        de = d1e + d2e

        return de * e, []


class CrossEntropyLayer(Layer):

    def forward_pass(self, input_matrices):
        """True class is input_0 and predicted values are input_1"""
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return np.sum(- true_class * np.log(predicted_values) , 1).reshape(predicted_values.shape[0] ,1)

    def backward_pass(self, loss_gradient, input_matrices):
        """True class is input_0 and predicted values are input_1"""
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return - loss_gradient * np.log(predicted_values), - loss_gradient * true_class * (1 / predicted_values) , []