import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward_pass(self, input_matrices):
        pass

    def backward_pass(self, loss_gradient, input_matrices):
        pass

    def apply_backprop(self, gradient, learning_rate=1):
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

    @staticmethod
    def relu(input_matrix):
        return np.maximum(0, input_matrix)


class HiddenLayer(Layer):

    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.activation = activation

    def forward_pass(self, input_matrices):
        output = input_matrices[0] @ self.weights
        if self.activation == 'tanh':
            return BaseMath.tanh(output)
        if self.activation == 'relu':
            return BaseMath.relu(output)
        if self.activation == 'linear':
            return output

    def backward_pass(self, loss_gradient, input_matrices):
        """Returns the gradient w.r.t layer parameters  (dx , dw) """
        if self.activation == 'tanh':
            dz = BaseMath.tanh_derivative(input_matrices[0] @ self.weights) * loss_gradient
        if self.activation == 'relu':
            dz = np.where(input_matrices[0] @ self.weights > 0, 1, 0) * loss_gradient
        if self.activation == 'linear':
            dz = loss_gradient
        return dz @ self.weights.transpose(), input_matrices[0].transpose() @ dz

    def apply_backprop(self, gradient, learning_rate=1):
        self.weights -= gradient * learning_rate


class BatchSumLayer(Layer):

    def forward_pass(self, input_matrices):
        return np.sum(input_matrices[0]) / (input_matrices[0].shape[0] )

    def backward_pass(self, loss_gradient, input_matrices):
        return (np.ones(input_matrices[0].shape) * loss_gradient / input_matrices[0].shape[0], [])

class RowSumLayer(Layer):

    def forward_pass(self, input_matrices):
        return np.sum(input_matrices , 1).reshape((input_matrices[0].shape[0] , 1))

    def backward_pass(self, loss_gradient, input_matrices):
        return  np.tile(loss_gradient , (1 , input_matrices[0].shape[1])) , []


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


class RMSLayer(Layer):

    def forward_pass(self, input_matrices):
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return np.sum((true_class - predicted_values) * (true_class - predicted_values), 1)

    def backward_pass(self, loss_gradient, input_matrices):
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return (true_class - predicted_values) * 2, (predicted_values - true_class) * 2, []


class ConcatenateLayer(Layer):

    def forward_pass(self, input_matrices):
        return np.concatenate(tuple(input_matrices), 1)

    def backward_pass(self, loss_gradient, input_matrices):
        if len(input_matrices) == 1:
            return input_matrices[0]
        split_point = [input_matrices[0].shape[1]]

        for i in range(1, len(input_matrices) - 1):
            split_point.append(split_point[-1] + input_matrices[i].shape[1])
        return np.split(loss_gradient, split_point, 1)


class BiasLayer(Layer):

    def __init__(self,  default_value=1):
        self.defualt_value = default_value
    def forward_pass(self, input_matrices):
        return np.concatenate((input_matrices[0], np.ones((input_matrices[0].shape[0], 1))) , 1)

    def backward_pass(self, loss_gradient, input_matrices):
        return loss_gradient[: , 0:-1],[]


class CrossEntropyLayer(Layer):

    def forward_pass(self, input_matrices):
        """True class is input_0 and predicted values are input_1"""
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return np.sum(- true_class * np.log(predicted_values), 1).reshape(predicted_values.shape[0], 1)

    def backward_pass(self, loss_gradient, input_matrices):
        """True class is input_0 and predicted values are input_1"""
        true_class = input_matrices[0]
        predicted_values = input_matrices[1]
        return - loss_gradient * np.log(predicted_values), - loss_gradient * true_class * (1 / predicted_values), []
