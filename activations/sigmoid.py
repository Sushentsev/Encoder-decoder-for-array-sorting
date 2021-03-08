from function import Function
from tensor import Tensor
from module import Module
import numpy as np


class Sigmoid(Function):
    """
    Implementation of sigmoid(x) = 1 / (1 + exp(-x))
    """

    def __init__(self, x: Tensor):
        self.x = x

    def __call__(self):
        self.sigmoid = 1.0 / (1.0 + np.exp(-self.x.data))
        return Tensor(self.sigmoid, self)

    def backward(self, grad: np.ndarray):
        grad = self.sigmoid * (1.0 - self.sigmoid) * grad
        self.x.backward(grad)


class SigmoidFunction(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return Sigmoid(x)()
