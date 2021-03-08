from function import Function
from tensor import Tensor
from module import Module
import numpy as np


class Tanh(Function):
    """
    Implementation of tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def __init__(self, x: Tensor):
        self.x = x

    def __call__(self):
        self.tanh = np.tanh(self.x.data)
        return Tensor(self.tanh, self)

    def backward(self, grad: np.ndarray):
        grad = (1 - self.tanh ** 2) * grad
        self.x.backward(grad)


class TanhFunction(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return Tanh(x)()
