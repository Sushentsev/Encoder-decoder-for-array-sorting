from function import Function
from tensor import Tensor
from module import Module
import numpy as np


class ReLU(Function):
    """
    Implementation of relu(x) = max(0, x)
    """

    def __init__(self, x: Tensor):
        self.x = x

    def __call__(self):
        self.relu = np.maximum(0, self.x.data)
        return Tensor(self.relu, self)

    def backward(self, grad: np.ndarray):
        grad = (self.x.data >= 0) * grad
        self.x.backward(grad)


class ReLUFunction(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return ReLU(x)()
