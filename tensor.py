from typing import Optional, Tuple
from function import Function
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, func: Optional[Function] = None):
        self.data = data
        self.grad = np.zeros_like(data)
        self.func = func

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is not None:
            assert grad.shape == self.grad.shape
            self.grad += grad
            if self.func:
                self.func.backward(grad)
        else:
            if self.func:
                self.func.backward()

    def zero_grad(self):
        self.grad[:] = 0.0

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __str__(self) -> str:
        return str(self.data)
