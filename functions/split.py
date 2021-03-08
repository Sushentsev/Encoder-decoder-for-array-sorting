from function import Function
from typing import List
from tensor import Tensor
import numpy as np


class Split(Function):

    def __init__(self, x: Tensor, times: int):
        self.x = x
        self.times = times
        self.grad = []

    def __call__(self) -> List[Tensor]:
        output = []

        for _ in range(self.times):
            x_ = Tensor(self.x.data, self)
            output.append(x_)

        return output

    def backward(self, grad: np.ndarray):
        self.grad.append(grad)

        if len(self.grad) == self.times:
            grad_acc = self.grad[0]

            for grad in self.grad[1:]:
                grad_acc += grad

            self.grad.clear()
            self.x.backward(grad_acc)
