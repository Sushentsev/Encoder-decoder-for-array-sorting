from optimizers.optimizer import Optimizer
from tensor import Tensor
from typing import List


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
