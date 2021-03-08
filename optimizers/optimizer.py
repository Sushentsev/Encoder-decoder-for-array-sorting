from typing import List
from tensor import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.001):
        self.params = params
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
