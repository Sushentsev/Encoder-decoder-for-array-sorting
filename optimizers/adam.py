from optimizers.optimizer import Optimizer
from tensor import Tensor
from typing import List, Tuple


class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps

    def step(self):
        raise NotImplementedError
