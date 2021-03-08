from module import Module
from modules import RNNUnit
from tensor import Tensor
from typing import List
import numpy as np


class RNNDecoder(Module):
    """
    Decoder implementation of seq_len RNN units
    """

    def __init__(self, input_size: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.unit = RNNUnit(input_size, hidden_size, input_size)
        self.register_parameters(self.unit)
        self.seq_len = seq_len

    def forward(self, x: Tensor, h: Tensor) -> List[Tensor]:
        output = []

        for _ in range(self.seq_len):
            y, h = self.unit(x, h)
            output.append(y)
            x = np.zeros_like(x.data)
            x[y.data.argmax()] = 1
            x = Tensor(x)

        # Временный костыль, но пока ничего не портит (всего +1 градиент назад)
        h.backward(np.zeros_like(h.data))
        return output
