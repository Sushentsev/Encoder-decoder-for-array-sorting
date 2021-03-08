from module import Module
from modules import RNNHiddenUnit
from tensor import Tensor
from typing import List


class RNNEncoder(Module):
    """
    Encoder implementation of seq_len RNN units
    """
    def __init__(self, input_size: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.unit = RNNHiddenUnit(input_size, hidden_size)
        self.register_parameters(self.unit)
        self.seq_len = seq_len

    def forward(self, x: List[Tensor], h: Tensor) -> Tensor:
        assert self.seq_len == len(x)

        for embedding in x:
            h = self.unit.forward(embedding, h)

        return h
