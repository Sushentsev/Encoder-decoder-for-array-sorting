from module import Module
from tensor import Tensor
from encoders import RNNEncoder
from decoders import RNNDecoder
from typing import List
import numpy as np


class RNNSortingNet(Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.encoder = RNNEncoder(input_size, hidden_size, seq_len)
        self.decoder = RNNDecoder(input_size, hidden_size, seq_len)
        self.register_parameters([self.encoder, self.decoder])

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        assert self.seq_len == len(x)
        h_0 = Tensor(np.zeros((self.hidden_size, 1)))
        BOS = Tensor(np.zeros((self.input_size, 1)))

        encoder_h = self.encoder(x, h_0)
        decoder_output = self.decoder(BOS, encoder_h)
        return decoder_output
