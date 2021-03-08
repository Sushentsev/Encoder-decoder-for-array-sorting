from tensor import Tensor
from functions import HiddenLinear, Split, Linear
from activations import ReLUFunction
from module import Module
import numpy as np


class RNNHiddenUnit(Module):
    """
    Implementation of RNN unit without output value
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(RNNHiddenUnit, self).__init__()
        self.W_hh = Tensor(np.random.normal(loc=0, scale=0.1, size=(hidden_size, hidden_size)))
        self.W_xh = Tensor(np.random.normal(loc=0, scale=0.1, size=(hidden_size, input_size)))
        self.b_h = Tensor(np.random.normal(loc=0, scale=0.1, size=(hidden_size, 1)))
        self.register_parameters([self.W_hh, self.W_xh, self.b_h])
        self.act = ReLUFunction()

    def forward(self, x: Tensor, h_prev: Tensor):
        h_raw = HiddenLinear(x, h_prev, self.W_hh, self.W_xh, self.b_h)()
        h_curr = self.act(h_raw)
        return h_curr


class RNNUnit(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_unit = RNNHiddenUnit(input_size, hidden_size)
        self.W_hy = Tensor(np.random.normal(loc=0, scale=0.1, size=(output_size, hidden_size)))
        self.b_y = Tensor(np.random.normal(loc=0, scale=0.1, size=(output_size, 1)))
        self.register_parameters([self.hidden_unit, self.W_hy, self.b_y])

    def forward(self, x: Tensor, h_prev: Tensor):
        h_curr = self.hidden_unit(x, h_prev)
        h_curr1, h_curr2 = Split(h_curr, times=2)()
        y = Linear(h_curr1, self.W_hy, self.b_y)()
        return y, h_curr2
