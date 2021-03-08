from tensor import Tensor
from function import Function
import numpy as np


class Linear(Function):
    """
    Implementation of linear function y_t = W_hy * h_t + b_y
    """

    def __init__(self, h_curr: Tensor, W_hy: Tensor, b_y: Tensor):
        self.h_curr = h_curr
        self.W_hy = W_hy
        self.b_y = b_y

    def __call__(self) -> Tensor:
        output = np.dot(self.W_hy.data, self.h_curr.data) + self.b_y.data
        return Tensor(output, self)

    def backward(self, dy: np.ndarray):
        dW_xy = np.dot(dy, self.h_curr.data.T)
        db_y = dy
        dh_curr = np.dot(self.W_hy.data.T, dy)
        self.W_hy.backward(dW_xy)
        self.b_y.backward(db_y)
        self.h_curr.backward(dh_curr)


class HiddenLinear(Function):
    """
    Implementation of linear function for hidden state h_raw = W_hh * h_{t-1} + W_xh * x_t + b_h
    """

    def __init__(self, x: Tensor, h_prev: Tensor, W_hh: Tensor, W_xh: Tensor, b_h: Tensor):
        self.x = x
        self.h_prev = h_prev
        self.W_hh = W_hh
        self.W_xh = W_xh
        self.b_h = b_h

    def __call__(self):
        output = np.dot(self.W_hh.data, self.h_prev.data) + np.dot(self.W_xh.data, self.x.data) + self.b_h.data
        return Tensor(output, self)

    def backward(self, dh_raw: np.ndarray):
        dW_hh = np.dot(dh_raw, self.h_prev.data.T)
        dW_xh = np.dot(dh_raw, self.x.data.T)
        db_h = dh_raw
        dx = np.dot(self.W_xh.data.T, dh_raw)
        dh_prev = np.dot(self.W_hh.data.T, dh_raw)
        self.W_hh.backward(dW_hh)
        self.W_xh.backward(dW_xh)
        self.b_h.backward(db_h)
        self.x.backward(dx)
        self.h_prev.backward(dh_prev)
