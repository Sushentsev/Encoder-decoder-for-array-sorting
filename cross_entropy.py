from function import Function
from tensor import Tensor
from module import Module
from softmax import softmax
from typing import List
from one_hot_encoding import one_hot_encoding
import numpy as np


class NLL(Function):
    def __init__(self, y_hat: List[Tensor], y: np.ndarray, eps: float = 1e-15):
        classes = y_hat[0].shape[0]
        self.y_hat = [softmax(y_hat_) for y_hat_ in y_hat]
        self.y = [one_hot_encoding(y_, classes) for y_ in y]
        self.eps = eps

    def __call__(self) -> Tensor:
        loss = 0

        for y_hat_, y_ in zip(self.y_hat, self.y):
            logs = np.log(y_hat_.data + self.eps)
            loss += np.multiply(-y_.data, logs).sum()
        return Tensor(np.array([[loss]]), self)

    def backward(self):
        for y_hat_, y_ in zip(self.y_hat, self.y):
            grad = y_hat_.data - y_.data
            y_hat_.backward(grad)


class CrossEntropyLoss(Module):
    def __init__(self, eps: float = 1e-15):
        super().__init__()
        self.eps = eps

    def forward(self, outputs: List[Tensor], targets: np.ndarray) -> Tensor:
        return NLL(outputs, targets, self.eps)()
