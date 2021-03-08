from tensor import Tensor
import numpy as np


def softmax(x: Tensor):
    data = x.data
    sm = np.exp(data - data.max()) / np.sum(np.exp(data - data.max()))
    return Tensor(sm, x.func)
