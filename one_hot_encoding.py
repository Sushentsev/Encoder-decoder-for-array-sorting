from tensor import Tensor
import numpy as np


def one_hot_encoding(x: int, classes: int) -> Tensor:
    assert x < classes
    encoded = np.zeros((classes, 1))
    encoded[x, 0] = 1
    return Tensor(encoded)
