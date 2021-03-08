class Function:
    def __call__(self) -> "Tensor":
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
