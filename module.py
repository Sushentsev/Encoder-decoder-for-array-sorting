from tensor import Tensor
from typing import Union, List


class Module:
    def __init__(self):
        self.parameters = []

    def register_parameter(self, param: Tensor):
        self.parameters.append(param)

    def register_parameters(self, param_list_or_module: Union[List[Tensor], "Module", List["Module"]]):
        if isinstance(param_list_or_module, List):
            for element in param_list_or_module:
                if isinstance(element, Tensor):
                    self.register_parameter(element)
                elif isinstance(element, Module):
                    for param in element.parameters:
                        self.register_parameter(param)
                else:
                    raise TypeError("Parameter should be of type Tensor")
        elif isinstance(param_list_or_module, Module):
            for param in param_list_or_module.parameters:
                self.register_parameter(param)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def forward(self, *arg, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
