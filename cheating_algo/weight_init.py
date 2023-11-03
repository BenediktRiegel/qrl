# This file was copied from the project https://github.com/UST-QuAntiL/qhana-plugin-runner in compliance with its license
from enum import Enum
import torch


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"

    def init_params(self, shape, dtype=torch.float64):  # Added this method
        """
        Initialises a torch.tensor according to the chosen WeightInitEnum and the given shape and dtype.
        :param shape: shape that the tensor should have
        :param dtype: dtype the tensor should have
        :return: torch.tensor
        """
        if self == WeightInitEnum.standard_normal:
            parameters = torch.pi * torch.randn(shape, dtype=dtype)
        elif self == WeightInitEnum.uniform:
            parameters = torch.pi * torch.rand(shape, dtype=dtype)
        elif self == WeightInitEnum.zero:
            parameters = torch.zeros(shape, dtype=dtype)
        else:
            raise NotImplementedError("Unknown weight init method")

        return parameters
