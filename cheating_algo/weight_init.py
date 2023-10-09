from enum import Enum
import torch


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"

    def init_params(self, shape, dtype=torch.float64):
        if self == WeightInitEnum.standard_normal:
            parameters = torch.pi * torch.randn(shape, dtype=dtype)
        elif self == WeightInitEnum.uniform:
            parameters = torch.pi * torch.rand(shape, dtype=dtype)
        elif self == WeightInitEnum.zero:
            parameters = torch.zeros(shape, dtype=dtype)
        else:
            raise NotImplementedError("Unknown weight init method")

        return parameters
