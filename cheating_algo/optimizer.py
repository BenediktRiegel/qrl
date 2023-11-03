# This file was copied from the project https://github.com/UST-QuAntiL/qhana-plugin-runner in compliance with its license
from enum import Enum
import torch.optim as optim


class OptimizerEnum(Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamW = "AdamW"     # Modified: Deleted comment
    adamax = "Adamax"
    asgd = "ASGD"       # Modified: Deleted comment
    n_adam = "NAdam"
    r_adam = "RAdam"
    rms_prob = "RMSprop"    # Modified: Deleted comment
    sgd = "SGD"     # Modified: Uncommented

    def get_optimizer(self, parameters, lr):
        """
        returns the optimizer specified by the enum

        :param optimizer: optimizer type (OptimizerEnum)
        :param model: the network to optimize
        :param lr: learning rate (float)
        """
        if self == OptimizerEnum.adadelta:
            return optim.Adadelta(parameters, lr=lr)
        elif self == OptimizerEnum.adagrad:
            return optim.Adagrad(parameters, lr=lr)
        elif self == OptimizerEnum.adam:
            return optim.Adam(parameters, lr=lr)
        elif self == OptimizerEnum.adamW:
            return optim.AdamW(parameters, lr=lr)   # Modified: Deleted block of comments
        elif self == OptimizerEnum.adamax:
            return optim.Adamax(parameters, lr=lr)
        elif self == OptimizerEnum.asgd:
            return optim.ASGD(parameters, lr=lr)    # Modified: Deleted block of comments
        elif self == OptimizerEnum.n_adam:
            return optim.NAdam(parameters, lr=lr)
        elif self == OptimizerEnum.r_adam:
            return optim.RAdam(parameters, lr=lr)
        elif self == OptimizerEnum.rms_prob:
            return optim.RMSprop(parameters, lr=lr)     # Modified: Deleted block of comments
        elif self == OptimizerEnum.sgd:
            return optim.SGD(parameters, lr=lr)
        else:
            raise NotImplementedError("Unkown optimizer")
