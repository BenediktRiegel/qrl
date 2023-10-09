from enum import Enum
import torch.optim as optim


class OptimizerEnum(Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamW = "AdamW"
    # sparse_adam = "SparseAdam"     # "Does not support dense gradiente, please consider Adam instead"
    adamax = "Adamax"
    asgd = "ASGD"
    # lbfgs = "LBFGS"                # "Step() missing 1 required positional argument: closure"
    n_adam = "NAdam"
    r_adam = "RAdam"
    rms_prob = "RMSprop"
    # r_prop = "Rprop"               # AttributeError('Rprop')
    sgd = "SGD"

    def get_optimizer(self, parameters, lr):
        """
        returns the optimizer specified by the enum

        optimizer: optimizer type (OptimizerEnum)
        model: the network to optimize
        lr: learning rate (float)
        """
        if self == OptimizerEnum.adadelta:
            return optim.Adadelta(parameters, lr=lr)
        elif self == OptimizerEnum.adagrad:
            return optim.Adagrad(parameters, lr=lr)
        elif self == OptimizerEnum.adam:
            return optim.Adam(parameters, lr=lr)
        elif self == OptimizerEnum.adamW:
            return optim.AdamW(parameters, lr=lr)
        # elif (
        #     optimizer == OptimizerEnum.sparse_adam
        # ):  # "RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead"
        #     return optim.SparseAdam(model.parameters(), lr=lr)
        elif self == OptimizerEnum.adamax:
            return optim.Adamax(parameters, lr=lr)
        elif self == OptimizerEnum.asgd:
            return optim.ASGD(parameters, lr=lr)
        # elif (
        #     optimizer == OptimizerEnum.lbfgs
        # ):  # step() missing 1 required argument: 'closure'
        #     return optim.LBFGS(model.parameters(), lr=lr)
        elif self == OptimizerEnum.n_adam:
            return optim.NAdam(parameters, lr=lr)
        elif self == OptimizerEnum.r_adam:
            return optim.RAdam(parameters, lr=lr)
        elif self == OptimizerEnum.rms_prob:
            return optim.RMSprop(parameters, lr=lr)
        # elif optimizer == OptimizerEnum.Rprop:
        #     # AttributeError('Rprop')
        #     return optim.Rprop(model.parameters(), lr=step)
        elif self == OptimizerEnum.sgd:
            # AttributeError('Rprop')
            return optim.SGD(parameters, lr=lr)
        else:
            raise NotImplementedError("Unkown optimizer")
