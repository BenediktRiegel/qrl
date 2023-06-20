import pandas
import plotly.express
from numpy import ceil, log2
from torch import save, load
import torch
from torch.optim.lr_scheduler import StepLR
from optimizer import OptimizerEnum
from quantum_backends import QuantumBackends
from train import train_with_two_opt
from qnns import *
from qnns.weight_init import WeightInitEnum
from environment.frozen_lake import FrozenField
from environment.frozen_lake_rot_swap import FrozenLakeRotSwap
from loss_function.rot_swap_loss import value_loss
from wire_utils import get_wires
# from visualize import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from visualize.rot_swap_value import get_action_probs, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss


def get_gradient_and_loss(
        action_qnn, value_qnn, param_idx, param_values,
        environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
        value_indices, value_qubit, swap_vector_qubits, loss_qubit, ancilla_qubits, backend, gamma, eps,
        unclean_qubits, precise, end_state_values, diff_method,
):
    losses = []
    gradients = []
    for idx, p in enumerate(param_values, start=1):
        print(f"parameter value {idx}/{len(param_values)}")
        value_qnn.in_q_parameters[param_idx[0]][param_idx[1]] = torch.nn.Parameter(p, requires_grad=False)
        loss = value_loss(
            action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits,
            next_y_qubits, value_indices, value_qubit, swap_vector_qubits, loss_qubit, ancilla_qubits,
            backend, gamma, eps, unclean_qubits, precise, end_state_values, diff_method
        )
        # loss.backward()
        losses.append(loss.item())
        # gradients.append(value_qnn.in_q_parameters[param_idx[0]][param_idx[1]].grad.item())

    return losses, gradients


def save_parameter_plot():
    num_iterations = 1
    precise = True
    end_state_values = False
    action_qnn_depth = 1
    value_qnn_depth = 1
    # value_optimizer_enum = OptimizerEnum.adam
    # action_optimizer_enum = OptimizerEnum.adam
    value_optimizer_enum = OptimizerEnum.sgd
    action_optimizer_enum = OptimizerEnum.sgd
    value_lr = 0.5
    action_lr = 0.5
    gamma = 0.8
    eps = 0.0
    lam = 0.8
    # backend_enum = QuantumBackends.pennylane_lightning_kokkos
    # backend_enum = QuantumBackends.pennylane_default_qubic
    backend_enum = QuantumBackends.pennylane_lightning_qubit
    shots = 10000000
    action_diff_method = "best"
    value_diff_method = "best"
    # action_diff_method = "parameter-shift"
    # value_diff_method = "parameter-shift"
    # shots = None

    if precise:
        backend_enum = QuantumBackends.pennylane_default_qubit
        shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    # slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    slip_probabilities = [1., 0., 0., 0.]
    maps = [
        [
            [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_end()],
            [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        ]
    ]
    for idx, map in enumerate(maps):
        print("prepare environment")
        map = [el for el in reversed(map)]
        environment = FrozenLakeRotSwap(map, slip_probabilities, r_qubit_is_clean=True)

        log_rows = int(ceil(log2(len(map))))
        log_cols = int(ceil(log2(len(map[0]))))

        # Loss function
        # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
        # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
        # Loss function2
        # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 10])
        wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 7])
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, ancilla_qubits = wires
        backend = backend_enum.get_pennylane_backend("", "", total_num_wires, shots)
        print(f"{total_num_wires} qubits")

        print("prepare qnn")
        # action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
        action_qnn = RYQNN_Excessive(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth,
                                     WeightInitEnum.zero)
        for p in action_qnn.parameters():
            p.requires_grad = False

        param_idx = (0, 2)
        add_value = torch.pi / 2.
        # add_value = 0
        value_qnn = CCRYQNN_One(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.zero, param_idx, add_value)

        param_values = torch.arange(0, 4*torch.pi, 0.5)
        # param_values = torch.tensor([torch.pi*2], dtype=torch.float)
        unclean_qubits = []
        losses, gradients = get_gradient_and_loss(
            action_qnn, value_qnn, param_idx, param_values,
            environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
            ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3:6], ancilla_qubits[6], ancilla_qubits[7:],
            backend, gamma, eps, unclean_qubits, precise, end_state_values, value_diff_method,
        )
        losses = torch.tensor(losses)
        # losses /= losses.max()
        # gradients = torch.tensor(gradients)
        # gradients /= 1 if gradients.max() == 1 else gradients.max()
        values = torch.tensor([torch.cos(angle/2.).item() for angle in param_values])
        # values /= values.max()
        true_losses = []
        true_gradients = []
        for p in param_values:
            p = torch.nn.Parameter(p, requires_grad=True)
            loss = torch.square(torch.cos(p/2.) - gamma * torch.cos(torch.zeros(1) + add_value/2.))
            # loss.backward()
            true_losses.append(loss.item())
            # true_gradients.append(p.grad.item())
        true_losses = torch.tensor(true_losses)
        # true_gradients = torch.tensor(true_gradients)
        # true_losses /= true_losses.max()
        # true_gradients /= true_gradients.max()

        print(f"losses: {losses}")
        print(f"true_losses: {true_losses}")

        loss_fig = plotly.express.line(pandas.DataFrame(dict(
            parameter=param_values,
            loss=losses,
            # gradient=gradients,
            value=values,
            true_loss=true_losses,
            # true_gradient=true_gradients,
        #)), x="parameter", y=["loss", "gradient", "value", "true_loss", "true_gradient"])
        )), x="parameter", y=["loss", "value", "true_loss"])
        with open(f"./plots/plotted_parameter_{param_idx}.html", "w", encoding="utf-8") as f:
            f.write(loss_fig.to_html())
            f.close()

        # gradient_fig = plotly.express.line(pandas.DataFrame(dict(
        #     parameter=param_values,
        #     gradient=gradients,
        # )), x="parameter", y="gradient")
        # with open(f"plotted_gradient_for_{param_idx}", "w", encoding="utf-8") as f:
        #     f.write(gradient_fig.to_html())
        #     f.close()

        for param in action_qnn.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    save_parameter_plot()
