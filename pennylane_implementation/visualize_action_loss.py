from numpy import ceil, log2
import plotly.graph_objects as go
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
from loss_function.rot_swap_loss import loss_function as rot_swap_loss_function, action_loss
from wire_utils import get_wires
# from visualize import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from visualize.rot_swap_value import get_action_probs, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss


def visualize_action_loss():
    precise = True
    end_state_values = False
    action_qnn_depth = 1
    value_qnn_depth = 1
    gamma = 0.8
    backend_enum = QuantumBackends.pennylane_lightning_kokkos
    # backend_enum = QuantumBackends.pennylane_default_qubic
    # backend_enum = QuantumBackends.pennylane_lightning_qubit
    shots = 100000
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
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    # slip_probabilities = [1., 0., 0., 0.]

    maps = [
        [
            [FrozenField.get_end(), FrozenField.get_ice()],
        ],
        # [
        #     [FrozenField.get_ice(), FrozenField.get_ice()],
        #     [FrozenField.get_hole(), FrozenField.get_end()],
        # ],
        # [
        #     [FrozenField.get_end(), FrozenField.get_ice(), FrozenField.get_hole()]
        # ],
        # [
        #     [FrozenField.get_ice(), FrozenField.get_ice()],
        #     [FrozenField.get_hole(), FrozenField.get_ice()],
        #     [FrozenField.get_ice(), FrozenField.get_ice()],
        #     [FrozenField.get_end(), FrozenField.get_hole()],
        # ],
        # [
        #     [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_end()],
        #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        # ],
        # [
        #     [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
        #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        #     [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        # ],
    ]
    for idx, map in enumerate(maps):
        plot_path = f"./plots/loss{idx}_mapped.html"
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
                                     WeightInitEnum.standard_normal)
        # action_qnn.in_q_parameters = load("./action_qnn/param0")
        action_qnn.in_q_parameters.requires_grad = False
        # action_qnn.in_q_parameters[0, 0, 2] = 0
        value_qnn = CCRYQNN_Excessive(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
        # value_qnn.in_q_parameters = load("./value_qnn/param0")
        # action_qnn = RYQNN_D(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth, WeightInitEnum.standard_normal)
        # value_qnn = CCRYQNN_D(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)

        params = torch.arange(0, 4*torch.pi, step=torch.pi*0.125)

        action_losses = torch.empty((len(params), len(params)))
        for i1, p1 in enumerate(params):
            action_qnn.in_q_parameters[0, 0, 0] = p1
            for i2, p2 in enumerate(params):
                print(f"{i1+1}/{len(params)}, {i2+1}/{len(params)}")
                action_qnn.in_q_parameters[0, 0, 1] = p2
                action_losses[i1, i2] = action_loss(
                    action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
                    ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3], ancilla_qubits[4:], backend,
                    gamma, unclean_qubits=[], precise=False, end_state_values=end_state_values,
                    diff_method=action_diff_method,
                )

        fig = go.Figure(go.Heatmap(
            x=params,
            y=params,
            z=action_losses,
        ))
        with open(plot_path, "w", encoding="utf-8") as f:
            f.write(fig.to_html())
            f.close()


if __name__ == "__main__":
    visualize_action_loss()
