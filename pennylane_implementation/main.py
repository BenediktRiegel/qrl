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
from loss_function.rot_swap_loss import loss_function as rot_swap_loss_function
from wire_utils import get_wires
# from visualize import get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from visualize.rot_swap_value import get_action_probs, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss


def rot_swap_main():
    num_iterations = 6
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(50, 2), (50, 1)]
    # sub_iterations = [(2, 2)]
    # sub_iterations = [(50, 4)]
    # sub_iterations = [(1, 2)]
    precise = False
    end_state_values = False
    action_qnn_depth = 1
    value_qnn_depth = 1
    value_optimizer_enum = OptimizerEnum.adam
    action_optimizer_enum = OptimizerEnum.adam
    # value_optimizer_enum = OptimizerEnum.sgd
    # action_optimizer_enum = OptimizerEnum.sgd
    value_lr = 0.5
    action_lr = 0.5
    gamma = 0.8
    eps = 0.0
    lam = 0.8
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
    # map = [
    #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
    #     [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
    #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
    #     [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    # ]
    # map = [
    #     [FrozenField.get_ice(), FrozenField.get_ice()],
    #     [FrozenField.get_hole(), FrozenField.get_end()],
    # ]
    # map = [
    #     [FrozenField(reward=-1), FrozenField(reward=1)],
    #     [FrozenField(reward=0), FrozenField(reward=0)]
    # ]
    # map = [[FrozenField(reward=-1), FrozenField(reward=0.5)]]
    # map = [
    #     [FrozenField.get_ice(), FrozenField.get_ice()],
    #     [FrozenField.get_hole(), FrozenField.get_ice()],
    #     [FrozenField.get_ice(), FrozenField.get_ice()],
    #     [FrozenField.get_end(), FrozenField.get_hole()],
    # ]
    maps = [
        # [
        #     [FrozenField.get_end(), FrozenField.get_ice()],
        # ],
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
        [
            [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_end()],
            [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        ],
        # [
        #     [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
        #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        #     [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        # ],
    ]
    for idx, map in enumerate(maps):
        fig_path = f"plots/test_fig{idx}.html"
        loss_path = f"plots/test_loss{idx}.html"
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
        value_qnn = CCRYQNN_Excessive(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
        # value_qnn.in_q_parameters = load("./value_qnn/param0")
        # action_qnn = RYQNN_D(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth, WeightInitEnum.standard_normal)
        # value_qnn = CCRYQNN_D(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
        print(f"value_qnn parameters: ")
        for p in value_qnn.parameters():
            print(p)
        # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
        value_optimizer = value_optimizer_enum.get_optimizer(value_qnn.parameters(), value_lr)
        value_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
        action_optimizer = action_optimizer_enum.get_optimizer(action_qnn.parameters(), action_lr)
        action_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
        loss_fn = rot_swap_loss_function

        # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
        # fig.show()

        loss_function_params = dict(
            environment=environment,
            x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
            next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=[],
            backend=backend,
            gamma=gamma, lam=lam,
            eps=eps,
            precise=precise,
            end_state_values=end_state_values,
            action_diff_method=action_diff_method,
            value_diff_method=value_diff_method,
        )

        frames, losses = train_with_two_opt(
            loss_fn, value_optimizer, action_optimizer, value_scheduler, action_scheduler,
            num_iterations, sub_iterations, action_qnn, value_qnn,
            loss_function_params, fig_path, loss_path
        )
        frames = frames + [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits), gamma,
                                        end_state_values)]

        for param in action_qnn.parameters():
            param.requires_grad = False
        temp = []
        for y in range(len(map)):
            temp.append([])
            for x in range(len(map[0])):
                probs = get_action_probs(x, y, action_qnn, len(x_qubits), len(y_qubits))
                temp[-1].append({"right": probs[0], "down": probs[1], "left": probs[2], "up": probs[3]})

        print(f"True movements:")
        print([el for el in reversed(temp)])

        for param in action_qnn.parameters():
            param.requires_grad = True

        for i, param in enumerate(action_qnn.parameters()):
            save(param, f"./action_qnn/param{i}")
        for i, param in enumerate(value_qnn.parameters()):
            save(param, f"./value_qnn/param{i}")

        fig = plot_animated_frozen_lake(environment, frames, gamma)
        with open(fig_path, "w", encoding="utf-8") as f:
            f.write(fig.to_html())
            f.close()

        fig = plot_loss(losses)
        with open(loss_path, "w", encoding="utf-8") as f:
            f.write(fig.to_html())
            f.close()


if __name__ == "__main__":
    rot_swap_main()
