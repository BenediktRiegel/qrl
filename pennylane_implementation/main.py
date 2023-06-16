from numpy import ceil, log2
from torch import save
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
    # sub_iterations = [(50, 2), (50, 1)]
    # sub_iterations = [(100, 2), (100, 1)]
    sub_iterations = [(50, 2), (50, 1)]
    # sub_iterations = [(50, 4)]
    precise = True
    action_qnn_depth = 1
    value_qnn_depth = 1
    value_optimizer = OptimizerEnum.adam
    action_optimizer = OptimizerEnum.adam
    value_lr = 0.05
    action_lr = 0.08
    gamma = 0.5
    eps = 0.0
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    # backend = QuantumBackends.pennylane_lightning_qubit
    shots = 10000
    # shots = None

    if precise:
        backend = QuantumBackends.pennylane_default_qubit
        shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    # slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    slip_probabilities = [1., 0., 0., 0.]
    # map = [
    #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
    #     [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
    #     [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
    #     [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    # ]
    # map = [
    #     [FrozenField.get_hole(), FrozenField.get_end()],
    #     [FrozenField.get_ice(), FrozenField.get_ice()],
    # ]
    # map = [
    #     [FrozenField(reward=-1), FrozenField(reward=1)],
    #     [FrozenField(reward=0), FrozenField(reward=0)]
    # ]
    # map = [[FrozenField(reward=-1), FrozenField(reward=0.5)]]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_hole(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_end(), FrozenField.get_hole()],
    ]
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
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    # action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    # action_qnn = RYQNN_Excessive(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth, WeightInitEnum.standard_normal)
    # value_qnn = CCRYQNN_Excessive(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
    action_qnn = RYQNN_D(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth, WeightInitEnum.standard_normal)
    value_qnn = CCRYQNN_D(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
    print(f"value_qnn parameters: ")
    for p in value_qnn.parameters():
        print(p)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    value_optimizer = value_optimizer.get_optimizer(value_qnn.parameters(), value_lr)
    action_optimizer = action_optimizer.get_optimizer(action_qnn.parameters(), action_lr)
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
    )

    frames, losses = train_with_two_opt(loss_fn, value_optimizer, action_optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits), gamma)] + frames

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
    with open("plots/fig.html", "w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()


if __name__ == "__main__":
    rot_swap_main()
