from numpy import ceil, log2
from torch import save
from optimizer import OptimizerEnum
from quantum_backends import QuantumBackends
from train import train
from qnns import QNN, RotQNN, CCRotQNN, CCRotQNN2, CCRYQNN
from qnns.weight_init import WeightInitEnum
from environment.frozen_lake import FrozenField, FrozenLake
# from environment.frozen_lake import FrozenLake2
from environment.frozen_lake2 import FrozenLake2
from environment.frozen_lake2_2 import FrozenLake2_2
from environment.frozen_lake5_3 import FrozenLake5_3
from environment.frozen_lake2_3 import FrozenLake2_3
from environment.frozen_lake_rot_swap import FrozenLakeRotSwap
from loss_function import loss_function, loss_function2
from loss_function.rot_loss import loss_function as rot_loss_function
from loss_function.rot_loss2_2 import loss_function as rot_loss_function2_2
from loss_function.rot_swap_loss import loss_function as rot_swap_loss_function
from wire_utils import get_wires
from visualize import plot_frozen_lake, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss


def main():
    num_iterations = 1
    optimizer = OptimizerEnum.adam
    lr = 0.5
    gamma = 0.9
    lam = 1
    backend = QuantumBackends.pennylane_lightning_qubit
    shots = 20
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    r_qubit_interpretation = ["0"]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField(end=True), FrozenField.get_ice(), FrozenField(end=True)],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField(end=True)],
        [FrozenField(end=True), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLake(map, slip_probabilities, r_qubit_interpretation)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    wires, total_num_wires = get_wires(
        [log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation) + 3,
         len(r_qubit_interpretation) + 3, 2, 2])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits, rotation_value_qubits, ancilla_qubits = wires
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), 4)
    value_qnn = QNN(len(x_qubits), len(action_qubits), 4)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    loss_fn = loss_function2

    # loss_function
    # loss_function_params = dict(
    #     action_qnn=action_qnn, value_qnn=value_qnn, environment=environment,
    #     x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
    #     next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits, r_qubits=r_qubits,
    #     value_qubits=value_qubits, next_value_qubits=next_value_qubits,
    #     ancilla_qubits=value_qubits + next_value_qubits,
    #     backend=backend,
    #     gamma=gamma, lam=lam,
    #     unclean_qubits=None,
    # )
    # loss_function2
    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits, r_qubits=r_qubits,
        value_qubits=value_qubits, next_value_qubits=next_value_qubits,
        rotation_value_qubits=rotation_value_qubits,
        ancilla_qubits=ancilla_qubits,
        backend=backend,
        gamma=gamma, lam=lam,
        unclean_qubits=None,
    )

    train(loss_fn, optimizer, num_iterations, [(1, None)], action_qnn, value_qnn, loss_function_params)


def rot_main():
    num_iterations = 5
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(5, 2), (15, 1)]
    # sub_iterations = [(50, 4)]
    action_qnn_depth = 4
    value_qnn_depth = 4
    optimizer = OptimizerEnum.adam
    lr = 0.5
    gamma = 0.8
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLake2(map, slip_probabilities, r_m=1, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 1, 1, 1, 1, 1, 3])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubit, value_qubit, next_value_qubit, loss1_qubit, loss2_qubit, ancilla_qubits = wires
    r_qubit, value_qubit, next_value_qubit, loss1_qubit, loss2_qubit = r_qubit[0], value_qubit[0], next_value_qubit[0], \
        loss1_qubit[0], loss2_qubit[0]
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = RotQNN(len(x_qubits) + len(y_qubits), value_qnn_depth)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    loss_fn = rot_loss_function

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits, r_qubit=r_qubit,
        value_qubit=value_qubit, next_value_qubit=next_value_qubit,
        loss1_qubit=loss1_qubit, loss2_qubit=loss2_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=[],
        backend=backend,
        gamma=gamma, lam=lam,
        use_exp_value=False,
    )

    frames, losses = train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits))] + frames

    for i, param in enumerate(action_qnn.parameters()):
        save(param, f"./action_qnn/param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, f"./value_qnn/param{i}")

    fig = plot_animated_frozen_lake(environment, frames)
    with open("plots/fig.html", "w") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w") as f:
        f.write(fig.to_html())
        f.close()


def rot_main2_2():
    num_iterations = 1
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(1, 2), (1, 1)]
    # sub_iterations = [(50, 4)]
    action_qnn_depth = 4
    value_qnn_depth = 4
    optimizer = OptimizerEnum.adam
    lr = 0.001
    gamma = 0.8
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLake2_2(map, slip_probabilities, r_m=1, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 1, 1, 3])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, loss1_qubit, loss2_qubit, ancilla_qubits = wires
    loss1_qubit, loss2_qubit = loss1_qubit[0], loss2_qubit[0]
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = CCRotQNN(len(x_qubits) + len(y_qubits), value_qnn_depth)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    loss_fn = rot_loss_function2_2

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits,
        loss1_qubit=loss1_qubit, loss2_qubit=loss2_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=[],
        backend=backend,
        gamma=gamma, lam=lam,
        use_exp_value=False,
    )

    frames, losses = train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits))] + frames

    for i, param in enumerate(action_qnn.parameters()):
        save(param, f"./action_qnn/param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, f"./value_qnn/param{i}")

    fig = plot_animated_frozen_lake(environment, frames)
    with open("plots/fig.html", "w") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w") as f:
        f.write(fig.to_html())
        f.close()


def rot_main5_3():
    num_iterations = 80
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(15, 2), (15, 1)]
    # sub_iterations = [(50, 4)]
    action_qnn_depth = 10
    value_qnn_depth = 10
    optimizer = OptimizerEnum.adam
    lr = 0.005
    gamma = 0.8
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLake5_3(map, slip_probabilities, r_m=1, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 1, 1, 4])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, loss1_qubit, loss2_qubit, ancilla_qubits = wires
    loss1_qubit, loss2_qubit = loss1_qubit[0], loss2_qubit[0]
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = CCRotQNN2(len(x_qubits) + len(y_qubits), value_qnn_depth)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    loss_fn = rot_loss_function2_2

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits,
        loss1_qubit=loss1_qubit, loss2_qubit=loss2_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=[],
        backend=backend,
        gamma=gamma, lam=lam,
        use_exp_value=False,
    )

    frames, losses = train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits))] + frames

    for i, param in enumerate(action_qnn.parameters()):
        save(param, f"./action_qnn/param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, f"./value_qnn/param{i}")

    fig = plot_animated_frozen_lake(environment, frames)
    with open("plots/fig.html", "w") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w") as f:
        f.write(fig.to_html())
        f.close()


def rot_main2_3():
    num_iterations = 1
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(1, 2), (1, 1)]
    # sub_iterations = [(50, 4)]
    action_qnn_depth = 4
    value_qnn_depth = 4
    optimizer = OptimizerEnum.adam
    lr = 0.001
    gamma = 0.8
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLake2_3(map, slip_probabilities, r_m=1, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 1, 1, 6])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, loss1_qubit, loss2_qubit, ancilla_qubits = wires
    loss1_qubit, loss2_qubit = loss1_qubit[0], loss2_qubit[0]
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = CCRotQNN(len(x_qubits) + len(y_qubits), value_qnn_depth)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    loss_fn = rot_loss_function2_2

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits,
        loss1_qubit=loss1_qubit, loss2_qubit=loss2_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=[],
        backend=backend,
        gamma=gamma, lam=lam,
        use_exp_value=False,
    )

    frames, losses = train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits))] + frames

    for i, param in enumerate(action_qnn.parameters()):
        save(param, f"./action_qnn/param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, f"./value_qnn/param{i}")

    fig = plot_animated_frozen_lake(environment, frames)
    with open("plots/fig.html", "w") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w") as f:
        f.write(fig.to_html())
        f.close()


def rot_swap_main():
    num_iterations = 40
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = [(5, 2), (5, 1)]
    # sub_iterations = [(50, 4)]
    action_qnn_depth = 4
    value_qnn_depth = 4
    optimizer = OptimizerEnum.adam
    lr = 0.1
    gamma = 0.8
    eps = 0.1
    lam = 0.8
    backend = QuantumBackends.pennylane_lightning_kokkos
    # backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
    environment = FrozenLakeRotSwap(map, slip_probabilities, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 10])
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 8])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, ancilla_qubits = wires
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = CCRYQNN(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
    print(f"value_qnn parameters: ")
    for p in value_qnn.parameters():
        print(p)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
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
    )

    frames, losses = train(loss_fn, optimizer, num_iterations, sub_iterations, action_qnn, value_qnn,
                           loss_function_params)
    frames = [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits))] + frames

    for i, param in enumerate(action_qnn.parameters()):
        save(param, f"./action_qnn/param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, f"./value_qnn/param{i}")

    fig = plot_animated_frozen_lake(environment, frames)
    with open("plots/fig.html", "w") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with open("plots/loss.html", "w") as f:
        f.write(fig.to_html())
        f.close()


def test_rot_swap():
    from loss_function.rot_swap_loss import value_loss
    action_qnn_depth = 4
    value_qnn_depth = 4
    optimizer = OptimizerEnum.adam
    lr = 0.1
    gamma = 0.8
    eps = 0.1
    lam = 0.8
    backend = QuantumBackends.pennylane_default_qubit
    shots = 10000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    map = [
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice()],
        [FrozenField.get_ice(), FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_hole()],
        [FrozenField.get_hole(), FrozenField.get_ice(), FrozenField.get_ice(), FrozenField.get_end()],
    ]
    print("prepare environment")
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
    action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    value_qnn = CCRYQNN(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.zero)

    loss = value_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
        ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3:6], ancilla_qubits[6], ancilla_qubits[7:], backend,
        snaps=True,
    )

    print(f"\n\nloss: {loss}\n")


if __name__ == "__main__":
    # main()
    # rot_main()
    # rot_main2_2()
    # rot_main5_3()
    # rot_main2_3()
    # rot_swap_main()
    test_rot_swap()
