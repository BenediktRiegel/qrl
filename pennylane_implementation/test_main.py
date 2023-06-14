from numpy import ceil, log2
from optimizer import OptimizerEnum
from quantum_backends import QuantumBackends
from qnns import QNN, CCRYQNN
from qnns.weight_init import WeightInitEnum
from environment.frozen_lake import FrozenField
from environment.frozen_lake_rot_swap import FrozenLakeRotSwap
from wire_utils import get_wires


def test_rot_swap_value():
    from loss_function.rot_swap_loss import value_loss
    action_qnn_depth = 4
    value_qnn_depth = 1
    optimizer = OptimizerEnum.adam
    lr = 0.1
    gamma = 0.8
    eps = 0.1
    lam = 0.8
    backend = QuantumBackends.pennylane_default_qubit
    shots = 100000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    # slip_probabilities = [1., 0., 0., 0.]
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


def test_rot_swap_action():
    from loss_function.rot_swap_loss import action_loss
    action_qnn_depth = 4
    value_qnn_depth = 1
    optimizer = OptimizerEnum.adam
    lr = 0.1
    gamma = 0.9
    eps = 0.1
    lam = 0.8
    backend = QuantumBackends.pennylane_default_qubit
    shots = 100000
    # shots = None
    # main direction, next are the directions in clockwise order,
    # e.g. main direction is right, then slip probs correspond to [right, down, left, up]
    # slip_probabilities = [1. / 3., 1. / 3., 0., 1. / 3.]
    # slip_probabilities = [0.5, 0.25, 0., 0.25]
    slip_probabilities = [1., 0., 0., 0.]
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

    loss = action_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
        ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3], ancilla_qubits[4:], backend,
        gamma, unclean_qubits=[],
        snaps=True
    )

    print(f"\n\nloss: {loss}\n")


if __name__ == "__main__":
    test_rot_swap_value()
    # test_rot_swap_action()
