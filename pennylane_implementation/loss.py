from typing import List
import pennylane as qml
from little_tree_loader import LittleTreeLoader
from copy import deepcopy
import numpy as np
import torch
from qnns import QNN
from frozen_lake import FrozenLake
from ccnot import adaptive_ccnot


def swap_test(swap_qubit: int, reg1: List[int], reg2: List[int]):
    """
    Performs a swap test, using swap_qubit as the control qubit and reg1 and reg2 are swapped
    :param swap_qubit: int single qubit
    :param reg1: list of qubits
    :param reg2: list of qubits
    """
    qml.Hadamard((swap_qubit,))
    for reg1_q, reg2_q in zip(reg1, reg2):
        qml.CSWAP(wires=(swap_qubit, reg1_q, reg2_q))
    qml.Hadamard((swap_qubit,))


def ccry(c_qubits: List[int], phi: float, target_qubit: int, ancilla_qubits: List[int], unclean_qubits: List[int]):
    """
    Implements a RY operation that may be controlled by an arbitrary amount of control qubits. This is done,
    be first using a ccnot controlled by c_qubits to flip an ancilla qubit h_0. Next we perform a controlled RY operation.
    It is controlled by qubit h_0. Then we undo the first ccnot.
    """
    adaptive_ccnot(c_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])
    qml.CRY(phi=phi, wires=(ancilla_qubits[0], target_qubit))
    adaptive_ccnot(c_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])


def value_loss_circuit(
        action_qnn: QNN, value_qnn: QNN, gradient_free_value_qnn: QNN, environment: FrozenLake,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int], unclean_qubits: List[int] = None,
        gamma: float = 0.9, eps: float = 0.0,
        end_state_values: bool = False,
):
    """
    Implements the value loss circuit for amplitude encoding presented in Quantum Reinforcement Learning using Entangled States,
    by Benedikt Riegel
    """
    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    ancilla_qubits = swap_vector_qubits + [loss_qubit] + ancilla_qubits
    # Determine next action
    if eps != 0:
        eps_vec = np.array([[eps / (2 ** len(action_qubits))] * (2 ** len(action_qubits))])
        eps_vec[0] += (1 - eps)
        eps_vec /= np.linalg.norm(eps_vec)
        LittleTreeLoader(
            eps_vec, action_qubits,
            ancilla_wires=ancilla_qubits,
            unclean_wires=unclean_qubits + x_qubits + y_qubits + next_x_qubits + next_y_qubits
        ).circuit()
    action_qnn.circuit(x_qubits + y_qubits, action_qubits, ancilla_qubits=ancilla_qubits, unclean_qubits=unclean_qubits)
    qml.Snapshot("Chose action")

    # Load Value Indices
    LittleTreeLoader(
        np.array([[0, 1, 1, 1]]) / np.sqrt(3), value_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + next_x_qubits + next_y_qubits + action_qubits
    ).circuit()

    # r_factor = [1 / v_max]
    r_factor = [1]

    # Transistion
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, value_indices_qubits, [value_qubit], r_factor,
        ancilla_qubits, unclean_qubits
    )
    qml.Snapshot(f"After env. Set r on indices {''.join([str(el) for el in value_indices_qubits])}=11")

    # Load values
    qml.PauliX((value_indices_qubits[0],))
    end_state_qubit = []
    if not end_state_values:
        end_state_qubit = ancilla_qubits[0:1]
        ancilla_qubits = ancilla_qubits[1:]

        # State Value
        environment.check_end_state(
            x_qubits, y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + next_x_qubits + next_y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ccry(end_state_qubit + value_indices_qubits, torch.pi, value_qubit, ancilla_qubits, unclean_qubits)
        qml.PauliX((end_state_qubit[0],))
    qml.Snapshot("Before value_qnn")
    value_qnn.circuit(
        x_qubits + y_qubits, value_qubit,
        control_qubits=end_state_qubit + value_indices_qubits,
        ancilla_qubits=ancilla_qubits, unclean_qubits=next_x_qubits + next_y_qubits + action_qubits
    )
    if not end_state_values:
        qml.PauliX((end_state_qubit[0],))
        environment.check_end_state(
            x_qubits, y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + next_x_qubits + next_y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
    qml.PauliX((value_indices_qubits[0],))
    qml.Snapshot(f"Set v(s) on indices {''.join([str(el) for el in value_indices_qubits])}=10")

    # Next State Value
    qml.PauliX((value_indices_qubits[1],))
    if not end_state_values:
        environment.check_end_state(
            next_x_qubits, next_y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ccry(end_state_qubit + value_indices_qubits, torch.pi, value_qubit, ancilla_qubits, unclean_qubits)
        qml.PauliX((end_state_qubit[0],))
    gradient_free_value_qnn.circuit(
        next_x_qubits + next_y_qubits, value_qubit,
        control_qubits=end_state_qubit + value_indices_qubits,
        ancilla_qubits=ancilla_qubits, unclean_qubits=x_qubits + y_qubits + action_qubits
    )
    if not end_state_values:
        qml.PauliX((end_state_qubit[0],))
        environment.check_end_state(
            next_x_qubits, next_y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ancilla_qubits = end_state_qubit + ancilla_qubits
    qml.PauliX((value_indices_qubits[1],))
    qml.Snapshot(f"Set v(s') on indices {''.join([str(el) for el in value_indices_qubits])}=01")

    # Load swap vector
    ancilla_qubits = ancilla_qubits[len(swap_vector_qubits):]
    vector = np.array([[0, 0, 1, 0, -1 * gamma, 0, -r_max/v_max, 0]])
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm
    LittleTreeLoader(
        vector, swap_vector_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + action_qubits + next_x_qubits + next_y_qubits
    ).circuit()
    qml.Snapshot(f"Loaded swap vector")

    # ancilla_qubits = ancilla_qubits[1:]  # Remove loss_qubit
    swap_test(loss_qubit, value_indices_qubits + [value_qubit], swap_vector_qubits)
    qml.Snapshot(f"Swap test")


def value_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        value_indices: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9, eps: float = 0.0,
        unclean_qubits: List[int] = None,
        end_state_values: bool = False,
        diff_method: str = "best",
):
    """
    Computes the value loss, by loading in all states with the same probability
    and then sampling the loss qubit of the value loss circuit. Then scales by the
    correct factors.
    """
    unclean_qubits = [] if unclean_qubits is None else unclean_qubits
    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))

        value_loss_circuit(
            action_qnn, value_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits,
            value_indices, value_qubit,
            swap_vector_qubits, loss_qubit,
            ancilla_qubits, unclean_qubits,
            gamma, eps,
            end_state_values=end_state_values,
        )

        return qml.probs(wires=loss_qubit)

    # backprop, parameter-shift
    result = qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()

    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    vector = np.array([[0, 0, 1, 0, -1 * gamma, 0, -r_max / v_max, 0]])
    vector_norm = np.linalg.norm(vector[0])

    return (result[0] - result[1]) * 3 * (v_max ** 2) * vector_norm**2  # * (2**(len(x_qubits) + len(y_qubits)))


def action_loss_circuit(
        action_qnn: QNN, gradient_free_value_qnn: QNN, environment: FrozenLake,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int, loss_qubit: int,
        ancilla_qubits: List[int], unclean_qubits: List[int] = None,
        gamma: float = 0.9, end_state_values: bool = False
):
    """
    Implements the action loss circuit for amplitude encoding presented in Quantum Reinforcement Learning using Entangled States,
    by Benedikt Riegel
    """
    ancilla_qubits = action_qubits + [loss_qubit] + [value_qubit] + ancilla_qubits + value_indices_qubits[1:]
    value_indices_qubits = value_indices_qubits[:1]  # Only one qubit needed to achieve (q_value, -, reward, -)
    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max


    ancilla_qubits = ancilla_qubits[len(action_qubits) + 2:]  # Remove action qubits, loss qubit and value qubit
    # Determine next action
    action_qnn.circuit(x_qubits + y_qubits, action_qubits, ancilla_qubits=ancilla_qubits, unclean_qubits=unclean_qubits)

    # Start swap test
    qml.Hadamard((loss_qubit,))

    qml.CRY(phi=np.pi / 2., wires=(loss_qubit, value_indices_qubits[0]))

    r_factor = [1]

    # Transistion
    # r is loaded into loss_qubit = |1> and value_indices_qubits = |0>
    qml.PauliX((value_indices_qubits[0],))
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, [loss_qubit] + value_indices_qubits,
        [value_qubit], r_factor,
        ancilla_qubits, unclean_qubits
    )
    qml.PauliX((value_indices_qubits[0],))

    # Load values
    # v(s) is loaded into loss_qubit = |1> and value_indices_qubits = |1>
    end_state_qubit = []
    if not end_state_values:
        end_state_qubit = ancilla_qubits[0:1]
        ancilla_qubits = ancilla_qubits[1:]
        environment.check_end_state(
            next_x_qubits, next_y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ccry(end_state_qubit + value_indices_qubits, torch.pi, value_qubit, ancilla_qubits, unclean_qubits)
        qml.PauliX((end_state_qubit[0],))
    gradient_free_value_qnn.circuit(
        next_x_qubits + next_y_qubits, value_qubit,
        control_qubits=end_state_qubit + [loss_qubit] + value_indices_qubits,
        ancilla_qubits=ancilla_qubits, unclean_qubits=x_qubits + y_qubits + action_qubits
    )
    if not end_state_values:
        qml.PauliX((end_state_qubit[0],))
        environment.check_end_state(
            next_x_qubits, next_y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ancilla_qubits = end_state_qubit + ancilla_qubits

    # Load swap vector
    # Load [1, 0, 1*gamma, 0] / ||vec||, if loss_qubit = |0>
    vector = np.array([[r_max/v_max, 0, 1 * gamma, 0]])
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm

    qml.PauliX((loss_qubit,))
    LittleTreeLoader(
        vector, value_indices_qubits + [value_qubit],
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + action_qubits + next_x_qubits + next_y_qubits
    ).circuit(control_qubits=[loss_qubit])

    # End swap test
    qml.Hadamard((loss_qubit,))


def action_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        loss_qubit: int,
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9,
        unclean_qubits: List[int] = None,

        end_state_values: bool = False,
        diff_method: str = "best",
):
    """
    Computes the value loss, by loading in all states with the same probability
    and then sampling the loss qubit of the value loss circuit. Then scales by the
    correct factors.
    """
    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))

        action_loss_circuit(
            action_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits,
            value_indices_qubits, value_qubit,
            loss_qubit,
            ancilla_qubits, unclean_qubits,
            gamma,
            end_state_values=end_state_values,
        )

        return qml.probs(wires=loss_qubit)

    # backprop, parameter-shift
    result = qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()


    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    vector_norm = np.linalg.norm([r_max/v_max, 0, 1 * gamma, 0])

    return (result[1] - result[0]) * vector_norm * np.sqrt(2) * v_max   # * (2 ** (len(x_qubits) + len(y_qubits)))


def loss_function(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9,
        lam: float = 1,
        eps: float = 0.0,
        unclean_qubits: List[int] = None,
        l_type: int = 3,
        end_state_values: bool = False,
        action_diff_method: str = "best",
        value_diff_method: str = "best",
):
    """Computes both the action and the value loss via the quantum circuits and returns them"""
    loss1 = action_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
        ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3], ancilla_qubits[4:], backend,
        gamma, unclean_qubits=unclean_qubits, end_state_values=end_state_values,
        diff_method=action_diff_method,
    )
    loss2 = value_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
        ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3:6], ancilla_qubits[6], ancilla_qubits[7:],
        backend, gamma, eps, unclean_qubits=unclean_qubits, end_state_values=end_state_values,
        diff_method=value_diff_method,
    )
    if l_type >= 4:
        return lam * loss1 + loss2

    return loss1, loss2
