from typing import List
import pennylane as qml
from load_data import TreeLoader, QAM, LittleTreeLoader
from copy import deepcopy
import numpy as np
import torch
from qnns import QNN
from environment.frozen_lake import FrozenLake
# from debug_utils import (
#     snapshots_to_debug_strings,
#     snapshots_to_debug_dict,
#     snapshots_to_prob_histogram,
#     snapshots_to_ausspur_dict,
#     vector_to_ket_expression
# )
# from utils import bitlist_to_int, int_to_bitlist
from ccnot import adaptive_ccnot


def swap_test(swap_qubit: int, reg1: List[int], reg2: List[int]):
    qml.Hadamard((swap_qubit,))
    for reg1_q, reg2_q in zip(reg1, reg2):
        qml.CSWAP(wires=(swap_qubit, reg1_q, reg2_q))
    qml.Hadamard((swap_qubit,))


def ccry(c_qubits: List[int], phi: float, target_qubit: int, ancilla_qubits: List[int], unclean_qubits: List[int]):
    adaptive_ccnot(c_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])
    qml.CRY(phi=phi, wires=(ancilla_qubits[0], target_qubit))
    adaptive_ccnot(c_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])


def sample_actions(states, action_qnn: QNN, action_backend, action_qubits, diff_method: str = "best"):
    def circuit(s):
        action_qnn.circuit([], s, action_qubits[0], action_qubits[1], action_qubits[2])
        return qml.sample(wires=action_qubits[0])

    return [qml.QNode(circuit, action_backend, interface="torch", diff_method=diff_method)(_s)[0] for _s in states]


def value_loss_circuit(
        state, action, next_state, reward, r_max: float,
        value_qnn: QNN, gradient_free_value_qnn: QNN,
        loss_qubit: int, value_indices_qubits: List[int],
        value_qubit: int, swap_vector_qubits: List[int],
        value_additional_qubits: List[int],
        swap_value_additional_qubits: List[int], ancilla_qubits: List[int],
        unclean_qubits: List[int] = None,
        gamma: float = 0.9, eps: float = 0.0,
        end_state_values: bool = False,
):
    v_max = r_max / (1 - gamma)

    ancilla_qubits = swap_vector_qubits + [loss_qubit] + ancilla_qubits

    # Prep value indices
    LittleTreeLoader(
        np.array([[0, 1, 1, 1]]) / np.sqrt(3), value_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits
    ).circuit()

    # r_factor = [1 / v_max]
    # Load reward 11
    ccry(value_indices_qubits, reward, value_qubit, ancilla_qubits, unclean_qubits)

    # Load values 10
    qml.PauliX((value_indices_qubits[0],))
    qml.Snapshot("Before value_qnn")
    value_qnn.circuit(value_indices_qubits, state, [value_qubit], value_additional_qubits, ancilla_qubits,
                      unclean_qubits)
    qml.PauliX((value_indices_qubits[0],))
    qml.Snapshot(f"Set v(s) on indices {''.join([str(el) for el in value_indices_qubits])}=10")

    # Next State Value 01
    qml.PauliX((value_indices_qubits[1],))
    gradient_free_value_qnn.circuit(value_indices_qubits, next_state, [value_qubit], value_additional_qubits, ancilla_qubits, unclean_qubits)
    qml.PauliX((value_indices_qubits[1],))
    qml.Snapshot(f"Set v(s') on indices {''.join([str(el) for el in value_indices_qubits])}=01")

    # Load swap vector
    ancilla_qubits = ancilla_qubits[len(swap_vector_qubits):]
    swap_indices_qubits = swap_vector_qubits[:-1]
    swap_value_qubit = swap_vector_qubits[-1]
    vector = np.array([[0, 1, -1 * gamma, -1]])
    # print(f"swap vector: {vector[0]}")
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm
    # print(f"normed swap vector: {vector[0]}")
    LittleTreeLoader(
        vector, swap_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits,
    ).circuit()
    qml.Snapshot(f"Loaded swap vector")

    # Load values 10
    qml.PauliX((swap_indices_qubits[0],))
    qml.Snapshot("Before value_qnn")
    value_qnn.conj_circuit(swap_indices_qubits, state, [swap_value_qubit], swap_value_additional_qubits, ancilla_qubits, unclean_qubits)
    qml.PauliX((swap_indices_qubits[0],))
    qml.Snapshot(f"Set v(s) on indices {''.join([str(el) for el in value_indices_qubits])}=10")

    # Next State Value 01
    qml.PauliX((swap_indices_qubits[1],))
    gradient_free_value_qnn.conj_circuit(swap_indices_qubits, next_state, [swap_value_qubit], swap_value_additional_qubits, ancilla_qubits, unclean_qubits)
    qml.PauliX((swap_indices_qubits[1],))

    qml.PauliZ((swap_value_qubit,))

    # ancilla_qubits = ancilla_qubits[1:]  # Remove loss_qubit
    swap_test(loss_qubit, value_indices_qubits + [value_qubit], swap_vector_qubits)
    qml.Snapshot(f"Swap test")


def value_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        loss_qubit: int, value_indices_qubits: List[int],
        value_qubit: int, swap_vector_qubits: List[int],
        value_additional_qubits: List[int],
        swap_value_additional_qubits: List[int],
        value_ancilla_qubits: List[int],
        value_backend,
        action_backend,
        action_qubits,
        shots: int,
        gamma: float = 0.9, eps: float = 0.0,
        unclean_qubits: List[int] = None,
        precise: bool = False,
        end_state_values: bool = False,
        diff_method: str = "best",
        snaps: bool = False,
):
    unclean_qubits = [] if unclean_qubits is None else unclean_qubits
    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    states = environment.get_random_state(shots)
    actions = sample_actions(states, action_qnn, action_backend, action_qubits)
    next_states_and_rewards = [environment.sample_transition(s, a) for (s, a) in zip(states, actions)]

    def circuit(state, action, next_state, reward):
        value_loss_circuit(
            state, action, next_state, reward,
            environment.r_m,
            value_qnn, gradient_free_value_qnn,
            loss_qubit, value_indices_qubits, value_qubit,
            swap_vector_qubits,
            value_additional_qubits, swap_value_additional_qubits,
            value_ancilla_qubits, unclean_qubits,
            gamma, eps,
            end_state_values=end_state_values,
        )

        return qml.sample(wires=loss_qubit)

    result = torch.tensor([qml.QNode(circuit, value_backend, interface="torch", diff_method=diff_method)(s, a, s_, r)
                           for (s, a, (s_, r)) in zip(states, actions, next_states_and_rewards)])
    result *= -2
    result += 1
    result = result.sum()

    r_max = environment.r_m
    v_max = r_max / (1 - gamma)

    return result * 3 * (v_max ** 2) * (2 + gamma ** 2)  # * (2**(len(x_qubits) + len(y_qubits)))


def action_loss_circuit(
        next_state, reward,
        gradient_free_value_qnn: QNN, r_max: float,
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        value_additional_qubits: List[int],
        swap_value_additional_qubits: List[int],
        ancilla_qubits: List[int], unclean_qubits: List[int] = None,
        gamma: float = 0.9, end_state_values: bool = False
):
    v_max = r_max / (1 - gamma)

    ancilla_qubits = swap_vector_qubits + [loss_qubit] + ancilla_qubits

    # Prep value indices 01, 10, 11 with equal prob
    LittleTreeLoader(
        np.array([[0, 1, 1, 1]]) / np.sqrt(3), value_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits
    ).circuit()

    # Load reward 11
    ccry(value_indices_qubits, reward, value_qubit, ancilla_qubits, unclean_qubits)

    # Next State Value 10
    qml.PauliX((value_indices_qubits[1],))
    gradient_free_value_qnn.circuit(value_indices_qubits, next_state, [value_qubit], value_additional_qubits, ancilla_qubits, unclean_qubits)
    qml.PauliX((value_indices_qubits[1],))

    # Load swap vector
    ancilla_qubits = ancilla_qubits[len(swap_vector_qubits):]
    swap_indices_qubits = swap_vector_qubits[:-1]
    swap_value_qubit = swap_vector_qubits[-1]
    # neg_r_factor = gamma-1    # r_factor = r_max / v_max = r_max / (r_max / (1 - gamma)) = 1-gamma
    # vector = np.array([[0, 2, -1 * gamma, neg_r_factor]])
    vector = np.array([[0, 2, -1 * gamma, -1]])
    # print(f"swap vector: {vector[0]}")
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm
    # print(f"normed swap vector: {vector[0]}")
    LittleTreeLoader(
        vector, swap_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits,
    ).circuit()
    qml.Snapshot(f"Loaded swap vector")

    # Next State Value 01
    qml.PauliX((swap_indices_qubits[1],))
    gradient_free_value_qnn.conj_circuit(swap_indices_qubits, next_state, [swap_value_qubit], swap_value_additional_qubits, ancilla_qubits, unclean_qubits)
    qml.PauliX((swap_indices_qubits[1],))

    qml.PauliZ((swap_value_qubit,))

    swap_test(loss_qubit, value_indices_qubits + [value_qubit], swap_vector_qubits)


def action_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        loss_qubit: int,
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int],
        value_additional_qubits: List[int],
        swap_value_additional_qubits: List[int],
        ancilla_qubits: List[int],
        value_backend,
        action_backend,
        action_qubits,
        shots: int,
        gamma: float = 0.9,
        unclean_qubits: List[int] = None,
        precise: bool = False,
        end_state_values: bool = False,
        diff_method: str = "best",
        snaps=False,
):
    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    states = environment.get_random_state(shots)
    actions = sample_actions(states, action_qnn, action_backend, action_qubits[:4], action_qubits[4:])
    next_states_and_rewards = [environment.sample_transition(s, a) for (s, a) in zip(states, actions)]

    def circuit(next_state, reward):
        action_loss_circuit(
            next_state, reward,
            gradient_free_value_qnn, environment.r_m,
            value_indices_qubits, value_qubit,
            swap_vector_qubits,
            loss_qubit, value_additional_qubits,
            swap_value_additional_qubits,
            ancilla_qubits, unclean_qubits,
            gamma,
            end_state_values=end_state_values,
        )

        return qml.samples(wires=loss_qubit)

    result = torch.tensor([qml.QNode(circuit, value_backend, interface="torch", diff_method=diff_method)(s_, r)
                           for (s_, r) in next_states_and_rewards])
    result *= -2
    result += 1
    result = result.sum()

    vector_norm = np.linalg.norm([0, 2, -1 * gamma, -1])

    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    return result * vector_norm * np.sqrt(2) * v_max  # * (2 ** (len(x_qubits) + len(y_qubits)))


def loss_function(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake,
        value_qubits: List[List[int]],
        action_qubits: List[List[int]],
        value_backend,
        action_backend,
        shots: int,
        gamma: float = 0.9,
        eps: float = 0.0,
        unclean_qubits: List[int] = None,
        precise: bool = False,
        end_state_values: bool = False,
        action_diff_method: str = "best",
        value_diff_method: str = "best",
):
    # Split value qubits [1, 3, 3, value_additional_qubits, value_additional_qubits, value_ancillas]
    loss_qubit = value_qubits[0][0]
    value_indices_qubits = value_qubits[1][:-1]
    value_qubit = value_qubits[1][-1]
    swap_vector_qubits = value_qubits[2]
    value_additional_qubits = value_qubits[3]
    swap_value_additional_qubits = value_qubits[4]
    value_ancilla_qubits = value_qubits[5]

    a_loss = action_loss(
        action_qnn, value_qnn, environment,
        loss_qubit, value_indices_qubits, value_qubit, swap_vector_qubits, value_additional_qubits,
        swap_value_additional_qubits,
        value_ancilla_qubits,
        value_backend,
        action_backend, action_qubits,
        shots,
        gamma, unclean_qubits=unclean_qubits, precise=precise, end_state_values=end_state_values,
        diff_method=action_diff_method,
    )
    # value_indices: List[int], value_qubit: int,
    # swap_vector_qubits: List[int], loss_qubit: int,
    v_loss = value_loss(
        action_qnn, value_qnn, environment,
        loss_qubit, value_indices_qubits, value_qubit, swap_vector_qubits, value_additional_qubits,
        swap_value_additional_qubits,
        value_ancilla_qubits,
        value_backend,
        action_backend, action_qubits,
        shots,
        gamma, eps, unclean_qubits=unclean_qubits, precise=precise, end_state_values=end_state_values,
        diff_method=value_diff_method,
    )

    return v_loss, a_loss
