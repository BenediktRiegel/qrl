from typing import List, Callable
import pennylane as qml
from load_data import TreeLoader, QAM, LittleTreeLoader
from copy import deepcopy
import numpy as np
import torch
from qnns import QNN
from environment.frozen_lake import FrozenLakeRotSwap
from debug_utils import (
    snapshots_to_debug_strings,
    snapshots_to_debug_dict,
    snapshots_to_prob_histogram,
    snapshots_to_ausspur_dict,
    vector_to_ket_expression
)
from utils import bitlist_to_int, int_to_bitlist
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


def load_value(
        environment, value_qnn_circuit: Callable,
        load_x_qubits: List[int], load_y_qubits: List[int],
        x_qubits: List[int], y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        ancilla_qubits: List[int],
        unclean_qubits: List[int] = None,
        end_state_values: bool = False,
        conjugate: bool = False
):
    end_state_qubit = []
    if not end_state_values:
        end_state_qubit = ancilla_qubits[0:1]
        ancilla_qubits = ancilla_qubits[1:]

        # State Value
        environment.check_end_state(
            x_qubits, y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits,
            oracle_qubit=end_state_qubit[0],
        )
        ccry(end_state_qubit + value_indices_qubits, torch.pi, value_qubit, ancilla_qubits, unclean_qubits)
        qml.PauliX((end_state_qubit[0],))
    qml.Snapshot("Before value_qnn")
    value_qnn_circuit(
        load_x_qubits + load_y_qubits, x_qubits + y_qubits, value_qubit,
        control_qubits=end_state_qubit + value_indices_qubits,
        ancilla_qubits=ancilla_qubits, unclean_qubits=unclean_qubits,
        conjugate=conjugate,
    )
    if not end_state_values:
        qml.PauliX((end_state_qubit[0],))
        environment.check_end_state(
            x_qubits, y_qubits,
            ancilla_qubits=ancilla_qubits,
            unclean_qubits=unclean_qubits,
            oracle_qubit=end_state_qubit[0],
        )


def value_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLakeRotSwap,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        extra_x_qubits: List[int], extra_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9, eps: float = 0.0,
        unclean_qubits: List[int] = None,
        precise: bool = False,
        end_state_values: bool = False,
        diff_method: str = "best",
        snaps: bool = False,
):
    # print(
    #     f"x_qubits: {x_qubits}, y_qubits: {y_qubits}, action_qubits: {action_qubits}, next_x_qubits: {next_x_qubits}, next_y_qubits: {next_y_qubits}, value_indices: {value_indices}, value_qubit: {value_qubit}, swap_vector_qubits: {swap_vector_qubits}, loss_qubit: {loss_qubit}, ancilla_qubits: {ancilla_qubits}")
    unclean_qubits = [] if unclean_qubits is None else unclean_qubits
    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))
        # for s_qubit, b in enumerate([1, 0, 1, 1]):
        #     if b == 1:
        #         qml.PauliX((s_qubit,))
        # qml.Snapshot("Load state 1011")
        # for s_qubit in x_qubits + y_qubits:
        #     qml.PauliX((s_qubit,))
        # qml.PauliX((x_qubits[1],))
        # qml.PauliX((y_qubits[0],))

        value_loss_circuit(
            action_qnn, value_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits,
            extra_x_qubits, extra_y_qubits,
            value_indices_qubits, value_qubit,
            swap_vector_qubits, loss_qubit,
            ancilla_qubits, unclean_qubits,
            gamma, eps,
            end_state_values=end_state_values,
        )

        return qml.expval(qml.PauliZ((loss_qubit,)))

    # backprop, parameter-shift
    if precise:
        qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()

        result = torch.zeros(2)
        for amp, ket in vector_to_ket_expression(backend._state.flatten()):
            result[int(ket[loss_qubit])] += torch.square(torch.abs(amp))
        result = result[0] - result[1]
    else:
        result = qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()

    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    # snaps_strings = qml.snapshots(qml.QNode(circuit, backend, interface="torch", diff_method="best"))()
    # for snap_str in snapshots_to_debug_strings(
    #         snaps_strings, show_zero_rounded=False,
    #         make_space_at=[x_qubits[0], action_qubits[0], next_x_qubits[0],
    #                        value_indices[0], value_qubit, swap_vector_qubits[0], loss_qubit]    # , ancilla_qubits[0]]
    # ):
    #     print(snap_str)
    # if snaps:
    #     snaps_strings = qml.snapshots(qml.QNode(circuit, backend, interface="torch", diff_method="best"))()
    #     temp = x_qubits + y_qubits + action_qubits + next_x_qubits + next_y_qubits
    #     temp.sort()
    #     num_uninteresting_qubits = len(temp)
    #     print(np.count_nonzero(snaps_strings["Loaded swap vector"]))
    #     print(np.array(snaps_strings["Loaded swap vector"]).size)
    #     probs = {k: v[-1] for (k, v) in snapshots_to_prob_histogram({"Loaded swap vector": snaps_strings["Loaded swap vector"]}, temp).items()}
    #     state = np.array(snaps_strings["Loaded swap vector"])
    #     # we have the loss qubit
    #     num_interesting = 2**(len(value_indices) + 1 + len(swap_vector_qubits))
    #     swap_state_size = 2**len(swap_vector_qubits)
    #     # state = state[::2]
    #     sum = 0
    #     nonzero_count = []
    #     # for i in range(len(state) // num_interesting):
    #     #     p = probs["".join([str(el) for el in int_to_bitlist(i, num_uninteresting_qubits)])]
    #     #     s = state[i*num_interesting:(i+1)*num_interesting]
    #     #     nonzero_count.append(np.count_nonzero(s))
    #     #     if p != 0:
    #     #         if np.count_nonzero(s) != 0:
    #     #             print(len(s))
    #     #             # print(f"s: {s}")
    #     #             print(f"p: {p}")
    #     #         s_sum = 0
    #     #         for j in range(0, swap_state_size, 2):
    #     #             # print(f"idx: {(swap_state_size + 1)*j}")
    #     #             s_sum += s[(swap_state_size + 1)*j]
    #     #         # print(f"s_sum: {s_sum}")
    #     #         sum += p * np.square(np.abs(s_sum))
    #
    #     print(np.bincount(np.abs(state) > 10e-12))
    #     total_num_wires = int(np.log2(len(state)))
    #     for idx, s in enumerate(state):
    #         if np.abs(s) >= 10e-12:
    #             temp = [str(el) for el in int_to_bitlist(idx, total_num_wires)]
    #             make_space_at = [x_qubits[0], y_qubits[0], next_x_qubits[0], next_y_qubits[0], action_qubits[0], value_qubit, value_indices[0], swap_vector_qubits[0], loss_qubit]
    #             make_space_at.sort()
    #             for k in reversed(make_space_at):
    #                 temp.insert(k, " ")
    #             print(f"{s} |{''.join(temp)}>")
    #
    #
    #     print(f"calculated result: {sum}")
    #     print(f"measured prob: {result}")
    #     true_prob = snapshots_to_prob_histogram(snaps_strings, [loss_qubit])
    #     true_prob = [true_prob['0'][-1], true_prob['1'][-1]]
    #     print(f"true prob: {true_prob}")
    #     print(f"precise result: {true_prob[0] - true_prob[1]}")
    #     print(f"result: {result[0] - result[1]}")
    #     print(f"precise rescaled loss: {(true_prob[0] - true_prob[1])} * 3 * {v_max ** 2} * {2 + gamma ** 2} = {(true_prob[0] - true_prob[1]) * 3} * {v_max ** 2} * {(2 + gamma ** 2)} = {(true_prob[0] - true_prob[1]) * 3 * (v_max ** 2)} * {2 + gamma ** 2} = {(true_prob[0] - true_prob[1]) * 3 * (v_max ** 2) * (2 + gamma**2)}")
    #     print(f"precise rescaled loss: {(result[0] - result[1])} * 3 * {v_max ** 2} * {2 + gamma ** 2} = {(result[0] - result[1]) * 3} * {v_max ** 2} * {(2 + gamma ** 2)} = {(result[0] - result[1]) * 3 * (v_max ** 2)} * {2 + gamma ** 2} = {(result[0] - result[1]) * 3 * (v_max ** 2) * (2 + gamma**2)}")

    return result * 3 * (v_max ** 2) * (2 + gamma ** 2)  # * (2**(len(x_qubits) + len(y_qubits)))


def value_loss_circuit(
        action_qnn: QNN, value_qnn: QNN, gradient_free_value_qnn: QNN, environment: FrozenLakeRotSwap,
        x_qubits: List[int], y_qubits: List[int],
        action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        extra_x_qubits: List[int], extra_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int], unclean_qubits: List[int] = None,
        gamma: float = 0.9, eps: float = 0.0,
        end_state_values: bool = False,
):
    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    ancilla_qubits = value_indices_qubits + next_x_qubits + next_y_qubits + [value_qubit] + extra_x_qubits + extra_y_qubits + swap_vector_qubits + [loss_qubit]

    # Determine next action
    action_qnn.circuit([], x_qubits + y_qubits, action_qubits, ancilla_qubits=ancilla_qubits, unclean_qubits=unclean_qubits)
    qml.Snapshot("Chose action")

    ancilla_qubits = ancilla_qubits[len(value_indices_qubits):]     # Remove value indices qubits
    # 3 Qubits, i.e. 8 possible states
    # possible states 000, 001, 010, 011, 100, 101, 110, 111
    # Loading in states 010, 011, 100, 101, 111
    # 010: state value, 011: conjugate state value,
    # 100: next state value, 101: conjugate next state value,
    # 111: reward
    indices_vector = np.array([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=np.float64)
    indices_vector /= np.linalg.norm(indices_vector[0])
    LittleTreeLoader(
        indices_vector, value_indices_qubits,
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + action_qubits
    ).circuit()

    r_factor = [1 / v_max]

    ancilla_qubits = ancilla_qubits[len(next_x_qubits) + len(next_y_qubits) + 1:]   # Remove next_x_qubits, next_y_qubits and value qubit
    # Transition
    # Reward is loaded into value_indices_qubits 111
    environment.circuit(
        [], [],
        x_qubits, y_qubits, action_qubits,
        next_x_qubits, next_y_qubits,
        value_indices_qubits, [value_qubit], r_factor,
        ancilla_qubits, unclean_qubits
    )
    qml.Snapshot(f"After env. Set r on indices {''.join([str(el) for el in value_indices_qubits])}=11")

    ancilla_qubits = ancilla_qubits[len(extra_x_qubits) + len(extra_y_qubits):]     # Remove extra_x_qubits and extra_y_qubits
    # Correctly prep extra x (resp. y) qubits register
    # 'Copy' binary state in order to load the values
    qml.PauliX((value_indices_qubits[0],))
    for x_c_q, y_c_q, x_t_q, y_t_q1 in zip(x_qubits, y_qubits, extra_x_qubits, extra_y_qubits):
        qml.Toffoli(wires=(value_indices_qubits[0], x_c_q, x_t_q))
        qml.Toffoli(wires=(value_indices_qubits[0], y_c_q, y_t_q1))
    qml.PauliX((value_indices_qubits[0],))

    # extra qubits for reward to |+>
    adaptive_ccnot(value_indices_qubits, ancilla_qubits[1:], unclean_qubits + x_qubits + y_qubits + action_qubits, ancilla_qubits[0])
    for extra_x_q, extra_y_q in zip(extra_x_qubits, extra_y_qubits):
        qml.CRY(phi=torch.pi, wires=(ancilla_qubits[0], extra_x_q))
        qml.CRY(phi=torch.pi, wires=(ancilla_qubits[0], extra_y_q))
    adaptive_ccnot(value_indices_qubits, ancilla_qubits[1:], unclean_qubits + x_qubits + y_qubits + action_qubits, ancilla_qubits[0])

    # Next state is already set
    qml.PauliX((value_indices_qubits[1],))
    for x_c_q, y_c_q, x_t_q, y_t_q in zip(x_qubits, y_qubits, extra_x_qubits, extra_y_qubits):
        adaptive_ccnot(value_indices_qubits[:2] + [x_c_q], ancilla_qubits, unclean_qubits, x_t_q)
        adaptive_ccnot(value_indices_qubits[:2] + [y_c_q], ancilla_qubits, unclean_qubits, y_t_q)
    qml.PauliX((value_indices_qubits[1],))

    # Load state value into value_indices_qubits 010
    qml.PauliX((value_indices_qubits[0],))
    qml.PauliX((value_indices_qubits[2],))
    load_value(
        environment, value_qnn.circuit,
        x_qubits, y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + next_x_qubits + next_y_qubits + action_qubits,
        end_state_values=end_state_values
    )
    qml.PauliX((value_indices_qubits[2],))

    # Load conjugate state value into value_indices_qubits 011
    load_value(
        environment, value_qnn.circuit,
        x_qubits, y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + next_x_qubits + next_y_qubits + action_qubits,
        end_state_values=end_state_values,
        conjugate=True,
    )
    qml.PauliX((value_indices_qubits[0],))
    qml.Snapshot(f"Set v(s) on indices {''.join([str(el) for el in value_indices_qubits])}=10")

    # Load next state value into value_indices_qubits 100
    qml.PauliX((value_indices_qubits[1],))
    qml.PauliX((value_indices_qubits[2],))
    load_value(
        environment, gradient_free_value_qnn.circuit,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
        end_state_values=end_state_values,
    )
    qml.PauliX((value_indices_qubits[2],))

    # Load conjugate next state value into value_indices_qubits 101
    load_value(
        environment, gradient_free_value_qnn.circuit,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
        end_state_values=end_state_values,
        conjugate=True,
    )
    qml.PauliX((value_indices_qubits[1],))

    qml.Snapshot(f"Set v(s') on indices {''.join([str(el) for el in value_indices_qubits])}=01")

    # Load swap vector
    ancilla_qubits = ancilla_qubits[len(swap_vector_qubits):]   # Remove swap_vector_qubits
    # Reminder 010: state value, 011: conjugate state value,
    # 100: next state value, 101: conjugate next state value,
    # 111: reward
    vector = np.array([[0, 0, 0.5, 0.5, -0.5 * gamma, -0.5 * gamma, 0, -1]])
    # print(f"swap vector: {vector[0]}")
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm

    # first swap_vector_qubits get set to every combination
    for s_q in swap_vector_qubits[:-4]:
        qml.Hadamard((s_q,))

    # last swap_vector_qubit stays zero
    LittleTreeLoader(
        vector, swap_vector_qubits[-4:-1],
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + action_qubits
    ).circuit()
    qml.Snapshot(f"Loaded swap vector")

    ancilla_qubits = ancilla_qubits[1:]  # Remove loss_qubit
    swap_test(loss_qubit, extra_x_qubits + extra_y_qubits + value_indices_qubits + [value_qubit], swap_vector_qubits)
    qml.Snapshot(f"Swap test")


def action_loss_circuit(
        action_qnn: QNN, gradient_free_value_qnn: QNN, environment: FrozenLakeRotSwap,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        extra_x_qubits: List[int], extra_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int], unclean_qubits: List[int] = None,
        gamma: float = 0.9, end_state_values: bool = False
):
    ancilla_qubits = action_qubits + value_indices_qubits + [value_qubit] + swap_vector_qubits + [loss_qubit] + ancilla_qubits
    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    # QAM(np.array([[1], [0]]), [], value_indices_qubits, ancilla_qubits, unclean_qubits)
    # LittleTreeLoader(
    #     np.array([[1, 1]]) / np.sqrt(2), value_indices_qubits[:1],
    #     ancilla_wires=ancilla_qubits,
    #     unclean_wires=unclean_qubits,    #  + x_qubits + y_qubits + next_x_qubits + next_y_qubits + action_qubits
    # ).circuit()

    ancilla_qubits = ancilla_qubits[len(action_qubits):]  # Remove action qubits
    # Determine next action
    action_qnn.circuit([], x_qubits + y_qubits, action_qubits, ancilla_qubits=ancilla_qubits, unclean_qubits=unclean_qubits)

    ancilla_qubits = ancilla_qubits[len(value_indices_qubits):]  # Remove value indices qubits
    # 2 Qubits, i.e. 4 mögliche Zustände
    # possible states 00, 01, 10, 11
    # Loading in all states
    # 00: max value,
    # 01: next state value, 10: conjugate next state value,
    # 11: reward
    for q in value_indices_qubits:
        qml.Hadamard((q,))

    r_factor = [1 / v_max]
    # r_factor = [0]

    ancilla_qubits = ancilla_qubits[1:]  # Remove value qubit
    # Transistion
    # r is loaded if value_indices_qubits = |11>
    # print(f"ancilla qubits for env: {ancilla_qubits}")
    environment.circuit(
        [], [],
        x_qubits, y_qubits, action_qubits,
        next_x_qubits, next_y_qubits,
        value_indices_qubits, [value_qubit], r_factor,
        ancilla_qubits, unclean_qubits
    )
    # qml.Toffoli(wires=(value_indices_qubits[0], value_indices_qubits[1], value_qubit))
    # qml.Snapshot("Reward")

    # Load next state value into value_indices_qubits 01
    qml.PauliX((value_indices_qubits[0],))
    load_value(
        environment, gradient_free_value_qnn.circuit,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
        end_state_values=end_state_values,
    )
    # qml.Toffoli(wires=(value_indices_qubits[0], value_indices_qubits[1], value_qubit))
    qml.PauliX((value_indices_qubits[0],))

    # Load conjugate next state value into value_indices_qubits 10
    qml.PauliX((value_indices_qubits[1],))
    load_value(
        environment, gradient_free_value_qnn.circuit,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        value_indices_qubits, value_qubit,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=unclean_qubits + x_qubits + y_qubits + action_qubits,
        end_state_values=end_state_values,
        conjugate=True,
    )
    # qml.Toffoli(wires=(value_indices_qubits[0], value_indices_qubits[1], value_qubit))
    qml.PauliX((value_indices_qubits[1],))

    for q in extra_x_qubits + extra_y_qubits:
        qml.Hadamard((q,))


    ancilla_qubits = ancilla_qubits[len(swap_vector_qubits):]
    # Load swap vector
    vector = np.array([[gamma + 1, -0.5*gamma, -0.5*gamma, -1]])
    vector_norm = np.linalg.norm(vector[0])
    vector /= vector_norm
    # print(f"normed vec: {vector}")

    for s_q in swap_vector_qubits[:-3]:
        qml.Hadamard((s_q,))

    LittleTreeLoader(
        vector, swap_vector_qubits[-3:-1],
        ancilla_wires=ancilla_qubits,
        unclean_wires=unclean_qubits + x_qubits + y_qubits + action_qubits + next_x_qubits + next_y_qubits + value_indices_qubits
    ).circuit()

    ancilla_qubits = ancilla_qubits[1:]  # Remove loss qubit
    swap_test(loss_qubit, extra_x_qubits + extra_y_qubits + value_indices_qubits + [value_qubit], swap_vector_qubits)

    qml.Snapshot("Result")


def action_loss(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLakeRotSwap,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        extra_x_qubits: List[int], extra_y_qubits: List[int],
        value_indices_qubits: List[int], value_qubit: int,
        swap_vector_qubits: List[int], loss_qubit: int,
        ancilla_qubits: List[int],
        backend,
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

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))
        # for s_qubit in x_qubits + y_qubits:
        #     qml.PauliX((s_qubit,))
        # qml.PauliX((x_qubits[0],))
        # qml.PauliX((y_qubits[0],))

        action_loss_circuit(
            action_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits,
            extra_x_qubits, extra_y_qubits,
            value_indices_qubits, value_qubit,
            swap_vector_qubits, loss_qubit,
            ancilla_qubits, unclean_qubits,
            gamma,
            end_state_values=end_state_values,
        )

        return qml.expval(qml.PauliZ((loss_qubit,)))
        # return qml.probs(loss_qubit)

    # backprop, parameter-shift
    if precise:
        qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()

        result = torch.zeros(2)
        for amp, ket in vector_to_ket_expression(backend._state.flatten()):
            result[int(ket[loss_qubit])] += torch.square(torch.abs(amp))
        result = result[0] - result[1]
    else:
        result = qml.QNode(circuit, backend, interface="torch", diff_method=diff_method)()

    vector_norm = np.linalg.norm([gamma + 1, -0.5*gamma, -0.5*gamma, -1])

    r_max = environment.r_m
    v_max = r_max / (1 - gamma) if end_state_values else r_max

    # if snaps:
    #     qnode = qml.QNode(circuit, backend, diff_method="best")
    #     snapshots = qml.snapshots(qnode)()
    #     probs = snapshots_to_prob_histogram({"Result": snapshots["Result"]}, wires=[loss_qubit])
    #     probs = [probs['0'][-1], probs['1'][-1]]
    #     print(f"v_max = {v_max}")
    #     print(f"vector_norm = {vector_norm}")
    #     print(f"true probs: {probs}")
    #     print(f"measured probs: {result}")
    #     print(f"({probs[0]} - {probs[1]}) * {vector_norm} * {v_max} = {probs[0] - probs[1]} * {vector_norm} * {v_max} = {(probs[0] - probs[1]) * vector_norm} * {v_max} = {(probs[0] - probs[1]) * vector_norm} * {v_max} = {(probs[0] - probs[1]) * vector_norm * v_max}")
    #     print(f"({result}) * {vector_norm} * {v_max} = {result} * {vector_norm} * {v_max} = {result * vector_norm} * {v_max} = {result * vector_norm} * {v_max} = {result * vector_norm * v_max}")
    #     print(f"{(probs[0] - probs[1]) * vector_norm * np.sqrt(2)} * {v_max} = {(probs[0] - probs[1]) * vector_norm * np.sqrt(2) * v_max}")
    #     p_wires = [loss_qubit] + value_indices_qubits[:1] + [value_qubit]
    #     # p_wires.sort()
    #     # print(f"loss_qubit: {loss_qubit}, value_indices: {value_indices_qubits[:1]}, value_qubit: {value_qubit}")
    #     # probs = snapshots_to_prob_histogram({"Reward": snapshots["Reward"]}, wires=p_wires)
    #     # print(f"reward: {probs}")
    #
    #     # print(qml.draw(qnode)())

    return torch.sqrt(result) * 2 * vector_norm * v_max   # * (2**((len(x_qubits) + len(y_qubits)))) # * (2 ** (len(x_qubits) + len(y_qubits)))


def loss_function(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLakeRotSwap,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        extra_x_qubits: List[int], extra_y_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9,
        lam: float = 1,
        eps: float = 0.0,
        unclean_qubits: List[int] = None,
        l_type: int = 3,
        precise: bool = False,
        end_state_values: bool = False,
        action_diff_method: str = "best",
        value_diff_method: str = "best",
):
    state_size = len(x_qubits) + len(y_qubits)
    loss1 = action_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        ancilla_qubits[:2], ancilla_qubits[2], ancilla_qubits[3:6+state_size], ancilla_qubits[6+state_size], ancilla_qubits[state_size+7:], backend,
        gamma, unclean_qubits=unclean_qubits, precise=precise, end_state_values=end_state_values,
        diff_method=action_diff_method,
        snaps=False,
    )
    loss2 = value_loss(
        action_qnn, value_qnn, environment, x_qubits, y_qubits, action_qubits,
        next_x_qubits, next_y_qubits,
        extra_x_qubits, extra_y_qubits,
        ancilla_qubits[:3], ancilla_qubits[3], ancilla_qubits[4:8+state_size], ancilla_qubits[state_size+8], ancilla_qubits[state_size+9:],
        backend, gamma, eps, unclean_qubits=unclean_qubits, precise=precise, end_state_values=end_state_values,
        diff_method=value_diff_method,
    )
    if l_type >= 4:
        return lam * loss1 + loss2

    return loss1, loss2
