from typing import List, Iterable
import numpy as np
from utils import int_to_bitlist


def vector_to_ket_expression(vec, no_zeros=True):
    num_qbits = int(np.ceil(np.log2(len(vec))))
    result = []
    for i in range(len(vec)):
        if vec[i] != 0:
            result.append((vec[i], int_to_bitlist(i, num_qbits)))
        elif not no_zeros:
            result.append((vec[i], int_to_bitlist(i, num_qbits)))
    return result


def ausspuren(state: List[int], ausspur_wires: Iterable[int]):
    state = np.array(state)
    total_qubits = int(np.log2(len(state)))
    ausspur_wires = set(ausspur_wires)
    ausspur_composition = [0]
    for i in range(total_qubits):
        if i not in ausspur_wires:
            if isinstance(ausspur_composition[-1], int):
                ausspur_composition.append(np.eye(2))
            else:
                ausspur_composition[-1] = np.kron(ausspur_composition[-1], np.eye(2))
        else:
            if not isinstance(ausspur_composition[-1], int):
                ausspur_composition.append(1)
            else:
                ausspur_composition[-1] += 1
    ausspur_sum = None
    for i in range(len(ausspur_wires)):
        basis_vec = int_to_bitlist(i, len(ausspur_wires))
        current_basis_bit = 0
        temp = np.array([1])
        for ausspur_component in ausspur_composition:
            if isinstance(ausspur_component, int):
                for _ in range(ausspur_component):
                    basis_bit = basis_vec[current_basis_bit]
                    current_basis_bit += 1
                    temp = np.kron(temp, np.array([1-basis_bit, basis_bit]))
            else:
                temp = np.kron(temp, ausspur_component)
        print(state.size)
        print(temp.size)
        temp = temp @ state
        ausspur_sum = temp if ausspur_sum is None else ausspur_sum + temp

    return ausspur_sum


def snapshots_to_ausspur_dict(snapshots, ausspur_wires: List[List[int]]):
    ausspur_dict = dict()
    for title, state in snapshots.items():
        if title != 'execution_results':
            ausspur_dict[title] = [ausspuren(state, wires) for wires in ausspur_wires]
    return ausspur_dict


def vector_list_to_ket_expression(vec_list):
    return [vector_to_ket_expression(vec) for vec in vec_list]


def snapshots_to_debug_dict(snapshots):
    debug_dict = dict()
    for title, state in snapshots.items():
        if title != 'execution_results':
            debug_dict[title] = vector_to_ket_expression(state)
    return debug_dict


def snapshots_to_debug_strings(snapshots, wires=None, make_space_at=None, show_zero_rounded=True):
    if make_space_at is None:
        make_space_at = []
    make_space_at.sort()
    if wires is None:
        debug_strings = []
        for title, state in snapshots.items():
            if title != 'execution_results':
                all_kets = vector_to_ket_expression(state)
                temp = [str(el) for el in all_kets[0][1]]
                for j in reversed(make_space_at):
                    temp.insert(j, " ")
                if show_zero_rounded or f"{all_kets[0][0]:.2f}" != "0.00+0.00j":
                    state_str = f"{all_kets[0][0]:.2f} |" + "".join(temp) + ">"
                else:
                    state_str = ""
                for i in range(1, len(all_kets)):
                    temp = [str(el) for el in all_kets[i][1]]
                    for j in reversed(make_space_at):
                        temp.insert(j, " ")
                    if show_zero_rounded or f"{all_kets[i][0]:.2f}" != "0.00+0.00j":
                        state_str += f" + {all_kets[i][0]:.2f} |" + "".join(temp) + ">"
                debug_strings.append(f"{title} {state_str}")
    else:
        debug_strings = []
        for title, state in snapshots.items():
            if title != 'execution_results':
                state_dict = {}
                all_kets = vector_to_ket_expression(state)
                for i in range(len(all_kets)):
                    state_vec = "".join([str(el) for el in np.array(all_kets[i][1])[wires]])
                    if state_vec not in state_dict:
                        state_dict[state_vec] = 0
                    state_dict[state_vec] += all_kets[i][0]
                state_strings = list()
                for state_vec, amplitude in state_dict.items():
                    if show_zero_rounded or f"{amplitude:.2f}" != "0.00+0.00j":
                        state_strings.append(f"{amplitude:.2f} |" + state_vec + ">")
                debug_strings.append(f"{title} {' + '.join(state_strings)}")
    return debug_strings


def snapshots_to_probability_strings(snapshots, wires=None):
    if wires is None:
        debug_strings = []
        for title, state in snapshots.items():
            if title != 'execution_results':
                all_kets = vector_to_ket_expression(state)
                state_str = f"{all_kets[0][0].imag**2 + all_kets[0][0].real**2} |" + "".join([str(el) for el in all_kets[0][1]]) + ">"
                for i in range(1, len(all_kets)):
                    state_str += f" + {all_kets[i][0].imag**2 + all_kets[i][0].real**2} |" + "".join([str(el) for el in all_kets[i][1]]) + ">"
                debug_strings.append(f"{title} {state_str}")
    else:
        debug_strings = []
        for title, state in snapshots.items():
            if title != 'execution_results':
                state_dict = {}
                all_kets = vector_to_ket_expression(state)
                for i in range(len(all_kets)):
                    state_vec = "".join([str(el) for el in np.array(all_kets[i][1])[wires]])
                    if state_vec not in state_dict:
                        state_dict[state_vec] = 0
                    state_dict[state_vec] += all_kets[i][0]
                state_strings = list()
                for state_vec, amplitude in state_dict.items():
                    state_strings.append(f"{amplitude.imag**2 + amplitude.real**2} |" + state_vec + ">")
                debug_strings.append(f"{title} {' + '.join(state_strings)}")
    return debug_strings


def snapshots_to_prob_histogram(snapshots, wires):
    prob_histogram = dict()
    for title, state in snapshots.items():
        if title != 'execution_results':
            all_kets = vector_to_ket_expression(state, no_zeros=False)
            temp_probs = dict()
            for entry in all_kets:
                state = np.array(entry[1])
                state = "".join([str(el) for el in state[wires]])

                if state not in temp_probs:
                    temp_probs[state] = 0
                temp_probs[state] += (entry[0].real**2 + entry[0].imag**2)
            for state, prob in temp_probs.items():
                if state not in prob_histogram:
                    prob_histogram[state] = list()
                prob_histogram[state].append(prob)
    return prob_histogram
