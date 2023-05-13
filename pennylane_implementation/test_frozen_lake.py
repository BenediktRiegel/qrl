from numpy import ceil, log2, max, abs
import numpy as np
import pennylane as qml
from utils import int_to_bitlist, bitlist_to_int
from quantum_backends import QuantumBackends
from wire_utils import get_wires
from environment.frozen_lake import FrozenField, FrozenLake, FrozenLake2
from environment.frozen_lake3 import FrozenLake3
from environment.frozen_lake4 import FrozenLake4
from environment.frozen_lake5 import FrozenLake5
from debug_utils import snapshots_to_prob_histogram, snapshots_to_probability_strings


def bit_list_to_str(bit_list):
    return str(bit_list).replace(",", "").replace(" ", "").replace("\"", "").replace("[", "").replace("]", "")


def sort_dict_str(d: dict):
    temp = [(key, value) for key, value in d.items()]
    temp.sort(key=lambda element: element[0])
    temp = [f"'{key}': {value}" for (key, value) in temp]
    return "{" + ", ".join(temp) + "}"


def compute_next_state(environment, state, action, log_cols, log_rows):
    x = state[:log_cols]
    x_int = bitlist_to_int(x)
    y = state[log_cols:]
    y_int = bitlist_to_int(y)
    slip_probs = np.roll(environment.slip_probabilities, bitlist_to_int(action))
    resulting_states = [
        int_to_bitlist(x_int+1, log_cols) + y if (x_int < (len(environment.map[0]) - 1)) else state,
        x + int_to_bitlist(y_int-1, log_rows) if (y_int > 0) else state,
        int_to_bitlist(x_int-1, log_cols) + y if (x_int > 0) else state,
        x + int_to_bitlist(y_int+1, log_rows) if (y_int < (len(environment.map) - 1)) else state,
    ]
    outcome = dict()
    for res_state, s_prob in zip(resulting_states, slip_probs):
        if s_prob != 0:
            s_str = bit_list_to_str(res_state)
            if s_str not in outcome:
                outcome[s_str] = 0
            outcome[s_str] += s_prob
    return outcome


def test_state_action(environment, state, action, log_cols, log_rows):
    print(f"({bit_list_to_str(state)}, {bit_list_to_str(action)}):")
    print(f"{sort_dict_str(compute_next_state(environment, state, action, log_cols, log_rows))}")
    backend = QuantumBackends.pennylane_default_qubit
    shots = None

    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 1, 5])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubit, ancilla_qubits = wires
    r_qubit = r_qubit[0]
    backend = backend.get_pennylane_backend("", "", total_num_wires, shots)

    def circuit():
        for q, s in zip(x_qubits+y_qubits, state):
            if s == 1:
                qml.PauliX((q,))
        for q, a in zip(action_qubits, action):
            if a == 1:
                qml.PauliX((q,))
        environment.circuit(x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubit, ancilla_qubits, [])
        qml.Snapshot("result")

        return qml.probs(x_qubits+y_qubits+action_qubits+next_x_qubits+next_y_qubits+[r_qubit])

    snaps = qml.snapshots(qml.QNode(circuit, backend))()
    prob_histogram = snapshots_to_prob_histogram(snaps, x_qubits+y_qubits+action_qubits+next_x_qubits+next_y_qubits+[r_qubit]+ancilla_qubits)
    pruned_probs = [dict() for _ in list(prob_histogram.values())[0]]
    for key, value in prob_histogram.items():
        # for idx, el in enumerate(value):
        idx = len(value)-1
        el = value[idx]
        temp = np.array([int(el) for el in key])
        if (temp[np.array(x_qubits+y_qubits+action_qubits)] == np.array(state+action)).all() and (temp[np.array(ancilla_qubits)] == 0).all():
            if el > 10e-5:
                new_key = "".join([str(el) for el in temp[np.array(next_x_qubits+next_y_qubits)]])
                new_key += f".{temp[r_qubit]}"
                pruned_probs[idx][new_key] = el
    for p in pruned_probs:
        print(sort_dict_str(p))
    prob_strings = snapshots_to_probability_strings(snaps, x_qubits+y_qubits+action_qubits+next_x_qubits+next_y_qubits+[r_qubit]+ancilla_qubits)
    for s in prob_strings:
        print(s)
    print()


def main():
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
    environment = FrozenLake5(map, slip_probabilities, r_m=max(abs([[el.reward for el in row] for row in map])), r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(environment.map))))
    log_cols = int(ceil(log2(len(environment.map[0]))))

    for y in [0]: # range(len(map)):
        for x in [0]: # range(len(map[0])):
            state = int_to_bitlist(x, log_cols) + int_to_bitlist(y, log_rows)
            for a in [0, 2]: # range(4):
                action = int_to_bitlist(a, 2)
                test_state_action(environment, state, action, log_cols, log_rows)


if __name__ == "__main__":
    main()
