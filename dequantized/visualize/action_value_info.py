from typing import List, Dict
import pennylane as qml
from quantum_backends import QuantumBackends
from qnns import QNN
from wire_utils import get_wires


def sum_2d_list(list: List[List[int]]) -> int:
    result = 0
    for sub_list in list:
        result += len(sub_list)
    return result


def get_action_probs(states, action_qnn, action_qubits: List[List[int]]) -> Dict[str, List[float]]:
    action_probs = dict()
    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend("", "", sum_2d_list(action_qubits), None)
    for s in states:
        def circuit(s):
            action_qnn.circuit([], s, action_qubits[0], action_qubits[1], action_qubits[2])
            return qml.probs(wires=action_qubits[0])

        action_probs[str(s)] = list(qml.QNode(circuit, backend)(s))

    return action_probs


def get_state_values(states, value_qnn: QNN, v_max: float) -> Dict[str, float]:
    state_values = dict()
    value_additional_qubits, value_ancilla_qubits = value_qnn.num_additional_and_ancilla_qubits()
    qubits = get_wires([1, value_additional_qubits, value_ancilla_qubits])
    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend("", "", sum_2d_list(qubits), None)
    for s in states:
        def circuit(s):
            value_qnn.circuit([], s, qubits[0], qubits[1], qubits[2], [])
            return qml.probs(wires=qubits[0])

        state_values[str(s)] = (2*qml.QNode(circuit, backend)(s)[0] - 1)*v_max

    return state_values
