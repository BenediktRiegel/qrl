import pennylane as qml
from ccnot import adaptive_ccnot


def set_state_to_ones(reg_qubits, state):
    for r_qubit, s in zip(reg_qubits, state):
        if s == 0:
            qml.PauliX((r_qubit,))


def simple_single_oracle(reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit):
    set_state_to_ones(reg_qubits, state)
    adaptive_ccnot(reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    set_state_to_ones(reg_qubits, state)


def cc_simple_single_oracle(control_qubits, reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit):
    set_state_to_ones(reg_qubits, state)
    adaptive_ccnot(control_qubits + reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    set_state_to_ones(reg_qubits, state)
