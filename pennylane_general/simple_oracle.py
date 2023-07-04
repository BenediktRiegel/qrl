import pennylane as qml
from ccnot import adaptive_ccnot


def set_state_to_ones(reg_qubits, state, debug_str=""):
    for r_qubit, s in zip(reg_qubits, state):
        if s == 0:
            qml.PauliX((r_qubit,))
        # qml.Snapshot(f"{debug_str} flipped {r_qubit}, if {s} == 0")


def simple_single_oracle(reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit):
    set_state_to_ones(reg_qubits, state)
    adaptive_ccnot(reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    set_state_to_ones(reg_qubits, state)


def cc_simple_single_oracle(control_qubits, reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit, debug_str=""):
    set_state_to_ones(reg_qubits, state, debug_str=f"{debug_str} reg {reg_qubits} from {''.join([str(el) for el in state])} to " + ("1"*len(state)))
    qml.Snapshot(f"{debug_str} reg {reg_qubits} from {''.join([str(el) for el in state])} to " + ("1"*len(state)))
    adaptive_ccnot(control_qubits + reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    qml.Snapshot(f"{debug_str} {''.join([str(el) for el in state])}: adaptive_ccnot(c={control_qubits + reg_qubits}, a:{ancilla_qubits}, u: {unclean_qubits}, o: {oracle_qubit}")
    set_state_to_ones(reg_qubits, state, debug_str=f"{debug_str} reg {reg_qubits} from {('1'*len(state))} to {''.join([str(el) for el in state])}")
    qml.Snapshot(f"{debug_str} reg {reg_qubits} from {('1'*len(state))} to {''.join([str(el) for el in state])}")
