import pennylane as qml
from ccnot import adaptive_ccnot


def set_state_to_ones(reg_qubits, state, debug_str=""):
    """
    Let state_j be the j'th bit of state, then this method executes:
    (X**state_0) x (X**state_1) x (X**state_2) ....
    """
    for r_qubit, s in zip(reg_qubits, state):
        if s == 0:
            qml.PauliX((r_qubit,))
        # qml.Snapshot(f"{debug_str} flipped {r_qubit}, if {s} == 0")


def simple_single_oracle(reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit):
    """
    If the reg_qubits are in the state state, then the oracle_qubit is flipped
    :param reg_qubits: list of qubits
    :param state: list of ints
    :param ancilla_qubits: list of qubits
    :param unclean_qubits: list of qubits
    :param oracle_qubit: int single qubit
    """
    set_state_to_ones(reg_qubits, state)
    adaptive_ccnot(reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    set_state_to_ones(reg_qubits, state)


def cc_simple_single_oracle(control_qubits, reg_qubits, state, ancilla_qubits, unclean_qubits, oracle_qubit, debug_str=""):
    """
    Executes the simple_single_oracle in a controlled manner.
    :param control_qubits: list of qubits controlling the circuit
    :param reg_qubits: list of qubits
    :param state: list of ints
    :param ancilla_qubits: list of qubits
    :param unclean_qubits: list of qubits
    :param oracle_qubit: int single qubit target qubit
    """
    set_state_to_ones(reg_qubits, state, debug_str=f"{debug_str} reg {reg_qubits} from {''.join([str(el) for el in state])} to " + ("1"*len(state)))
    qml.Snapshot(f"{debug_str} reg {reg_qubits} from {''.join([str(el) for el in state])} to " + ("1"*len(state)))
    adaptive_ccnot(control_qubits + reg_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
    qml.Snapshot(f"{debug_str} {''.join([str(el) for el in state])}: adaptive_ccnot(c={control_qubits + reg_qubits}, a:{ancilla_qubits}, u: {unclean_qubits}, o: {oracle_qubit}")
    set_state_to_ones(reg_qubits, state, debug_str=f"{debug_str} reg {reg_qubits} from {('1'*len(state))} to {''.join([str(el) for el in state])}")
    qml.Snapshot(f"{debug_str} reg {reg_qubits} from {('1'*len(state))} to {''.join([str(el) for el in state])}")
