import pennylane as qml
import numpy as np
from load_data import LittleTreeLoader
from wire_utils import get_wires
from debug_utils import snapshots_to_probability_strings


def main():
    wires, total_num_wires = get_wires([1, 1, 1, 1])
    c_qubits_a, data_qubit, c_qubits, ancilla_qubits = wires

    backend = qml.device("default.qubit", wires=total_num_wires)

    def circuit():
        qml.PauliX((c_qubits[0],))
        qml.PauliX((c_qubits_a[0],))
        LittleTreeLoader(np.array([[0., 1.]]), data_qubit, ancilla_qubits).circuit(c_qubits + c_qubits_a)
        qml.Snapshot("result")
        return qml.probs(wires=[0, 1])

    snaps = qml.snapshots(qml.QNode(circuit, backend))()
    for s in snapshots_to_probability_strings(snaps):
        print(s)


if __name__ == "__main__":
    main()
