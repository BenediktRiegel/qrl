import pennylane as qml
import numpy as np


def get_controlled_one_qubit_unitary(U):
    import qiskit

    qc = qiskit.QuantumCircuit(1)
    qc.unitary(U, [0])
    c_qc = qc.control()
    sv_backend = qiskit.Aer.get_backend('statevector_simulator')
    qc_transpiled = qiskit.transpile(c_qc, backend=sv_backend, basis_gates=sv_backend.configuration().basis_gates,
                              optimization_level=3)
    converted = qml.from_qiskit(qc_transpiled)

    def circuit(c_wire, t_wire):
        converted(wires=(t_wire, c_wire))

    return circuit


def qml_get_controlled_one_qubit_unitary(U):
    from typing import Union
    U = U.copy()

    def circuit(c_wire, t_wire):
        if not isinstance(c_wire, Union[int]):
            c_wire = [c_wire]
        qml.ControlledQubitUnitary(U, wires=t_wire, control_wires=c_wire)

    return circuit


def compute_global_phase(U1, U2):
    global_phase = U1 / U2
    for i in range(2):
        for j in range(2):
            if global_phase[i, j] != 0:
                return global_phase[i, j]
    return 0


def compute_zyz_decomposition(U):
    U = np.array(U)
    rot = qml.transforms.zyz_decomposition(U, 0)[0]
    rot_matrix = np.array(rot.matrix())
    global_phase = compute_global_phase(U, rot_matrix)
    alpha = np.arccos(max(min(global_phase.real, 1), -1))
    # TASK_LOGGER.info(f"global_phase = {global_phase} alpha = {np.arccos(alpha)} other angles = {rot.single_qubit_rot_angles()}")
    angles = [np.arccos(alpha)] + list(rot.single_qubit_rot_angles())
    print(f"U = {U} and Rot = {rot_matrix*global_phase}")
    return angles


def controlled_one_qubit_unitary(zyz_angles, wires):
    wires = wires[:2]
    gamma_half = zyz_angles[2] / 2.
    qml.RZ(zyz_angles[1], wires=(wires[1],))
    qml.RY(gamma_half, wires=(wires[1],))
    qml.CNOT(wires)
    qml.RY(-1*gamma_half, wires=(wires[1],))
    qml.RZ(-0.5*(zyz_angles[3] + zyz_angles[1]), wires=(wires[1],))
    qml.CNOT(wires)
    qml.RZ(0.5*(zyz_angles[3] - zyz_angles[1]), wires=(wires[1],))
    qml.RZ(zyz_angles[0], wires=(wires[0],))
