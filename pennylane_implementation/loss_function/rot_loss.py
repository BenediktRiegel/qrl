from typing import List
import pennylane as qml
from numpy import pi
from copy import deepcopy
from qnns import QNN
from environment import Environment


def loss_circuit(
    action_qnn: QNN, value_qnn: QNN, gradient_free_value_qnn: QNN, environment: Environment,
    x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
    next_x_qubits: List[int], next_y_qubits: List[int], r_qubit: int,
    value_qubit: int, next_value_qubit: int,
    loss1_qubit: int, loss2_qubit: int,
    ancilla_qubits: List[int], unclean_qubits: List[int] = None,
    gamma: float = 0.9,
    l_type: int = 0,
):
    r_max = environment.r_m
    v_max = r_max / (1 - gamma)     #v_max (<)= r_max + gamma * v_max => v_max = r_max / (1-gamma)
    p_max = r_max + gamma * v_max  # max value for r + gamma * v <= r_max + gamma * v_max <= r_max + gamma * r_max / (1-gamma) = (1 + gamma/(1-gamma))*r_max
    q_max = r_max + (1+gamma)*v_max  # max value for |v(s) - (r + gamma*v(s'))| <= v_max + r_max + gamma * v_max <= 2r_max + r_max + gamma * 2r_max = (3 + 2*gamma) * r_max

    ancilla_qubits = [value_qubit, next_value_qubit, loss1_qubit, loss2_qubit] + ancilla_qubits
    # Determine next action
    action_qnn.circuit(x_qubits + y_qubits, action_qubits)

    # Transistion
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubit, ancilla_qubits, unclean_qubits
    )
    ancilla_qubits = ancilla_qubits[4:]

    # Determine state values
    value_qnn.circuit(x_qubits+y_qubits, value_qubit)
    gradient_free_value_qnn.circuit(next_x_qubits+next_y_qubits, next_value_qubit)

    # Rotate rotation loss1 qubit   (action_qnn's loss)
    # if l_type != 2:
    qml.CRY(phi=(gamma * pi/p_max), wires=(next_value_qubit, loss1_qubit))
    qml.CRY(phi=(pi/p_max), wires=(r_qubit, loss1_qubit))

    # Rotate rotation loss2 qubit   (value_qnn's loss)
    # if l_type != 1:
    qml.CRY(phi=(gamma * 2), wires=(next_value_qubit, loss2_qubit))
    qml.CRY(phi=((1 - gamma) * 2), wires=(r_qubit, loss2_qubit))
    qml.CRY(phi=(-1*2), wires=(value_qubit, loss2_qubit))


def loss_function(
        action_qnn: QNN, value_qnn: QNN, environment: Environment,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int], r_qubit: int,
        value_qubit: int, next_value_qubit: int,
        loss1_qubit: int, loss2_qubit: int,
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9,
        lam: float = 1,
        unclean_qubits: List[int] = None,
        use_exp_value: bool = False,
        l_type: int = 0,
):

    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))

        loss_circuit(
            action_qnn, value_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits, r_qubit,
            value_qubit, next_value_qubit,
            loss1_qubit, loss2_qubit,
            ancilla_qubits, unclean_qubits,
            gamma,
            l_type=l_type,
        )

        if use_exp_value:
            return [qml.expval(qml.PauliZ(wires=loss1_qubit)), qml.expval(qml.PauliZ(wires=loss2_qubit))]
        else:
            return [qml.probs(wires=loss1_qubit), qml.probs(wires=loss2_qubit)]

    # backprop, parameter-shift
    result = qml.QNode(circuit, backend, interface="torch", diff_method="best")()
    if use_exp_value:
        # result[0] = result[0] + 1
        # result[1] = (-1 * result[1]) + 1
        result[0] = (-1 * result[0]) + 1
        result[1] = result[1] + 1
        # result[0] = result[0] + 1
        # result[1] = result[1] + 1
    else:
        result = [
            result[0][0],
            result[1][1],
        ]

    if l_type == 1:
        return result[0]
    if l_type == 2:
        return result[1]
    if l_type == 3:
        return result

    return lam * result[0] + result[1]
