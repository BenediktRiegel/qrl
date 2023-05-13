from typing import List
import pennylane as qml
from numpy import pi, arcsin, sqrt
from copy import deepcopy
from qnns import QNN
from environment.frozen_lake2_2 import FrozenLake2_2


def loss_circuit(
    action_qnn: QNN, value_qnn: QNN, gradient_free_value_qnn: QNN, environment: FrozenLake2_2,
    x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
    next_x_qubits: List[int], next_y_qubits: List[int],
    loss1_qubit: int, loss2_qubit: int,
    ancilla_qubits: List[int], unclean_qubits: List[int] = None,
    gamma: float = 0.9,
    l_type: int = 0,
):
    r_max = environment.r_m
    v_max = r_max / (1 - gamma)     #v_max (<)= r_max + gamma * v_max => v_max = r_max / (1-gamma)
    p_max = r_max + v_max  # max value for r + gamma * v <= r_max + gamma * v_max <= r_max + gamma * r_max / (1-gamma) = (1 + gamma/(1-gamma))*r_max
    q_max = r_max + 2*v_max  # max value for |v(s) - (r + v(s'))| <= v_max + r_max + gamma * v_max <= 2r_max + r_max + gamma * 2r_max = (3 + 2*gamma) * r_max

    ancilla_qubits = ancilla_qubits
    # Determine next action
    action_qnn.circuit(x_qubits + y_qubits, action_qubits)

    r_factor = [pi / p_max, pi / q_max]
    # r_factor = [q_max, q_max]

    # Transistion
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, [loss1_qubit, loss2_qubit], r_factor, ancilla_qubits, unclean_qubits
    )

    # Determine state values
    gamma_qubit = ancilla_qubits[0]
    ancilla_qubits = ancilla_qubits[1:]
    qml.RY(phi=2*arcsin(sqrt(gamma)), wires=(gamma_qubit,))
    gradient_free_value_qnn.circuit(next_x_qubits+next_y_qubits, loss2_qubit, [gamma_qubit], ancilla_qubits, unclean_qubits=x_qubits+y_qubits+action_qubits+[loss1_qubit])
    qml.adjoint(value_qnn.circuit)(x_qubits+y_qubits, loss2_qubit, [], ancilla_qubits, unclean_qubits=next_x_qubits+next_y_qubits+action_qubits+[loss1_qubit])

    gradient_free_value_qnn.circuit(next_x_qubits+next_y_qubits, loss1_qubit, [gamma_qubit], ancilla_qubits, unclean_qubits=x_qubits+y_qubits+action_qubits+[loss2_qubit])


def loss_function(
        action_qnn: QNN, value_qnn: QNN, environment: FrozenLake2_2,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int],
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
            next_x_qubits, next_y_qubits,
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
