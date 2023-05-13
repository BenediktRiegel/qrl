from typing import List
import pennylane as qml
import numpy as np
from utils import get_float_by_interpretation
from qnns import QNN
from environment import Environment
from copy import deepcopy


def loss_circuit(
    action_qnn: QNN, value_qnn: QNN, environment: Environment,
    x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
    next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
    value_qubits: List[int], next_value_qubits: List[int],
    ancilla_qubits: List[int], unclean_qubits: List[int] = None
):
    # Determine next action
    action_qnn.circuit(x_qubits + y_qubits, action_qubits)

    # Transistion
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, ancilla_qubits, unclean_qubits
    )

    # Determine state values
    value_qnn.circuit(x_qubits+y_qubits, value_qubits)
    value_qnn.circuit(next_x_qubits+next_y_qubits, next_value_qubits)


def process_sample(sample, qubit_registers: List[List[int]], reg_interpretation: List[List[str]]):
    results = []
    for reg, interpretation in zip(qubit_registers, reg_interpretation):
        results.append(get_float_by_interpretation([sample[idx] for idx in reg], interpretation))
    return results


def get_value_interpretation(r_qubit_interpretation, value_qubits):
    value_interpretation = r_qubit_interpretation[:len(value_qubits)]
    num_missing_values = len(value_qubits) - len(value_interpretation)
    min_interpretation = np.array([int(el) for el in value_interpretation if el != "sign"]).min()
    value_interpretation += [str(el) for el in list(range(min_interpretation - 1, min_interpretation - num_missing_values - 1, -1))]
    return value_interpretation


def get_probabilities_and_values(sample_values):
    prob_and_val_dict = {}
    for s_values in sample_values:
        str_name = ",".join([str(el.item()) for el in s_values])
        if str_name not in prob_and_val_dict:
            prob_and_val_dict[str_name] = [0, s_values]
        prob_and_val_dict[str_name][0] += 1

    for k, v in prob_and_val_dict.items():
        prob_and_val_dict[k][0] = v[0] / len(sample_values)

    return prob_and_val_dict


def loss_function(
    action_qnn: QNN, value_qnn: QNN, environment: Environment,
    x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
    next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
    value_qubits: List[int], next_value_qubits: List[int],
    ancilla_qubits: List[int],
    backend,
    gamma: float = 0.9, lam: float = 1,
    unclean_qubits: List[int] = None,
):
    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))

        loss_circuit(
            action_qnn, value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits, r_qubits,
            value_qubits, next_value_qubits,
            ancilla_qubits, unclean_qubits
        )
        return qml.sample(wires=x_qubits + y_qubits + value_qubits + next_x_qubits + next_y_qubits + r_qubits + next_value_qubits)

    samples = qml.QNode(circuit, backend, interface="torch", diff_method="best")()
    temp_registers = [x_qubits, y_qubits, value_qubits, next_x_qubits, next_y_qubits, r_qubits, next_value_qubits]
    registers = []
    total = 0
    for t_reg in temp_registers:
        registers.append(list(range(total, total+len(t_reg))))
        total += len(t_reg)
    reg_interpretations = [
        [str(el) for el in range(len(x_qubits) - 1, -1, -1)], [str(el) for el in range(len(y_qubits) - 1, -1, -1)],
        get_value_interpretation(environment.r_qubit_interpretation, value_qubits),
        [str(el) for el in range(len(next_x_qubits) - 1, -1, -1)], [str(el) for el in range(len(next_y_qubits) - 1, -1, -1)],
        environment.r_qubit_interpretation,
        get_value_interpretation(environment.r_qubit_interpretation, next_value_qubits)
    ]
    sample_values = [process_sample(s, registers, reg_interpretations) for s in samples]
    probabilities_and_values = get_probabilities_and_values(sample_values)

    loss_value = None
    for k, p_and_v in probabilities_and_values.items():
        p = p_and_v[0]
        v = p_and_v[1]
        s_value = v[2]
        r = v[5]
        next_s_value = v[6]
        if loss_value is None:
            loss_value = lam * np.abs(s_value - p * (r + gamma*next_s_value)) - p * (r + gamma*next_s_value)
        else:
            loss_value += lam * np.abs(s_value - p * (r + gamma*next_s_value)) - p * (r + gamma*next_s_value)

    return loss_value


def rotate_values(
    value_qubits: List[int],
    value_interpretation: List[str],
    rotate_qubit: int,
    ancilla_qubits: List[int],
    factor: float = 1.
):
    sign_qubit = ancilla_qubits[0]
    ancilla_qubits = ancilla_qubits[1:]

    # Check sign
    for v_qubit, interpretation in zip(value_qubits, value_interpretation):
        if interpretation == "sign":
            qml.CNOT((v_qubit, sign_qubit))

    # Rotate
    for v_qubit, interpretation in zip(value_qubits, value_interpretation):
        if interpretation != "sign":
            # Add if sign_qubit is 0
            qml.PauliX((sign_qubit,))
            qml.Toffoli((v_qubit, sign_qubit, ancilla_qubits[0]))
            qml.CRY(phi=2**int(interpretation)*factor, wires=(ancilla_qubits[0], rotate_qubit))
            qml.Toffoli((v_qubit, sign_qubit, ancilla_qubits[0]))
            qml.PauliX((sign_qubit,))
            # Subtract if sign_qubit is 1
            qml.Toffoli((v_qubit, sign_qubit, ancilla_qubits[0]))
            qml.CRY(phi=-1*2**int(interpretation)*factor, wires=(ancilla_qubits[0], rotate_qubit))
            qml.Toffoli((v_qubit, sign_qubit, ancilla_qubits[0]))

    # Undo sign check
    for v_qubit, interpretation in zip(value_qubits, value_interpretation):
        if interpretation == "sign":
            qml.CNOT((v_qubit, sign_qubit))


def loss_circuit2(
    action_qnn: QNN, value_qnn: QNN, gradient_free_value_qnn: QNN, environment: Environment,
    x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
    next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
    value_qubits: List[int], next_value_qubits: List[int],
    max_interpretation: float,
    rotation_value_qubits: List[int],
    ancilla_qubits: List[int], unclean_qubits: List[int] = None,
    gamma: float = 0.9,
):
    ancilla_qubits = value_qubits + next_value_qubits + rotation_value_qubits + ancilla_qubits
    # Determine next action
    action_qnn.circuit(x_qubits + y_qubits, action_qubits)

    # Transistion
    environment.circuit(
        x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, ancilla_qubits, unclean_qubits
    )
    ancilla_qubits = ancilla_qubits[len(value_qubits) + len(next_value_qubits):]

    # Determine state values
    value_qnn.circuit(x_qubits+y_qubits, value_qubits)
    gradient_free_value_qnn.circuit(next_x_qubits+next_y_qubits, next_value_qubits)

    # Rotate rotation value qubits
    ancilla_qubits = ancilla_qubits[len(rotation_value_qubits):]
    # Rotate v(s) - r + gamma * v(s_next). we start at 0 and can move to pi, with 0 being the best
    # Rotate by v(s)
    rotate_values(
        value_qubits,
        get_value_interpretation(environment.r_qubit_interpretation, value_qubits),
        rotation_value_qubits[0],
        ancilla_qubits,
        factor=np.pi/max_interpretation
    )
    # Rotate by -1 * gamma * v(s_next)
    rotate_values(
        next_value_qubits,
        get_value_interpretation(environment.r_qubit_interpretation, next_value_qubits),
        rotation_value_qubits[0],
        ancilla_qubits,
        factor=-1*gamma*np.pi / max_interpretation
    )
    # Rotate by -1 * r
    rotate_values(
        r_qubits,
        environment.r_qubit_interpretation,
        rotation_value_qubits[0],
        ancilla_qubits,
        factor=-1*np.pi / max_interpretation
    )

    # Rotate -r - gamma * v(s_next). we start at pi/2 and can move to 0 or pi, with 0 being the best
    qml.RY(phi=np.pi/2., wires=rotation_value_qubits[1])
    # Rotate by -1 * gamma * v(s_next)
    rotate_values(
        value_qubits,
        get_value_interpretation(environment.r_qubit_interpretation, next_value_qubits),
        rotation_value_qubits[0],
        ancilla_qubits,
        factor=-1 * gamma * np.pi/2. / max_interpretation
    )
    # Rotate by -1 * r
    rotate_values(
        value_qubits,
        get_value_interpretation(environment.r_qubit_interpretation, value_qubits),
        rotation_value_qubits[0],
        ancilla_qubits,
        factor=-1 * np.pi/2. / max_interpretation
    )


def loss_function2(
        action_qnn: QNN, value_qnn: QNN, environment: Environment,
        x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
        value_qubits: List[int], next_value_qubits: List[int],
        rotation_value_qubits: List[int],
        ancilla_qubits: List[int],
        backend,
        gamma: float = 0.9,
        lam: float = 1,
        unclean_qubits: List[int] = None,
        use_exp_value: bool = False,
):
    value_interpretation = get_value_interpretation(environment.r_qubit_interpretation, value_qubits)
    max_value = np.sum([2**int(el) for el in value_interpretation if el != "sign"])
    next_value_interpretation = get_value_interpretation(environment.r_qubit_interpretation, next_value_qubits)
    max_next_value = np.sum([2**int(el) for el in next_value_interpretation if el != "sign"])
    max_reward = np.sum([2**int(el) for el in environment.r_qubit_interpretation if el != "sign"])
    max_interpretation = np.max([max_value, max_next_value, max_reward, max_reward + gamma * max_next_value])

    gradient_free_value_qnn = deepcopy(value_qnn)
    for parameter in gradient_free_value_qnn.parameters():
        parameter.requires_grad = False

    def circuit():
        for s_qubit in x_qubits + y_qubits:
            qml.Hadamard((s_qubit,))

        loss_circuit2(
            action_qnn, value_qnn, gradient_free_value_qnn, environment,
            x_qubits, y_qubits, action_qubits,
            next_x_qubits, next_y_qubits, r_qubits,
            value_qubits, next_value_qubits,
            max_interpretation,
            rotation_value_qubits,
            ancilla_qubits, unclean_qubits,
            gamma,
        )

        if use_exp_value:
            return [qml.probs(qml.PauliZ(wires=rot_v_qubit)) for rot_v_qubit in rotation_value_qubits]
        else:
            return [qml.probs(wires=rot_v_qubit) for rot_v_qubit in rotation_value_qubits]

    result = qml.QNode(circuit, backend, interface="torch", diff_method="best")()
    if use_exp_value:
        result[0] = (-1*result[0]) + 1
        result[1] = (-1 * result[1]) + 1
    else:
        result = [
            result[0][0],
            result[1][0],
        ]

    return lam * result[0] + result[1]
