from typing import List
from abc import ABCMeta, abstractmethod
import pennylane as qml


class Environment(metaclass=ABCMeta):
    @abstractmethod
    def circuit(
        self, x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
        ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        """Transistions a quantum agent"""

    def get_circuit(
        self, x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
        ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        def circuit():
            self.circuit(
                x_qubits, y_qubits, action_qubits,
                next_x_qubits, next_y_qubits, r_qubits,
                ancilla_qubits, unclean_qubits
            )
        return circuit

    def get_circuit_as_tensor(
        self, x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
        next_x_qubits: List[int], next_y_qubits: List[int], r_qubits: List[int],
        ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        return qml.matrix(
            self.circuit(
                x_qubits, y_qubits, action_qubits,
                next_x_qubits, next_y_qubits, r_qubits,
                ancilla_qubits, unclean_qubits=unclean_qubits
            )
        )()
