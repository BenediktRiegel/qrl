from typing import List, Tuple
from abc import ABCMeta, abstractmethod


class QNN(metaclass=ABCMeta):
    def __init__(
            self, num_input_bits: int, num_result_qubits: int, depth: int
    ):
        self.num_input_bits = num_input_bits
        self.num_result_qubits = num_result_qubits
        self.depth = depth

    @abstractmethod
    def circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        """
        Implements the quantum circuit of the qnn
        :control_qubits: A list of qubits which control the qnn execution
        :input_bits: Classical bits that serve as the input of the QNN
        :result_qubits: A list of qubits which store the important output of the qnn
        :additional_qubits: A list of qubits that help to implement the qnn and will be unclean after
        :ancilla_qubits: A list of qubits that help to implement the qnn and will be clean after, i.e. |0>
        """
        raise NotImplementedError("Method circuit is not implemented")

    def get_circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        """
        Returns a method that implements the quantum circuit of the qnn. The returned method requires no parameters.
        :control_qubits: A list of qubits which control the qnn execution
        :input_bits: Classical bits that serve as the input of the QNN
        :result_qubits: A list of qubits which store the important output of the qnn
        :additional_qubits: A list of qubits that help to implement the qnn and will be unclean after
        :ancilla_qubits: A list of qubits that help to implement the qnn and will be clean after, i.e. |0>
        """
        def circuit():
            return self.circuit(control_qubits, input_bits, result_qubits, additional_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    @abstractmethod
    def conj_circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        """
        Implements the conjugated quantum circuit of the qnn
        :control_qubits: A list of qubits which control the qnn execution
        :input_bits: Classical bits that serve as the input of the QNN
        :result_qubits: A list of qubits which store the important output of the qnn
        :additional_qubits: A list of qubits that help to implement the qnn and will be unclean after
        :ancilla_qubits: A list of qubits that help to implement the qnn and will be clean after, i.e. |0>
        """
        raise NotImplementedError("Method conj_circuit is not implemented")

    def get_conj_circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        """
        Returns a method that implements the conjugated quantum circuit of the qnn. The returned method requires no parameters.
        :control_qubits: A list of qubits which control the qnn execution
        :input_bits: Classical bits that serve as the input of the QNN
        :result_qubits: A list of qubits which store the important output of the qnn
        :additional_qubits: A list of qubits that help to implement the qnn and will be unclean after
        :ancilla_qubits: A list of qubits that help to implement the qnn and will be clean after, i.e. |0>
        """
        def circuit():
            return self.circuit(control_qubits, input_bits, result_qubits, additional_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    @abstractmethod
    def parameters(self):
        """
        Returns all parameters used by the qnn
        """
        raise NotImplementedError("Method parameters is not implemented")

    @abstractmethod
    def num_additional_and_ancilla_qubits(self) -> Tuple[int, int]:
        """
        Returns a tuple with the amount of additional and ancilla qubits needed
        """
