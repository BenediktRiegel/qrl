import pennylane as qml
from typing import List
import torch
from utils import int_to_bitlist
from simple_oracle import cc_simple_single_oracle, simple_single_oracle
from enum import Enum


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class QNN:
    """
    This QNN does nothing. It was easier to implement stuff, using this as a parameter type
    """
    def circuit(
            self, input_qubits: List[int], result_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        raise NotImplementedError("Not implemented")

    def get_circuit(
            self, input_qubits: List[int], result_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return []



class CCRYQNN_Excessive:
    def __init__(
            self, num_input_qubits: int, depth: int, weight_init: WeightInitEnum
    ):
        """
        Initialise parameters with shape (depth, 2 ** num_input_qubits) and the specified weight initialisation method.
        :param num_input_qubits: int
        :param depth: int
        :param weight_init: WeightInitEnum
        """
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.randn((depth, 2 ** num_input_qubits)),
                                                      requires_grad=True)
        elif weight_init == WeightInitEnum.uniform:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.rand((depth, 2 ** num_input_qubits)),
                                                      requires_grad=True)
        elif weight_init == WeightInitEnum.zero:
            self.in_q_parameters = torch.nn.Parameter(torch.zeros((depth, 2 ** num_input_qubits)), requires_grad=True)
        else:
            raise NotImplementedError("Unknown weight init method")

    def layer(
            self, layer_num: int,
            control_qubits: List[int],
            input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        """
        Implements a single layer of the QNN. If the input qubits are in state |j>, then it performs a RY rotation
        with the $j$'th parameter
        """
        for idx, q_param in enumerate(self.in_q_parameters[layer_num]):
            bits = int_to_bitlist(idx, self.num_input_qubits)
            cc_simple_single_oracle(control_qubits, input_qubits, bits, ancilla_qubits[1:], unclean_qubits,
                                    ancilla_qubits[0])
            qml.CRY(phi=q_param, wires=(ancilla_qubits[0], result_qubit))
            cc_simple_single_oracle(control_qubits, input_qubits, bits, ancilla_qubits[1:], unclean_qubits,
                                    ancilla_qubits[0])

    def circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        """
        Constructs the quantum circuit of the QNN layer for layer
        :param input_qubits: list of qubits
        :param result_qubit: int single qubit
        :param control_qubits: list of qubits
        :param ancilla_qubits: list of qubits
        :param unclean_qubits: list of qubits
        """
        control_qubits = [] if control_qubits is None else control_qubits
        ancilla_qubits = [] if ancilla_qubits is None else ancilla_qubits
        unclean_qubits = [] if unclean_qubits is None else unclean_qubits
        for d in range(self.depth):
            self.layer(d, control_qubits, input_qubits, result_qubit, ancilla_qubits, unclean_qubits)

    def get_circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        """
        Returns the circuit as a function that does not require any more inputs
        :param input_qubits: list of qubits
        :param result_qubit: int qubit
        :param control_qubits: list of qubits
        :param ancilla_qubits: list of qubits
        :param unclean_qubits list of qubits
        :return: function
        """
        def circuit():
            return self.circuit(input_qubits, result_qubit, control_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        """
        Returns parameters
        :return: list of parameters
        """
        return [self.in_q_parameters]


class RYQNN_Excessive:
    def __init__(
            self, num_input_qubits: int, num_output_qubits: int, depth: int, weight_init: WeightInitEnum
    ):
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            self.in_q_parameters = torch.nn.Parameter(
                torch.pi * torch.randn((depth, 2 ** self.num_input_qubits, num_output_qubits + 1)), requires_grad=True)
        elif weight_init == WeightInitEnum.uniform:
            self.in_q_parameters = torch.nn.Parameter(
                torch.pi * torch.rand((depth, 2 ** self.num_input_qubits, num_output_qubits + 1)), requires_grad=True)
        elif weight_init == WeightInitEnum.zero:
            self.in_q_parameters = torch.nn.Parameter(torch.zeros((depth, 2 ** self.num_input_qubits, num_output_qubits)),
                                                      requires_grad=True)
        else:
            raise NotImplementedError("Unknown weight init method")

    def layer(
            self, layer_num: int,
            input_qubits: List[int], output_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        for idx, q_out_params in enumerate(self.in_q_parameters[layer_num]):
            bits = int_to_bitlist(idx, self.num_input_qubits)
            simple_single_oracle(input_qubits, bits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])
            for out_q, q_param in zip(output_qubits, q_out_params[:-1]):
                qml.CRY(phi=q_param, wires=(ancilla_qubits[0], out_q))
                qml.CRY(phi=q_out_params[-1], wires=(ancilla_qubits[0], out_q))
            simple_single_oracle(input_qubits, bits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])

    def circuit(
            self,
            input_qubits: List[int], output_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        ancilla_qubits = [] if ancilla_qubits is None else ancilla_qubits
        unclean_qubits = [] if unclean_qubits is None else unclean_qubits
        for d in range(self.depth):
            self.layer(d, input_qubits, output_qubits, ancilla_qubits, unclean_qubits)

    def get_circuit(
            self,
            input_qubits: List[int], output_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, output_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.in_q_parameters]
