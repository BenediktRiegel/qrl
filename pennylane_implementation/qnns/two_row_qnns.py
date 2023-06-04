import pennylane as qml
from typing import List
from ccnot import adaptive_ccnot
import torch
from qnns.weight_init import WeightInitEnum


class QNN:
    def __init__(
            self, num_input_qubits: int, num_result_qubits: int, depth: int
    ):
        self.num_input_qubits = num_input_qubits
        self.num_result_qubits = num_result_qubits
        self.depth = depth
        self.q_parameters = torch.nn.Parameter(torch.rand((depth, num_result_qubits, 2)), requires_grad=True)

    def loading_layer(self, input_qubits: List[int], result_qubits: List[int]):
        for idx, c_qubit in enumerate(input_qubits):
            qml.CNOT((c_qubit, result_qubits[idx % len(result_qubits)]))

    def entanglement_layer(self, result_qubits: List[int]):
        for c_qubit, t_qubit in zip(result_qubits[:-1], result_qubits[1:]):
            qml.CNOT((c_qubit, t_qubit))
        qml.CNOT((result_qubits[-1], result_qubits[0]))

    def parameter_layer(self, layer_num: int, result_qubits: List[int]):
        for r_qubit, params in zip(result_qubits, self.q_parameters[layer_num]):
            qml.RY(params[0], r_qubit)
            qml.RX(params[1], r_qubit)

    def layer(
            self, layer_num: int,
            input_qubits: List[int], result_qubits: List[int]
    ):
        self.loading_layer(input_qubits, result_qubits)
        self.parameter_layer(layer_num, result_qubits)
        self.entanglement_layer(result_qubits)

    def circuit(
            self, input_qubits: List[int], result_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        for d in range(self.depth):
            self.layer(d, input_qubits, result_qubits)

    def get_circuit(
            self, input_qubits: List[int], result_qubits: List[int],
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.q_parameters]


class RotQNN:
    def __init__(
            self, num_input_qubits: int, depth: int
    ):
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        self.in_q_parameters = torch.nn.Parameter(torch.rand((depth, num_input_qubits)), requires_grad=True)
        self.out_q_parameters = torch.nn.Parameter(torch.rand((depth, 2)), requires_grad=True)

    def layer(
            self, layer_num: int,
            input_qubits: List[int], result_qubit: int,
    ):
        for idx, in_qubit in enumerate(input_qubits):
            qml.CRX(phi=self.in_q_parameters[layer_num, idx], wires=(in_qubit, result_qubit))

        qml.RZ(phi=self.out_q_parameters[layer_num, 1], wires=(result_qubit,))
        qml.RY(phi=self.out_q_parameters[layer_num, 0], wires=(result_qubit,))

    def circuit(
            self, input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        for d in range(self.depth):
            self.layer(d, input_qubits, result_qubit)

    def get_circuit(
            self, input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubit, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.in_q_parameters, self.out_q_parameters]


class CCRotQNN:
    def __init__(
            self, num_input_qubits: int, depth: int
    ):
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        self.in_q_parameters = torch.nn.Parameter(torch.rand((depth, num_input_qubits)), requires_grad=True)
        self.out_q_parameters = torch.nn.Parameter(torch.rand((depth, 2)), requires_grad=True)

    def layer(
            self, layer_num: int,
            control_qubits: List[int],
            input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        if len(control_qubits) == 0:
            for idx, in_qubit in enumerate(input_qubits):
                qml.CRX(phi=self.in_q_parameters[layer_num, idx], wires=(in_qubit, result_qubit))

            qml.RZ(phi=self.out_q_parameters[layer_num, 1], wires=(result_qubit,))
            qml.RY(phi=self.out_q_parameters[layer_num, 0], wires=(result_qubit,))
        else:
            oracle_qubit = ancilla_qubits[0]
            ancilla_qubits = ancilla_qubits[1:]
            for idx, in_qubit in enumerate(input_qubits):
                adaptive_ccnot(control_qubits + [in_qubit], ancilla_qubits, unclean_qubits, oracle_qubit)
                qml.CRX(phi=self.in_q_parameters[layer_num, idx], wires=(oracle_qubit, result_qubit))
                adaptive_ccnot(control_qubits + [in_qubit], ancilla_qubits, unclean_qubits, oracle_qubit)

            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
            qml.CRZ(phi=self.out_q_parameters[layer_num, 1], wires=(oracle_qubit, result_qubit))
            qml.CRY(phi=self.out_q_parameters[layer_num, 0], wires=(oracle_qubit, result_qubit))
            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)

    def circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        control_qubits = [] if control_qubits is None else control_qubits
        for d in range(self.depth):
            self.layer(d, control_qubits, input_qubits, result_qubit, ancilla_qubits, unclean_qubits)

    def get_circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubit, control_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.in_q_parameters, self.out_q_parameters]


class CCRotQNN2:
    def __init__(
            self, num_input_qubits: int, depth: int
    ):
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        # self.in_q_parameters = torch.nn.Parameter(torch.rand((depth, num_input_qubits)), requires_grad=True)
        self.out_q_parameters = torch.nn.Parameter(torch.rand((depth, 2)), requires_grad=True)

    def layer(
            self, layer_num: int,
            control_qubits: List[int],
            input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        value = 1
        factor = torch.pi / 2**(len(input_qubits)+1)
        if len(control_qubits) == 0:
            for in_qubit in input_qubits:
                qml.CRX(phi=value * factor, wires=(in_qubit, result_qubit))
                value *= 2

            qml.RZ(phi=self.out_q_parameters[layer_num, 1], wires=(result_qubit,))
            qml.RY(phi=self.out_q_parameters[layer_num, 0], wires=(result_qubit,))
        else:
            oracle_qubit = ancilla_qubits[0]
            ancilla_qubits = ancilla_qubits[1:]
            for in_qubit in input_qubits:
                qml.CRX(phi=value * factor, wires=(in_qubit, result_qubit))
                value *= 2

            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
            qml.CRZ(phi=self.out_q_parameters[layer_num, 1], wires=(oracle_qubit, result_qubit))
            qml.CRY(phi=self.out_q_parameters[layer_num, 0], wires=(oracle_qubit, result_qubit))
            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)

    def circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        control_qubits = [] if control_qubits is None else control_qubits
        for d in range(self.depth):
            self.layer(d, control_qubits, input_qubits, result_qubit, ancilla_qubits, unclean_qubits)

    def get_circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubit, control_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.out_q_parameters]


class CCRYQNN:
    def __init__(
            self, num_input_qubits: int, depth: int, weight_init: WeightInitEnum
    ):
        self.num_input_qubits = num_input_qubits
        self.depth = depth
        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.randn((depth, num_input_qubits)), requires_grad=True)
            self.out_q_parameters = torch.nn.Parameter(torch.pi * torch.randn((depth,)), requires_grad=True)
        elif weight_init == WeightInitEnum.uniform:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.rand((depth, num_input_qubits)), requires_grad=True)
            self.out_q_parameters = torch.nn.Parameter(torch.pi * torch.rand((depth,)), requires_grad=True)
        elif weight_init == WeightInitEnum.zero:
            self.in_q_parameters = torch.nn.Parameter(torch.zeros((depth, num_input_qubits)), requires_grad=True)
            self.out_q_parameters = torch.nn.Parameter(torch.zeros((depth,)), requires_grad=True)
        else:
            raise NotImplementedError("Unknown weight init method")

    def layer(
            self, layer_num: int,
            control_qubits: List[int],
            input_qubits: List[int], result_qubit: int,
            ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        # value = 1
        # factor = torch.pi / 2**(len(input_qubits)+1)
        if len(control_qubits) == 0:
            for idx, in_qubit in enumerate(input_qubits):
                # qml.CRY(phi=value * factor, wires=(in_qubit, result_qubit))
                qml.CRY(phi=self.in_q_parameters[layer_num, idx], wires=(in_qubit, result_qubit))
                # value *= 2

            qml.RY(phi=self.out_q_parameters[layer_num], wires=(result_qubit,))
        else:
            oracle_qubit = ancilla_qubits[0]
            ancilla_qubits = ancilla_qubits[1:]
            for idx, in_qubit in enumerate(input_qubits):
                # qml.CRY(phi=value * factor, wires=(in_qubit, result_qubit))
                adaptive_ccnot(control_qubits + [in_qubit], ancilla_qubits, unclean_qubits, oracle_qubit)
                qml.CRY(phi=self.in_q_parameters[layer_num, idx], wires=(oracle_qubit, result_qubit))
                adaptive_ccnot(control_qubits + [in_qubit], ancilla_qubits, unclean_qubits, oracle_qubit)
                # value *= 2

            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)
            qml.CRY(phi=self.out_q_parameters[layer_num], wires=(oracle_qubit, result_qubit))
            adaptive_ccnot(control_qubits, ancilla_qubits, unclean_qubits, oracle_qubit)

    def circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None,
    ):
        control_qubits = [] if control_qubits is None else control_qubits
        for d in range(self.depth):
            self.layer(d, control_qubits, input_qubits, result_qubit, ancilla_qubits, unclean_qubits)

    def get_circuit(
            self,
            input_qubits: List[int], result_qubit: int,
            control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None
    ):
        def circuit():
            return self.circuit(input_qubits, result_qubit, control_qubits, ancilla_qubits, unclean_qubits)

        return circuit

    def parameters(self):
        return [self.in_q_parameters, self.out_q_parameters]
