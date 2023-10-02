from typing import List
from qnns import QNN
from qnns.weight_init import WeightInitEnum
from utils import bitlist_to_int
import torch
import pennylane as qml
from ccnot import adaptive_ccnot


class RYQNN_Excessive(QNN):
    def __init__(
            self, num_input_bits: int, num_result_qubits: int, depth: int, weight_init: WeightInitEnum
    ):
        QNN.__init__(self, num_input_bits, num_result_qubits, depth)

        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.randn((depth, self.num_result_qubits, 2 ** self.num_input_bits)),
                                                      requires_grad=True)
        elif weight_init == WeightInitEnum.uniform:
            self.in_q_parameters = torch.nn.Parameter(torch.pi * torch.rand((depth, self.num_result_qubits, 2 ** self.num_input_bits)),
                                                      requires_grad=True)
        elif weight_init == WeightInitEnum.zero:
            self.in_q_parameters = torch.nn.Parameter(torch.zeros((depth, self.num_result_qubits, 2 ** self.num_input_bits)), requires_grad=True)
        else:
            raise NotImplementedError("Unknown weight init method")

    def layer(
            self, layer_num: int,
            r_idx: int,
            control_qubits: List[int],
            input_bits: List[int],
            result_qubit: int,
            ancilla_qubits: List[int],
            unclean_qubits: List[int],
    ):
        q_param = self.in_q_parameters[layer_num, r_idx, bitlist_to_int(input_bits)]
        if control_qubits:
            adaptive_ccnot(control_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])
            qml.CRY(phi=q_param, wires=(ancilla_qubits[0], result_qubit))
            adaptive_ccnot(control_qubits, ancilla_qubits[1:], unclean_qubits, ancilla_qubits[0])
        else:
            qml.RY(phi=q_param, wires=(result_qubit,))

    def circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        unclean_qubits = [] if unclean_qubits is None else unclean_qubits
        unclean_qubits += additional_qubits
        for d in range(self.depth):
            for r_idx, r_qubit in enumerate(result_qubits[:self.num_result_qubits]):
                self.layer(d, r_idx, control_qubits, input_bits, r_qubit, ancilla_qubits, unclean_qubits)

    def conj_circuit(self, control_qubits: List[int], input_bits: List[int], result_qubits: List[int], additional_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int] = None):
        self.circuit(control_qubits, input_bits, result_qubits, additional_qubits, ancilla_qubits, unclean_qubits)

    def parameters(self):
        return [self.in_q_parameters]

    def num_additional_and_ancilla_qubits(self):
        return 0, 1
