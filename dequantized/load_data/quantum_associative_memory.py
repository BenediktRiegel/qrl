import pennylane as qml
import numpy as np
from typing import List
from ccnot import adaptive_ccnot
# from controlled_unitaries import get_controlled_one_qubit_unitary
from controlled_unitaries import qml_get_controlled_one_qubit_unitary as get_controlled_one_qubit_unitary
from utils import is_binary
from wire_utils import check_wires_uniqueness, check_num_wires


class QAM:
    def __init__(
        self,
        X: np.ndarray,
        c_wires: List[int],
        register_wires: List[int],
        ancilla_wires: List[int],
        unclean_wires: List[int] = None,
        amplitudes: List[float] = None,
        additional_bits: List[int] = None,
        additional_wires: List[int] = None,
    ):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if amplitudes is None:
            self.X = X
        else:
            amplitudes = np.array(amplitudes)
            non_zero_amps = np.nonzero(amplitudes)
            self.X = X[non_zero_amps]
            amplitudes = amplitudes[non_zero_amps].tolist()

        if not is_binary(self.X):
            raise ValueError(
                "A QAM (Quantum Associative Memory) can only load binary data"
            )
        self.xor_X = self.create_xor_X(self.X)

        if additional_bits is not None:
            if not is_binary(additional_bits):
                raise ValueError(
                    "A QAM (Quantum Associative Memory) can only load binary additional bits"
                )
            if not isinstance(additional_bits, np.ndarray):
                additional_bits = np.array(additional_bits)
            self.xor_additional_bits = self.create_xor_X(additional_bits)
        else:
            self.xor_additional_bits = None

        self.c_wires = c_wires
        self.register_wires = list(register_wires)
        self.ancilla_wires = list(ancilla_wires)
        self.additional_wires = [] if additional_wires is None else additional_wires
        self.unclean_wires = (
            [] if unclean_wires is None else unclean_wires
        )  # unclean wires are like ancilla wires, but they are not guaranteed to be 0

        ancillas_needed = (
            2 if self.X.shape[1] < 3 else 3
        )  # If we have less than 3 data qubits, we can use a Toffoli or a CNOT and thus we do not need an ancilla qubit for a CCCNOT
        wire_types = ["register", "ancilla", "additional", "unclean"]
        num_wires = [self.X.shape[1], ancillas_needed]
        error_msgs = ["the points' dimensionality.", str(ancillas_needed) + "."]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-2], num_wires, error_msgs)

        if additional_bits is not None:
            if (
                additional_wires is None
                or len(additional_wires) < additional_bits.shape[1]
            ):
                raise qml.QuantumFunctionError(
                    "The number of additional wires must be at least the same as the dimension of the additional bits."
                )

        self.load_wire = self.ancilla_wires[0]
        self.control_wire = self.ancilla_wires[1]
        self.ancilla_wires = self.ancilla_wires[2:]

        if amplitudes is None:
            self.amplitudes = [1 / np.sqrt(self.X.shape[0], dtype=np.float64)] * self.X.shape[0]
        else:
            self.amplitudes = amplitudes

        self.rotation_circuits = self.prepare_rotation_circuits()

    def create_xor_X(self, X: np.ndarray) -> np.ndarray:
        shifted_X = np.zeros(X.shape, dtype=int)
        shifted_X[1:] = X[
            :-1
        ]  # Thus, shifted_X[i] = X[i+1] and shifted_X[0] = [0...0], i.e. all zeros
        return np.bitwise_xor(X, shifted_X)  # Creating xor_X

    def abs2(self, x: float):
        return x.real**2 + x.imag**2

    def prepare_rotation_circuits(self) -> List:
        rotation_circuits = []
        prev_sum = 1
        rotation_matrix = np.zeros((2, 2), dtype=np.complex128)
        for amp in self.amplitudes[:-1]:
            next_sum = prev_sum - self.abs2(amp)
            diag = np.sqrt(next_sum / prev_sum)
            rotation_matrix[0, 0] = diag
            rotation_matrix[1, 1] = diag

            sqrt_prev_sum = np.sqrt(prev_sum)
            rotation_matrix[0, 1] = amp / sqrt_prev_sum
            rotation_matrix[1, 0] = -amp.conjugate() / sqrt_prev_sum
            rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))
            prev_sum = next_sum

        rotation_matrix[0, 0] = 0
        rotation_matrix[1, 1] = 0

        sqrt_prev_sum = np.sqrt(prev_sum)
        rotation_matrix[0, 1] = self.amplitudes[-1] / sqrt_prev_sum
        rotation_matrix[1, 0] = -self.amplitudes[-1].conjugate() / sqrt_prev_sum
        rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))

        return rotation_circuits

    def cc_flip_control_if_reg_equals_x(self, c_wires: List[int], x: np.ndarray):
        # Flip all wires to one, if reg == x
        for i, x_ in enumerate(x):
            if x_ == 0:
                qml.PauliX((self.register_wires[i],))
        # Check if all wires are one
        adaptive_ccnot(
            c_wires + self.register_wires, self.ancilla_wires, self.unclean_wires, self.control_wire
        )
        # Uncompute flips
        for i, x_ in enumerate(x):
            if x_ == 0:
                qml.PauliX((self.register_wires[i],))

    def load_x(self, x: np.ndarray):
        for idx, x_ in enumerate(x):
            if x_ == 1:
                qml.CNOT((self.load_wire, self.register_wires[idx]))

    def load_additional_bits(self, bits: np.ndarray):
        for idx, bit in enumerate(bits):
            if bit == 1:
                qml.CNOT((self.load_wire, self.additional_wires[idx]))

    def circuit(self, c_wires: List[int] = None, debug="", snaps=False):
        c_wires = [] if c_wires is None else c_wires
        adaptive_ccnot(c_wires + self.c_wires, self.ancilla_wires, self.unclean_wires, self.load_wire)
        xor_additional_bits = (
            self.xor_additional_bits
            if self.xor_additional_bits is not None
            else [None] * len(self.xor_X)
        )
        for xor_x, xor_add_bit, x, rot_circuit in zip(
            self.xor_X, xor_additional_bits, self.X, self.rotation_circuits
        ):
            # Load xi
            self.load_x(xor_x)
            if xor_add_bit is not None:
                self.load_additional_bits(xor_add_bit)
            if snaps: qml.Snapshot(f"{debug} load {x} with {xor_x}")

            # CNOT load to control
            qml.CNOT(wires=(self.load_wire, self.control_wire))
            if snaps: qml.Snapshot(f"{debug}, {x} cnot load to control")

            # Controlled rotation i from control on load
            rot_circuit(self.control_wire, self.load_wire)
            if snaps: qml.Snapshot(f"{debug}, {x} rot load, if control 1")

            # Flip control wire, if register == xi
            self.cc_flip_control_if_reg_equals_x(c_wires + self.c_wires, x)
            if snaps: qml.Snapshot(f"{debug}, {x} flip control wire, if register is xi")

            # Since we are using xor_X, we don't need to uncompute the register, if load == 1
            # xor_X[0] = X[0], but xor_X[i] = X[i] xor X[i-1] for all i > 0
            # After iteration i, reg=X[i]. Now loading xor_X[i+1] results in
            # reg = X[i] xor xor_X[i+1] = X[i] xor (X[i+1] xor X[i]) = (X[i] xor X[i]) xor X[i+1] = 0 xor X[i+1] = X[i+1]

    def get_circuit(self):
        return self.circuit

    def inv_circuit(self):
        self.get_inv_circuit()()

    def get_inv_circuit(self):
        return qml.adjoint(self.circuit)


class DefaultValueQAM:
    def __init__(
        self,
        X: np.ndarray,
        default_value: np.ndarray,
        c_wires: List[int],
        register_wires: List[int],
        value_wires: List[int],
        ancilla_wires: List[int],
        unclean_wires: List[int] = None,
        amplitudes: List[float] = None,
        additional_bits: List[int] = None,
        additional_wires: List[int] = None,
    ):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.X = X
        if not is_binary(self.X):
            raise ValueError(
                "A QAM (Quantum Associative Memory) can only load binary data"
            )
        self.xor_X = self.create_xor_X(X)

        self.default_value = default_value

        if additional_bits is not None:
            if not is_binary(additional_bits):
                raise ValueError(
                    "A QAM (Quantum Associative Memory) can only load binary additional bits"
                )
            if not isinstance(additional_bits, np.ndarray):
                additional_bits = np.array(additional_bits)
            self.xor_additional_bits = self.create_xor_X(additional_bits)
        else:
            self.xor_additional_bits = None

        self.c_wires = c_wires
        self.register_wires = list(register_wires)
        self.value_wires = list(value_wires)
        self.ancilla_wires = list(ancilla_wires)
        self.additional_wires = [] if additional_wires is None else additional_wires
        self.unclean_wires = (
            [] if unclean_wires is None else unclean_wires
        )  # unclean wires are like ancilla wires, but they are not guaranteed to be 0

        ancillas_needed = (
            2 if X.shape[1] < 3 else 3
        )  # If we have less than 3 data qubits, we can use a Toffoli or a CNOT and thus we do not need an ancilla qubit for a CCCNOT
        wire_types = ["register", "ancilla", "additional", "unclean"]
        num_wires = [X.shape[1], ancillas_needed]
        error_msgs = ["the points' dimensionality.", str(ancillas_needed) + "."]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-2], num_wires, error_msgs)

        if additional_bits is not None:
            if (
                additional_wires is None
                or len(additional_wires) < additional_bits.shape[1]
            ):
                raise qml.QuantumFunctionError(
                    "The number of additional wires must be at least the same as the dimension of the additional bits."
                )

        self.load_wire = self.ancilla_wires[0]
        self.control_wire = self.ancilla_wires[1]
        self.ancilla_wires = self.ancilla_wires[2:]

        if amplitudes is None:
            self.amplitudes = [1 / np.sqrt(X.shape[0], dtype=np.float64)] * X.shape[0]
        else:
            self.amplitudes = amplitudes

        self.rotation_circuits = self.prepare_rotation_circuits()

    def create_xor_X(self, X: np.ndarray) -> np.ndarray:
        shifted_X = np.zeros(X.shape, dtype=int)
        shifted_X[1:] = X[
            :-1
        ]  # Thus, shifted_X[i] = X[i+1] and shifted_X[0] = [0...0], i.e. all zeros
        return np.bitwise_xor(X, shifted_X)  # Creating xor_X

    def abs2(self, x: float):
        return x.real**2 + x.imag**2

    def prepare_rotation_circuits(self) -> List:
        rotation_circuits = []
        prev_sum = 1
        rotation_matrix = np.zeros((2, 2), dtype=np.complex128)
        for amp in self.amplitudes[:-1]:
            next_sum = prev_sum - self.abs2(amp)
            diag = np.sqrt(next_sum / prev_sum)
            rotation_matrix[0, 0] = diag
            rotation_matrix[1, 1] = diag

            sqrt_prev_sum = np.sqrt(prev_sum)
            rotation_matrix[0, 1] = amp / sqrt_prev_sum
            rotation_matrix[1, 0] = -amp.conjugate() / sqrt_prev_sum
            rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))
            prev_sum = next_sum

        rotation_matrix[0, 0] = 0
        rotation_matrix[1, 1] = 0

        sqrt_prev_sum = np.sqrt(prev_sum)
        rotation_matrix[0, 1] = self.amplitudes[-1] / sqrt_prev_sum
        rotation_matrix[1, 0] = -self.amplitudes[-1].conjugate() / sqrt_prev_sum
        rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))

        return rotation_circuits

    def flip_control_if_reg_equals_x(self, x: np.ndarray):
        # Flip all wires to one, if reg == x
        for i, x_ in enumerate(x):
            if x_ == 0:
                qml.PauliX((self.register_wires[i],))
        # Check if all wires are one
        adaptive_ccnot(
            self.register_wires, self.ancilla_wires, self.unclean_wires, self.control_wire
        )
        # Uncompute flips
        for i, x_ in enumerate(x):
            if x_ == 0:
                qml.PauliX((self.register_wires[i],))

    def load_x(self, one_value: np.ndarray, zero_value: np.ndarray, check_wire):
        qml.PauliX((check_wire, ))
        for idx, x_ in enumerate(zero_value):
            if x_ == 1:
                qml.Toffoli((check_wire, self.load_wire, self.register_wires[idx]))
        qml.PauliX((check_wire,))
        for idx, x_ in enumerate(one_value):
            if x_ == 1:
                qml.Toffoli((check_wire, self.load_wire, self.register_wires[idx]))

    def load_additional_bits(self, bits: np.ndarray):
        for idx, bit in enumerate(bits):
            if bit == 1:
                qml.CNOT((self.load_wire, self.additional_wires[idx]))

    def circuit(self, c_wires, debug=""):
        adaptive_ccnot(c_wires + self.c_wires, self.ancilla_wires, self.unclean_wires, self.load_wire)
        xor_additional_bits = (
            self.xor_additional_bits
            if self.xor_additional_bits is not None
            else [None] * len(self.xor_X)
        )
        for xor_x, xor_add_bit, x, rot_circuit, v_wire in zip(
            self.xor_X, xor_additional_bits, self.X, self.rotation_circuits, self.value_wires
        ):
            # Load xi
            self.load_x(xor_x, np.bitwise_xor(np.bitwise_xor(xor_x, x), self.default_value), v_wire)
            if xor_add_bit is not None:
                self.load_additional_bits(xor_add_bit)
            qml.Snapshot(f"{debug} load {x}")

            # CNOT load to control
            qml.CNOT(wires=(self.load_wire, self.control_wire))
            qml.Snapshot(f"{debug}, {x} cnot load to control")

            # Controlled rotation i from control on load
            rot_circuit(self.control_wire, self.load_wire)
            qml.Snapshot(f"{debug}, {x} rot load, if control 1")

            # Flip control wire, if register == xi
            self.flip_control_if_reg_equals_x(x)
            qml.Snapshot(f"{debug}, {x} flip control wire, if register is xi")

            # Since we are using xor_X, we don't need to uncompute the register, if load == 1
            # xor_X[0] = X[0], but xor_X[i] = X[i] xor X[i-1] for all i > 0
            # After iteration i, reg=X[i]. Now loading xor_X[i+1] results in
            # reg = X[i] xor xor_X[i+1] = X[i] xor (X[i+1] xor X[i]) = (X[i] xor X[i]) xor X[i+1] = 0 xor X[i+1] = X[i+1]

    def get_circuit(self):
        return self.circuit

    def inv_circuit(self):
        self.get_inv_circuit()()

    def get_inv_circuit(self):
        return qml.adjoint(self.circuit)

