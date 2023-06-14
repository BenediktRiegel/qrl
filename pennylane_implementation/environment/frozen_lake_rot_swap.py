from typing import List
import numpy as np
import pennylane as qml
from .frozen_lake import FrozenField, simple_single_oracle, cc_simple_single_oracle
from . import Environment
from utils import int_to_bitlist, get_bit_by_interpretation, bitlist_to_int
from q_arithmetic import add_classical_quantum_registers, add_registers
from ccnot import adaptive_ccnot
from load_data import QAM, LittleTreeLoader


class FrozenLakeRotSwap:
    def __init__(
            self, map: List[List[FrozenField]], slip_probabilities: List[float],
            r_qubit_is_clean: bool = False,
    ):
        super().__init__()
        self.map = map
        self.r_m = np.abs(map[0][0].reward)
        for row in map:
            for field in row:
                temp = np.abs(field.reward)
                if temp > self.r_m:
                    self.r_m = temp

        # possible edge cases
        self.possible_edge_cases = self.get_possible_edge_cases()
        x_ceil_log2 = int(np.ceil(np.log2(len(map[0]))))
        self.x_xor_bits = [0 if el == 1 else 1 for el in int_to_bitlist(len(map[0]) - 1, x_ceil_log2)] if len(map[0]) > 1 else []
        y_ceil_log2 = int(np.ceil(np.log2(len(map))))
        self.y_xor_bits = [0 if el == 1 else 1 for el in int_to_bitlist(len(map) - 1, y_ceil_log2)] if len(map) > 1 else []
        do_nothing = [0] * (x_ceil_log2 + y_ceil_log2)
        go_right = [0] * max((x_ceil_log2 - 1), 0) + [1] + [0] * y_ceil_log2
        go_down = [0] * x_ceil_log2 + [1] * y_ceil_log2
        go_left = [1] * x_ceil_log2 + [0] * y_ceil_log2
        go_up = [0] * x_ceil_log2 + [0] * max((y_ceil_log2 - 1), 0)
        self.moves = np.array([go_right, go_down, go_left, go_up, do_nothing])
        print(f"self.moves: {self.moves}")
        self.slip_probabilities = np.array(slip_probabilities)

        self.r_qubit_is_clean = r_qubit_is_clean

    def get_possible_edge_cases(self):
        """
        [right, down, left, up]
        :return:
        """
        possible_edge_cases = []
        if len(self.map) == 1 and len(self.map[0]) == 1:
            possible_edge_cases = [[1, 1, 1, 1]]
        elif len(self.map) == 1 and len(self.map[0]) > 1:
            # one row
            possible_edge_cases = [[0, 1, 1, 1], [1, 1, 0, 1]]
            if len(self.map[0]) > 2:
                possible_edge_cases.append([0, 1, 0, 1])
        elif len(self.map) > 1 and len(self.map[0]) == 1:
            # one column
            possible_edge_cases = [[1, 0, 1, 1], [1, 1, 1, 0]]
            if len(self.map) > 2:
                possible_edge_cases.append([1, 0, 1, 0])
        elif len(self.map) > 1 and len(self.map[0]) > 1:
            possible_edge_cases = [[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]]
            if len(self.map) > 2:
                # at least 3 rows
                possible_edge_cases += [[0, 0, 1, 0], [1, 0, 0, 0]]
            if len(self.map[0]) > 2:
                # at least 3 columns
                possible_edge_cases += [[0, 1, 0, 0], [0, 0, 0, 1]]
                if len(self.map) > 2:
                    # At least a 3x3 field
                    possible_edge_cases.append([0, 0, 0, 0])
        return possible_edge_cases

    def check_edges(self, x_qubits, y_qubits, edge_qubits, ancilla_qubits, unclean_qubits):
        # Edge on the right
        for q, b in zip(x_qubits, self.x_xor_bits):
            if b == 1:
                qml.PauliX((q,))
        adaptive_ccnot(x_qubits, ancilla_qubits, unclean_qubits, edge_qubits[0])
        for q, b in zip(x_qubits, self.x_xor_bits):
            if b == 1:
                qml.PauliX((q,))

        # Edge below
        for q in y_qubits:
            qml.PauliX((q,))
        adaptive_ccnot(y_qubits, ancilla_qubits, unclean_qubits, edge_qubits[1])
        for q in y_qubits:
            qml.PauliX((q,))

        # Edge on the left
        for q in x_qubits:
            qml.PauliX((q,))
        adaptive_ccnot(x_qubits, ancilla_qubits, unclean_qubits, edge_qubits[2])
        for q in x_qubits:
            qml.PauliX((q,))

        # Edge above
        for q, b in zip(y_qubits, self.y_xor_bits):
            if b == 1:
                qml.PauliX((q,))
        adaptive_ccnot(y_qubits, ancilla_qubits, unclean_qubits, edge_qubits[3])
        for q, b in zip(y_qubits, self.y_xor_bits):
            if b == 1:
                qml.PauliX((q,))

    def check_edge_case(self, edge_case, x_qubits, y_qubits, e_qubit, ancilla_qubits, unclean_qubits):
        edge_qubits = ancilla_qubits[:4]
        ancilla_qubits = ancilla_qubits[4:]
        self.check_edges(x_qubits, y_qubits, edge_qubits, [e_qubit] + ancilla_qubits, unclean_qubits)
        for q, b in zip(edge_qubits, edge_case):
            if b == 0:
                qml.PauliX((q,))
        adaptive_ccnot(edge_qubits, ancilla_qubits, unclean_qubits, e_qubit)
        for q, b in zip(edge_qubits, edge_case):
            if b == 0:
                qml.PauliX((q,))
        self.check_edges(x_qubits, y_qubits, edge_qubits, [e_qubit] + ancilla_qubits, unclean_qubits)

    def check_end_state(
            self, x_qubits: List[int], y_qubits: List[int],
            ancilla_qubits: List[int], unclean_qubits: List[int], oracle_qubit: int
    ):
        for y_idx, row in enumerate(self.map):
            for x_idx, field in enumerate(row):
                if field.end:
                    simple_single_oracle(
                        x_qubits + y_qubits,
                        int_to_bitlist(x_idx, len(x_qubits)) + int_to_bitlist(y_idx, len(y_qubits)),
                        ancilla_qubits[1:], unclean_qubits, oracle_qubit
                    )

    def move_in_direction(
            self, slip_probs: np.ndarray, x_qubits: List[int], y_qubits: List[int],
            next_x_qubits: List[int], next_y_qubits: List[int],
            action_qubits: List[int], action: List[int],
            ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        end_state_qubit = ancilla_qubits[0]
        e_qubit = ancilla_qubits[1]
        ancilla_qubits = ancilla_qubits[2:]

        # Check if movement is necessary
        for a_qubit, a_bit in zip(action_qubits, action):
            if a_bit == 0:
                qml.PauliX((a_qubit,))

        # Move
        for edge_case in self.possible_edge_cases:
            # Get modified slip amps
            modified_slip_amps = np.zeros(len(self.moves))
            for j, (b, prob) in enumerate(zip(edge_case, slip_probs)):
                if b == 1:
                    modified_slip_amps[-1] += prob
                else:
                    modified_slip_amps[j] = np.sqrt(prob)
            modified_slip_amps[-1] = np.sqrt(modified_slip_amps[-1])

            self.check_edge_case(edge_case, x_qubits, y_qubits, e_qubit, [end_state_qubit] + ancilla_qubits,
                                 action_qubits + next_x_qubits + next_y_qubits + unclean_qubits)

            # Check if in end state
            self.check_end_state(
                x_qubits, y_qubits,
                next_x_qubits + next_y_qubits + ancilla_qubits,
                action_qubits + [e_qubit] + unclean_qubits,
                end_state_qubit,
            )
            qml.PauliX((end_state_qubit,))

            # QAM(
            #     self.moves,
            #     [e_qubit, end_state_qubit] + action_qubits,
            #     next_x_qubits + next_y_qubits,
            #     ancilla_qubits,
            #     x_qubits + y_qubits + unclean_qubits,
            #     amplitudes=modified_slip_amps.tolist()
            # ).circuit()
            move_vec = np.zeros((1, 2**len(self.moves[0])))
            for move, amp in zip(self.moves, modified_slip_amps):
                move_vec[0, bitlist_to_int(move)] += amp
            temp = [e_qubit, end_state_qubit] + action_qubits
            temp.sort()
            qml.Snapshot(f"direction: {action}, edge case: {edge_case}, before state prep {move_vec} on qubits {next_x_qubits + next_y_qubits} controlled by qubits {temp}")
            LittleTreeLoader(
                move_vec, next_x_qubits + next_y_qubits, ancilla_qubits, x_qubits + y_qubits + unclean_qubits
            ).circuit(control_qubits=[e_qubit, end_state_qubit] + action_qubits)

            qml.Snapshot(f"direction: {action}, edge case: {edge_case}, prepared state {move_vec} on qubits {next_x_qubits + next_y_qubits} controlled by qubits {temp}")

            # Undo end state check
            qml.PauliX((end_state_qubit,))
            self.check_end_state(
                x_qubits, y_qubits,
                ancilla_qubits,
                next_x_qubits + next_y_qubits + action_qubits + [e_qubit] + unclean_qubits,
                end_state_qubit,
            )

            qml.adjoint(self.check_edge_case)(edge_case, x_qubits, y_qubits, e_qubit,
                                              [end_state_qubit] + ancilla_qubits,
                                              action_qubits + next_x_qubits + next_y_qubits + unclean_qubits)

        # Undo movement check
        for a_qubit, a_bit in zip(action_qubits, action):
            if a_bit == 0:
                qml.PauliX((a_qubit,))

    def move(
            self, x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
            next_x_qubits: List[int], next_y_qubits: List[int],
            ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        slip_probs = self.slip_probabilities.copy()
        self.move_in_direction(slip_probs, x_qubits, y_qubits, next_x_qubits, next_y_qubits, action_qubits, [0, 0],
                               ancilla_qubits, unclean_qubits)
        slip_probs = np.roll(slip_probs, shift=1)
        self.move_in_direction(slip_probs, x_qubits, y_qubits, next_x_qubits, next_y_qubits, action_qubits, [0, 1],
                               ancilla_qubits, unclean_qubits)
        slip_probs = np.roll(slip_probs, shift=1)
        self.move_in_direction(slip_probs, x_qubits, y_qubits, next_x_qubits, next_y_qubits, action_qubits, [1, 0],
                               ancilla_qubits, unclean_qubits)
        slip_probs = np.roll(slip_probs, shift=1)
        self.move_in_direction(slip_probs, x_qubits, y_qubits, next_x_qubits, next_y_qubits, action_qubits, [1, 1],
                               ancilla_qubits, unclean_qubits)

        qml.Snapshot("next_qubits prepared")

        # Add state
        add_registers(
            x_qubits, next_x_qubits,
            ancilla_qubits,
            y_qubits + next_y_qubits + action_qubits + unclean_qubits,
            indicator_is_zero=True
        )
        add_registers(
            y_qubits, next_y_qubits,
            ancilla_qubits,
            x_qubits + next_x_qubits + action_qubits + unclean_qubits,
            indicator_is_zero=True
        )

        qml.Snapshot("Added next_qubits and state_qubits")

    def compute_rewards(
            self, control_qubits: List[int], x_qubits: List[int], y_qubits: List[int],
            r_qubits: List[int], factors: List[float],
            ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        """
        This method executes a quantum circuit that, depending on a state, encodes the reward in the reward qubit.
        Let r be the reward received, when transitioning into the state s = (x, y). This method rotates the r_qubit
        by RY((r / r_m + 1) * π/2), where r_m := max_{r'} |r'|. Thus, the interval [-r_m, r_m] is linearly mapped to the
        interval [0, π] and then we rotated r_qubit according to this method. Hence, if r_qubit = |1>, then we received
        the maximum reward and if r_qubit = |0>, then we received -r_m.
        If we first rotate r_qubit by RY(π/2) and then by RY((r / r_m) * π/2) we get the same result, but we do not need
        to do a controlled RY rotation for cases in which the reward is 0.
        :param x_qubits: qubits with x coordinates of state
        :param y_qubits: qubits with y coordinates of state
        :param r_qubit: qubit to encode the reward with
        :param ancilla_qubits: qubits to aid execution (They are in state |0>)
        :param unclean_qubits: qubits to aid execution (They may be in any state)
        :return: none
        """
        oracle_qubit = ancilla_qubits[0]
        adaptive_ccnot(control_qubits, ancilla_qubits[1:], unclean_qubits, oracle_qubit)
        for r_q in r_qubits:
            qml.CRY(phi=np.pi, wires=(oracle_qubit, r_q))
        adaptive_ccnot(control_qubits, ancilla_qubits[1:], unclean_qubits, oracle_qubit)
        for x_idx, row in enumerate(self.map):
            for y_idx, field in enumerate(row):
                if field.reward != 0:
                    cc_simple_single_oracle(
                        control_qubits,
                        x_qubits + y_qubits,
                        int_to_bitlist(x_idx, len(x_qubits)) + int_to_bitlist(y_idx, len(y_qubits)),
                        ancilla_qubits[1:], unclean_qubits, oracle_qubit
                    )
                    for r_q, fac in zip(r_qubits, factors):
                        qml.CRY(phi=(np.arccos(field.reward * fac) * 2 - np.pi), wires=(oracle_qubit, r_q))
                    cc_simple_single_oracle(
                        control_qubits,
                        x_qubits + y_qubits,
                        int_to_bitlist(x_idx, len(x_qubits)) + int_to_bitlist(y_idx, len(y_qubits)),
                        ancilla_qubits[1:], unclean_qubits, oracle_qubit
                    )

    def circuit(
            self, x_qubits: List[int], y_qubits: List[int], action_qubits: List[int],
            next_x_qubits: List[int], next_y_qubits: List[int], r_control_qubits: List[int], r_qubits: List[int], r_factor: List[float],
            ancilla_qubits: List[int], unclean_qubits: List[int] = None
    ):
        """
        00: Right
        01: Down
        10: Left
        11: Up
        :return:
        """
        # Move
        self.move(
            x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits,
            r_qubits + ancilla_qubits if self.r_qubit_is_clean else ancilla_qubits,
            unclean_qubits,
        )

        # Rewards
        self.compute_rewards(
            r_control_qubits,
            next_x_qubits, next_y_qubits,
            r_qubits, r_factor,
            ancilla_qubits,
            x_qubits + y_qubits + action_qubits + unclean_qubits,
        )
