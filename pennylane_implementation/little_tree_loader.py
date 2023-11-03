# This file was copied from the project https://github.com/UST-QuAntiL/qhana-plugin-runner in compliance with its license
from typing import List
import pennylane as qml
import numpy as np
from utils import int_to_bitlist
from ccnot import adaptive_ccnot
from wire_utils import check_wires_uniqueness, check_num_wires


class BinaryTreeNode:
    """A Node of a binary tree"""
    def __init__(self, bit_str, value, neg_sign=None):
        self.bit_str = bit_str
        self.value = value
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.neg_sign = neg_sign


class LittleTreeLoader:
    """Allows to load an arbitrary quantum state"""
    def __init__(   # Modified: simplified parameters
        self,
        data,
        data_wires,
        ancilla_wires,
        unclean_wires=None,
    ):
        # 3 Ancillas needed
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data

        self.data_wires = data_wires        # Modified: Since we receive different Parameters, have to deal with them differently
        self.ancilla_wires = ancilla_wires
        self.unclean_wires = (
            [] if unclean_wires is None else unclean_wires
        )  # unclean wires are like ancilla wires, but they are not guaranteed to be 0

        wire_types = ["data", "ancilla", "unclean"]
        num_wires = [
            int(np.ceil(np.log2(data.shape[1]))),
            1,
        ]
        error_msgs = [
            "ceil(log2(datas' dimensionality)).",
            "1.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-2], num_wires, error_msgs)

        self.binary_trees = self.build_binary_tree_list()
        self.prepare_tree_list_values()

    def prepare_tree_values(self, node: BinaryTreeNode, sqrt_parent_value=1.0):
        """
        Changes values in binary tree, such that they can be directly used for a RY rotation.
        """
        sqrt_value = np.sqrt(node.value)
        if node.parent is None:
            node.value = 0
        else:
            node.value = np.arccos(sqrt_value / sqrt_parent_value) * 2.0

        if node.left_child is not None:
            self.prepare_tree_values(node.left_child, sqrt_value)
        if node.right_child is not None:
            self.prepare_tree_values(node.right_child, sqrt_value)

    def prepare_tree_list_values(self):     # Modified: Added doc string
        """
        prepares the values for each tree
        """
        for tree in self.binary_trees:
            self.prepare_tree_values(tree)

    def build_binary_tree_list(self):
        binary_trees = []
        if len(self.data.shape) == 1:
            binary_trees.append(self.build_binary_tree(self.data))
        else:
            for i in range(self.data.shape[0]):
                binary_trees.append(self.build_binary_tree(self.data[i]))
        return binary_trees

    def build_binary_tree(self, state):
        """
        Create a binary tree. The leafs consist of the probabilities of the different states. Going up the tree the
        probabilities get added up. The root has a value equal to one.
        Example:
        Let |Psi> = 0.4 |00> + 0.4 |01> + 0.8 |10> + 0.2 |11> be the input state.
        Then the binary tree looks as follows
                   ┇1.00┇
                   ╱    ╲
                 ╱        ╲
            ┇0.32┇        ┇0.68┇
            ╱    ╲        ╱    ╲
        ┇0.16┇  ┇0.16┇┇0.64┇  ┇0.04┇

        If the left and right child have a value of zero, we remove them both.
        Example:
        Let |Psi> = 1.414 |00> + 1.414 |01> + 0. |10> + 0. |11> be the input state.
        Then the binary tree looks as follows
                   ┇1.00┇
                   ╱    ╲
                 ╱        ╲
            ┇1.00┇        ┇0.00┇
            ╱    ╲
        ┇0.50┇  ┇0.50┇

        :param state: The quantum state that needs to be loaded
        :return: returns a binary tree of the quantum state
        """
        tree_depth = int(np.log2(len(state)))
        tree_nodes = [
            BinaryTreeNode(
                "".join([str(el) for el in int_to_bitlist(i, tree_depth)]),
                state[i] ** 2,
                neg_sign=(state[i] < 0.0),
            )
            for i in range(len(state))
        ]

        for depth in range(tree_depth):
            new_tree_nodes = []
            for i in range(0, len(tree_nodes), 2):
                new_tree_nodes.append(
                    BinaryTreeNode(
                        tree_nodes[i].bit_str[:-1],
                        tree_nodes[i].value + tree_nodes[i + 1].value,
                    )
                )
                if tree_nodes[i].value != 0 or tree_nodes[i + 1].value != 0:
                    # Set new node to the parent node
                    tree_nodes[i].parent = new_tree_nodes[-1]
                    tree_nodes[i + 1].parent = new_tree_nodes[-1]
                    # Set new parent nodes children
                    new_tree_nodes[-1].left_child = tree_nodes[i]
                    new_tree_nodes[-1].right_child = tree_nodes[i + 1]
            tree_nodes = new_tree_nodes
        return tree_nodes[-1]

    def get_sign_rotation(self, tree_node: BinaryTreeNode):
        if tree_node.left_child is not None:
            if tree_node.left_child.neg_sign is not None:
                """
                Depending, if an amplitude should be negative or positive, the RY rotations needs to be rotated by
                additional 2pi and a Z operations needs to be added. Since this concerns the amplitudes, only the leafs of
                the binary tree can have neg_sin == True.
                0: + +
                2pi: - -
                0 + z: + -
                2pi + z: - +
                """
                if tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 2.0 * np.pi, False
                elif tree_node.left_child.neg_sign and not tree_node.right_child.neg_sign:
                    return 2.0 * np.pi, True
                elif not tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 0, True
        return 0, False

    def qubit_rotations(
        self, qubit_idx: int, tree_node: BinaryTreeNode, control_qubits: List[int], ancilla_qubits: List[int], unclean_qubits: List[int], right=True, tree_idx=0
    ):
        """
        Recursively rotates the qubits in the data register to produce the correct state.
        """
        if tree_node.left_child is not None:
            temp_control_qubits = control_qubits.copy()
            # rotate qubit
            sign_rot, use_z = self.get_sign_rotation(tree_node)
            if tree_node.parent is not None:
                temp_control_qubits += self.data_wires[:qubit_idx]

            adaptive_ccnot(
                temp_control_qubits,
                self.ancilla_wires[1:] + ancilla_qubits,
                self.unclean_wires + self.data_wires[qubit_idx + 1:] + unclean_qubits,
                self.ancilla_wires[0],
            )
            qml.CRY(
                tree_node.left_child.value + sign_rot,
                wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]),
            )
            if use_z:
                qml.CZ(wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]))

            adaptive_ccnot(
                temp_control_qubits,
                self.ancilla_wires[1:] + ancilla_qubits,
                self.unclean_wires + self.data_wires[qubit_idx + 1:] + unclean_qubits,
                self.ancilla_wires[0],
            )

            # left child
            qml.PauliX((self.data_wires[qubit_idx],))
            self.qubit_rotations(qubit_idx + 1, tree_node.left_child, control_qubits, ancilla_qubits, unclean_qubits, right=False)
            qml.PauliX((self.data_wires[qubit_idx],))

            # right child
            self.qubit_rotations(qubit_idx + 1, tree_node.right_child, control_qubits, ancilla_qubits, unclean_qubits, right=True)

    def load_tree(self, tree_idx: int, control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None):   # Modified: New Doc Str
        """
        Loads the quantum state controlled, by the control_qubtis
        """
        control_qubits = [] if control_qubits is None else control_qubits
        ancilla_qubits = [] if ancilla_qubits is None else ancilla_qubits
        unclean_qubits = [] if unclean_qubits is None else unclean_qubits
        self.qubit_rotations(0, self.binary_trees[tree_idx], control_qubits, ancilla_qubits, unclean_qubits, tree_idx=tree_idx)     # Modified: checks. Since we load only one tree without caring about indices, we simplified this to this onliner

    def circuit(self, control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None):    # Modified: New Doc Str
        """
        Execute the TreeLoaders quantum circuit
        """
        self.load_tree(0, control_qubits, ancilla_qubits, unclean_qubits)   # Modified: Removed for loop, since we are loading only one tree

    def get_circuit(self, control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None):
        def circuit():
            self.circuit(control_qubits, ancilla_qubits, unclean_qubits)
        return circuit

    def inv_circuit(self, control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None):
        self.get_inv_circuit(control_qubits, ancilla_qubits, unclean_qubits)()

    def get_inv_circuit(self, control_qubits: List[int] = None, ancilla_qubits: List[int] = None, unclean_qubits: List[int] = None):
        def circuit():
            qml.adjoint(self.circuit)(control_qubits, ancilla_qubits, unclean_qubits)
        return circuit

    # Modified: removed 'get_necessary_wires' method