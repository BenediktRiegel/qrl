# This file was copied from the project https://github.com/UST-QuAntiL/qhana-plugin-runner in compliance with its license
from pennylane import QuantumFunctionError


def check_wires_uniqueness(obj_with_wires, wire_types): # Modification: Added Doc Str
    """
    Given an object with wires, we check, if the wires that belong to at least one of the wire types in wire_types, if
    they are unique
    """
    for idx1, wire_type1 in enumerate(wire_types):
        wires1 = getattr(obj_with_wires, wire_type1 + '_wires')
        for wire_type2 in wire_types[idx1 + 1:]:
            wires2 = getattr(obj_with_wires, wire_type2 + '_wires')
            if any(wire in wires1 for wire in wires2):
                raise QuantumFunctionError(
                    f"The {wire_type1} wires must be different from the {wire_type2} wires"
                )


def check_num_wires(obj_with_wires, wire_types, num_wires, error_msgs): # Modification: Added Doc Str
    """
    Given an object that has wires, we retrieve a certain 'wire_type' (the type is just naming convention) by
    getattr on the object, with the variable name {w_type}_wires. Next, we check, if it has more equally as manny
    wires as necessary, according to num_wires. If not then we through its error message from error_msg.
    The j'th wire-type, the j'th num wires and the j'th error message belong together.
    :param obj_with_wires: object
    :param wire_types: list of str
    :param num_wires: list of in
    :param error_msgs: list of str
    """
    # print(f"obj_with_wires = {obj_with_wires.__dict__}\nwire_types = {wire_types}\nnum_wires = {num_wires}\nerror_msgs = {error_msgs}")
    for w_type, n_wires, e_msg in zip(wire_types, num_wires, error_msgs):
        wires = getattr(obj_with_wires, w_type + '_wires')
        # print(f"{w_type}, {n_wires}, {e_msg}")
        if len(wires) < n_wires:
            error = f"The number of {w_type} wires has to be greater or equal to {e_msg} Expected {n_wires}, but got {len(wires)}."
            raise QuantumFunctionError(error)


def get_wires(wires):   # Modified: Added this function
    """
    Given a list of integers, this method returns the corresponding amount of wires, e.g.
    Input [1, 3, 5] => Output [[0], [1, 2, 3], [4, 5 , 6, 7, 8]]
    :return: 2d list containing no duplicate wires.
    """
    result = []
    total_wires = 0
    for no_wires in wires:
        result.append(list(range(total_wires, total_wires + no_wires)))
        total_wires += no_wires
    return result, total_wires
