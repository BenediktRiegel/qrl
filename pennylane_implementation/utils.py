import numpy as np
from typing import List


def bitlist_to_int(bitlist):
    """
    Compute an int from a bitlist. The bitlist is interpret to be in big endian
    :param bitlist: list of bits
    :return: int
    """
    if bitlist is None:
        return None
    out = 0
    for bit in bitlist:
        bit = int(bit)
        out = (out << 1) | bit
    return out


def int_to_bitlist(num: int, length: int):
    """
    Given an int, this method returns the big endian encoding of num. The returned list has the length length.
    :param bitlist: list of bits
    :return: int
    """
    if length == 0:
        return []
    binary = bin(num)[2:]
    result = [0]*length
    for i in range(-1, -len(binary)-1, -1):
        result[i] = int(binary[i])
    return result


def compute_zyz_decomposition(U):
    """
    Compute the zyz decomposition of a unitary U
    """
    from pennylane.transforms import zyz_decomposition
    from numpy import arccos
    rot = zyz_decomposition(U, 0)[0]
    rot_matrix = rot.matrix()
    global_phase = U[0] / rot_matrix[0]
    angles = [arccos(global_phase.real)] + list(rot.single_qubit_rot_angles())
    return angles


def is_binary(data):
    """
    checks if every entry in data is binary
    """
    return np.array_equal(data, data.astype(bool))
