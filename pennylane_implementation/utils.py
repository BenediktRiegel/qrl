import numpy as np
from typing import List


def bitlist_to_int(bitlist):
    if bitlist is None:
        return None
    out = 0
    for bit in bitlist:
        bit = int(bit)
        out = (out << 1) | bit
    return out


def int_to_bitlist(num: int, length: int):
    binary = bin(num)[2:]
    result = [0]*length
    for i in range(-1, -len(binary)-1, -1):
        result[i] = int(binary[i])
    return result


def compute_zyz_decomposition(U):
    from pennylane.transforms import zyz_decomposition
    from numpy import arccos
    rot = zyz_decomposition(U, 0)[0]
    rot_matrix = rot.matrix()
    global_phase = U[0] / rot_matrix[0]
    angles = [arccos(global_phase.real)] + list(rot.single_qubit_rot_angles())
    return angles


def is_binary(data):
    return np.array_equal(data, data.astype(bool))


def get_bit_by_exponent(num: float, exp: int):
    exp = -1*(exp+1)
    num *= (2**exp)
    num = num - int(num)
    num = int(num*2)
    return num


def get_bit_by_interpretation(num: float, interpretation: str):
    if interpretation == "sign":
        return 1 if num < 0 else 0
    else:
        num *= -1 if num < 0 else 1
        return get_bit_by_exponent(num, int(interpretation))


def float_to_bitlist(num: float, interpretation: List[str]):
    return [get_bit_by_interpretation(num, exp) for exp in interpretation]


def get_float_by_interpretation(bitlist: List[int], interpretation: List[str]):
    value = 0.
    signs = 1
    for bit, interp in zip(bitlist, interpretation):
        if interp == "sign":
            signs *= -1*bit
        else:
            value += (bit * 2**int(interp))
    return value


def main():
    pass


if __name__ == '__main__':
    main()
