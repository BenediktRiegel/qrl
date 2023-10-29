import torch
from plot import get_matplotlib_heatmap
import matplotlib.pyplot as plt
import numpy as np


def get_qpe_output_prob(delta, l, N):
    exp1 = N*delta - l
    exp2 = delta - l/N
    if exp2 == 0:
        return 1.0
    result = (1-torch.exp(2j*torch.pi*exp1)) / (1-torch.exp(2j*torch.pi*exp2)) / N
    return torch.square(result.real) + torch.square(result.imag)


def best_phi_approximation(phi: torch.tensor, num_bits: int):
    phi = phi.clone().detach()
    b_bits = torch.zeros(num_bits)
    for bit in range(1, num_bits+1):
        phi *= 2
        if phi >= 1.0:
            b_bits[bit-1] = 1
            phi -= 1
        if phi <= 0:
            break
    small_b = torch.tensor(0., dtype=phi.dtype)
    # big_b = torch.tensor(0., dtype=phi.dtype)
    for little_end_bit, big_end_bit in zip(reversed(b_bits), b_bits):
        small_b /= 2.
        small_b += (little_end_bit/2.0)
        # big_b *= 2.
        # big_b += big_end_bit

    return small_b  # , big_b

qpe_result_prob = dict()

def get_qpe_result_prob(phi: float, num_qubits: int):
    global qpe_result_prob
    if phi.item() in qpe_result_prob:
        print(f"{phi.item()} already in qpe_result_prob")
        return qpe_result_prob[phi.item()]
    N = 2 ** num_qubits
    b = best_phi_approximation(phi, num_qubits)  # best phi approximation
    delta = phi - b
    if delta == 0:
        result_prob = torch.zeros(N, dtype=torch.float64)
        result_prob[0] = 1
        result_value = torch.arange(0, N) / N
        result_value += b
        result_value[result_value >= 1.0] -= 1.0
        result_value[result_value < 0.0] += 1.0
        return result_prob, result_value
    if delta < 0:
        # return torch.zeros(shots) + (b / N)
        raise ValueError(f"delta must be greater or equal to 0, but got {phi} - {b} = {delta}")
    # print(f"phi-b/N = {phi - b/N}")
    result_prob = torch.empty(N)
    result_value = torch.empty(N)
    idx = 0
    for l1, l2 in zip(range(N // 2), range(1, N // 2 + 1)):
        # for l1 in range(N):
        result_prob[idx] = get_qpe_output_prob(delta, l1, N)
        new_angle = b + l1 / N
        if new_angle >= 1:
            new_angle -= 1
        result_value[idx] = new_angle.item()
        idx += 1

        result_prob[idx] = get_qpe_output_prob(delta, -l2, N)
        new_angle = b - l2 / N
        if new_angle < 0:
            new_angle += 1
        result_value[idx] = new_angle.item()
        idx += 1
    qpe_result_prob[phi] = (result_prob, result_value)
    return result_prob, result_value


def get_prob_of_receiving_max(phi1, phi2, num_qubits):
    result_prob1, result_value1 = get_qpe_result_prob(phi1, num_qubits)
    result_prob1_minus, result_value1_minus = get_qpe_result_prob(1 - phi1, num_qubits)
    for idx, v1 in enumerate(result_value1):
        for p1_minus, v1_minus in zip(result_prob1_minus, result_value1_minus):
            if v1 == v1_minus:
                result_prob1[idx] += p1_minus
    result_prob1 /= 2

    result_prob2, result_value2 = get_qpe_result_prob(phi2, num_qubits)
    result_prob2_minus, result_value2_minus = get_qpe_result_prob(1 - phi2, num_qubits)
    for idx, v2 in enumerate(result_value2):
        for p2_minus, v2_minus in zip(result_prob2_minus, result_value2_minus):
            if v2 == v2_minus:
                result_prob2[idx] += p2_minus
    result_prob2 /= 2

    for idx, value in enumerate(result_value1):
        result_value1[idx] = 1 - value if value < 0.5 else value
    for idx, value in enumerate(result_value2):
        result_value2[idx] = 1 - value if value < 0.5 else value

    max1_prob = 0
    equal_prob = 0
    max2_prob = 0
    for v1, p1 in zip(result_value1, result_prob1):
        for v2, p2 in zip(result_value2, result_prob2):
            if v1 > v2:
                max1_prob += p1*p2
            elif v1 < v2:
                max2_prob += p1*p2
            else:
                equal_prob += p1*p2
    return max1_prob, equal_prob, max2_prob, equal_prob+max2_prob


def get_prob_of_opt_max(phi1, phi2):
    if phi1 > 0.5:
        phi1 = 1 - phi1
    if phi2 > 0.5:
        phi2 = 1 - phi2

    if phi1 > phi2:
        return 1, 0, 0, 1
    elif phi1 < phi2:
        return 0, 0, 1, 1
    else:
        return 0, 1, 0, 1


def main():
    num_qubits = 4
    num_phis = 2**(num_qubits+3)
    phi1 = torch.arange(0, num_phis+1, dtype=torch.float64) / num_phis / 2.
    phi2 = torch.arange(0, num_phis+1, dtype=torch.float64) / num_phis / 2.
    z = np.empty((len(phi1), len(phi2), 4))
    for idx1, p1 in enumerate(phi1):
        for idx2, p2 in enumerate(phi2):
            if (idx2 % 20) == 0:
                print(f"{idx1+1}/{len(phi1)}, {idx2+1}/{len(phi2)}")
            z[idx1, idx2, :] = get_prob_of_receiving_max(p1, p2, num_qubits)
    np.save("./z", z)

    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), z[:, :, 0], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./max_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), z[:, :, 1], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./equal.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), z[:, :, 2], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./max_phi1.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), z[:, :, 3], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./less_equal_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()

    # True max
    true_z = np.empty((len(phi1), len(phi2), 4))
    for idx1, p1 in enumerate(phi1):
        for idx2, p2 in enumerate(phi2):
            if (idx2 % 20) == 0:
                print(f"{idx1+1}/{len(phi1)}, {idx2+1}/{len(phi2)}")
            if idx1 > idx2:
                temp = (0, 0, 1, 1)
            elif idx1 < idx2:
                temp = (1, 0, 0, 0)
            else:
                temp = (0, 1, 0, 1)
            true_z[idx1, idx2, :] = temp

    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), true_z[:, :, 0], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./true_max_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), true_z[:, :, 1], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./true_equal.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), true_z[:, :, 2], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./true_max_phi1.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), true_z[:, :, 3], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./true_less_equal_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()
    np.save("./true_z", z)

    # Diff max
    diff_z = np.abs(z - true_z)
    np.save("./diff_z", z)
    if (diff_z < 0).any():
        print("diff_z < 0")
    else:
        print("all diff_z > 0")
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), diff_z[:, :, 0], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./diff_max_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), diff_z[:, :, 1], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./diff_equal.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), diff_z[:, :, 2], "$\\varphi_0$", "$\\varphi_1$")
    plt.tight_layout()
    plt.savefig("./diff_max_phi1.pdf", dpi="figure", format="pdf")
    plt.clf()
    get_matplotlib_heatmap(phi1.tolist(), phi2.tolist(), diff_z[:, :, 3], "$\\varphi_0$", "$\\varphi_1$", "")
    plt.tight_layout()
    plt.savefig("./diff_less_equal_phi0.pdf", dpi="figure", format="pdf")
    plt.clf()


if __name__ == "__main__":
    main()
