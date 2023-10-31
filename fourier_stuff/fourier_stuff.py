import numpy as np
from plot import get_fig, get_heatmap, get_seaborn_heatmap
import plotly.express as px
import pandas as pd


def get_qft_matrix(n_qubits: int):
    """
    Returns the matrix of the Quantum Fourier Transform (QFT)
    :param n_qubits: number of qubits involved in the QFT
    :return: numpy array of QFT
    """
    N = 2**n_qubits
    qft = np.ones((N, N), dtype=complex)
    omega = np.exp(2j*np.pi/N)
    for i in range(1, N):
        qft[1, i] = qft[1, i-1] * omega
    for i in range(2, N):
        qft[i, 1:] = np.power(qft[1, 1:], i)

    return qft


def get_probs(n_qubits, phase):
    """
    Given a number of qubits and a phase, this function returns the probabilities, of what would happen, if a
    Quantum Phase Estimation was executed and the phase was the given phase.
    :param n_qubits: number of qubits
    :param phase: phase
    :return: array containing probabilities for state |0>, |1>, ...
    """
    N = 2**n_qubits
    qft = get_qft_matrix(n_qubits)
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    power = 2j*np.pi*phase
    for i in range(N):
        state[i] *= np.exp(power*i)

    temp = qft.T.conj() @ state
    temp /= np.linalg.norm(temp)
    return np.square(temp.real) + np.square(temp.imag)


def create_plot(n_qubits, n_phases):
    """
    Given two integers, one specifying the number of qubits used and the other the number of phases. The function
    generates the specified amount of phases equidistantly in the interval [-1, 1]. Next it computes for each phase
    the probabilities for each possible outcome of a QPE, if it was executed with this phase. Finally, it returns
    a heatmap. The phases are on the y-axis, the possible outcomes are on the x-axis and the probabilities are the
    colours.
    :param n_qubits: number of qubits
    :param n_phases: number of phases
    :return: heatmap via plotly
    """
    N = 2**n_qubits
    x = np.array(range(N))
    y = np.linspace(-1, 1, n_phases)
    z = np.empty((n_phases, N))
    for i in range(z.shape[0]):
        z[i] = get_probs(n_qubits, y[i])

    return get_heatmap(x, y, z)


#TODO Remove from final version
def create_2d_plot(n_qubits, n_phases):
    N = 2**n_qubits
    x = np.linspace(0, 1, n_phases, endpoint=True)
    y = np.empty(n_phases, dtype=float)
    for idx, p in enumerate(x):
        temp = get_probs(n_qubits, p)
        if p == 0 or p == 1:
            y[idx] = temp[0]
        else:
            y[idx] = temp[int(np.ceil(p * N)) % N] + temp[int(p * N) % N]

    df = pd.DataFrame(dict(
        phase=x,
        closest_prob=y,
    ))
    return px.line(df, x="phase", y="closest_prob", title='')


def create_seaborn_plot(n_qubits, n_phases):
    """
    Given two integers, one specifying the number of qubits used and the other the number of phases. The function
    generates the specified amount of phases equidistantly in the interval [-1, 1]. Next it computes for each phase
    the probabilities for each possible outcome of a QPE, if it was executed with this phase. Finally, it returns
    a heatmap. The phases are on the y-axis, the possible outcomes are on the x-axis and the probabilities are the
    colours.
    :param n_qubits: number of qubits
    :param n_phases: number of phases
    :return: heatmap via seaborn
    """
    import matplotlib.mathtext as mathtext
    N = 2**n_qubits
    x = np.array(range(N))
    y = np.linspace(-1, 1, n_phases)
    z = np.empty((n_phases, N))
    for i in range(z.shape[0]):
        z[i] = get_probs(n_qubits, y[i])

    return get_seaborn_heatmap(x, y, z, f"Estimated Phase times $2^{n_qubits}$", "True Phase")


def main():
    """
    Creates plot of QPE output probabilities via matplotlib and saves it as pdf
    """
    import matplotlib.pyplot as plt
    create_seaborn_plot(4, 2**20)
    plt.tight_layout()
    plt.savefig("./fourier.pdf", dpi="figure", format="pdf")


if __name__ == "__main__":
    main()
