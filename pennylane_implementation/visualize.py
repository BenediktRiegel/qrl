import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pennylane as qml
import pandas as pd
from quantum_backends import QuantumBackends
from utils import int_to_bitlist
from wire_utils import get_wires


def get_arrow(x_start, y_start, x_end, y_end, color=(255, 51, 0)):
    """
    Returns a plotly arrow, starting at the coordinates (x_start, y_start) and ending in (x_enc, y_end)
    :param x_start: float
    :param y_start: float
    :param x_end: float
    :param y_end: float
    :return: arrow for plotly
    """
    # print(f"({x_start}, {y_start}) -> ({x_end}, {y_end})")
    arrow = go.layout.Annotation(dict(
        x=x_end,
        y=y_end,
        xref="x", yref="y",
        text="",
        showarrow=True,
        axref="x", ayref='y',
        ax=x_start,
        ay=y_start,
        arrowhead=3,
        arrowwidth=3,
        arrowcolor=f'rgb({color[0]},{color[1]},{color[2]})', )
    )
    return arrow


def get_action_probs(x, y, action_qnn, num_x_qubits, num_y_qubits):
    """
    Estimates the action probabilities, given the state (x, y) and the action QNN
    :param x: int
    :param y: int
    :param action_qnn: action QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :return: probabilities
    """
    wires, total_num_wires = get_wires([num_x_qubits, num_y_qubits, 2, 2])
    x_qubits, y_qubits, action_qubits, ancilla_qubits = wires

    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend(total_num_wires, None)
    x_bits = int_to_bitlist(x, num_x_qubits)
    y_bits = int_to_bitlist(y, num_y_qubits)

    def circuit():
        for (x_q, x_b) in zip(x_qubits, x_bits):
            if x_b == 1:
                qml.PauliX((x_q,))
        for (y_q, y_b) in zip(y_qubits, y_bits):
            if y_b == 1:
                qml.PauliX((y_q,))
        action_qnn.circuit(x_qubits + y_qubits, action_qubits, ancilla_qubits=ancilla_qubits)

        return [qml.probs(wires=action_qubits)]

    return qml.QNode(circuit, backend)()[0].numpy()


def get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits, slip_probs):
    """
    Returns policy arrows and slip arrows for state (x, y), given a certain action QNN.
    The policy arrows are proportional to the probability of them being chosen.
    The slip arrows are proportional to the probability of the agent actually transitioning to that field.
    :param x: int
    :param y: int
    :param action_qnn: action QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :return: probabilities
    """
    probs = get_action_probs(x, y, action_qnn, num_x_qubits, num_y_qubits)

    slip_probs = np.array(slip_probs)
    slipped_probs = np.zeros(4)
    for a_prob in probs:
        slipped_probs += (a_prob * slip_probs)
        slip_probs = np.roll(slip_probs, shift=1)
    # 00: Right
    # 01: Down
    # 10: Left
    # 11: Up
    x_offset = 0.04
    y_offset = 0.04
    direction = [np.array([1-2*x_offset, 0]), np.array([0, -1+2*y_offset]), np.array([-1+2*x_offset, 0]), np.array([0, 1-2*y_offset])]

    start_pos = np.array([[x+x_offset, y+y_offset], [x-x_offset, y-y_offset], [x-x_offset, y+y_offset], [x-x_offset, y+y_offset]])
    end_pos = [pos + (p * d) / 2. for pos, p, d in zip(start_pos, probs, direction)]
    arrows = [get_arrow(s_p[0], s_p[1], e_p[0], e_p[1]) for s_p, e_p in zip(start_pos, end_pos)]

    slipped_start_pos = np.array([[x+x_offset, y-y_offset], [x+x_offset, y-y_offset], [x-x_offset, y-y_offset], [x+x_offset, y+y_offset]])
    slipped_end_pos = [pos + (p * d) / 2. for pos, p, d in zip(slipped_start_pos, slipped_probs, direction)]
    arrows += [get_arrow(s_p[0], s_p[1], e_p[0], e_p[1], color=(0, 0, 0)) for s_p, e_p in zip(slipped_start_pos, slipped_end_pos)]

    return arrows


def get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, v_m):
    """
    Estimates the value of the state (x, y) and returns it.
    :param x: x coordinate
    :param y: y coordinate
    :param value_qnn: a Value QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :param r_m: the maximum of the absolute rewards in the FrozenLake environment
    :return: float
    """
    wires, total_num_wires = get_wires([num_x_qubits, num_y_qubits, 1, 1, 2])
    x_qubits, y_qubits, value_qubit, swap_qubit, ancilla_qubits = wires
    value_qubit, swap_qubit = value_qubit[0], swap_qubit[0]
    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend(total_num_wires, None)
    x_qubits = list(range(num_x_qubits))
    y_qubits = list(range(num_x_qubits, num_x_qubits + num_y_qubits))
    value_qubit = num_x_qubits + num_y_qubits
    x_bits = int_to_bitlist(x, num_x_qubits)
    y_bits = int_to_bitlist(y, num_y_qubits)

    def circuit():
        for (x_q, x_b) in zip(x_qubits, x_bits):
            if x_b == 1:
                qml.PauliX((x_q,))
        for (y_q, y_b) in zip(y_qubits, y_bits):
            if y_b == 1:
                qml.PauliX((y_q,))

        qml.Hadamard((swap_qubit,))
        value_qnn.circuit(
            x_qubits + y_qubits, value_qubit, control_qubits=[swap_qubit],
            ancilla_qubits=ancilla_qubits
        )
        qml.Hadamard((swap_qubit,))

        return [qml.probs(wires=(swap_qubit,))]

    probs = qml.QNode(circuit, backend)()[0].numpy()

    value = v_m * (probs[0] - probs[1])

    return value


def plot_frozen_lake(environment, action_qnn, num_x_qubits, num_y_qubits):
    """
    Returns a heatmap of the FrozenLake, includeing policy arrows
    :return: plotly heatmap
    """
    for param in action_qnn.parameters():
        param.requires_grad = False
    Z = [[el.reward for el in row] for row in environment.map]
    lake_fig = go.Figure(
        go.Heatmap(
            z=Z,
            x=list(range(len(Z[0]))),
            y=list(range(len(Z))),
            showscale=True,
            colorscale=px.colors.sequential.Viridis,
            hoverinfo="all",
            opacity=0.55,
        )
    )
    policy_arrows = []
    for y in range(len(Z)):
        for x in range(len(Z[0])):
            policy_arrows += get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits, environment.slip_probabilities)

    lake_fig.update_layout(annotations=policy_arrows)

    for param in action_qnn.parameters():
        param.requires_grad = True
    return lake_fig


def get_frozen_lake_action_frame(environment, action_qnn, num_x_qubits, num_y_qubits):
    """
    Gathers the policy arrows of each state and returns them as a plotly Frame.
    :param environment: FrozenLake environment
    :param action_qnn: action QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :return: plotly frame
    """
    param_requires_grad_value = action_qnn.parameters()[0].requires_grad
    for param in action_qnn.parameters():
        param.requires_grad = False

    policy_arrows = []
    for y in range(len(environment.map)):
        for x in range(len(environment.map[0])):
            policy_arrows += get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits, environment.slip_probabilities)

    for param in action_qnn.parameters():
        param.requires_grad = param_requires_grad_value
    return go.Frame(layout=go.Layout(annotations=policy_arrows))


def get_frozen_lake_value_frame(environment, value_qnn, num_x_qubits, num_y_qubits, gamma, end_state_values):
    """
    Computes a heatmap of the FrozenLake environment. One state consists of 4 fields. The top right corner is set to
    the reward received, when transitioning onto this field. The other subfields are assigned the value of each state,
    given by the value QNN. This is returned as a Frame
    :param environment: FrozenLake environment
    :param value_qnn: value QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :param gamma: float discount factor
    :param end_state_values: bool if True, then terminal state also estimate state values
    :return: plotly frame
    """
    # v_max = environment.r_m / (1 - gamma)
    v_max = environment.r_m / (1 - gamma) if end_state_values else environment.r_m

    param_requires_grad_value = value_qnn.parameters()[0].requires_grad
    for param in value_qnn.parameters():
        param.requires_grad = False

    heatmap = np.empty((2 * len(environment.map), 2 * len(environment.map[0])))
    for y, row in enumerate(environment.map):
        for x, field in enumerate(row):
            value = get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, v_max)
            heatmap[2 * y, 2 * x] = value
            heatmap[2 * y, 2 * x + 1] = value
            heatmap[2 * y + 1, 2 * x] = np.cos(value_qnn.in_q_parameters.detach()[0, x*(2**int(np.log2(len(environment.map)))) + y] / 2.)
            heatmap[2 * y + 1, 2 * x + 1] = environment.default_reward if field.reward is None else field.reward

    for param in value_qnn.parameters():
        param.requires_grad = param_requires_grad_value

    value_fig = go.Heatmap(z=heatmap)

    return go.Frame(data=value_fig)


def get_frozen_lake_frame(environment, action_qnn, value_qnn, num_x_qubits, num_y_qubits, gamma, end_state_values):
    """
    Gets frame of policy and frame of values and combines them. This combined result is returned.
    :param environment: FrozenLake environment
    :param action_qnn: action QNN
    :param value_qnn: value QNN
    :param num_x_qubits: int number of qubits required to encode the x coordinate of any state of the FrozenLake
    :param num_y_qubits: int number of qubits required to encode the y coordinate of any state of the FrozenLake
    :param gamma: float discount factor
    :param end_state_values: bool if True, then terminal state also estimate state values
    :return: plotly frame
    """
    action_frame = get_frozen_lake_action_frame(environment, action_qnn, num_x_qubits, num_y_qubits)
    value_frame = get_frozen_lake_value_frame(environment, value_qnn, num_x_qubits, num_y_qubits, gamma, end_state_values)
    value_frame.update(action_frame)
    return value_frame


def plot_animated_frozen_lake(environment, frames, gamma, end_state_values: bool = False):
    """
    Given the FrozenLake frames it creates an animated FrozenLake
    :param environment: FrozenLake
    :param frames: plotly frames
    :param gamma: float discount factor
    :param end_state_values: bool if True, then terminal state also estimate state values
    """
    for idx, frame in enumerate(frames):
        frame["name"] = idx
    heatmap = np.empty((2 * len(environment.map), 2 * len(environment.map[0])))
    for y in range(len(environment.map)):
        for x in range(len(environment.map[0])):
            heatmap[2 * y, 2 * x] = environment.map[y][x].reward
            heatmap[2 * y, 2 * x + 1] = environment.map[y][x].reward
            heatmap[2 * y + 1, 2 * x] = environment.map[y][x].reward
            heatmap[2 * y + 1, 2 * x + 1] = environment.map[y][x].reward
    lake_fig = go.Figure(
        # data=frames[0]["data"]
        data=go.Heatmap(
            z=heatmap,
            x=np.array(list(range(len(heatmap[0]))), dtype=float) / 2. - 0.25,
            y=np.array(list(range(len(heatmap))), dtype=float) / 2. - 0.25,
            zmin=-environment.r_m / (1-gamma) if end_state_values else environment.r_m,
            zmax=environment.r_m / (1-gamma) if end_state_values else environment.r_m,
            showscale=True,
            colorscale=px.colors.sequential.Viridis,
            hoverinfo="all",
            opacity=0.55,
        ),
        frames=frames,
    )
    lake_fig.update_layout(
        annotations=frames[0]["layout"]["annotations"],
        xaxis=dict(showgrid=True, zeroline=True,
                   linecolor='black', ),
        yaxis=dict(showgrid=True, zeroline=True,
                   linecolor='black', ),
    )

    lake_fig.update_layout(
        updatemenus=[
            {'buttons': [{'args': [None, {'frame': {'duration':
                                                        500, 'redraw': True},
                                          'mode': 'immediate',
                                          'fromcurrent': True,
                                          'transition': {'duration':
                                                             500, 'easing': 'linear'}}],
                          'label': '&#9654;',
                          'method': 'animate'},
                         {'args': [[None], {'frame':
                                                {'duration': 0, 'redraw':
                                                    True}, 'mode': 'immediate',
                                            'fromcurrent': True,
                                            'transition': {'duration': 0,
                                                           'easing': 'linear'}}],
                          'label': '&#9724;',
                          'method': 'animate'}
                         ],
             'direction': 'left',
             'pad': {'r': 10, 't': 70},
             'showactive': True,
             'type': 'buttons',
             'x': 0.1,
             'xanchor': 'right',
             'y': 0,
             'yanchor': 'top'}
        ]
    )
    lake_fig.update_layout({
        'sliders':
            [
                {'active': 0,
                 'currentvalue': {'prefix': 'Iteration='},
                 'len': 0.9,
                 'pad': {'b': 10, 't': 60},
                 'steps': [{'args': [[frame["name"]], {'frame': {'duration':
                                                                     0, 'redraw': True}, 'mode':
                                                           'immediate', 'fromcurrent': True,
                                                       'transition': {'duration': 0,
                                                                      'easing': 'linear'}}],
                            'label': frame["name"],
                            'method': 'animate'} for frame in lake_fig.frames],
                 'x': 0.1,
                 'xanchor': 'left',
                 'y': 0,
                 'yanchor': 'top'}
            ]
    })

    # lake_fig.layout.height = 800
    # lake_fig.layout.width = 800

    return lake_fig


def plot_loss(losses):
    """
    Given a 2d List of both losses, this method plots them in the same plot and returns it
    :param losses: action and value losses in a 2d list
    :return: plotly line plot
    """
    losses = np.array(losses)
    if len(losses.shape) == 2:
        if losses.shape[1] == 2:
            df = pd.DataFrame(losses, columns=[f"action_loss", "value_loss"])
            fig = px.line(df, title="loss")
            return fig
        else:
            df = pd.DataFrame(losses, columns=[f"loss {i}" for i in range(losses.shape[1])])
            fig = px.line(df, title="loss")
            return fig
    df = pd.DataFrame(losses, columns=[f"loss"])
    fig = px.line(df, title="loss")
    return fig
