import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pennylane as qml
import pandas as pd
from quantum_backends import QuantumBackends
from utils import int_to_bitlist


def get_arrow(x_start, y_start, x_end, y_end):
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
        arrowcolor='rgb(255,51,0)', )
    )
    return arrow


def get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits):
    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend("", "", num_x_qubits + num_y_qubits + 2,
                                                                            None)
    x_qubits = list(range(num_x_qubits))
    y_qubits = list(range(num_x_qubits, num_x_qubits + num_y_qubits))
    action_qubits = list(range(num_x_qubits + num_y_qubits, num_x_qubits + num_y_qubits + 2))
    x_bits = int_to_bitlist(x, num_x_qubits)
    y_bits = int_to_bitlist(y, num_y_qubits)

    def circuit():
        for (x_q, x_b) in zip(x_qubits, x_bits):
            if x_b == 1:
                qml.PauliX((x_q,))
        for (y_q, y_b) in zip(y_qubits, y_bits):
            if y_b == 1:
                qml.PauliX((y_q,))
        action_qnn.circuit(x_qubits + y_qubits, action_qubits)

        return [qml.probs(wires=action_qubits)]

    probs = qml.QNode(circuit, backend)()[0].numpy()
    # 00: Right
    # 01: Down
    # 10: Left
    # 11: Up
    pos = np.array([x, y])
    direction = [np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])]
    end_pos = [pos + (p * d) / 2. for p, d in zip(probs, direction)]
    arrows = [get_arrow(x, y, e_p[0], e_p[1]) for e_p in end_pos]

    return arrows


def get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, r_m):
    backend = QuantumBackends.pennylane_default_qubit.get_pennylane_backend("", "", num_x_qubits + num_y_qubits + 1,
                                                                            None)
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
        value_qnn.circuit(x_qubits + y_qubits, value_qubit)

        return [qml.probs(wires=value_qubit)]

    probs = qml.QNode(circuit, backend)()[0].numpy()
    # Determine theta, such that P(RY(theta)|0> = |0>) = P(value_qubit = |0>)
    # |cos(theta/2)|^2 = P(RY(theta)|0> = |0>) => theta = 2*arccos(sqrt(P(RY(theta)|0> = |0>)))
    theta = 2 * np.arccos(np.sqrt(probs[0]))
    # Compute back the value from theta
    # theta = (v / r_m + 1) π / 2 => v = ((2/π * theta) - 1) * r_m
    return ((2. / np.pi * theta) - 1) * r_m


def plot_frozen_lake(environment, action_qnn, num_x_qubits, num_y_qubits):
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
            policy_arrows += get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits)

    lake_fig.update_layout(annotations=policy_arrows)

    for param in action_qnn.parameters():
        param.requires_grad = True
    return lake_fig


def get_frozen_lake_action_frame(environment, action_qnn, num_x_qubits, num_y_qubits):
    param_requires_grad_value = action_qnn.parameters()[0].requires_grad
    for param in action_qnn.parameters():
        param.requires_grad = False

    policy_arrows = []
    for y in range(len(environment.map)):
        for x in range(len(environment.map[0])):
            policy_arrows += get_policy_arrows(x, y, action_qnn, num_x_qubits, num_y_qubits)

    for param in action_qnn.parameters():
        param.requires_grad = param_requires_grad_value
    return go.Frame(layout=go.Layout(annotations=policy_arrows))


def get_frozen_lake_value_frame(environment, value_qnn, num_x_qubits, num_y_qubits):
    param_requires_grad_value = value_qnn.parameters()[0].requires_grad
    for param in value_qnn.parameters():
        param.requires_grad = False

    heatmap = np.empty((2 * len(environment.map), 2 * len(environment.map[0])))
    for y in range(len(environment.map)):
        for x in range(len(environment.map[0])):
            heatmap[2 * y, 2 * x] = get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, environment.r_m)
            heatmap[2 * y, 2 * x + 1] = get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, environment.r_m)
            heatmap[2 * y + 1, 2 * x] = get_value(x, y, value_qnn, num_x_qubits, num_y_qubits, environment.r_m)
            heatmap[2 * y + 1, 2 * x + 1] = environment.map[y][x].reward

    for param in value_qnn.parameters():
        param.requires_grad = param_requires_grad_value

    value_fig = go.Heatmap(z=heatmap)

    return go.Frame(data=value_fig)


def get_frozen_lake_frame(environment, action_qnn, value_qnn, num_x_qubits, num_y_qubits):
    action_frame = get_frozen_lake_action_frame(environment, action_qnn, num_x_qubits, num_y_qubits)
    value_frame = get_frozen_lake_value_frame(environment, value_qnn, num_x_qubits, num_y_qubits)
    value_frame.update(action_frame)
    return value_frame


def plot_animated_frozen_lake(environment, frames):
    for idx, frame in enumerate(frames):
        frame["name"] = idx
    heatmap = np.empty((2 * len(environment.map), 2 * len(environment.map[0])))
    for y in range(len(environment.map)):
        for x in range(len(environment.map[0])):
            heatmap[2 * y, 2 * x] = environment.map[y][x].reward
            heatmap[2 * y, 2 * x] = environment.map[y][x].reward
            heatmap[2 * y, 2 * x] = environment.map[y][x].reward
            heatmap[2 * y + 1, 2 * x + 1] = environment.map[y][x].reward
    lake_fig = go.Figure(
        # data=frames[0]["data"]
        data=go.Heatmap(
            z=heatmap,
            x=np.array(list(range(len(heatmap[0]))), dtype=float) / 2. - 0.25,
            y=np.array(list(range(len(heatmap))), dtype=float) / 2. - 0.25,
            zmin=-2,
            zmax=2,
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
                   linecolor='black',),
        yaxis=dict(showgrid=True, zeroline=True,
                   linecolor='black',),
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

    return lake_fig


def plot_loss(losses):
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
