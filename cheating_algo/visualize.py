import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from load_config import _load_config, save_config
from logger import load_log
from solve_matrix_form import calculate_policy_quality, calculate_optimal_policy_quality
import matplotlib.pyplot as plt
from datetime import datetime


# terminal_states = {5, 7, 11, 12, 15}
# goal_states = {15}
terminal_states = {0, 3, 7, 9, 11}
goal_states = {3}
v_max = None

play_speed = 500 // 8


def get_v_max(gamma, end_state_values):
    """
    Given the parameter gamma and end_state_values, this method returns |v|_max if end_state_values is True
    as 1/(1-gamma), else as 1
    :param gamma: discount factor
    :param end_state_values: boolean determining if the terminal states of the environment are assigned none zero values
    """
    global v_max
    if v_max is None:
        v_max = (1. / (1-gamma)) if end_state_values else 1.
    return v_max


def get_state_visual(x, y, gamma, end_state_values):
    """
    Returns the visualization value of the state (x, y) given gamma and end_state_value.
    In the visualization of the FrozenLake, holes are represented by the value -v_max, goals by v_max and normal fields
    by 0.
    :param x: x coordinate of a state
    :param y: y coordinate of a state
    :param gamma: discount factor
    :param end_state_values: boolean determining if the terminal states of the environment are assigned none zero values
    """
    v_max = get_v_max(gamma, end_state_values)
    s = y*4 + x
    if s in goal_states:
        return v_max
    elif s in terminal_states:
        return -v_max
    else:
        return 0

def get_arrow(x_start, y_start, x_end, y_end, color=(255, 51, 0)):
    """
    Returns an arrow going from the coordinates (x_start, y_start) to the coordinate (x_end, y_end). The color of the
    arrow can be changed optionally.
    :param x_start: x coordinate of the starting position
    :param y_start: y coordinate of the starting position
    :param x_end: x coordinate of the end position
    :param y_end: y coordinate of the end position
    :param color: optional rgb values to determine the color of the arrow
    :return: arrow as a go.layout.Annotation object
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


def get_policy_arrows(x, y, action_probs):
    """
    Given a state (x, a), this function returns 8 arrows. Four of these arrows are black and represent the actions
    chosen by the policy. The other four arrows are red and represent the direction the agent actually transition.
    This direction differs from the chosen ones, since the agent might slip. The length of the arrows are proportional
    to the transition probabilities (red) and the action probabilities (black).
    All arrows start at about (x, y). A small offset might be added in either direction,to avoid overlapping.

    :param x: x coordinate of the state
    :param y: y coordinate of the state.
    :param action_probs: matrix of size |S| x |A| containing the probabilities of taking action a, while residing in
    state s
    """
    slip_probs = np.array([1., 1., 0., 1.]) / 3.
    slipped_probs = np.zeros(4)
    for a_prob in action_probs:
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
    end_pos = [pos + (p * d) / 2. for pos, p, d in zip(start_pos, action_probs, direction)]
    arrows = [get_arrow(s_p[0], s_p[1], e_p[0], e_p[1]) for s_p, e_p in zip(start_pos, end_pos)]

    slipped_start_pos = np.array([[x+x_offset, y-y_offset], [x+x_offset, y-y_offset], [x-x_offset, y-y_offset], [x+x_offset, y+y_offset]])
    slipped_end_pos = [pos + (p * d) / 2. for pos, p, d in zip(slipped_start_pos, slipped_probs, direction)]
    arrows += [get_arrow(s_p[0], s_p[1], e_p[0], e_p[1], color=(0, 0, 0)) for s_p, e_p in zip(slipped_start_pos, slipped_end_pos)]

    return arrows


def plot_frozen_lake(action_qnn, gamma):
    """
    Given the action QNN and the discount factor gamma, this returns a heatmap of size 4x4, with the z value of each
    (x, y) coordinate being the value of get_state_visual. Additionally, it adds the policy arrows from
    get_policy_arrows.
    :param action_qnn: action QNN
    :param gamma: discount factor gamma
    :return: a heatmap as go.Figure
    """
    Z = [[get_state_visual(x, y, gamma) for x in range(4)] for y in range(4)]
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
            policy_arrows += get_policy_arrows(x, y, action_qnn)

    lake_fig.update_layout(annotations=policy_arrows)

    for param in action_qnn.parameters():
        param.requires_grad = True
    return lake_fig


def get_frozen_lake_action_frame(action_probs):
    """
    Returns a go.Frame containing the policy and slip arrows of get_policy_arrows of a 4x4 FrozenLake, given
    the action probabilities of each state.
    :param action_probs: matrix of size |S|x|A|. Entry y•4 + x is a vector containing the probability of taking action a
    when residing in state (x, y) at entry a.
    :return: go.Frame with policy and slip arrows
    """
    policy_arrows = []
    for y in range(4):
        for x in range(4):
            policy_arrows += get_policy_arrows(x, y, action_probs[y*4 + x])

    return go.Frame(layout=go.Layout(annotations=policy_arrows))


def get_frozen_lake_value_frame(state_values, gamma, end_state_values):
    """
    Returns a frame containing a heatmap of size 8x8 for a 4x4 FrozenLake. One state is represented by 4 subfields.
    The top right subfield is coloured after the state visuals of the function ``get_state_visual``. The other three
    subfields are coloured after the learned state value of the agent.
    :param state_values: vector of size |S| containing the state values
    :param gamma: discount factor
    :param end_state_values: Boolean determining whether terminal states are assigned a value.
    :return: a go.Frame object
    """
    # v_max = environment.r_m / (1 - gamma)
    # v_max = 1 / (1 - gamma) if end_state_values else 1

    heatmap = np.empty((8, 8))
    for y in range(4):
        for x in range(4):
            value = state_values[4*y + x]
            heatmap[2 * y, 2 * x] = value
            heatmap[2 * y, 2 * x + 1] = value
            heatmap[2 * y + 1, 2 * x] = value
            heatmap[2 * y + 1, 2 * x + 1] = get_state_visual(x, y, gamma, end_state_values)

    value_fig = go.Heatmap(z=heatmap)

    return go.Frame(data=value_fig)


def get_frozen_lake_frame(action_probs, state_values, gamma, end_state_values):
    """
    Retrieves action and value frame and merges them into one. Returns this new frame.
    :param action_probs: matrix of size |S|x|A|. Entry y•4 + x is a vector containing the probability of taking action a
    when residing in state (x, y) at entry a.
    :param state_values: vector of size |S| containing the state values
    :param gamma: discount factor
    :param end_state_values: Boolean determining whether terminal states are assigned a value.
    :return: a go.Frame object
    """
    action_frame = get_frozen_lake_action_frame(action_probs)
    value_frame = get_frozen_lake_value_frame(state_values, gamma, end_state_values)
    value_frame.update(action_frame)
    return value_frame


def plot_animated_frozen_lake(frames, gamma, end_state_values):
    """
    Creates an animated frozenLake trainings session, given the frames. The max and min values of the heatmap
    are set to v_max and -v_max respectfully.
    :param frames: list of frozenLake frames (see get_frozen_lake_frame)
    :param gamma: discount factor
    :paran end_state_values: Boolean determining whether terminal states are assigned a value.
    :return: animated heatmap
    """
    for idx, frame in enumerate(frames):
        frame["name"] = idx
    heatmap = np.empty((8, 8))
    for y in range(4):
        for x in range(4):
            r = get_state_visual(x, y, gamma, end_state_values)
            heatmap[2 * y, 2 * x] = r
            heatmap[2 * y, 2 * x + 1] = r
            heatmap[2 * y + 1, 2 * x] = r
            heatmap[2 * y + 1, 2 * x + 1] = r

    v_max = get_v_max(gamma, end_state_values)
    lake_fig = go.Figure(
        # data=frames[0]["data"]
        data=go.Heatmap(
            z=heatmap,
            x=np.array(list(range(len(heatmap[0]))), dtype=float) / 2. - 0.25,
            y=np.array(list(range(len(heatmap))), dtype=float) / 2. - 0.25,
            zmin=-v_max,
            zmax=v_max,
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
                                                        play_speed, 'redraw': True},
                                          'mode': 'immediate',
                                          'fromcurrent': True,
                                          'transition': {'duration':
                                                             play_speed, 'easing': 'linear'}}],
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
    Plots losses during training. If the input parameter losses contains exactly two losses, then they are named
    ``action_loss`` and ``value_loss``. For more than two, the j'th loss is named ``loss j``. Otherwise, its just named
    ``loss``.
    :param losses: matrix of size T x |L|, where T represents the total steps during training. |L| is the number of
    losses.
    :return: line plot
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


def plot_max_grad(grads):
    """
    Given a list of size T x 2 x #parameters containing the gradients of all parameters of the action and value QNN
    at each training step t. It plots the maximum gradient of the action QNN at each training step and the same
    for the value QNN.
    :param grads: list of size T x 2 x #parameters. T is the number of training steps. First index is the training
    step, second index is the action QNN and the value QNN and last index is the parameter idx.
    :return: line plot
    """
    max_grads = np.array([[np.abs(np.array(entry[0])).max(), np.abs(np.array(entry[1])).max()] for entry in grads])
    df = pd.DataFrame(max_grads, columns=["action", "value"])
    fig = px.line(df, title="max grad")
    return fig


def plot_value_grads(value_grads):
    """
    Given the gradient of each parameter of the value QNN at each training step, this function returns a line plot of
    these gradients.
    :param: List of gradients of the value QNN at each training step
    :return: line plot
    """
    value_grads = np.array(value_grads)
    df = pd.DataFrame(value_grads, columns=[f"param{i}" for i in range(value_grads.shape[1])])
    fig = px.line(df, title="Value Grads")
    return fig


def plot_max_params_change(max_params_change):
    """
    Given the maximum change of the parameters of the value and action QNN at each training step, this function plots
    them in a line graph.
    :param max_params_change: List of size T x 2, containing the maximum gradient of the parameter of the value and
    action QNN at each time step.
    """
    max_params_change = np.array(max_params_change)
    df = pd.DataFrame(max_params_change, columns=["action", "value"])
    fig = px.line(df, title="Max Parameter Change")
    return fig


def plot_policy_quality(policy_quality):
    """
    Given the policy quality, this function creates a line plot.
    :param policy_quality: List of the policy quality during each training step
    :return: line plot
    """
    policy_quality = np.array(policy_quality)
    df = pd.DataFrame(policy_quality)
    fig = px.line(df, title="Quality of the Policy over Episodes")
    return fig


def dict_subset_of_dict(sub_dict: dict, big_dict: dict):
    """
    Checks, if ``sub_dict`` is a sub-dictionary of ``big_dict``. For this, for each key k in ``sub_dict``, ``big_dict``
    has to also contain the key k. Both dicts have to have the same value, for a given key k, if both dicts contain the
    key k.
    :param sub_dict: dict
    :param big_dict: dict
    :return: boolean. It is True, if the check succeeds and False otherwise.
    """
    for k, v in sub_dict.items():
        if k not in big_dict:
            return False
        if v is not None and big_dict[k] is not None:
            if v != big_dict[k]:
                return False
        if v is not None and big_dict[k] is None:
            return False
        if v is None and big_dict[k] is not None:
            return False
    return True


def retrieve_result_paths(config, start_dir, result_path="./results"):
    """
    Returns a list of paths to training results in the path 'result_path'. Each path has to fulfill the property that
    its last directory comes after the ``start_dir``, when sorted lexicographically. Additionally, the ``config`` has
    to be a sub-directory of the config within the path.
    :param config: dict
    :param start_dir: str
    :param result_path: str | Path object. Default value is './results'
    :return: list of paths
    """
    import os.path
    result_path = Path(result_path)
    dirs = [d for d in os.listdir(result_path) if os.path.isdir(result_path / d)]
    dirs.sort()
    start_idx = 0
    for idx, d in enumerate(dirs):
        if d == start_dir:
            start_idx = idx
            break
    dirs = dirs[start_idx:]
    result_dirs = []
    for d in dirs:
        config_path = (result_path / d) / "config.json"
        if config_path.exists():
            d_config = _load_config(config_path)
            if dict_subset_of_dict(config, d_config):
                result_dirs.append(result_path / d)

    return result_dirs


def retrieve_policy_qualities(paths, recalculate=False):
    """
    Given a list of paths to results of the training process, this retrieves the policy qualities for each result.
    If the policy quality is not there, it calculates them and saves them as a npy file. If the policy qualities do not
    have the same length, it repeats the last quality of a result, until the length matches that of the result, with
    the most policy quality entries.
    It returns a matrix of size #path x policy quality length
    :param paths: list of paths
    :param recalculate: optional bool. Forces recalculation of the policy qualities, if True.
    :return: numpy array
    """
    policy_qualities = []
    max_length = 0
    # Go through directories and load policy qualities
    for d in paths:
        path_to_quality = Path(d) / "policy_quality.npy"
        if path_to_quality.exists() and not recalculate:
            quality = np.load(d / "policy_quality.npy")
        else:
            quality = create_policy_quality_file(d)
        max_length = max(max_length, len(quality))
        policy_qualities.append(quality)

    # Adjust lengths
    final_result = np.zeros((len(policy_qualities), max_length))
    for idx, quality in enumerate(policy_qualities):
        final_result[idx, :len(quality)] = quality
        final_result[idx, len(quality):] = [quality[-1]] * (max_length - len(quality))

    return final_result


def create_policy_quality_file(result_path, gamma=None, log=None, action_probs=None):
    """
    Given a path, this method calculates an array of the policy qualities of the training process saved in the path.
    It save this array as a npy file in the path and returns it as well.
    :param result_path: path
    :param gamma: optional gamma value, if known
    :param log: optional log in path, if known
    :param action_probs: optional action probabilities, if known
    :return: list of policy qualities at ever training step
    """
    result_path = Path(result_path)
    if gamma is None:
        config = _load_config(result_path / "config.json")
        gamma = config["gamma"]
    if action_probs is None:
        if log is None:
            log = load_log(result_path / "log.txt")
        action_probs = [entry["action_probs"] for entry in log]
    policy_quality = np.array([calculate_policy_quality(policy, gamma) for policy in action_probs])
    np.save(result_path / "policy_quality.npy", policy_quality)
    return policy_quality


def plot_mean_std(data, label=""):
    """
    Given a 2d np.array, the second dimension is the "x-axis" and the first dimension encodes the values per step.
    This method takes the mean and the std along axis=0 and plots the mean as a bold line and the std is a shaded are
    around the mean with thickness of two stds, i.e. mean-std to mean+std
    :param data: 2d np array
    :param label: optional label for the mean
    :return: fig, axis and x ticks
    """
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    fig, ax = plt.subplots()
    ax.fill_between(x, mean + std, mean - std, alpha=0.2)
    ax.plot(x, mean, label=label)
    ax.margins(x=0)
    return fig, ax, x


def save_mean_std_plot(data, save_path, file_name, x_label, y_label):
    """
    Given a 2d np.array, this function plots the mean and the std in a plot and saves it to the specified save_path.
    The name of the saved file is set to the parameter file_name and the x-axis and y-axis are labeled with x_label and
    y_label respectfully.
    :param data: 2d np.array containing the data
    :param save_path: str | Path path to to save the file to
    :param file_name: str name the file should have
    :param x_label: str label for the x-axis
    :param y_label: str label for the y-axis
    """
    fig, ax, x = plot_mean_std(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{file_name}.pdf", dpi="figure", format="pdf")
    plt.clf()


def save_matplotlib_policy_quality(policy_qualities, save_path, gamma=None):
    """
    Given the policy qualities of multiple trainings, this function computes the mean and the std at each
    training step. Next it plots the mean as a line and the std as a shaded area via matplotlib and saves the resulting
    plot in the specified path as a pdf file called "policy_quality.pdf". If the optional parameter gamma is not none,
    then it will also compute the optimal value and include it as a single line within the plot
    :param policy_qualities: matrix of size #trainings x #training steps containing the policy qualities
    :param save_path: the path to save the file to
    :param gamma: optional float
    """
    fig, ax, x = plot_mean_std(policy_qualities, label="Mean Quality")

    if gamma is not None and isinstance(gamma, float) and gamma >= 0 and gamma < 1:
        optimal_v = calculate_optimal_policy_quality(gamma)
        y_opt = np.array([optimal_v]*len(x))
        ax.plot(x, y_opt, label="Optimal Quality")

    plt.xlabel("Training step")
    plt.ylabel("Quality")
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "policy_quality.pdf", dpi="figure", format="pdf")
    plt.clf()


def save_policy_qualities(config, start_dir, result_path="./results", output_path=None, copy_config=True):
    """
    Gets all policy qualities of training processes with the same sub-config, created after start_dir.
    Next it saves a plot of these policy qualities to the path specified by output_path. If output_path is None or an
    empty string, then it saves it to './policy_qualities/{current date and time}' instead.
    :param config: dict. Is the sub-config that every other config needs to contain
    :param start_dir: str. Directories of training process's must come after start_dir, lexicographically.
    :param result_path: str | Path object. Default value is './results'
    :param output_path: str | Path object. Default = ""
    """
    directories = retrieve_result_paths(config, start_dir, result_path)
    print(f"retrieved {len(directories)} result paths")
    policy_qualities = retrieve_policy_qualities(directories, True)
    if not output_path:
        output_path = Path("./policy_qualities/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    save_matplotlib_policy_quality(policy_qualities, output_path, config.get("gamma", None))
    if copy_config:
        save_config(output_path / "config.json", config)


def save_two_data_as_plot(data1, data2, output_path, plot_name, label1, label2, x_label, y_label1, y_label2):
    """
    Given two 2d np.arrays, this function plots the mean as a bold line and the std as an area that is 2 std's thick.
    Each dataset has its own y-axis, but they share the x-axis. The data1 is plotted blue and data2 is plotted
    orange. The other parameters are used as follows:
    - label1 (resp. label2) is the label of the mean, in a legend
    - x_label is the label of the x-axis
    - y_label1 (resp. y_label2) is the label of the y-axis for data1 (resp. data2)
    - output_path is the path the result will be saved to
    - plot_name is the name of the output file
    :param data1: 2d np.array
    :param data2: 2d np.array
    :param output_path: str | Path
    :param plot_name: str
    :param label1: str
    :param label2: str
    :param x_label: str
    :param y_label1: str
    :param y_label2: str
    """

    color1 = 'tab:blue'
    x = np.arange(data1.shape[1])
    mean1 = np.mean(data1, axis=0)
    std1 = np.std(data1, axis=0)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(x_label)
    ax1.fill_between(x, mean1 + std1, mean1 - std1, alpha=0.2, color=color1)
    ax1.plot(x, mean1, label=label1, color=color1)
    ax1.set_ylabel(y_label1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.margins(x=0)

    color2 = "tab:orange"
    ax2 = ax1.twinx()
    x2 = np.arange(data2.shape[1])
    mean2 = np.mean(data2, axis=0)
    std2 = np.std(data2, axis=0)
    ax2.fill_between(x2, mean2 + std2, mean2 - std2, alpha=0.2, color=color2)
    ax2.plot(x2, mean2, label=label2, color=color2)
    ax2.set_ylabel(y_label2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{plot_name}.pdf", dpi="figure", format="pdf")
    plt.clf()


def save_log_param_as_plot(training_result_paths, log_param, output_path, plot_name, x_label, y_label):
    """
    Given a list of paths, containing training results and a log_param, the method goes through the training results,
    retrieves the parameter log_param from the logs. Then it plots this data with save_mean_std_plot and saves the
    resulting to 'output_path/plot_name.pdf'. It also labels the x- and y-axis with x_label and y_label
    :param training_result_paths: list of paths
    :param log_param: str
    :param output_path: str | Path
    :param plot_name: str name of output file
    :param x_label: str
    :param y_label: str
    """
    data = []
    max_length = 0
    for p in training_result_paths:
        log = load_log(p / "log.txt")
        data.append([entry[log_param] for entry in log])
        max_length = max(max_length, len(data[-1]))

    # Adjust lengths
    adjusted_data = np.zeros((len(data), max_length))
    for idx, d in enumerate(data):
        adjusted_data[idx, :len(d)] = d
        if max_length - len(d) > 0:
            adjusted_data[idx, len(d):] = [d[-1]] * (max_length - len(d))

    save_mean_std_plot(adjusted_data, output_path, plot_name, x_label, y_label)


def save_two_log_param_as_plot(training_result_paths, log_param1, log_param2, output_path, plot_name, x_label, y_label1, y_label2):
    """
    Given a list of paths, containing training results and two log parameters log_param1 and log_param2, the method
    goes through the training results, retrieves the parameter log_param1 and log_param2 from the logs.
    Then it plots both datas in the same plot, but with different y-axis. The plot is the mean along axis=0
    and a shaded area of the standard deviations std along axis=0. The shaded area is 2 std thick. The plot is then
    saved in the path output_path as plot_name.pdf. It also labels the x-axis with x_label and the two
    y-axes with y_label1 and y_label2 respectfully.
    :param training_result_paths: list of paths
    :param log_param1: str
    :param log_param2: str
    :param output_path: str | Path
    :param plot_name: str name of output file
    :param x_label: str
    :param y_label1: str
    :param y_label2: str
    """
    data1 = []
    max_length1 = 0
    for p in training_result_paths:
        log = load_log(p / "log.txt")
        data1.append([entry[log_param1] for entry in log])
        max_length1 = max(max_length1, len(data1[-1]))

    # Adjust lengths
    adjusted_data1 = np.zeros((len(data1), max_length1))
    for idx, d in enumerate(data1):
        adjusted_data1[idx, :len(d)] = d
        if max_length1 - len(d) > 0:
            adjusted_data1[idx, len(d):] = [d[-1]] * (max_length1 - len(d))

    data2 = []
    max_length2 = 0
    for p in training_result_paths:
        log = load_log(p / "log.txt")
        data2.append([entry[log_param2] for entry in log])
        max_length2 = max(max_length2, len(data2[-1]))

    # Adjust lengths
    adjusted_data2 = np.zeros((len(data2), max_length2))
    for idx, d in enumerate(data2):
        adjusted_data2[idx, :len(d)] = d
        if max_length2 - len(d) > 0:
            adjusted_data2[idx, len(d):] = [d[-1]] * (max_length2 - len(d))

    save_two_data_as_plot(adjusted_data1, adjusted_data2, output_path, plot_name, "", "", x_label, y_label1, y_label2)


def save_action_and_value_losses(config, start_dir, result_path="./results", output_path = None, copy_config=True):
    """
    Gets all losses of training processes with the same sub-config, created after start_dir.
    Next it saves two plots of these losses (action loss and value loss) to the path
    './losses/{current date and time}'.
    :param config: dict. Is the sub-config that every other config needs to contain
    :param start_dir: str. Directories of training process's must come after start_dir, lexicographically.
    :param result_path: str | Path object. Default value is './results'
    """
    training_result_paths = retrieve_result_paths(config, start_dir, result_path)
    print(f"retrieved {len(training_result_paths)} result paths")
    # Get losses
    if not output_path:
        output_path = Path("./losses/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    save_log_param_as_plot(training_result_paths, "value_loss", output_path, "value_loss")
    save_log_param_as_plot(training_result_paths, "action_loss", output_path, "action_loss")
    if copy_config:
        save_config(output_path / "config.json", config)


def retrieve_max_action_and_value_grads(training_result_paths):
    """
    Given a list of paths containing training results, this function retrieves the gradient of the action QNN and the
    value QNN. For each gradient, it takes the maximum value, after first taking the absolute value of the gradients
    at each training step.
    :param training_result_paths:
    :return: 2d np.array, 2d np.array First is the result of the action gradients and second is the result of the value
    gradients.
    """
    action_gradients = []
    value_gradients = []
    max_action_length = 0
    max_value_length = 0
    for p in training_result_paths:
        log = load_log(p / "log.txt")
        action_gradients.append([np.absolute(np.array(entry["action_grad"])).max() for entry in log])
        value_gradients.append([np.absolute(np.array(entry["value_grad"])).max() for entry in log])
        max_action_length = max(max_action_length, len(action_gradients[-1]))
        max_value_length = max(max_value_length, len(value_gradients[-1]))

    # Adjust lengths
    adjusted_action_gradients = np.zeros((len(action_gradients), max_action_length))
    for idx, d in enumerate(action_gradients):
        adjusted_action_gradients[idx, :len(d)] = d
        if max_action_length - len(d) > 0:
            adjusted_action_gradients[idx, len(d):] = [d[-1]] * (max_action_length - len(d))

    adjusted_value_gradients = np.zeros((len(value_gradients), max_value_length))
    for idx, d in enumerate(value_gradients):
        adjusted_value_gradients[idx, :len(d)] = d
        if max_value_length - len(d) > 0:
            adjusted_value_gradients[idx, len(d):] = [d[-1]] * (max_value_length - len(d))
    return adjusted_action_gradients, adjusted_value_gradients


def save_matplotlib_visualizations(config, start_dir, result_path="./results"):
    """
    Given a config, it searchs in the result_path for training processes for which this config is a sub-config of the
    training process. It starts its search lexicographically, after the directory start_dir. Given these training
    processes, it plots the following:
    - mean + std of policy quality
    - mean + std of action and value loss (different y-axis)
    - mean + std of value loss
    - mean + std of action loss
    - mean + std of maximum of the absolute gradient. Action and value in one plot with two y-axes.
    - mean + std of maximum of the absolute gradient of the action parameters
    - mean + std of maximum of the absolute gradient of the value parameters
    These plots are all saved into the directory './visualizations/{current_datetime}/
    :param config: config used to search training processes
    :param start_dir: directory to start the search from (lexicographically)
    :param result_path: str | Path where this method will search for matching training processes
    """
    plt.rcParams.update({'font.size': 15, 'lines.linewidth': 2})
    output_path = Path("./visualizations/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    print("save policy qualities")
    save_policy_qualities(config, start_dir, result_path=result_path, output_path=output_path, copy_config=False)
    training_result_paths = retrieve_result_paths(config, start_dir, result_path)
    save_two_log_param_as_plot(training_result_paths, "action_loss", "value_loss", output_path, "action_value_loss", "Trainings step", "Action loss", "Value loss")
    print("save value_loss")
    save_log_param_as_plot(training_result_paths, "value_loss", output_path, "value_loss", "Trainings step", "Loss")
    print("save action_loss")
    save_log_param_as_plot(training_result_paths, "action_loss", output_path, "action_loss", "Trainings step", "Loss")
    action_gradients, value_gradients = retrieve_max_action_and_value_grads(training_result_paths)
    save_two_data_as_plot(action_gradients, value_gradients, output_path, "abs_max_action_value_grads", "", "", "Training steps", "Maximum absolute Action Gradient", "Maximum absolute Value Gradient")
    print("save abs_max_action_grads")
    save_mean_std_plot(action_gradients, output_path, "abs_max_action_grads", "Trainings step", "Maximum absolute gradient")
    print("save abs_max_value_grads")
    save_mean_std_plot(value_gradients, output_path, "abs_max_value_grads", "Trainings step", "Maximum absolute gradient")

    save_config(output_path / "config.json", config)


def create_visualizations(result_path: Path):
    """
    Given a path to the results of a trainings process, this function creates visualizations of the losses,
    the maximum gradients, the gradients of the value QNN, the maximum parameter change,
    the policy quality and the animated trainings process. It saves these visualizations in the specified path of the
    results
    :param result_path: Path containing the results of a trainings process
    """
    config = _load_config(result_path / "config.json")
    gamma = config["gamma"]
    end_state_values = config["end_state_values"]

    log = load_log(result_path / "log.txt")

    print("create loss fig")
    losses = [[entry["action_loss"], entry["value_loss"]] for entry in log]
    loss_fig = plot_loss(losses)
    with (result_path / "fig_loss.html").open("w", encoding="utf-8") as f:
        f.write(loss_fig.to_html())
        f.close()

    print("create max grad fig")
    grads = [[entry["action_grad"], entry["value_grad"]] for entry in log]
    max_grad_fig = plot_max_grad(grads)
    with (result_path / "fig_max_grad.html").open("w", encoding="utf-8") as f:
        f.write(max_grad_fig.to_html())
        f.close()

    print("create value grad fig")
    with (result_path / "fig_value_grad.html").open("w", encoding="utf-8") as f:
        f.write(plot_value_grads([entry["value_grad"] for entry in log]).to_html())
        f.close()

    print("create max params change fig")
    with (result_path / "fig_max_params_change.html").open("w", encoding="utf-8") as f:
        f.write(plot_max_params_change([[entry["action_params_change"], entry["value_params_change"]] for entry in log]).to_html())
        f.close()

    print("Calculate and save quality of the policy")
    policy_quality = create_policy_quality_file(result_path, gamma, log)

    print("create policy quality fig")
    with (result_path / "fig_policy_quality.html").open("w", encoding="utf-8") as f:
        f.write(plot_policy_quality(policy_quality).to_html())
        f.close()


    # print("create training fig")
    # frames = [get_frozen_lake_frame(entry["action_probs"], entry["state_values"], gamma, end_state_values) for entry in
    #           log]
    # training_fig = plot_animated_frozen_lake(frames, gamma, end_state_values)
    # with (result_path / "fig_training.html").open("w", encoding="utf-8") as f:
    #     f.write(training_fig.to_html())
    #     f.close()


def main():
    """
    Depending on the parameter ``latest`` declared within this method:
    - ``latest = True``: executes create_visualizations with the latest directory in ./results
    - ``latest = False``: executes create_visualizations with the a directory specified in this method
    """
    latest = True
    if latest:
        import os.path
        result_path = Path("./results/")
        dirs = [d for d in os.listdir(result_path) if os.path.isdir(result_path / d)]
        dirs.sort()
        result_path = result_path / dirs[-1]
    else:
        result_path = Path("./interesting_results/sampling/1000000/2023.11.02_08.49.30/")
    create_visualizations(result_path)


if __name__ == "__main__":
    # main()
    # result_path = Path("./results/2023.10.29_23.23.19/")
    # create_visualizations(result_path)
    config = dict(
        num_iterations=12,
        sub_iterations=[[0.0001, 25, 2], [0.0001, 25, 1]],
        end_state_values=True,
        value_optimizer="Adam",
        action_optimizer="Adam",
        value_lr=0.5,
        action_lr=0.5,
        gamma=0.8,
        eps=0.0,
        shots=10000000,
        qpe_qubits=0,
        max_qpe_prob=0.8,
    )
    # save_action_and_value_losses(config, "", "./interesting_results/qpe/16_0,999")
    save_matplotlib_visualizations(config, "", result_path="./interesting_results/qpe/50_0,9999/")
    # paths = retrieve_result_paths(config, "2023.10.27_21.03.05")
    #
    # import os
    # import shutil
    #
    # destination_path = Path("./interesting_results/sampling/100000/")
    # for p in paths:
    #     print(f"Copying content of {p} to {destination_path/p.name}")
    #     for file_name in os.listdir(p):
    #         # construct full file path
    #         extra_dir = p.name
    #         source = p / file_name
    #         dest_path = destination_path / extra_dir
    #         destination = dest_path / file_name
    #         # copy only files
    #         dest_path.mkdir(parents=True, exist_ok=True)
    #         if os.path.isfile(source):
    #             shutil.copy(source, destination)
    #
    # print(f"copied {len(paths)} paths")
