import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from load_config import _load_config, save_config
from logger import load_log
from solve_matrix_form import calculate_policy_quality
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


def retrieve_result_paths(config, start_dir):
    """
    Returns a list of paths. Each path has to fulfill the property that its last directory comes after the
    ``start_dir``, when sorted lexicographically. Additionally, the ``config`` has to be a sub-directory of the config within
    the path.
    :param config: dict
    :param start_dir: str
    :return: list of paths
    """
    import os.path
    result_path = Path("./results/")
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


def save_matplotlib_policy_quality(policy_qualities, save_path):
    """
    Given the policy qualities of multiple trainings, this function computes the mean and the std at each
    training step. Next it plots the mean as a line and the std as a shaded area via matplotlib and saves the resulting
    plot in the specified path as a pdf file called "policy_quality.pdf".
    :param policy_qualities: matrix of size #trainings x #training steps containing the policy qualities
    :param save_path: the path to save the file to
    """
    data = policy_qualities
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    fig, ax = plt.subplots()
    ax.fill_between(x, mean + std, mean - std, alpha=0.2)
    ax.plot(x, mean)
    ax.margins(x=0)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    plt.tight_layout()
    plt.savefig(save_dir / "policy_quality.pdf", dpi="figure", format="pdf")
    plt.clf()


def save_policy_qualities(config, start_dir):
    """
    Gets all policy qualities of training processes with the same sub-config, created after start_dir.
    Next it saves a plot of these policy qualities to the path ./policy_qualities/{current date and time}.
    :param config: dict. Is the sub-config that every other config needs to contain
    :param start_dir: str. Directories of training process's must come after start_dir, lexicographically.
    """
    directories = retrieve_result_paths(config, start_dir)
    policy_qualities = retrieve_policy_qualities(directories, True)
    output_dir = Path("./policy_qualities/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    save_matplotlib_policy_quality(policy_qualities, output_dir)
    save_config(output_dir / "config.json", config)


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


    print("create training fig")
    frames = [get_frozen_lake_frame(entry["action_probs"], entry["state_values"], gamma, end_state_values) for entry in
              log]
    training_fig = plot_animated_frozen_lake(frames, gamma, end_state_values)
    with (result_path / "fig_training.html").open("w", encoding="utf-8") as f:
        f.write(training_fig.to_html())
        f.close()


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
        result_path = Path("./results/2023.10.05_16.42.21/")
    create_visualizations(result_path)


if __name__ == "__main__":
    # main()
    # result_path = Path("./results/2023.10.29_23.23.19/")
    # create_visualizations(result_path)
    # config = dict(
    #     num_iterations=12,
    #     sub_iterations=[[0.0001, 25, 2], [0.0001, 25, 1]],
    #     end_state_values=True,
    #     value_optimizer="Adam",
    #     action_optimizer="Adam",
    #     value_lr=0.5,
    #     action_lr=0.5,
    #     gamma=0.8,
    #     eps=0.0,
    #     shots=None,
    #     qpe_qubits=0,
    #     max_qpe_prob=0.8,
    # )
    # save_policy_qualities(config, "2023.10.29_23.05.16")
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
        shots=None,
        qpe_qubits=0,
    )
    paths = retrieve_result_paths(config, "2023.10.27_16.37.01")

    import os
    import shutil

    destination_path = Path("./interesting_results/perfect/")
    for p in paths:
        print(f"Copying content of {p}")
        for file_name in os.listdir(p):
            # construct full file path
            extra_dir = p.name
            source = p / file_name
            dest_path = destination_path / extra_dir
            destination = dest_path / file_name
            # copy only files
            dest_path.mkdir(parents=True, exist_ok=True)
            if os.path.isfile(source):
                shutil.copy(source, destination)
