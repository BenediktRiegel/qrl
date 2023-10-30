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
    global v_max
    if v_max is None:
        v_max = (1. / (1-gamma)) if end_state_values else 1.
    return v_max


def get_reward(x, y, gamma, end_state_values):
    v_max = get_v_max(gamma, end_state_values)
    s = y*4 + x
    if s in goal_states:
        return v_max
    elif s in terminal_states:
        return -v_max
    else:
        return 0

def get_arrow(x_start, y_start, x_end, y_end, color=(255, 51, 0)):
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
    Z = [[get_reward(x, y, gamma) for x in range(4)] for y in range(4)]
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
    policy_arrows = []
    for y in range(4):
        for x in range(4):
            policy_arrows += get_policy_arrows(x, y, action_probs[y*4 + x])

    return go.Frame(layout=go.Layout(annotations=policy_arrows))


def get_frozen_lake_value_frame(state_values, gamma, end_state_values):
    # v_max = environment.r_m / (1 - gamma)
    # v_max = 1 / (1 - gamma) if end_state_values else 1

    heatmap = np.empty((8, 8))
    for y in range(4):
        for x in range(4):
            value = state_values[4*y + x]
            heatmap[2 * y, 2 * x] = value
            heatmap[2 * y, 2 * x + 1] = value
            heatmap[2 * y + 1, 2 * x] = value
            heatmap[2 * y + 1, 2 * x + 1] = get_reward(x, y, gamma, end_state_values)

    value_fig = go.Heatmap(z=heatmap)

    return go.Frame(data=value_fig)


def get_frozen_lake_frame(action_probs, state_values, gamma, end_state_values):
    action_frame = get_frozen_lake_action_frame(action_probs)
    value_frame = get_frozen_lake_value_frame(state_values, gamma, end_state_values)
    value_frame.update(action_frame)
    return value_frame


def plot_animated_frozen_lake(frames, gamma, end_state_values):
    for idx, frame in enumerate(frames):
        frame["name"] = idx
    heatmap = np.empty((8, 8))
    for y in range(4):
        for x in range(4):
            r = get_reward(x, y, gamma, end_state_values)
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
    max_grads = np.array([[np.abs(np.array(entry[0])).max(), np.abs(np.array(entry[1])).max()] for entry in grads])
    df = pd.DataFrame(max_grads, columns=["action", "value"])
    fig = px.line(df, title="max grad")
    return fig


def plot_value_grads(value_grads):
    max_grads = np.array(value_grads)
    df = pd.DataFrame(max_grads, columns=[f"param{i}" for i in range(max_grads.shape[1])])
    fig = px.line(df, title="Value Grads")
    return fig


def plot_max_params_change(max_params_change):
    max_params_change = np.array(max_params_change)
    df = pd.DataFrame(max_params_change, columns=["action", "value"])
    fig = px.line(df, title="Max Parameter Change")
    return fig


def plot_policy_quality(policy_quality):
    policy_quality = np.array(policy_quality)
    df = pd.DataFrame(policy_quality)
    fig = px.line(df, title="Quality of the Policy over Episodes")
    return fig


def dict_subset_of_dict(sub_dict: dict, big_dict: dict):
    for k, v in sub_dict.items():
        if k not in big_dict:
            return False
        if v != big_dict[k] and v is not None and big_dict is not None:
            return False
    return True


def retrieve_result_dirs(config, start_dir):
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


def retrieve_policy_qualities(directories, recalculate=False):
    policy_qualities = []
    max_length = 0
    # Go through directories and load policy qualities
    for d in directories:
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


def save_matplotlib_policy_quality(policy_qualities, save_dir):
    data = policy_qualities
    x = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    fig, ax = plt.subplots()
    ax.fill_between(x, mean + std, mean - std, alpha=0.2)
    ax.plot(x, mean)
    ax.margins(x=0)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)
    plt.tight_layout()
    plt.savefig(save_dir / "policy_quality.pdf", dpi="figure", format="pdf")
    plt.clf()


def save_policy_qualities(config, start_dir):
    directories = retrieve_result_dirs(config, start_dir)
    policy_qualities = retrieve_policy_qualities(directories, True)
    output_dir = Path("./policy_qualities/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    save_matplotlib_policy_quality(policy_qualities, output_dir)
    save_config(output_dir / "config.json", config)


def create_visualizations(result_path: Path):
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
    result_path = Path("./results/2023.10.29_23.23.19/")
    create_visualizations(result_path)
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
        max_qpe_prob=0.8,
    )
    # save_policy_qualities(config, "2023.10.29_23.05.16")
