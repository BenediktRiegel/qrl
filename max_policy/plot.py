import plotly.graph_objects as go
import seaborn as sns
from numpy import flip


def get_fig(x, y, z):
    """
    Given the x ticks, the y ticks and a 2d list containing values, this function returns a Surface figure of plotly,
    with the x ticks, y ticks and it uses the 2d list as the height as the coordinate (x, y)
    :param x: list
    :param y: list
    :param z: 2d list
    """
    fig = go.Figure(data=[go.Surface(
        x=x,
        y=y,
        z=z,
    )])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(autosize=True,
                      margin=dict(l=65, r=50, b=65, t=90),
    )

    return fig


def get_heatmap(x, y, z):
    """
    Plots Plotly heatmap given x, y and z.
    :param x: list
    :param y: list
    :param z: 2d list
    """
    return go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False))


def get_matplotlib_heatmap(x, y, z, xlabel, ylabel, title=""):
    """
    Plots a heatmap given x, y and z with seaborn. It labels the x-axis with xlabel and the y-label with ylabel.
    Optionally, it also includes a title.
    :param x: list
    :param y: list
    :param z: 2d list
    :param xlabel: str
    :param ylabel: str
    :param title: optional str
    :return: plot
    """
    z = flip(z.T, axis=0)
    y = flip(y, axis=0)
    plot = sns.heatmap(z, xticklabels=x, yticklabels=y, linewidths=0.0, rasterized=True, vmin=0, vmax=1)
    plot.set(xlabel=xlabel, ylabel=ylabel)
    plot.set(title=title)

    phaseticklabels = ["0", "0.125", "0.25", "0.375", "0.5"]
    ticks = [0, (len(y)+1)//4, (len(y)+1)//2, 3*(len(y)+1)//4, len(y)]
    # yticklabels = ["-1", "-0.75", "-0.5", "-0.25", "0", "0.25", "0.5", "0.75", "1"]
    # yticks = [len(y), 7*(len(y)+1)//8, 3*(len(y)+1)//4, 5*(len(y)+1)//8, (len(y)+1)//2, 3*(len(y)+1)//8, (len(y)+1)//4, (len(y)+1)//8, 0]
    plot.set_xticks(ticks)
    plot.set_xticklabels(phaseticklabels, rotation=0)
    ticks = list(reversed(ticks))
    plot.set_yticks(ticks)
    plot.set_yticklabels(phaseticklabels, rotation=0)
    return plot
