import plotly.graph_objects as go
import seaborn as sns
from numpy import flip


def get_fig(x, y, z):
    """
    Given x, y and z, this function returns a 3d plot with contours, made via plotly
    :param x: list of values on the x-axis
    :param y: list of values on the y-axis
    :param z: 2d list for heatmap colour
    :return: 3d plot made via plotly
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
    Given x, y and z, this function returns a heatmap via plotly
    :param x: list of values on the x-axis
    :param y: list of values on the y-axis
    :param z: 2d list for heatmap colour
    :return: heatmap made via plotly
    """
    return go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False))


def get_seaborn_heatmap(x, y, z, xlabel, ylabel):
    """
    Given x, y and z, this function returns a heatmap via plotly. It also sets the labels for the x- and y-axis
    :param x: list of values on the x-axis
    :param y: list of values on the y-axis
    :param z: 2d list for heatmap colour
    :param xlabel: label of the x-axis
    :param ylabel: label of the y-axis
    :return: heatmap made via seaborn
    """
    z = flip(z, axis=0)
    plot = sns.heatmap(z, xticklabels=x, linewidths=0.0, rasterized=True)
    plot.set(xlabel=xlabel, ylabel=ylabel)

    yticklabels = ["-1", "-0.75", "-0.5", "-0.25", "0", "0.25", "0.5", "0.75", "1"]
    yticks = [len(y), 7*(len(y)+1)//8, 3*(len(y)+1)//4, 5*(len(y)+1)//8, (len(y)+1)//2, 3*(len(y)+1)//8, (len(y)+1)//4, (len(y)+1)//8, 0]
    plot.set_yticks(yticks)
    plot.set_yticklabels(yticklabels, rotation=0)
    return plot
