import plotly.graph_objects as go


def get_fig(x, y, z):
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
    return go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False))
