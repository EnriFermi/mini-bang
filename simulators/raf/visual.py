
import plotly.graph_objects as go

def plot_traces(times, traces, to_plot=None, ylim=None, title=''):
    if to_plot is None:
        to_plot = list(traces.keys())

    fig = go.Figure()
    for name in to_plot:
        fig.add_trace(go.Scatter(
            x=times, y=traces[name],
            mode='lines',
            name=name,
            line=dict(width=1),    # thin
            opacity=0.6
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Count',
        legend=dict(itemclick='toggle', itemdoubleclick='toggleothers')  # default behavior
    )
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    fig.show()