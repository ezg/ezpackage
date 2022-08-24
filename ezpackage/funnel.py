import plotly.graph_objects as go


def funnel(data,
           targets,
           event_col,
           index_col,
           groups=None,
           group_names=None):

    # if group not specified select all users
    if groups is None:
        groups = [data[index_col].unique()]
        group_names = ['all users']
    elif group_names is None:
        group_names = [f"group {i}" for i in range(len(groups))]

    # pre-process targets

    # format targets to list of lists:
    for n, i in enumerate(targets):
        if type(i) != list:
            targets[n] = [i]

    # generate target_names:
    target_names = []
    for t in targets:
        # get name
        target_names.append(' | '.join(t).strip(' | '))

    res_dict = {}
    for group, group_name in zip(groups, group_names):

        # isolate users from group
        group_data = data[data[index_col].isin(group)]

        vals = []
        for target in targets:
            # define how many users have particular target:
            vals.append(group_data[group_data[event_col].isin(
                target)][index_col].nunique())
        res_dict[group_name] = {'targets': target_names, 'values': vals}

    return res_dict


def plot(groups):
    data = []

    for t in groups.keys():
        trace = go.Funnel(
            name=t,
            y=groups[t]['targets'],
            x=groups[t]['values'],
            textinfo="value+percent initial+percent previous"
        )
        data.append(trace)

    layout = go.Layout(margin={"l": 180, "r": 0, "t": 30, "b": 0, "pad": 0},
                       funnelmode="stack",
                       showlegend=True,
                       hovermode='closest',
                       legend=dict(orientation="v",
                                   bgcolor='#E2E2E2',
                                   xanchor='left',
                                   font=dict(
                                       size=12)
                                   )
                       )
    fig = go.Figure(data, layout)
    fig.show()
