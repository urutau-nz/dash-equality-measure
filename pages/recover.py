import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

from utils import Header, make_dash_table
import pandas as pd
import numpy as np

# mapbox token
mapbox_access_token = open(".mapbox_token").read()

amenities = ['supermarket','gas_station']
amenity_names = {'supermarket':'Supermarket','gas_station':'Service Station'}

# Load data
df_dist = pd.read_csv('./data/recover_md_map_8.csv',dtype={"geoid10": str})
df_dist['raw_distance'] = df_dist['raw_distance']/1000
df_dist['raw_distance'] = df_dist['raw_distance'].replace(np.inf, 999)
df_dist['change_distance'] = df_dist['change_distance']/1000
df_dist['change_distance'] = df_dist['change_distance'].replace(np.inf, 999)

# load destinations
df_dest = pd.read_csv('./data/dest_outages_md_8.csv')

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 3
            html.Div(
                [
                    # Buttons
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.A(
                                        html.Button("Overview", id="learn-more-button"),
                                        href="./overview",
                                    ),
                                    html.A(
                                        html.Button("Resilience & Access", id="learn-more-button"),
                                        href="./resilience",
                                    ),
                                    html.A(
                                        html.Button("Measuring Equity", id="learn-more-button"),
                                        href="./equity",
                                    ),
                                    html.A(
                                        html.Button("Recover", id="learn-more-button", className="current-button"),
                                        # href="./recover",
                                    ),
                                    html.A(
                                        html.Button("Transformation", id="learn-more-button"),
                                        href="./transform",
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row buttons",
                        style={"margin-bottom": "35px"},
                    ),
                    # Title
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Recovery and response"),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(
                                        ['''
                                    One of the dimensions of resilience relates to recovery.
                                    It has been refered to as the capacity to 'quickly restore desired functionality...'

                                    The challenge is two-fold, first recovery to the previous state is often not "desired functionality" (it is either inequitable or exposed),
                                    but second, the recovery can often exacerbate the inequities.

                                    We argue that this is not an inherent flaw with resilience, but with how we've been using it.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Response"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    A first, brief note: We can use the EDE and assessing access to determine the locations of pop-up aid and supply drops, and shelters.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Recovery"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    We simulate disruptive events in a city and model the recovery.
                                    We present three approaches to the recovery: 1) randomly opening the grocery stores; 2) optimize reopening
                                    in a manner than increases the average distance; 3) optimize the reopening to improve the EDE.
                                    '''],
                                    ),
                                    html.H6(
                                        ["What we measure matters!"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Manner resilience quantifications assess the change in functionality
                                    and endeavour to return to "normal" (0).
                                    However, consider the following map. On the lefthandside we present the
                                    distance to nearest grocery store in a city immediately following a disruption
                                    and on the righthandside we show the *change in* distance to nearest grocery store.

                                    What you see is that in many cases, the areas with the largest change in - who would be
                                    and often are prioritized in recovery efforts - in fact still have better access at their
                                    worst point, than other areas have on a normal day.
                                    This is even worse if we were to use percentage change, which is sometimes used and appropriate in cases
                                    of economic loss.
                                    '''],
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Maps
                    html.Div(
                        [
                            html.Div(
                                id="map-container",
                                children=[
                                    html.H6("Disruption: Actual distance"),
                                    dcc.Graph(
                                        id="map",
                                        figure = generate_map(df_dist),
                                        config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian',
                                            'pan',"zoomIn2d", "zoomOut2d","lasso2d","select2d"],
                                    },
                                ),
                                ],
                                className=" six columns",
                            ),
                            html.Div(
                                [
                                    html.H6("Disruption: Change in distance"),
                                    dcc.Graph(
                                        id="map",
                                        figure=generate_map(df_dist, change=True),
                                        config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian'],
                                    },
                                ),
                                ],
                                className=" six columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Guiding recovery"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    In a similar fashion, if we focus on the average distance when we optimize our recovery,
                                    we could inadverently prioritize restoring regions that are typically well-off.
                                    For example, when considering the mean/average, the average distance improves the same amount if
                                    we improve one person's proximity from 1km to 0.5km as it would by increasing someone from
                                    3km to 2.5km.

                                    An EDE (as it is based on a welfare function - see Atkinson (1970) for details) prioritizes
                                    interventions that support the worse-off.

                                    Here we demonstrate the recovery following several simulated disruptions.
                                    '''],
                                    className="my_list"
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.H6(
                                ["Recovery after a disruption"], className="subtitle padded"
                            ),
                            html.Div(
                                    id="ecdf-container",
                                    children=[
                                        html.H6("Ranking cities based on access to grocery stores"),
                                        dcc.Graph(id="recovery-md",
                                        config={"scrollZoom": True, "displayModeBar": True,
                                                "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian',
                                                'pan',"zoomIn2d", "zoomOut2d","lasso2d","select2d"],
                                                },
                                            ),
                                        html.H6("Select the simulation"),
                                        dcc.Checklist(
                                            id="simulation-select",
                                            options=[
                                                {'label': '1', 'value': '1'},
                                                {'label': '2', 'value': '2'},
                                                {'label': '3', 'value': '3'},
                                                {'label': '4', 'value': '4'},
                                                {'label': '5', 'value': '5'},
                                            ],
                                            value=['1','2','3'],
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                        html.H6("Select the metric"),
                                        dcc.Checklist(
                                            id="metric-select",
                                            options=[
                                                {'label': 'Average', 'value': 'mean'},
                                                {'label': 'Equally-distributed equivalent', 'value': 'ede'},
                                            ],
                                            value=['ede'],
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                        html.H6("Select the group"),
                                        dcc.Checklist(
                                            id="group-select",
                                            options=[
                                                {'label': 'Everyone', 'value': 'all'},
                                                {'label': 'The 1st quintile (the 20% of residents who normally have the best access)', 'value': 'quin1'},
                                                {'label': 'The 2nd quintile', 'value': 'quin2'},
                                                {'label': 'The 3rd quintile', 'value': 'quin3'},
                                                {'label': 'The 4th quintile', 'value': 'quin4'},
                                                {'label': 'The 5th quintile (worst-access)', 'value': 'quin5'},
                                            ],
                                            value=['quin1','quin5'],
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                    ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Thoughts"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    A couple of things to note:
                                    * guided recovery (the optimized ones) is significantly better than random recovery
                                    * people with good access (in these simulations) seldom have worse access than the people with the worse access usually have
                                    * when we optimize for the average, we see faster recovery for the quintiles that are better off than if we optimize for the EDE

                                    Fundamentally, however, we still confront the issue that recovery (optimized or not)
                                    is returning to the previous state and that is inequitable in many cities.
                                    The benefit of presenting resilience that shows the true access (rather than the change in)
                                    is that we are in an undesirable state and this is highlighted more by the use of
                                    the EDE. The EDE means that decision-makers cannot be complacent by thinking the
                                    state of the average person is OK.

                                    Nevertheless, if we are true to the wider themes of resilience, which
                                    often includes transformation, we need to consider how we can transform
                                    these communities.
                                    '''],
                                    className="my_list"
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 5
                    html.Div(
                        [
                            # html.Img(
                            #     src=app.get_asset_url("dash-financial-logo.png"),
                            #     className="logo",
                            # ),
                            html.A(
                                html.Button("Transformation", id="learn-more-button"),
                                href="./transform",
                            ),
                        ],
                        className="row",
                        style={"text-align": "right"},
                    ),
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Limitations and opportunities to implement"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    The demographic data is based on census data. However, during an emergency people
                                    will evacuate or otherwise relocate. However, in implementation, this approach can rely
                                    data based on the knowledge of emergency managers or other information. Additionally, it
                                    can be supplemented with cellphone data.

                                    Additionally, we rely on Twitter data and other internet feeds for information on
                                    service closures, however it would be feasible to create a online means for essential services
                                    to update their status and infrastructure dependencies (e.g., a store may be undamaged but have
                                    no power) and this can guide recovery efforts.
                                    '''],
                                    className="my_list"
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Further information"],
                                        className="subtitle padded",
                                    ),
                                    html.P(
                                        ["We are still working on this project, but\
                                        feel free to contact us if you have questions."
                                        ]),
                                    html.P(
                                        ["More information on equally-distributed equivalents:"]
                                        ),
                                    html.P(
                                        ["Atkinson, A. B. (1970). On the measurement of inequality. Journal of Economic Theory, 2(3), 244–263"
                                        ],
                                        style={'padding-left': '22px', 'text-indent': '-22px', 'font-weight':400}
                                    ),
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )


# Assign color to legend
colors = ['#EA5138','#E4AE36','#1F386B','#507332']
colormap = {}
for ind, amenity in enumerate(amenities):
    colormap[amenity] = colors[ind]

pl_deep=[[0.0, 'rgb(253, 253, 204)'],
         [0.1, 'rgb(201, 235, 177)'],
         [0.2, 'rgb(145, 216, 163)'],
         [0.3, 'rgb(102, 194, 163)'],
         [0.4, 'rgb(81, 168, 162)'],
         [0.5, 'rgb(72, 141, 157)'],
         [0.6, 'rgb(64, 117, 152)'],
         [0.7, 'rgb(61, 90, 146)'],
         [0.8, 'rgb(65, 64, 123)'],
         [0.9, 'rgb(55, 44, 80)'],
         [1.0, 'rgb(39, 26, 44)']]



def generate_map(dff_dist, change=False):
    """
    Generate map showing the distance to services and the locations of them
    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """
    # print(dff_dist['geoid10'].tolist())
    dff_dist = dff_dist.reset_index()

    if change:
        dist_var = 'change_distance'
        hover_txt = 'Change in distance'
    else:
        dist_var = 'raw_distance'
        hover_txt = 'Distance'

    layout = go.Layout(
        clickmode="none",
        dragmode="zoom",
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat = 39.292126, lon = -76.613632),
            pitch=0,
            zoom=10.5,
            style="basic", #"dark", #
        ),
        legend=dict(
            # bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="black"),
            x=0,
            y=0,
            yanchor="top",
        ),
    )


    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = 'https://raw.githubusercontent.com/urutau-nz/dash-evaluating-proximity/master/data/block.geojson',
        locations = dff_dist['geoid'].tolist(),
        z = dff_dist[dist_var].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        text=np.repeat(hover_txt, len(dff_dist)),
        hovertemplate="%{text}: %{z:.2f}km<br>" +
                        "<extra></extra>",
    ))


    # scatterplot of the amenity locations
    dest_open = df_dest[df_dest['operational']==True]
    dest_closed = df_dest[df_dest['operational']==False]

    if len(dest_open) > 0:
        data.append(go.Scattermapbox(
            lat=dest_open["st_y"],
            lon=dest_open["st_x"],
            mode="markers",
            marker={"color": [colormap[amenity]]*len(dest_open), "size": 9},
            name=amenity_names[amenity],
            hoverinfo="skip", hovertemplate="",
        ))

    if len(dest_closed) > 0:
        data.append(go.Scattermapbox(
            lat=dest_closed["st_y"],
            lon=dest_closed["st_x"],
            mode="markers",
            marker={"color": ['black']*len(dest_closed), "size": 9},
            name='Closed {}'.format(amenity_names[amenity]),
            hoverinfo="skip", hovertemplate="",
        ))

    return {"data": data, "layout": layout}

def plot_recovery(dff_recovery):
    """
    :return: Figure object
    """
    ylimit = dff_recovery.value.max()/1000
    layout = dict(
        xaxis=dict(
            title="time".upper(),
            zeroline=False,
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        yaxis=dict(
            title="Distance (km)".format(amenity_names[amenity]).upper(),
            zeroline=False,
            range=(ylimit,0),
            # autorange='reversed',
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        font=dict(size=13),
        showlegend=True,
        legend_title_text='Optimized for',
        margin=dict(l=40, r=0, t=10, b=30),
        # transition = {'duration': 500},
    )

    data = []
    metric_names = {1:'random',2:'average',6:'EDE'}
    # Assign color to legend
    colormap = {1:'#EA5138',2:'#E4AE36',6:'#1F386B'}
    group_names = {'mean_all': 'Average for everyone', 'ede_all':'EDE for everyone',
                    'mean_quin1': 'Average for 1st Q', 'ede_quin1':'EDE for 1st Q',
                    'mean_quin2': 'Average for 2nd Q', 'ede_quin2':'EDE for 2nd Q',
                    'mean_quin3': 'Average for 3rd Q', 'ede_quin3':'EDE for 3rd Q',
                    'mean_quin4': 'Average for 4th Q', 'ede_quin4':'EDE for 4th Q',
                    'mean_quin5': 'Average for 5th Q', 'ede_quin5':'EDE for 5th Q',}

    for recovery_metric in [1,2,6]:
        j = 0
        for group in dff_recovery.access_metric.unique():
            df_plot = dff_recovery.loc[(dff_recovery.recovery_metric==recovery_metric)&(dff_recovery.access_metric==group),].pivot_table(index='time_step',columns='sim_id',values='value')
            # add the sim
            for i in list(df_plot):
                new_trace = go.Scatter(
                        x=df_plot.index, y=df_plot[i]/1000,
                        opacity=0.8,
                        name=metric_names[recovery_metric],
                        line=dict(color=colormap[recovery_metric],),
                        text=[group_names[group]]*len(df_plot),
                        hovertemplate = "%{text}: %{y:.1f}km<br>" + "<extra></extra>",
                        hoverlabel = dict(font_size=20),
                        showlegend= j == 0,
                        )
                data.append(new_trace)
                j += 1


    return {"data": data, "layout": layout}
