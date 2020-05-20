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
df_dist = pd.read_csv('./data/recover_md_map.csv',dtype={"geoid10": str})
df_dist['raw_distance'] = df_dist['raw_distance']/1000
df_dist['raw_distance'] = df_dist['raw_distance'].replace(np.inf, 999)
df_dist['change_distance'] = df_dist['change_distance']/1000
df_dist['change_distance'] = df_dist['change_distance'].replace(np.inf, 999)

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
                                        # href="./soon",
                                    ),
                                    html.A(
                                        html.Button("Transformation", id="learn-more-button"),
                                        href="./soon",
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
                                    html.H6("Select a distance range to identify those areas"),
                                    dcc.Graph(id="map",
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
                                    html.H6("Resilience curve"),
                                    dcc.Graph(
                                        id="map",
                                        figure=generate_map_change(df_dist),
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
                                    dcc.Markdown(
                                        ['''
                                    In a similar fashion, if we focus on the average distance when we optimize our recovery,
                                    we could inadverently prioritize restoring regions that are typically well-off.
                                    For example, when considering the mean/average, the average distance improves the same amount if
                                    we improve one person's proximity from 1km to 0.5km as it would by increasing someone from
                                    3km to 2.5km.
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
                                    '''],
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
                                ["Example: Ranking with the EDE"], className="subtitle padded"
                            ),
                            html.Div(
                                    id="ecdf-container",
                                    children=[
                                        html.H6("Ranking cities based on access to grocery stores"),
                                        dcc.Graph(id="food_ranking",
                                        config={"scrollZoom": True, "displayModeBar": True,
                                                "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian',
                                                'pan',"zoomIn2d", "zoomOut2d","lasso2d","select2d"],
                                                },
                                            ),
                                        html.H6("Select the demographic groups"),
                                        dcc.Checklist(
                                            id="race-select-2",
                                            options=[
                                                {'label': 'All', 'value': 'H7X001'},
                                                {'label': 'White', 'value': 'H7X002'},
                                                {'label': 'Black', 'value': 'H7X003'},
                                                {'label': 'Am. Indian', 'value': 'H7X004'},
                                                {'label': 'Asian', 'value': 'H7X005'},
                                                {'label': 'Latino/Hispanic', 'value': 'H7Y003'},
                                            ],
                                            value=['H7X001','H7X002','H7X003'],
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                        html.H6("Order by"),
                                        dcc.RadioItems(
                                            id="race-order",
                                            options=[
                                                {'label': 'All', 'value': 'H7X001'},
                                                {'label': 'White', 'value': 'H7X002'},
                                                {'label': 'Black', 'value': 'H7X003'},
                                                {'label': 'Am. Indian', 'value': 'H7X004'},
                                                {'label': 'Asian', 'value': 'H7X005'},
                                                {'label': 'Latino/Hispanic', 'value': 'H7Y003'},
                                            ],
                                            value='H7X001',
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
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
                            html.Div(
                                [
                                    html.H6(
                                        ["Further information"],
                                        className="subtitle padded",
                                    ),
                                    html.P(
                                        ["We are still working on this paper, but\
                                        feel free to contact us if you have questions."
                                        ]),
                                    html.P(
                                        ["Logan, T. M., Anderson, M. J., Williams, T., & Conrow, L. (In Progress). Measuring inequality in the built environment: Evaluating grocery store accessfor planning policy and intervention."],
                                        style={'padding-left': '22px', 'text-indent': '-22px', 'font-weight':400}
                                        ),
                                    html.P(
                                        ["Information on the inequality measure can also be found here:"]
                                        ),
                                    html.P(
                                        ["Sheriff, G., & Maguire, K. (2013). Ranking Distributions of Environmental Outcomes Across Population Groups. Working paper, EPA."
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



def generate_map(dff_dist):
    """
    Generate map showing the distance to services and the locations of them
    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """
    # print(dff_dist['geoid10'].tolist())
    dff_dist = dff_dist.reset_index()


    layout = go.Layout(
        clickmode="none",
        dragmode="zoom",
        showlegend=False,
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
    )


    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = 'https://raw.githubusercontent.com/urutau-nz/dash-evaluating-proximity/master/data/block.geojson',
        locations = dff_dist['geoid'].tolist(),
        z = dff_dist['raw_distance'].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        hovertemplate="Distance: %{z:.2f}km<br>" +
                        "<extra></extra>",
    ))

    return {"data": data, "layout": layout}

def generate_map_change(dff_dist):
    """
    Generate map showing the distance to services and the locations of them
    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """
    # print(dff_dist['geoid10'].tolist())
    dff_dist = dff_dist.reset_index()


    layout = go.Layout(
        clickmode="none",
        dragmode="zoom",
        showlegend=False,
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
    )


    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = 'https://raw.githubusercontent.com/urutau-nz/dash-evaluating-proximity/master/data/block.geojson',
        locations = dff_dist['geoid'].tolist(),
        z = dff_dist['change_distance'].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        hovertemplate="$\delta$ Distance: %{z:.2f}km<br>" +
                        "<extra></extra>",
    ))

    return {"data": data, "layout": layout}
