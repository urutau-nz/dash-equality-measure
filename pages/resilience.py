import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from utils import Header, make_dash_table, build_graph_title
import pandas as pd

import numpy as np
from urllib.request import urlopen
import json
import geopandas as gpd
import datetime

amenities = ['supermarket','gas_station']
amenity_names = {'supermarket':'Supermarket','gas_station':'Service Station'}

# mapbox token
mapbox_access_token = open(".mapbox_token").read()

# Load data
df_dist = pd.read_csv('./data/distance_to_nearest.csv',dtype={"geoid10": str})
df_dist['distance'] = df_dist['distance']/1000
df_dist['distance'] = df_dist['distance'].replace(np.inf, 999)

destinations = pd.read_csv('./data/destinations.csv')

df_recovery = pd.read_csv('./data/recovery_nc.csv')

# days since land landfall
days = np.unique(df_recovery['day'])

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


def generate_ecdf_plot(amenity_select, dff_dist, x_range=None):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    amenity = amenity_select
    if x_range is None:
        x_range = [dff_dist.distance.min(), dff_dist.distance.max()]


    layout = dict(
        xaxis=dict(
            title="distance (km)".format(amenity_names[amenity]).upper(),
            range=(0,15),
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        yaxis=dict(
            title="% of residents".upper(),
            range=(0,100),
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        font=dict(size=13),
        dragmode="select",
        paper_bgcolor = 'rgba(255,255,255,1)',
		plot_bgcolor = 'rgba(0,0,0,0)',
        bargap=0.05,
        showlegend=False,
        margin=dict(l=40, r=0, t=10, b=30),
        transition = {'duration': 500},
        # height= 300

    )
    data = []
    # add the cdf for that amenity
    counts, bin_edges = np.histogram(dff_dist.distance, bins=100, density = True)#, weights=df.W.values)
    dx = bin_edges[1] - bin_edges[0]
    new_trace = go.Scatter(
            x=bin_edges, y=np.cumsum(counts)*dx*100,
            opacity=1,
            line=dict(color=colormap[amenity],),
            text=[amenity_names[amenity].lower()]*len(dff_dist),
            hovertemplate = "%{y:.0f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )

    data.append(new_trace)

    # histogram
    multiplier = 300 if amenity=='supermarket' else 150
    counts, bin_edges = np.histogram(dff_dist.distance, bins=25, density=True)#, weights=df.W.values)
    opacity = []
    for i in bin_edges:
        if i >= x_range[0] and i <= x_range[1]:
            opacity.append(0.6)
        else:
            opacity.append(0.1)
    new_trace = go.Bar(
            x=bin_edges, y=counts*multiplier,
            marker_opacity=opacity,
            marker_color=colormap[amenity],
            hoverinfo="skip", hovertemplate="",)
    data.append(new_trace)


    # add the cdf for that amenity
    dff_dist = df_dist[(df_dist.day==days[0]) & (df_dist.service==amenity)]
    counts, bin_edges = np.histogram(dff_dist.distance, bins=100, density = True)#, weights=df.W.values)
    dx = bin_edges[1] - bin_edges[0]
    new_trace = go.Scatter(
            x=bin_edges, y=np.cumsum(counts)*dx*100,
            opacity=0.5,
            line=dict(color=colormap[amenity]),
            text=[amenity_names[amenity].lower()]*len(dff_dist),
            # hovertemplate = "%{y:.2f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
            # hoverlabel = dict(font_size=20),
            hoverinfo="skip", hovertemplate="",
            )

    data.append(new_trace)


    return {"data": data, "layout": layout}


def recovery_plot(amenity_select, dff_recovery, day):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    amenity = amenity_select
    if amenity == 'supermarket':
        ylimit = 15
    else:
        ylimit = 8

    layout = dict(
        xaxis=dict(
            title="days since landfall".upper(),
            zeroline=False,
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        yaxis=dict(
            title="Distance (km)".format(amenity_names[amenity]).upper(),
            zeroline=False,
            range=(ylimit,0),
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        font=dict(size=13),
        paper_bgcolor = 'rgba(255,255,255,1)',
		plot_bgcolor = 'rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=40, r=0, t=10, b=30),
        transition = {'duration': 500},
    )

    data = []
    # add the average
    new_trace = go.Scatter(
            x=dff_recovery.day, y=dff_recovery['average']/1000,
            opacity=1,
            line=dict(color=colormap[amenity],),
            text=[amenity_names[amenity].lower()]*len(dff_recovery),
            hovertemplate = "The average distance to the nearest %{text} was %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)
    # add the percentiles
    new_trace = go.Scatter(
            x=dff_recovery.day, y=dff_recovery['p5']/1000,
            opacity=.50,
            line=dict(color=colormap[amenity],dash='dash'),
            text=dff_recovery.service,
            hovertemplate = "5th % = %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)
    # add the percentiles
    new_trace = go.Scatter(
            x=dff_recovery.day, y=dff_recovery['p95']/1000,
            opacity=.50,
            line=dict(color=colormap[amenity],dash='dash'),
            text=dff_recovery.service,
            hovertemplate = "95th % = %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)

    # add date line
    new_trace = go.Scatter(
            x=[day, day],
            y=[0,ylimit+2],
            opacity=.50,
            mode='lines',
            line=dict(color='black',dash='dash'),
            hoverinfo="skip", hovertemplate="",
            # transition = {'duration': 500},
            )
    data.append(new_trace)

    return {"data": data, "layout": layout}


def generate_map(amenity, dff_dist, dff_dest, x_range=None):
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
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        # height= 561,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat = 34.245580, lon = -77.872072),
            pitch=0,
            zoom=10.5,
            style="basic", #"dark", #
        ),
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="top",
        ),
    )

    if x_range:
        # get the indices of the values within the specified range
        idx = dff_dist.index[dff_dist['distance'].between(x_range[0],x_range[1], inclusive=True)].tolist()
    else:
        idx = dff_dist.index.tolist()

    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = 'https://raw.githubusercontent.com/urutau-nz/dash-recovery-florence/master/data/block.geojson',
        locations = dff_dist['geoid10'].tolist(),
        z = dff_dist['distance'].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        hovertemplate="Distance: %{z:.2f}km<br>" +
                        "<extra></extra>",
        selectedpoints=idx,
    ))

    # scatterplot of the amenity locations
    dest_open = dff_dest[dff_dest['operational']==True]
    dest_closed = dff_dest[dff_dest['operational']==False]

    if len(dest_open) > 0:
        data.append(go.Scattermapbox(
            lat=dest_open["lat"],
            lon=dest_open["lon"],
            mode="markers",
            marker={"color": [colormap[amenity]]*len(dest_open), "size": 9},
            name=amenity_names[amenity],
            hoverinfo="skip", hovertemplate="",
        ))

    if len(dest_closed) > 0:
        data.append(go.Scattermapbox(
            lat=dest_closed["lat"],
            lon=dest_closed["lon"],
            mode="markers",
            marker={"color": ['black']*len(dest_closed), "size": 9},
            name='Closed {}'.format(amenity_names[amenity]),
            hoverinfo="skip", hovertemplate="",
        ))
    # point_color = [colormap[amenity] if i==True else 'black' for i in dff_dest['operational']]



    return {"data": data, "layout": layout}


def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.A(
                                        html.Button("Overview", id="learn-more-button"),
                                        href="./overview",
                                    ),
                                    html.A(
                                        html.Button("Resilience & Access", id="learn-more-button", className="current-button"),
                                        # href="./resilience",
                                    ),
                                    html.A(
                                        html.Button("Measuring Equity", id="learn-more-button"),
                                        href="./equity",
                                    ),
                                    html.A(
                                        html.Button("Recover", id="learn-more-button"),
                                        href="./recover",
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
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Resilience as access to essentials"),
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
                                    In [Logan & Guikema (2020)](https://onlinelibrary.wiley.com/doi/full/10.1111/risa.13492)
                                    we made the case for ''access to essential services'' being considered as a
                                    pillar of community resilience.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Everyday services"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    The geography literature tells us that community cohesion, the generation of social
                                    capital, and community sustainability is fostered by access to opportunities
                                    and resources: specifically equitable access to those opportunities (Dempsey, 2011).
                                    These everyday amenities include water, power, sanitation, and communications, but communities also require
                                    access to food, education, and health care (Winter, 1997).
                                    (While access includes dimensions of availability, acceptability, afforability, adequacy, and awareness (Penchasky 1981; Saurman 2016),
                                    we begin by considering proximity.)"
                                    '''],
                                    ),
                                    html.H6(
                                        ["Community capacity"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Common to the many definitions of resilience is community capacity to anticipate, prepare, absorb,
                                    adapt, and transform.
                                    To develop these capacities, a community needs cohesion and social capital.
                                    It is therefore necessary to ensure there is equitable access to essential services.
                                    We have seen that communities without access to everyday services will simply collapse (Contreras, 2017).
                                    That is, a community without resources and the trust that arises from equitable opportunities, will struggle
                                    to develop the capacities identified as providing the foundation of resilience.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Functionality"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Resilience in terms of access not only supports the capacity for resilience prior to a disruption,
                                    but supports decision making during a disaster.
                                    At these times the focus of resilience shifts from the lens of capacity to that of functionality and impact.
                                    Understanding how people's access has changed can enable emergency managers to quickly restore
                                    the access that was lost and provide temporary access to those in-need.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Outcome-centric"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    This thinking of resilience is outcome-based.
                                    Where infrastructure has been the focus because of traditional engineering resilience approaches
                                     (due to its critical importance in supporting everyday services) 
                                    we focus on the outcomes for communities.
                                    For instance we can explore interventions that can
                                    improve the resilience of communities, such as decentralization or infrastructure independence (e.g., solar panels or generators).
                                    '''],
                                    ),
                                    html.H6(
                                        ["Spatial dimension"], className="subtitle padded"
                                    ),
                                    html.P(
                                        ["\
                                    Finally, this conceptualization is spatially explicit, \
                                    so resilience-enhancing decisions can be integrated into\
                                    land-use planning and hazard exposure mapping."
                                    ],
                                    ),
                                    html.H6(
                                        ["Applicable throughout the hazard cycle"], className="subtitle padded"
                                    ),
                                    html.P(
                                        ["\
                                    Thinking about resilience in this way can be used to support decision-making throughout the hazard cycle."
                                    ],
                                    ),
                                    html.Div(
                                        [
                                    html.Img(
                                        src='./assets/hazard_cycle.png',
                                        alt='How we can support at each stage of the hazard cycle',
                                        width='70%'
                                    ),
                                    ],
                                    style={"text-align": "center"},
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Example of Wilmington, NC and Hurricane Florence"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.P(
                                        ["\
                                    Below is an interactive example of assessing access over\
                                    the course of a hurricane. The map enables you to explore\
                                    how access is spatially distributed. You can highlight areas of\
                                    deprivation by dragging to select a range of values on the histogram.\
                                    The resilience curve shows how quickly access was restored; the resilience\
                                    curve includes the 5th and 95th percentiles so that we don't\
                                    ignore the people who are worst-off.\
                                    In the next page, we propose an alternative way to capture these\
                                    people."
                                    ],
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="amenity-select",
                                                options=[
                                                    {"label": amenity_names[i].upper(), "value": i}
                                                    for i in amenities
                                                ],
                                                value=amenities[0],
                                            ),
                                        ],
                                        # style={"overflow-x": "auto"},
                                    ),
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(

                                id="map-container",
                                    children=[
                                        html.H6("What is the state of people's access to services?"),
                                        # build_graph_title("How has people's access to services changed?"),
                                        dcc.Graph(
                                            id="map",
                                            figure={
                                                "layout": {
                                                    # "paper_bgcolor": "#192444",
                                                    # "plot_bgcolor": "#192444",
                                                }
                                            },
                                            config={"scrollZoom": True, "displayModeBar": True,
                                                    "modeBarButtonsToRemove":["lasso2d","select2d"],
                                            },
                                        ),
                                    ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                    id="ecdf-container",
                                    children=[
                                        html.H6("Select the day since hurricane landfall"),
                                        dcc.Slider(
                                            id="day-select",
                                            min=np.min(days),
                                            max=np.max(days),
                                            # step=2,
                                            marks={i: str(i) for i in range(np.min(days),np.max(days),1)},
                                            value=-2,
                                        ),
                                    ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # # Row 4
                    html.Div(
                        [
                            html.Div(
                                id="ecdf-container",
                                children=[
                                    html.H6("Select a distance range to identify those areas"),
                                    dcc.Graph(id="ecdf",
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
                                    dcc.Graph(id="recovery",
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
                                        ["For more information and the reference list, see ",
                                        html.A("Logan & Guikema (2020).", href="https://onlinelibrary.wiley.com/doi/full/10.1111/risa.13492",)
                                        ],
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
