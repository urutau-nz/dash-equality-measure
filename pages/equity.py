import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

from utils import Header, make_dash_table
import pandas as pd
import numpy as np

# mapbox token
mapbox_access_token = open(".mapbox_token").read()

cities_dict = {'baltimore':'bal','chicago':'chi','detroit':'det','seattle':'sea',
            'portland':'por','denver':'den','miami':'mia','atlanta':'atl',
            'new orleans':'new','houston':'hou'}

coords_dict = {'baltimore':[39.29501800942237, -76.61361529218145],
                'chicago':'chi',
                'detroit':'det','seattle':'sea',
            'portland':'por','denver':'den','miami':'mia','atlanta':'atl',
            'new orleans':'new','houston':'hou'}

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


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 3
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Measuring equality and equity in urban planning"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    One major approach to evaluate equity in urban planning is equity mapping.
                                    While mapping is essential for communicating to decision-makers, there are occassions where
                                    quantifying a distribution and its equality with a single value would be useful.
                                    Such cases include 1) ranking of interventions or regions,\
                                    2) regression analysis, 3) optimization, 4) and enhancing resilience.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Evaluating a distribution"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Existing approaches to evaluate a distribution include the average value and percentage of population within a threshold.
                                    These approaches essentially ignore the residents who are worst-off.
                                    Therefore we need a measure that captures the inequality.
                                    Such measures include the Gini Index, the Atkinson Index, and the recently proposed Kolm-Pollak "EDE" (Sheriff, 2013).
                                    '''],
                                    ),
                                    html.H6(
                                        ["The Kolm-Pollak EDE"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    An EDE is an equally-distributed equivalent;
                                    this is the value that would, if everyone had that same value, provide the same level of welfare as the existing distribution.
                                    That is, it measures the distribution and penalizes for inequality.

                                    The Kolm-Pollak EDE measure is well suited for urban planning application compared to the Gini or Atkinson because
                                    * it can be used for distributions of both desirable and undesirable quantities (e.g., income or exposure respectively)
                                    * it can be used in lieu of the average value (so no changes to the analysis process are required)
                                    * it is an absolute (as opposed to relative) measure, which is necessary in planning
                                    * it enables subgroup comparisons for equity analysis.
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
                                ["Example: Evaluating grocery store access in ten USA cities"], className="subtitle padded"
                            ),
                            html.P(
                                ["\
                            To illustrate using the K-P approach, we will evaluate grocery store access in ten USA cities.\
                            We begin by calculating the network walking distance from every census-block to the nearest\
                            grocery store.\
                            "
                            ],
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="city-select",
                                        options=[
                                            {'label': 'Baltimore', 'value': 'baltimore'},
                                            {'label': 'Chicago', 'value': 'chicago'},
                                            {'label': 'Detroit', 'value': 'detroit'},
                                            {'label': 'Seattle', 'value': 'seattle'},
                                            {'label': 'Portland', 'value': 'portland'},
                                            {'label': 'Denver', 'value': 'denver'},
                                            {'label': 'Miami', 'value': 'miami'},
                                            {'label': 'Atlanta', 'value': 'atlanta'},
                                            {'label': 'New Orleans', 'value': 'new orleans'},
                                            {'label': 'Houston', 'value': 'houston'},
                                        ],
                                        multi=False,
                                        value="baltimore"
                                    ),
                                ],
                                # style={"overflow-x": "auto"},
                            ),
                            html.Div(
                                id="map-container",
                                    children=[
                                        html.H6("What is the state of people's access to services?"),
                                        dcc.Graph(
                                            id="map",
                                            figure={
                                                "layout": {
                                                }
                                            },
                                            config={"scrollZoom": True, "displayModeBar": True,
                                                    "modeBarButtonsToRemove":["lasso2d","select2d"],
                                            },
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
                                ["Example: Evaluating grocery store access in ten USA cities"], className="subtitle padded"
                            ),
                            html.P(
                                ["\
                            To illustrate using the K-P approach, we will evaluate grocery store access in ten USA cities.\
                            We begin by calculating the network walking distance from every census-block to the nearest\
                            grocery store.\
                            We plot the distribution of access in the figure below.\
                            This shows what percentage of the population live within x-distance to their nearest store.\
                            In this figure, compare the different cities and demographic groups.\
                            We then calculate the Kolm-Pollak EDE so we have a single metric we can use to compare these groups (bottom figure).\
                            Note that this analysis is not possible with other equality indices because the Atkinson index is suited only for quantities of which you want more of (e.g., money, but not distance)\
                            and the Gini index is a relative measure of equality so it would say a city with no supermarkets is better because everyone is equally poorly off.\
                            "
                            ],
                            ),
                            html.Div(
                                    id="ecdf-container",
                                    children=[
                                        html.H6("Select the cities"),
                                        dcc.Dropdown(
                                            id="cities-select",
                                            options=[
                                                {'label': 'Baltimore', 'value': 'baltimore'},
                                                {'label': 'Chicago', 'value': 'chicago'},
                                                {'label': 'Detroit', 'value': 'detroit'},
                                                {'label': 'Seattle', 'value': 'seattle'},
                                                {'label': 'Portland', 'value': 'portland'},
                                                {'label': 'Denver', 'value': 'denver'},
                                                {'label': 'Miami', 'value': 'miami'},
                                                {'label': 'Atlanta', 'value': 'atlanta'},
                                                {'label': 'New Orleans', 'value': 'new orleans'},
                                                {'label': 'Houston', 'value': 'houston'},
                                            ],
                                            multi=True,
                                            value=["chicago"]
                                        ),
                                        html.H6("Select the demographic groups"),
                                        dcc.Checklist(
                                            id="race-select",
                                            options=[
                                                {'label': 'All', 'value': 'H7X001'},
                                                {'label': 'White', 'value': 'H7X002'},
                                                {'label': 'Black', 'value': 'H7X003'},
                                                {'label': 'Am. Indian', 'value': 'H7X004'},
                                                {'label': 'Asian', 'value': 'H7X005'},
                                                {'label': 'Latino/Hispanic', 'value': 'H7Y003'},
                                            ],
                                            value=['H7X001',],
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                        html.Div(
                                            id="ecdf-container",
                                            children=[
                                                html.H6("How access to grocery stores is distributed across residents"),
                                                dcc.Graph(id="food_ecdf",
                                                config={"scrollZoom": True, "displayModeBar": True,
                                                        "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian',
                                                        'pan',"zoomIn2d", "zoomOut2d","lasso2d","select2d"],
                                                },
                                            ),
                                            ],
                                            className=" twelve columns",
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
                            html.H6(
                                ["Comparisons and rankings"], className="subtitle padded"
                            ),
                            html.Div(
                                    id="ecdf-container",
                                    children=[
                                        html.H6("We can rank the cities based on access to grocery stores"),
                                        dcc.Graph(id="food_ranking",
                                        config={"scrollZoom": True, "displayModeBar": True,
                                                "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian',
                                                'pan',"zoomIn2d", "zoomOut2d","lasso2d","select2d"],
                                                },
                                            ),
                                        html.H6("Which metric should be used:"),
                                        dcc.RadioItems(
                                            id="metric",
                                            options=[
                                                {'label': 'Kolm-Pollak EDE', 'value': 'ede'},
                                                {'label': 'Mean', 'value': 'mean'},
                                                {'label': 'Gini', 'value': 'gini'},
                                            ],
                                            value='ede',
                                            labelStyle={'display': 'inline-block', 'font-weight':400}
                                        ),
                                        html.H6('If you selected Kolm-Pollak EDE, you can select the options below.'),
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
                                        ["Logan, T. M., Anderson, M. J., Williams, T., & Conrow, L. (Under review). Measuring inequalities in urban systems: An approach for evaluating the distribution of amenities and burdens. Computers, Environmental, and Urban Systems."],
                                        style={'padding-left': '22px', 'text-indent': '-22px', 'font-weight':400}
                                        ),
                                    html.P(
                                        [
                                        html.A(
                                                "Sheriff, G., & Maguire, K. B. (2020). Health Risk, Inequality Indexes, and Environmental Justice. Risk Analysis: An Official Publication of the Society for Risk Analysis.",
                                                href="https://doi.org/10.1111/risa.13562",
                                                target='_blank'
                                            )
                                        ],
                                        style={'padding-left': '22px', 'text-indent': '-22px', 'font-weight':400}
                                    ),
                                    html.H6(
                                        ["Python package"],
                                        className="subtitle padded",
                                    ),
                                    html.P(
                                        [
                                        "This package lets you calculate the Kolm-Pollak, Atkinson, and Gini metrics for your data, in Python.",
                                        html.Br(),
                                        html.A(
                                                "Inequalipy",
                                                href="https://pypi.org/project/inequalipy/",
                                                target='_blank'
                                            )
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

race_dict = {'H7X001':'All', 'H7X002': 'White', 'H7X003':'Black', 'H7X004':'Am. Indian', 'H7X005':'Asian', 'H7Y003':'Hispanic'}

def generate_ecdf_plot(dff_dist, race_select, cities_select):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    layout = dict(
        xaxis=dict(
            title="distance to grocery (km)".upper(),
            # range=(0,15),
            # fixedrange=True,
            titlefont=dict(size=12)
            ),
        yaxis=dict(
            title="% of residents".upper(),
            range=(0,100),
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        font=dict(size=13),
        # dragmode="select",
        # paper_bgcolor = 'rgba(255,255,255,1)',
		# plot_bgcolor = 'rgba(0,0,0,0)',
        # bargap=0.05,
        showlegend=True,
        margin=dict(l=40, r=0, t=10, b=30),
        # transition = {'duration': 500},
        # legendgroup='city'
        # hovermode ='closest',
        # height= 300

    )
    data = []
    i = 0
    # loop the cities
    for city_select in cities_select:
        color = px.colors.qualitative.Safe[i]
        i += 1
        df_plot = dff_dist[dff_dist.city==city_select]
        city_select = city_select.capitalize()
    # loop the group
        j = 0
        for group_select in race_select:
            # add the cdf for that amenity
            counts, bin_edges = np.histogram(df_plot.distance, bins=100, density = True, weights=df_plot[group_select])
            dx = bin_edges[1] - bin_edges[0]
            new_trace = go.Scatter(
                    x=bin_edges, y=np.cumsum(counts)*dx*100,
                    opacity=1,
                    line=dict(color=color,),
                    name=city_select,
                    legendgroup=city_select,
                    showlegend= j == 0,
                    text=np.core.defchararray.add(
                            np.repeat('{} '.format(race_dict[group_select]),len(df_plot)),
                            np.repeat(city_select,len(df_plot)),
                            ),
                    hovertemplate = "%{y:.0f}% of %{text} residents live within %{x:.1f}km <br>" + "<extra></extra>",
                    hoverlabel = dict(font_size=20),
                    )
            j += 1
            data.append(new_trace)

    return {"data": data, "layout": layout}


def generate_ranking_plot(dff_rank, race_select):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    layout = dict(
        xaxis=dict(
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        yaxis=dict(
            title="EDE: distance (km)".upper(),
            range=(0,5),
            fixedrange=True,
            titlefont=dict(size=12)
            ),
        font=dict(size=13),
        showlegend=True,
        margin=dict(l=40, r=0, t=10, b=60),
        transition = {'duration': 500},
    )

    data = []
    # loop the group
    for group_select in race_select:
        df_plot = dff_rank[group_select]
        # add the cdf for that amenity
        new_trace = go.Scatter(
                x=df_plot.index, y=df_plot,
                opacity=1,
                # line=dict(color=color,),
                name=race_dict[group_select],
                showlegend = True,
                text=np.repeat(race_dict[group_select],len(df_plot)),
                hovertemplate = "%{text}: %{y:.1f}km <br>" + "<extra></extra>",
                hoverlabel = dict(font_size=20),
                )
        data.append(new_trace)

    return {"data": data, "layout": layout}

def generate_map(city_select, dff_dist, dff_dest, x_range=None):
    """
    Generate map showing the distance to services and the locations of them
    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """
    # print(dff_dist['geoid10'].tolist())
    dff_dist = dff_dist.reset_index()
    coord = coords_dict[city_select]
    block_data = 'https://raw.githubusercontent.com/urutau-nz/dash-equality-measure/master/data/block_{}.geojson'.format(cities_dict[city_select])

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
            center=go.layout.mapbox.Center(lat = coord[0], lon = coord[1]),
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

    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = block_data,
        locations = dff_dist['index'].tolist(),
        z = dff_dist['supermarket'].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        hovertemplate="Distance: %{z:.2f}km<br>" +
                        "<extra></extra>",
    ))

    data.append(go.Scattermapbox(
        lat=dff_dest["lat"],
        lon=dff_dest["lon"],
        mode="markers",
        marker={"color": ['#EA5138']*len(dff_dest), "size": 9},
        name='supermarket',
        hoverinfo="skip", hovertemplate="",
    ))


    return {"data": data, "layout": layout}
