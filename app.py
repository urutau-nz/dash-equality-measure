# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd

import numpy as np
from urllib.request import urlopen
import json
import geopandas as gpd
import datetime

from pages import (
    overview,
    resilience,
    equity,
    recover,
    transform,
    comingsoon
)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    url_base_pathname='/resilience-equity/',
)
server = app.server

app.config.suppress_callback_exceptions = True

app.title = 'Embedding equity into resilience'

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"),
                [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/resilience-equity/resilience":
        return resilience.create_layout(app)
    elif pathname == "/resilience-equity/equity":
        return equity.create_layout(app)
    elif pathname == "/resilience-equity/recover":
        return recover.create_layout(app)
    elif pathname == "/resilience-equity/transform":
        return transform.create_layout(app)
    elif pathname == "/resilience-equity/soon":
        return comingsoon.create_layout(app)
    elif pathname == "/resilience-equity/all":
        return (
            overview.create_layout(app),
            resilience.create_layout(app),
            equity.create_layout(app),
            recover.create_layout(app),
            transform.create_layout(app),
        )
    else:
        return overview.create_layout(app)


#####
# resilience
#####
# mapbox token
mapbox_access_token = open(".mapbox_token").read()

# Load data
df_dist = pd.read_csv('./data/distance_to_nearest.csv',dtype={"geoid10": str})
df_dist['distance'] = df_dist['distance']/1000
df_dist['distance'] = df_dist['distance'].replace(np.inf, 999)

destinations = pd.read_csv('./data/destinations.csv')

df_recovery = pd.read_csv('./data/recovery.csv')




# Update access map
@app.callback(
    Output("map", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
        Input("ecdf", "selectedData"),
    ],
)
def update_map(
    amenity_select, day, ecdf_selectedData
):
    x_range = None
    day = int(day)
    # subset the desination df
    dff_dest = destinations[(destinations.dest_type==amenity_select) & (destinations['day']==day)]
    dff_dist = df_dist[(df_dist['service']==amenity_select) & (df_dist['day']==day)]
    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2

    return resilience.generate_map(amenity_select, dff_dist, dff_dest, x_range=x_range)


# Update ecdf
@app.callback(
    Output("ecdf", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
        Input("ecdf", "selectedData"),
    ],
)
def update_ecdf(
    amenity_select, day, ecdf_selectedData
    ):
    x_range = None
    # day = int(day)

    # subset data
    dff_dist = df_dist[(df_dist['service']==amenity_select) & (df_dist['day']==day)]

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2

    return resilience.generate_ecdf_plot(amenity_select, dff_dist, x_range)

# Update ecdf
@app.callback(
    Output("recovery", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
    ],
)
def update_recovery(
    amenity_select, day
    ):
    x_range = None
    day = int(day)

    # subset data
    dff_recovery = df_recovery[(df_recovery['service']==amenity_select)]

    return resilience.recovery_plot(amenity_select, dff_recovery, day)

#####
# Equity
#####
# Load data
df_dist_grocery = pd.read_csv('./data/supermarket_distance.csv')
df_dist_grocery['distance'] = df_dist_grocery['distance']/1000
df_dist_grocery['distance'] = df_dist_grocery['distance'].replace(np.inf, 999)

df_rank = pd.read_csv('./data/ede_subgroups_-1.0.csv')
df_rank = df_rank.pivot(index='city',columns='group',values='ede')
# print(df_rank)

# Update ecdf
@app.callback(
    Output("food_ecdf", "figure"),
    [
        Input("race-select", "value"),
        Input("city-select", "value"),
    ],
)
def update_ecdf(
    race_select, cities_select
    ):

    # subset data
    dff_dist = df_dist_grocery[df_dist_grocery['city'].isin(cities_select)][['city','distance']+race_select]

    return equity.generate_ecdf_plot(dff_dist, race_select, cities_select)

# Update ranking
@app.callback(
    Output("food_ranking", "figure"),
    [
        Input("race-select-2", "value"),
        Input("race-order", "value"),
    ],
)
def update_ecdf(
    race_select, race_order
    ):

    # order
    df_rank.sort_values(by=[race_order], inplace=True)

    # subset data
    dff_rank = df_rank[[i for i in race_select]]

    return equity.generate_ranking_plot(dff_rank, race_select)
















if __name__ == "__main__":
    # app.run_server(debug=True, port=9006)
    app.run_server(port=9006)
