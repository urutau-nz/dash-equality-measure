# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import inequalipy as ineq
import numpy as np
from urllib.request import urlopen
import json

from pages import (
    # overview,
    # resilience,
    equity,
    # recover,
    # transform,
    # comingsoon
)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    url_base_pathname='/equality-measure/',
)
server = app.server

app.config.suppress_callback_exceptions = True

app.title = 'Measuring inequality in urban systems'

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"),
                [Input("url", "pathname")])
def display_page(pathname):
    return equity.create_layout(app)

cities_dict = {'baltimore':'bal','chicago':'chi','detroit':'det','seattle':'sea',
            'portland':'por','denver':'den','miami':'mia','atlanta':'atl',
            'new orleans':'new','houston':'hou'}
#####
# resilience
#####
# mapbox token
mapbox_access_token = open(".mapbox_token").read()


# Update access map
@app.callback(
    Output("map", "figure"),
    [
        Input("city-select", "value"),
    ],
)
def update_map(
    city_select
):
    x_range = None
    # import data
    dff_dest = pd.read_csv('./data/destinations_{}.csv'.format(cities_dict[city_select]))

    dist = pd.read_csv('./data/distance_{}.csv'.format(cities_dict[city_select]),dtype={"index": str})
    dist['supermarket'] = dist['supermarket']/1000
    dist['supermarket'] = dist['supermarket'].replace(np.inf, 999)
    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    return equity.generate_map(city_select, dist, dff_dest, x_range=x_range)



#####
# Equity
#####
# Load data
df_dist_grocery = pd.read_csv('./data/supermarket_distance.csv')
df_dist_grocery['distance'] = df_dist_grocery['distance']/1000
df_dist_grocery['distance'] = df_dist_grocery['distance'].replace(np.inf, 999)

#
# df_rank = pd.read_csv('./data/ede_subgroups_-1.0.csv')
# df_rank = df_rank.pivot(index='city',columns='group',values='ede')
# print(df_rank)

# Update ecdf
@app.callback(
    Output("food_ecdf", "figure"),
    [
        Input("race-select", "value"),
        Input("cities-select", "value"),
        Input("metrics-select-ecdf", "value"),
    ],
)
def update_ecdf(
    race_select, cities_select, metrics_select
    ):

    # subset data
    dff_dist = df_dist_grocery[df_dist_grocery['city'].isin(cities_select)][['city','distance']+race_select]

    return equity.generate_ecdf_plot(dff_dist, race_select, cities_select, metrics_select)

# Update ranking
@app.callback(
    Output("food_ranking", "figure"),
    [
        Input("race-select-2", "value"),
        Input("race-order", "value"),
        Input("metric-select", "value"),
        Input("epsilon", "value"),
    ],
)
def update_rank(
    race_select, race_order, metric_select, epsilon
    ):
    if metric_select == 'ede':
        kappa = ineq.kolmpollak.calc_kappa(df_dist_grocery.distance, weights=df_dist_grocery.H7X001,epsilon=epsilon)
        # initialize dataframe
        results = []
        for city in cities_dict.keys():
        # loop through the subgroups
            for group in race_select:
                # Gets the df for specific state
                import code
                # code.interact(local=locals())
                df = df_dist_grocery[df_dist_grocery.city==city][['distance']+ race_select].copy()
                # drop data that has 0 weight
                df = df.iloc[np.array(df[group]) > 0].copy()
                # calculate the ede
                ede = ineq.kolmpollak.ede(list(df.distance), kappa = kappa, weights = list(df[group]))
                # new result
                result_i = [city, group, ede]
                results.append(result_i)
        # make list of lists a dataframe
        df_rank = pd.DataFrame(results, columns = ['city','group','ede'])
    elif metric_select == 'mean':
        # initialize dataframe
        results = []
        for city in cities_dict.keys():
        # loop through the subgroups
            for group in race_select:
                # Gets the df for specific state
                df = df_dist_grocery[df_dist_grocery.city==city][['distance']+ race_select].copy()
                # drop data that has 0 weight
                df = df.iloc[np.array(df[group]) > 0].copy()
                # calculate the ede
                ede = np.average(list(df.distance), weights = list(df[group]))
                # new result
                result_i = [city, group, ede]
                results.append(result_i)
        # make list of lists a dataframe
        df_rank = pd.DataFrame(results, columns = ['city','group','ede'])
    elif metric_select == 'gini':
        race_select = ['H7X001']
        race_order = 'H7X001'
        # initialize dataframe
        results = []
        for city in cities_dict.keys():
        # loop through the subgroups
            for group in race_select:
                # Gets the df for specific state
                df = df_dist_grocery[df_dist_grocery.city==city][['distance']+ race_select].copy()
                # drop data that has 0 weight
                df = df.iloc[np.array(df[group]) > 0].copy()
                # calculate the ede
                ede = ineq.gini(list(df.distance), weights = list(df[group]))
                # new result
                result_i = [city, group, ede]
                results.append(result_i)
        # make list of lists a dataframe
        df_rank = pd.DataFrame(results, columns = ['city','group','ede'])

    df_rank = df_rank.pivot(index='city',columns='group',values='ede')
    # order
    df_rank.sort_values(by=[race_order], inplace=True)

    # subset data
    dff_rank = df_rank[[i for i in race_select]]

    return equity.generate_ranking_plot(dff_rank, race_select,metric_select)



if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server(port=9010, debug=True)
