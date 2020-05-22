import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table

import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


df_fund_facts = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))
df_price_perf = pd.read_csv(DATA_PATH.joinpath("df_price_perf.csv"))


def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Abstract"),
                                    html.Br([]),
                                    html.P(
                                        ["\
                                    Resilience is occassionally criticized as being an undesirable property of a system\
                                    because some of the ways it is characterized can limit potential for transformation\
                                    and even, inadverently, exacerbate inequality.\
                                    In ",
                                    html.A("Logan & Guikema (2020)", href="https://onlinelibrary.wiley.com/doi/full/10.1111/risa.13492",
                                        style={'color':'white'}),
                                    " we \
                                    made the case for ''access to essential services'' being considered as a \
                                    pillar of community resilience.\
                                    We now demonstrate how we can embed equity into community\
                                    resilience planning using a metric recently proposed in \
                                    the risk community. \
                                    We show how this approach could be used to not only\
                                    recover from a disruption, but also transform both prior to and following\
                                    such a disruption.\
                                    This approach to community resilience presents an opportunity to support\
                                    communities build better and actively improve urban form in a manner\
                                    that enhances the resilience and quality of life for all residents."],
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    # concluding thoughts
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(
                                        ['''
                                    This web app is created as my contribution
                                    for the 2020 AScUS
                                    (Actionable Science for Urban Sustainability)
                                    virtual unconference, hosted 3-5 June 2020.
                                    https://ascus.metabolismofcities.org/
                                    '''],
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
