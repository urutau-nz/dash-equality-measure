import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

from utils import Header, make_dash_table
import pandas as pd
import numpy as np

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
                                        html.Button("Recover", id="learn-more-button"),
                                        href="./recover",
                                    ),
                                    html.A(
                                        html.Button("Transformation", id="learn-more-button", className="current-button"),
                                        # href="./transform",
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
                                    html.H5("Transformation and preparation"),
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
                                    Key to resilience is the notion of "desired functionality" and the capacity to adapt or transform.
                                    Given that aspects of resilience rely on community characteristics, approaches that enhance community
                                    cohesion, social capital, and community sustainability all will contribute to a community's resilience.

                                    We identify three ways for transforming communities:
                                    '''],
                                    ),
                                    html.H6(
                                        ["Improving the 'normal' conditions"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Measuring access and the EDE enables decision-makers to evaluate areas
                                    of need. We can also use this to optimize facility/service locations, based on
                                    need, to improve people's access to resources and opportunities.
                                    Providing equitable access to opportunities has been identified as an important
                                    condition of community sustainability (Dempsey 2011).
                                    '''],
                                    ),
                                    html.H6(
                                        ["Preparation"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Building immediately on from the recovery discussion: we can use this data-driven planning
                                    approach to simulate failures and either strengthen existing facilities or build redunancy.
                                    Using simulation enables us to explore scenarios, and discuss and plan how we can respond.
                                    '''],
                                    ),
                                    html.H6(
                                        ["Transforming communities"], className="subtitle padded"
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Ultimately, with the changing environment we need to begun to rethink the spatial
                                    arrangement of our communities.
                                    We are working on integrating these approaches with hazard maps so to form a robust
                                    approach to guiding locations of not just amenities, but further development and densification.
                                    '''],
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # concluding thoughts
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Concluding thoughts"],
                                        className="subtitle padded",
                                    ),
                                    dcc.Markdown(
                                        ['''
                                    Much of this work is ongoing, and I am keen to discuss this at the AScUS unconference.

                                    Especially with the significant investment slated for kick-starting economic recovery,
                                    now is an important time to ensure that our investment not only improves resiliences,
                                    but ensures it is equitable. Where suitable and appropriate, we should use data to
                                    inform these decisions.

                                    I look forward to talking with you all.
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
