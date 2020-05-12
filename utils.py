import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.A([
                        html.Img(
                            src=app.get_asset_url("UC_logo.png"),
                            className="logo",
                        )],
                        href='https://canterbury.ac.nz/civil'),
                    # html.A(
                    #     html.Button("Learn More", id="learn-more-button"),
                    #     href="https://plot.ly/dash/pricing/",
                    # ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("Embedding equity into community resilience through data-driven urban planning")],
                        className="nine columns main-title",
                    ),
                    # html.Div(
                    #     [
                    #         dcc.Link(
                    #             "Full View",
                    #             href="/dash-financial-report/full-view",
                    #             className="full-view-link",
                    #         )
                    #     ],
                    #     className="five columns",
                    # ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Tom Logan",
                href="https://tomlogan.co.nz/",
                className="tab first",
            ),
            dcc.Link(
                "Mitchell Anderson",
                href="https://linkedin.com/in/man112",
                className="tab",
            ),
            dcc.Link(
                "Dai Kiddle",
                href="http://linkedin.com/in/dai-kiddle",
                className="tab",
            ),
            # dcc.Link(
            #     "Fees & Minimums", href="/dash-financial-report/fees", className="tab"
            # ),
            # dcc.Link(
            #     "Distributions",
            #     href="/dash-financial-report/distributions",
            #     className="tab",
            # ),
            # dcc.Link(
            #     "News & Reviews",
            #     href="/dash-financial-report/news-and-reviews",
            #     className="tab",
            # ),
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
