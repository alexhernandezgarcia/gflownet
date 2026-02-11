import sqlite3
import warnings
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme

from gflownet.utils.vislogger.plot_utils import Plotter


def run_dashboard(
    data: str,
    text_to_img_fn: callable,
    state_aggregation_fn: callable,
    s0: str = "#",
    debug_mode: bool = False,
):
    """Runs the dashboard on http://127.0.0.1:8050/.

    Parameters
    ----------
    data : str
        Folder of the logged data.
    text_to_img_fn : callable
        Function to convert the texts representing the states to base64 encoded svg
        images. Used to identify states on the dashboard.
    state_aggregation_fn : callable
        Function to aggregate states. In the state space it might help to see what
        states in a bin have in common. This function provides this aggregation.
        This can be eg the Maximum Common Substructure in Molecules / Graphs, paths that
        must have been taken for all states in Games, etc. Must take the states as a
        list of strings and return one state as a string. If not specified the longest
        common substring of all strings will be used. There is no guarantee that this is
        a valid state.
    s0: str
        Gives the option to specify the start state.
        By default '#' is used and treated as an empty state.
        If s0 is meaningful in your environment (e.g. the position [0, 0] in a grid),
        you can specify how your start state looks here.
        In that case it will be displayed via the text_to_img_fn.
    debug_mode : bool, optional
        Whether to display in debug mode for error handling. By default, False.
    """

    data_path = Path(data)
    data_folder = data_path.parent if data_path.suffix else data_path
    if data_folder.exists() and data_folder.is_dir():
        data_path = data_folder / "data.db"
    else:
        raise FileNotFoundError(f"Folder does not exist: {data_folder}")
    if not data_path.exists() or (not data_path.is_file()):
        raise FileNotFoundError(f"Data does not exist: {data_path}")

    if text_to_img_fn is None:
        warnings.warn(
            "No text-to-image function provided. "
            "Identifying states will not be possible. "
            "Provide text_to_img_fn to allow converting text representations of "
            "states to images."
        )

        def image_fn(state):
            return None

    else:

        def image_fn(state):
            if state == "#":
                return None
            result = text_to_img_fn(state)
            if isinstance(result, list):
                return result
            return f"data:image/svg+xml;base64,{text_to_img_fn(state)}"

    if state_aggregation_fn is None:
        warnings.warn("No state-aggregation-function provided. ")

        def agg_fn(states):
            return None

    else:

        def agg_fn(states):
            if len(states) == 0:
                return None
            elif len(states) == 1:
                return image_fn(states[0])
            else:
                agg = state_aggregation_fn(states)
                if isinstance(agg, str):
                    return f"data:image/svg+xml;base64,{agg}"
                return agg

    plotter = Plotter(data_path, image_fn, agg_fn, s0)

    # get provided metrics and feature columns
    conn = sqlite3.connect(data_path)
    data_cols = pd.read_sql_query(
        "SELECT name FROM pragma_table_info('trajectories');", conn
    )["name"].tolist()
    final_object_metrics = data_cols[
        data_cols.index("total_reward") : data_cols.index("logprobs_forward")
    ]
    feature_cols = data_cols[data_cols.index("features_valid") + 1 :]

    # create empty testset if it is missing and get column names
    cur = conn.cursor()
    cur.execute("""
            CREATE TABLE IF NOT EXISTS testset (
                id TEXT PRIMARY KEY,
                total_reward REAL,
                features_valid REAL
            )
        """)
    testset_cols = pd.read_sql_query(
        "SELECT name FROM pragma_table_info('testset');", conn
    )["name"].tolist()
    feature_cols_testset = testset_cols[testset_cols.index("features_valid") + 1 :]
    testset_metrics = testset_cols[
        testset_cols.index("total_reward") : testset_cols.index("features_valid")
    ]

    # get iteration range
    query = "SELECT MIN(iteration) AS min, MAX(iteration) AS max FROM trajectories"
    iteration_range = pd.read_sql_query(query, conn)
    iteration_step = iteration_range["max"][0] - iteration_range["min"][0]
    iteration_marks = [0, 0.25, 0.5, 0.75, 1]
    iteration_marks = [i * iteration_step for i in iteration_marks]
    iteration_marks = [i + iteration_range["min"][0] for i in iteration_marks]
    iteration_marks = [int(i) for i in iteration_marks]
    iteration_marks_dict = dict()
    for i in iteration_marks:
        iteration_marks_dict[i] = str(i)

    conn.commit()
    conn.close()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # Load extra layouts for cytoscape
    cyto.load_extra_layouts()

    app.layout = html.Div(
        [
            dcc.Store(id="selected-objects", data=[]),
            dcc.Store(id="build-ids", data=["#"]),
            dcc.Store(id="max-frequency", data=0),
            dcc.Store(id="dag-overview-tid-list", data=[]),
            dcc.Store(id="dag-overview-edge-list", data=[]),
            dcc.Store(id="hexbin-size", data=8),
            dcc.Store(id="dag-overview-page", data=0),
            # ================= LEFT SIDEBAR (12%) =================
            html.Div(
                [
                    # -------- TAB SELECTOR --------
                    html.A(
                        html.Button(
                            "How to Use",
                            style={
                                "border-radius": "8px",
                            },
                        ),
                        href="https://github.com/florianholeczek/GFlowNet_Training_Vis_Pilot/blob/master/Dashboard_Introduction.md",
                        target="_blank",
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "margin-bottom": "50px",
                            "textDecoration": "none",
                        },
                    ),
                    html.H4("View"),
                    html.Div(
                        [
                            html.Button(
                                "Final Objects",
                                id="tab-state-space",
                                n_clicks=0,
                            ),
                            html.Button(
                                "DAG",
                                id="tab-dag-view",
                                n_clicks=0,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            # "gap": "12px"
                        },
                    ),
                    dcc.Store(id="active-tab", data="state-space"),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "50px",
                            "height": "40px",
                        }
                    ),
                    html.H4("General"),
                    html.Div(
                        [
                            html.Button(
                                "Clear selection",
                                id="clear-selection",
                                n_clicks=0,
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                    "border-radius": "8px",
                                    "border": "2px solid #e5e7eb",
                                },
                            ),
                            # -------- Iterations --------
                            html.Div(
                                [
                                    html.Div(
                                        "Iterations", style={"textAlign": "center"}
                                    ),
                                    dcc.RangeSlider(
                                        id="iteration",
                                        min=iteration_range["min"][0],
                                        max=iteration_range["max"][0],
                                        step=1,
                                        value=[
                                            iteration_range["min"][0],
                                            iteration_range["max"][0],
                                        ],
                                        marks=iteration_marks_dict,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "50px",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "50px",
                            "height": "40px",
                        }
                    ),
                    html.H4("Projection", id="sidebar-tab-header"),
                    # --------------- Projection Controls ---------------
                    html.Div(
                        [
                            html.Div(
                                [
                                    # -------- Final Object Metric --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Object Metric",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.Dropdown(
                                                id="fo-metric",
                                                options=final_object_metrics,
                                                value="total_reward",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Ranking Metric --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Ranking Metric",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.Dropdown(
                                                id="ranking-metric",
                                                options=[
                                                    "highest over all",
                                                    "highest per iter",
                                                    "lowest over all",
                                                    "lowest per iter",
                                                ],
                                                value="highest over all",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Use Testset --------
                                    dcc.Checklist(
                                        [" Use Testset"],
                                        [],
                                        id="use-testset",
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- State Space Style --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "State Space Style",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.Dropdown(
                                                id="state-space-style",
                                                options=[
                                                    "Hex Ratio",
                                                    "Hex Obj. Metric",
                                                    "Hex Correlation",
                                                    "Scatter",
                                                ],
                                                value="Hex Ratio",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Projection method --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Dimensionality reduction",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.Dropdown(
                                                id="projection-method",
                                                options=[
                                                    {"label": "UMAP", "value": "umap"},
                                                    {"label": "t-SNE", "value": "tsne"},
                                                ],
                                                value="tsne",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Projection param --------
                                    html.Div(
                                        [
                                            html.Div(
                                                id="projection-param-label",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.Slider(
                                                id="projection-param",
                                                min=5,
                                                max=50,
                                                step=1,
                                                value=15,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "50px",
                                },
                            )
                        ],
                        id="sidebar-projection",
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "50px",
                        },
                    ),
                    # --------------- DAG Controls ---------------
                    html.Div(
                        [
                            html.Div(
                                [
                                    # -------- Layout --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Layout", style={"textAlign": "center"}
                                            ),
                                            dcc.Dropdown(
                                                id="dag-layout",
                                                options=[
                                                    {"label": "Klay", "value": "klay"},
                                                    {
                                                        "label": "Dagre",
                                                        "value": "dagre",
                                                    },
                                                    {
                                                        "label": "Breadthfirst",
                                                        "value": "breadthfirst",
                                                    },
                                                ],
                                                value="klay",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Metric --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Metric", style={"textAlign": "center"}
                                            ),
                                            dcc.Dropdown(
                                                id="dag-metric",
                                                options=[
                                                    "highest",
                                                    "lowest",
                                                    "variance",
                                                    "frequency",
                                                ],
                                                value="highest",
                                                clearable=False,
                                                style={"color": "black"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                    # -------- Direction --------
                                    html.Div(
                                        [
                                            html.Div(
                                                "Direction",
                                                style={"textAlign": "center"},
                                            ),
                                            dcc.RadioItems(
                                                id="dag-direction",
                                                options=["forward", "backward"],
                                                value="forward",
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "6px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "50px",
                                },
                            )
                        ],
                        id="sidebar-dag",
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "50px",
                        },
                    ),
                ],
                style={
                    "width": "12%",
                    "minWidth": "180px",
                    "maxWidth": "250px",
                    "padding": "12px",
                    "height": "100vh",
                    # "borderRight": "1px solid #ddd",
                    "overflow": "auto",
                },
            ),
            # ================= RIGHT CONTENT AREA (88%) =================
            html.Div(
                [
                    # ================= STATE-SPACE TAB =================
                    html.Div(
                        [
                            # TOP ROW
                            html.Div(
                                [
                                    # TOP LEFT
                                    html.Div(
                                        [
                                            html.Div(
                                                dcc.Graph(
                                                    id="bumpchart",
                                                    clear_on_unhover=True,
                                                ),
                                                style={
                                                    "height": "100%",
                                                    "width": "100%",
                                                },
                                            ),
                                            dcc.Tooltip(
                                                id="image-tooltip3", direction="left"
                                            ),
                                        ],
                                        style={
                                            "flex": 1,
                                            # "border": "1px solid #ddd",
                                            "padding": "5px",
                                            "height": "35vh",
                                            "boxSizing": "border-box",
                                            "overflow": "hidden",
                                        },
                                    ),
                                    # TOP RIGHT
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "width": "100%",
                                },
                            ),
                            # BOTTOM ROW
                            html.Div(
                                [
                                    # BOTTOM LEFT
                                    html.Div(
                                        [
                                            html.Div(
                                                dcc.Graph(
                                                    id="state-space-plot",
                                                    clear_on_unhover=True,
                                                ),
                                                style={
                                                    "height": "100%",
                                                    "width": "100%",
                                                },
                                            ),
                                            dcc.Tooltip(
                                                id="image-tooltip1", direction="top"
                                            ),
                                        ],
                                        style={
                                            "flex": 1,
                                            # "border": "1px solid #ddd",
                                            "padding": "5px",
                                            "height": "65vh",
                                            "boxSizing": "border-box",
                                            "overflow": "hidden",
                                        },
                                    ),
                                    # BOTTOM RIGHT
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "width": "100%",
                                },
                            ),
                        ],
                        id="state-space-tab",
                        style={"display": "block"},
                    ),
                    # ================= DAG TAB =================
                    html.Div(
                        [
                            html.Div(
                                [
                                    # DAG Overview
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Top 150",
                                                        id="dag-overview-button-top",
                                                        style={
                                                            "flex": 1,
                                                            "height": "40px",
                                                            "border-top-left-radius": "8px",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Previous 150",
                                                        id="dag-overview-button-prev",
                                                        style={
                                                            "flex": 1,
                                                            "height": "40px",
                                                            "border-top-right-radius": "8px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "margin-top": "10px",
                                                },
                                            ),
                                            # --- Graph (fills remaining height) ---
                                            html.Div(
                                                dcc.Graph(
                                                    id="dag-overview",
                                                    clear_on_unhover=True,
                                                    style={
                                                        "height": "100%",
                                                        "width": "100%",
                                                    },
                                                    config={"responsive": True},
                                                ),
                                                style={"flex": 1},
                                            ),
                                            # --- Bottom button ---
                                            html.Button(
                                                "Next 150",
                                                id="dag-overview-button-next",
                                                style={
                                                    "height": "40px",
                                                    "width": "100%",
                                                    "border-bottom-left-radius": "8px",
                                                    "border-bottom-right-radius": "8px",
                                                    "margin-bottom": "10px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "flex": "0 0 400px",
                                            "height": "100vh",
                                            "width": "400px",
                                            "gap": "8px",
                                        },
                                    ),
                                    # DAG AREA
                                    html.Div(
                                        [
                                            html.Div(
                                                "DAG-title",
                                                id="dag-title",
                                                style={
                                                    "height": "20px",
                                                    # "border": "1px solid #ddd",
                                                    "boxSizing": "border-box",
                                                    "padding-top": "3px",
                                                    "font-size": "12px",
                                                    "font-weight": "bold",
                                                    "margin-top": "2px",
                                                },
                                            ),
                                            html.Div(
                                                "DAG-subtitle",
                                                id="dag-subtitle",
                                                style={
                                                    "height": "24px",
                                                    # "border": "1px solid #ddd",
                                                    "boxSizing": "border-box",
                                                    "padding-top": "3px",
                                                    "font-size": "10px",
                                                    "margin-bottom": "13px",
                                                    "margin-top": "2px",
                                                    "whiteSpace": "pre-line",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    cyto.Cytoscape(
                                                        id="dag-graph",
                                                        layout={
                                                            "name": "klay",
                                                            "directed": True,
                                                            "spacingFactor": 0.5,
                                                            "animate": False,
                                                        },
                                                        style={
                                                            "flex": "1",
                                                            "height": "100%",
                                                            "width": "0px",
                                                            "background-color": "#222222",
                                                        },
                                                        elements=[],
                                                        stylesheet=[],
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flexDirection": "row",
                                                    "flex": 1,
                                                    # "border": "1px solid #ddd",
                                                    "boxSizing": "border-box",
                                                },
                                            ),
                                        ],
                                        style={
                                            "flex": 1,
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "height": "100vh",
                                            "margin-right": "10px",
                                            "margin-left": "10px",
                                        },
                                    ),
                                    # RIGHT SIDE - DATA TABLE
                                    dash_table.DataTable(
                                        id="dag-table",
                                        columns=[
                                            {
                                                "name": "Image",
                                                "id": "image",
                                                "presentation": "markdown",
                                            },
                                            {
                                                "name": "Final",
                                                "id": "node_type",
                                                "type": "any",
                                            },
                                            {
                                                "name": "Metric",
                                                "id": "logprobs",
                                                "type": "numeric",
                                                "format": Format(
                                                    precision=4, scheme=Scheme.fixed
                                                ),
                                            },
                                            {
                                                "name": "Reward",
                                                "id": "reward",
                                                "type": "numeric",
                                                "format": Format(
                                                    precision=4, scheme=Scheme.fixed
                                                ),
                                            },
                                        ],
                                        row_selectable="multi",
                                        filter_action="native",
                                        sort_action="native",
                                        selected_row_ids=[],
                                        page_size=10,
                                        markdown_options={"html": True},
                                        style_cell={
                                            "fontFamily": "Arial",
                                            "backgroundColor": "#222222",
                                        },
                                        style_header={
                                            "backgroundColor": "#222222",
                                            "fontWeight": "bold",
                                        },
                                        style_table={
                                            "width": "500px",
                                            "height": "95vh",
                                            "flex": "0 0 400px",
                                            "overflow": "auto",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "height": "98vh",
                                },
                            )
                        ],
                        id="dag-tab",
                        style={"display": "none"},
                    ),
                ],
                style={"width": "88%", "height": "100vh", "overflow": "hidden"},
            ),
            dcc.Tooltip(
                id="image-tooltip4",
                direction="right",
                style={"zIndex": 999, "pointerEvents": "none", "overflow": "visible"},
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "width": "100vw",
            "height": "100vh",
        },
    )

    # ================= CALLBACK TO SWITCH TABS =================
    @app.callback(
        [
            Output("state-space-tab", "style"),
            Output("dag-tab", "style"),
            Output("active-tab", "data"),
            Output("tab-state-space", "style"),
            Output("tab-dag-view", "style"),
            Output("sidebar-projection", "style"),
            Output("sidebar-dag", "style"),
            Output("sidebar-tab-header", "children"),
        ],
        [Input("tab-state-space", "n_clicks"), Input("tab-dag-view", "n_clicks")],
        [State("active-tab", "data")],
    )
    def switch_tabs(state_clicks, dag_clicks, current_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = "tab-state-space"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "tab-state-space":
            active_tab = "state-space"
        elif button_id == "tab-dag-view":
            active_tab = "dag-view"
        else:
            active_tab = current_tab

        # Base button styles
        active_style = {
            "border": "2px solid #3b82f6",
            "backgroundColor": "#3b82f6",
            "color": "white",
            "transition": "all 0.3s ease",
        }
        inactive_style = {
            "border": "2px solid #e5e7eb",
            "backgroundColor": "white",
            "color": "#6b7280",
            "transition": "all 0.3s ease",
        }
        top_style = {"border-top-left-radius": "8px", "border-top-right-radius": "8px"}
        bottom_style = {
            "border-bottom-left-radius": "8px",
            "border-bottom-right-radius": "8px",
        }

        # Apply styles based on active tab
        if active_tab == "state-space":

            return (
                {"display": "block"},
                {"display": "none"},
                "state-space",
                active_style | top_style,
                inactive_style | bottom_style,
                {"display": "block"},
                {"display": "none"},
                "Final Objects",
            )
        else:
            return (
                {"display": "none"},
                {"display": "block"},
                "dag-view",
                inactive_style | top_style,
                active_style | bottom_style,
                {"display": "none"},
                {"display": "block"},
                "DAG",
            )

    # Downprojection parameter header
    @app.callback(
        Output("projection-param-label", "children"),
        Output("projection-param", "min"),
        Output("projection-param", "max"),
        Output("projection-param", "step"),
        Output("projection-param", "marks"),
        Output("projection-param", "value"),
        Input("projection-method", "value"),
    )
    def update_projection_param(method):
        if method == "umap":
            return (
                "n_neighbors",
                2,
                200,
                1,
                {2: "2", 50: "50", 100: "100", 200: "200"},
                15,
            )
        else:
            return "perplexity", 5, 50, 1, {5: "5", 25: "25", 50: "50"}, 30

    Input("dag-overview", "selectedData"),

    # Main selection update
    @app.callback(
        Output("selected-objects", "data"),
        Output("dag-table", "data"),
        Output("dag-table", "selected_rows"),
        Input("clear-selection", "n_clicks"),
        Input("state-space-plot", "selectedData"),
        Input("state-space-plot", "clickData"),
        Input("bumpchart", "selectedData"),
        Input("dag-graph", "tapNodeData"),
        Input("dag-overview", "selectedData"),
        State("selected-objects", "data"),
        State("build-ids", "data"),
        State("dag-overview-tid-list", "data"),
        prevent_initial_call=True,
    )
    def update_selected_objects(
        clear_clicks,
        ss_select,
        ss_click,
        bump_select,
        dag_node,
        selected_tids,
        current_ids,
        build_ids,
        tid_list,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        trigger = ctx.triggered[0]["prop_id"]

        # -------- Clear button --------
        if "clear-selection" in trigger:
            return [], None, []

        # ---------- State-space lasso ----------
        if "state-space-plot.selectedData" in trigger:
            if not ss_select or not ss_select.get("points"):
                return no_update

            selected_ids = {pt["customdata"][0] for pt in ss_select["points"]}
            return list(selected_ids), None, []

        # ---------- State-space click ----------
        if "state-space-plot.clickData" in trigger:
            if (
                not ss_click
                or not ss_click["points"]
                or ss_click["points"][0]["customdata"][0] != "usehex"
            ):
                return no_update

            _, hex_q, hex_r, _, _, _ = ss_click["points"][0]["customdata"]
            conn = sqlite3.connect(data_path)
            query = "SELECT id FROM current_dp WHERE hex_q = ? AND hex_r = ?"
            selected_ids = pd.read_sql_query(query, conn, params=[hex_q, hex_r])[
                "id"
            ].tolist()

            return selected_ids, None, []

        # ---------- Bump chart lasso ----------
        elif "bumpchart.selectedData" in trigger:
            if not bump_select or not bump_select.get("points"):
                return no_update, None, []

            selected_ids = {pt["customdata"][0] for pt in bump_select["points"]}
            return list(selected_ids), None, []

        # ---------- DAG node click ----------
        elif "dag-graph.tapNodeData" in trigger:
            if not dag_node:
                return no_update
            if dag_node.get("id") == "#":
                # root selection clears selection
                return [], None, []

            iteration0 = int(dag_node.get("iteration0"))
            iteration1 = int(dag_node.get("iteration1"))

            # expand dag, build table
            if dag_node.get("node_type") == "handler":
                text = dag_node.get("id")[8:]
                conn = sqlite3.connect(data_path)

                if dag_node.get("metric") in ["highest", "lowest", "variance"]:
                    col = "logprobs_" + dag_node.get("direction")
                    query = f"""
                        WITH ranked AS (
                            SELECT
                                target,
                                iteration,
                                {col} AS logprobs,
                                AVG({col}) OVER (PARTITION BY target) AS avg_logprobs,
                                ROW_NUMBER() OVER (
                                    PARTITION BY target
                                    ORDER BY iteration DESC
                                ) AS rn
                            FROM edges
                            WHERE source = ?
                            AND iteration BETWEEN ? AND ?
                        )
                        SELECT
                            target,
                            logprobs AS latest_logprobs,
                            avg_logprobs
                        FROM ranked
                        WHERE rn = 1;
                        """
                else:  # frequency
                    query = """
                        SELECT
                            target,
                            COUNT(*) AS latest_logprobs
                        FROM edges
                        WHERE source = ?
                          AND iteration BETWEEN ? AND ?
                        GROUP BY target;
                        """
                children_e = pd.read_sql_query(
                    query, conn, params=[text, iteration0, iteration1]
                )

                if dag_node.get("metric") == "variance":
                    children_e["latest_logprobs"] -= children_e["avg_logprobs"]
                children_e.rename(
                    columns={"target": "id", "latest_logprobs": "logprobs"},
                    inplace=True,
                )
                children_e = children_e[["id", "logprobs"]]

                targets = list(children_e["id"])
                placeholders = ",".join("?" for _ in targets)
                query = f"""
                            SELECT DISTINCT
                                id, node_type, reward
                            FROM nodes
                            WHERE id IN ({placeholders})
                        """
                children_n = pd.read_sql_query(query, conn, params=targets)
                conn.close()
                children = pd.merge(children_n, children_e, on="id")

                def image_cell(id):
                    state = image_fn(id)
                    if not state:
                        return ""
                    if type(state) is list:
                        return "<br>".join(state)
                    return f'<img src="{state}" ' f'style="height:80px;" />'

                children["image"] = children["id"].apply(image_cell)
                children["node_type"] = children["node_type"].eq("final")
                selected_row_ids = list(
                    set.intersection(set(build_ids), set(list(children["id"])))
                )
                selected_rows = [
                    idx
                    for idx, row in enumerate(children.to_dict("records"))
                    if row["id"] in selected_row_ids
                ]
                return no_update, children.to_dict("records"), selected_rows

            # selected trajectories
            else:
                text = dag_node.get("id")
                conn = sqlite3.connect(data_path)
                query = """
                    SELECT DISTINCT
                        trajectory_id
                    FROM edges
                    WHERE target = ?
                      AND iteration BETWEEN ? AND ?
                """
                selected_ids = pd.read_sql_query(
                    query, conn, params=[text, iteration0, iteration1]
                )
                selected_ids = list(selected_ids["trajectory_id"])
                conn.close()
                return (selected_ids, None, []) if selected_ids else ([], None, [])

        elif "dag-overview.selectedData" in trigger:
            if not selected_tids or "range" not in selected_tids:
                return no_update

            y_range = np.round(selected_tids["range"]["y"]).astype(int)
            t_ids = tid_list[y_range[0] : y_range[1] + 1]
            t_ids = list({elem for sublist in t_ids for elem in sublist})
            iterations = np.round(selected_tids["range"]["x"]).astype(int).tolist()

            # get trajectory ids that are also in iteration range
            # and then all node ids for these trajectory_ids
            conn = sqlite3.connect(data_path)
            placeholders = ",".join("?" for _ in t_ids)
            query = f"""
            SELECT DISTINCT trajectory_id
            FROM edges
            WHERE trajectory_id IN ({placeholders})
                AND iteration BETWEEN ? AND ?
            """
            params = t_ids + [iterations[0], iterations[1]]
            selected_ids = pd.read_sql_query(query, conn, params=params)
            selected_ids = list(set(selected_ids["trajectory_id"].to_list()))
            conn.close()
            return selected_ids, None, []

        return no_update

    # Bump Callback
    @app.callback(
        Output("bumpchart", "figure"),
        Input("iteration", "value"),
        Input("selected-objects", "data"),
        Input("fo-metric", "value"),
        Input("ranking-metric", "value"),
    )
    def bump_callback(iteration, selected_ids, fo_metric, rank_metric):
        n_top = 30
        order = "DESC" if "highest" in rank_metric else "ASC"
        conn = sqlite3.connect(data_path)
        query = f"""
                SELECT iteration, text, metric, final_id, rank
                FROM (
                    SELECT
                        iteration,
                        text,
                        {fo_metric} as metric,
                        final_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY iteration
                            ORDER BY {fo_metric} {order}
                        ) AS rank
                    FROM trajectories
                    WHERE iteration BETWEEN ? AND ?
                    AND final_object = 1
                ) sub
                WHERE rank <= {n_top}
                ORDER BY iteration, rank;
            """
        df = pd.read_sql_query(query, conn, params=iteration)
        conn.close()

        order = "highest" if "highest" in rank_metric else "lowest"
        if "iter" in rank_metric:
            return plotter.update_bump_iter(df, selected_ids, fo_metric, order)
        return plotter.update_bump_all(df, n_top, selected_ids, fo_metric, order)

    # State Space Callback
    @app.callback(
        Output("state-space-plot", "figure"),
        Output("hexbin-size", "data"),
        Input("selected-objects", "data"),
        Input("projection-method", "value"),
        Input("projection-param", "value"),
        Input("iteration", "value"),
        Input("use-testset", "value"),
        Input("fo-metric", "value"),
        Input("state-space-style", "value"),
        State("hexbin-size", "data"),
    )
    def update_projection_plots(
        selected_ids,
        method,
        param_value,
        iteration,
        use_testset,
        metric,
        ss_style,
        hexbin_size,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update
        trigger = ctx.triggered[0]["prop_id"]
        metric_lists = (final_object_metrics, testset_metrics)
        new_hexbin_size = hexbin_size

        # fetch from db
        if (
            trigger == "selected-objects.data"
            or trigger == "fo-metric.value"
            or trigger == "state-space-style.value"
        ):
            conn = sqlite3.connect(data_path)
            if "Hex" in ss_style:
                df = plotter.get_hexbin_data(conn, ss_style, metric)
            else:
                query = f"""
                    SELECT
                        id,
                        x,
                        y,
                        hex_q,
                        hex_r,
                        text,
                        iteration,
                        istestset,
                        {', '.join(final_object_metrics)}
                    FROM current_dp
                """
                df = pd.read_sql_query(query, conn)
            conn.close()
        else:
            # downproject and write to db
            df, new_hexbin_size = plotter.create_dp_table(
                feature_cols,
                iteration,
                use_testset,
                feature_cols_testset,
                method,
                param_value,
                metric_lists,
            )
            if "Hex" in ss_style:
                conn = sqlite3.connect(data_path)
                df = plotter.get_hexbin_data(conn, ss_style, metric)
                conn.close()

        if "Hex" in ss_style:
            return (
                plotter.update_hex(
                    df, selected_ids, ss_style, metric, use_testset, new_hexbin_size
                ),
                new_hexbin_size,
            )
        return plotter.update_state_space(df, selected_ids, metric), new_hexbin_size

    # DAG Callback
    @app.callback(
        [
            Output("dag-graph", "elements"),
            Output("dag-graph", "stylesheet"),
            Output("dag-graph", "layout"),
            Output("dag-title", "children"),
            Output("dag-subtitle", "children"),
        ],
        Input("dag-layout", "value"),
        Input("dag-direction", "value"),
        Input("dag-metric", "value"),
        Input("iteration", "value"),
        Input("selected-objects", "data"),
        Input("build-ids", "data"),
        Input("max-frequency", "data"),
    )
    def update_dag(
        layout_name, direction, metric, iteration, selected_objects, build_ids, max_freq
    ):
        add_handlers = True
        if selected_objects:
            # If final objects are selected via another vis,
            # display the full dag of these
            conn = sqlite3.connect(data_path)
            placeholders = ",".join("?" for _ in selected_objects)
            query = f"""
                        SELECT DISTINCT
                            target
                        FROM edges
                        WHERE trajectory_id IN ({placeholders})
                          AND iteration BETWEEN ? AND ?
                    """
            build_ids = pd.read_sql_query(
                query, conn, params=selected_objects + [iteration[0], iteration[1]]
            )
            build_ids = list(build_ids["target"]) + ["#"]
            conn.close()
            add_handlers = False
        elif not build_ids:
            build_ids = ["#"]

        result = plotter.update_DAG(
            iteration,
            direction,
            metric,
            max_freq,
            add_handlers,
            build_ids=build_ids,
        )

        # Configure layout based on selection
        layout_config = {"name": layout_name, "directed": True, "animate": False}

        # Add layout-specific parameters
        if layout_name == "klay":
            layout_config["spacingFactor"] = 1.2
        elif layout_name == "dagre":
            layout_config["spacingFactor"] = 1
            layout_config["rankDir"] = "LR"  # Left to right
        elif layout_name == "breadthfirst":
            layout_config["spacingFactor"] = 1.2
            layout_config["roots"] = '[id = "#"]'

        if add_handlers:
            title = "Directed Acyclic Graph, Mode: Expand"
            subtitle = (
                "Click on 'Select children' nodes to expand the Graph"
                "and click on the root to collapse it."
                "Select a node or items from other visuals to switch to selection mode."
                "Edge coloring: "
            )
        else:
            title = "Directed Acyclic Graph, Mode: Selection"
            subtitle = (
                "hows all trajectories going through the selected items."
                "Clear selection or select the root to switch to expanding mode."
                "Edge coloring: "
            )
        if metric in ["highest", "lowest"]:
            subtitle += (
                f"{metric.capitalize()} {direction} logprobabilities of the edge"
                "over selected iterations."
            )
        elif metric == "variance":
            subtitle += (
                "Latest {direction} logprobability"
                "of the edge (in selected iterations) -"
                f"mean ({direction} logprobabilities) over selected iterations."
            )
        elif metric == "frequency":
            subtitle += "Frequency of the edge over selected iterations."
        return result["elements"], result["stylesheet"], layout_config, title, subtitle

    # Callback for dag-table
    @app.callback(
        Output("build-ids", "data"),
        Input("dag-table", "selected_row_ids"),
        Input("dag-graph", "tapNodeData"),
        State("dag-table", "data"),
        State("build-ids", "data"),
    )
    def save_selected_rows(selected_rows, node_select, table_data, build_ids):
        # reset build ids if root selected
        if "dag-graph.tapNodeData" in dash.callback_context.triggered[0]["prop_id"]:
            if node_select.get("id") == "#":
                return ["#"]
            else:
                return no_update
        # update build ids from table
        if selected_rows or table_data:
            children = set([r["id"] for r in table_data])
            unselected = children - set(selected_rows)
            new_build_ids = set(build_ids) - unselected
            if len(new_build_ids) < len(build_ids):
                removed_node = set(build_ids) - new_build_ids
                assert len(removed_node) == 1, "Bug in DAG selection"
                build_ids = plotter.dag_remove_node_prune(
                    build_ids, list(removed_node)[0]
                )
            build_ids = set(selected_rows) | set(build_ids) | set(["#"])
            return list(build_ids)
        else:
            return ["#"]

    # dag overview
    @app.callback(
        Output("dag-overview", "figure"),
        Output("max-frequency", "data"),
        Output("dag-overview-tid-list", "data"),
        Output("dag-overview-edge-list", "data"),
        Input("dag-direction", "value"),
        Input("dag-metric", "value"),
        Input("iteration", "value"),
        Input("dag-overview-page", "data"),
    )
    def update_dag_overview(direction, metric, iteration, page):
        fig, max_freq, ids, edge_list = plotter.update_DAG_overview(
            direction, metric, iteration, page
        )
        if max_freq:
            return fig, max_freq, ids, edge_list
        return fig, no_update, ids, edge_list

    # hover state space
    @app.callback(
        Output("image-tooltip1", "show"),
        Output("image-tooltip1", "bbox"),
        Output("image-tooltip1", "children"),
        Input("state-space-plot", "hoverData"),
        State("fo-metric", "value"),
        State("state-space-style", "value"),
        State("use-testset", "value"),
    )
    def display_image_tooltip1(hoverData, metric, ss_style, usetestset):
        if hoverData is None:
            return False, None, None
        bbox = hoverData["points"][0]["bbox"]
        customdata = hoverData["points"][0]["customdata"]
        if customdata is None:
            return False, None, None

        if customdata[0] == "usehex":
            _, hex_q, hex_r, metric_value, n_samples, n_test = customdata
            if ss_style == "Hex Ratio":
                if usetestset:
                    metric_title = f"Log Odds Ratio: {metric_value:.2f}"
                else:
                    metric_title = ""
            elif ss_style == "Hex Obj. Metric":
                metric_title = f"Average {metric} (Samples): {metric_value:.4f}"
            else:
                if metric_value is None or np.isnan(metric_value):
                    metric_title = "Not enough samples for Correlation"
                else:
                    metric_title = (
                        f"Correlation log reward and logprobs: {metric_value:.2f}"
                    )

            figures, texts = plotter.hex_hover_figures(
                hex_q, hex_r, metric, ss_style, metric in testset_metrics, usetestset
            )
            fig_graphs = [
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False},
                    style={"marginTop": "20px"},
                )
                for fig in figures
                if fig is not None
            ]
            aggregation = plotter.agg_fn(texts)
            if isinstance(aggregation, list):
                aggregation = [
                    html.Div(i, style={"color": "black", "marginTop": "5px"})
                    for i in aggregation
                ]
            elif isinstance(aggregation, str):
                aggregation = html.Img(
                    src=aggregation, style={"width": "150px", "height": "150px"}
                )

            children = [
                html.Div(
                    [
                        *fig_graphs,
                        html.Div(
                            [
                                html.Div(aggregation, style={"marginRight": "15px"}),
                                html.Div(
                                    [
                                        html.Div(
                                            f" Unique Samples: {n_samples}",
                                            style={
                                                "color": "black",
                                                "marginTop": "5px",
                                            },
                                        ),
                                        html.Div(
                                            f" Testset Objects: {n_test}",
                                            style={
                                                "color": "black",
                                                "marginTop": "5px",
                                            },
                                        ),
                                        html.Div(
                                            metric_title,
                                            style={
                                                "color": "black",
                                                "marginTop": "5px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",  # vertical alignment
                            },
                        ),
                    ]
                )
            ]

        else:
            _, iteration, metric_data, text = customdata
            metric_data = float("nan") if metric_data is None else metric_data
            if ss_style == "Hex Ratio":
                texts = [
                    html.Div(f"Iteration: {iteration}", style={"color": "black"}),
                ]
            else:
                texts = [
                    html.Div(f"Iteration: {iteration}", style={"color": "black"}),
                    html.Div(f"{metric}: {metric_data:.4f}", style={"color": "black"}),
                ]

            children = [html.Div([*plotter.html_from_imagefn(text), *texts])]

        return True, bbox, children

    # hover bump plot
    @app.callback(
        Output("image-tooltip3", "show"),
        Output("image-tooltip3", "bbox"),
        Output("image-tooltip3", "children"),
        Input("bumpchart", "hoverData"),
        State("fo-metric", "value"),
    )
    def display_image_tooltip3(hoverData, fo_metric):
        if hoverData is None:
            return False, None, None

        point = hoverData["points"][0]

        # Check if this point has customdata (skip shading area)
        if "customdata" not in point or point["customdata"] is None:
            return False, None, None

        bbox = point["bbox"]
        _, rank, metric, text = point["customdata"]

        children = [
            html.Div(
                [
                    *plotter.html_from_imagefn(text),
                    html.Div(f"Rank: {rank}", style={"color": "black"}),
                    html.Div(f"{fo_metric}: {metric:.4f}", style={"color": "black"}),
                ]
            )
        ]

        return True, bbox, children

    @app.callback(
        Output("image-tooltip4", "show"),
        Output("image-tooltip4", "bbox"),
        Output("image-tooltip4", "children"),
        Input("dag-overview", "hoverData"),
        State("dag-overview-edge-list", "data"),
    )
    def display_image_tooltip4(hoverData, edge_list):
        if hoverData is None or hoverData["points"][0]["z"] is None:
            return False, None, None

        value = hoverData["points"][0]["z"]
        bbox = hoverData["points"][0]["bbox"]
        idx = hoverData["points"][0]["y"]
        source, target = edge_list[idx]

        # get data
        conn = sqlite3.connect(data_path)
        query = """
            SELECT DISTINCT
                iteration,
                logprobs_forward,
                logprobs_backward
            FROM edges
            WHERE source = ? AND target = ?
        """
        edge_data = pd.read_sql_query(query, conn, params=[source, target])
        conn.close()

        # make images
        source_img = plotter.html_from_imagefn(source)
        target_img = plotter.html_from_imagefn(target)

        def make_img(img):
            if img is None:
                return html.Div("s0")
            return img

        children = html.Div(
            [
                html.Div(f"Value: {value}", style={"fontWeight": "bold"}),
                html.Div(
                    [
                        html.Div(
                            [html.Div("Source:"), *make_img(source_img)],
                            style={
                                "marginTop": "5px",
                                "marginRight": "50px",
                                "display": "flex",
                                "alignItems": "center",
                                "flexDirection": "column",
                                "minHeight": "80px",
                                "justifyContent": "flex-start",
                            },
                        ),
                        html.Div(
                            [html.Div("Target:"), *make_img(target_img)],
                            style={"marginTop": "5px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "marginTop": "5px",
                        "alignItems": "flex-start",
                    },
                ),
                dcc.Graph(
                    figure=plotter.edge_hover_fig(edge_data),
                    config={"displayModeBar": False},
                    style={"marginTop": "10px"},
                ),
            ],
            style={
                "color": "black",
                "backgroundColor": "white",
                "padding": "10px",
                "height": "auto",
            },
        )

        return True, bbox, children

    @app.callback(
        Output("dag-overview-page", "data"),
        Output("dag-overview-button-top", "disabled"),
        Output("dag-overview-button-prev", "disabled"),
        Output("dag-overview-button-next", "disabled"),
        Input("dag-overview-button-top", "n_clicks"),
        Input("dag-overview-button-next", "n_clicks"),
        Input("dag-overview-button-prev", "n_clicks"),
        Input("iteration", "value"),
        Input("dag-metric", "value"),
        State("dag-overview-page", "data"),
    )
    def dag_overview_buttons(top, nxt, prev, iterations, metric, page):
        """Dag overview buttons."""
        trigger = dash.callback_context.triggered[0]["prop_id"]
        if "top" in trigger or "iteration" in trigger or "metric" in trigger:
            page = 0
        elif "prev" in trigger:
            page = max(page - 1, 0)
        elif "next" in trigger:
            page += 1
        disabled_top = page == 0
        conn = sqlite3.connect(data_path)
        if metric == "variance":
            query = """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT source, target
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    HAVING COUNT(*) > 1
                ) t;
            """
        else:
            query = """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT DISTINCT source, target
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                );
            """

        n_edges = pd.read_sql_query(query, conn, params=iterations)["count"].tolist()[0]
        disabled_bottom = (page + 1) * 150 >= n_edges
        return page, disabled_top, disabled_top, disabled_bottom

    # Run the dashboard
    app.run(debug=debug_mode)
