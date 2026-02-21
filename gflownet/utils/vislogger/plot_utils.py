import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots
from sklearn import manifold
from umap import UMAP


class Plotter:

    def __init__(self, data, image_fn, state_aggregation_fn, s0):
        """Init datapath, functions and colorscales."""
        self.data = data
        self.s0 = s0
        self.image_fn = image_fn
        self.agg_fn = state_aggregation_fn

        # colorscales
        self.cs_main = px.colors.sequential.YlGn
        self.cs_iteration = px.colors.sequential.Teal
        self.cs_diverging_testset = px.colors.diverging.PRGn
        self.cs_diverging_edgechange = px.colors.diverging.PiYG
        self.cs_diverging_dir = px.colors.diverging.balance_r

    def html_from_imagefn(self, text):
        """
        Create the html for a hover when passing a text to the imagefn
        :param text:
        :return:
        """
        out = self.image_fn(text)
        if type(out) is list:
            out = [
                html.Div(i, style={"color": "black", "marginTop": "5px"}) for i in out
            ]
        elif type(out) is str:
            out = [html.Img(src=out, style={"width": "150px", "height": "150px"})]
        return out

    def hex_distplot(self, data, testdata, name):
        """Plot the histogram for hover on hexbins."""
        x_min, x_max = np.inf, -np.inf
        if len(data) > 0:
            x_min = min(x_min, data.min())
            x_max = max(x_max, data.max())
        if testdata is not None and len(testdata) > 0:
            x_min = min(x_min, testdata.min())
            x_max = max(x_max, testdata.max())
        if x_min == np.inf:
            return None
        if x_max == x_min:
            delta = 0.5 if x_min == 0 else abs(x_min) * 0.05 or 0.5
            x_min -= delta
            x_max += delta

        fig = go.Figure()
        if len(data) > 0 and testdata is not None and len(testdata) > 0:
            opacity = 0.6
        else:
            opacity = 1

        if len(data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name="Samples",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    opacity=opacity,
                    nbinsx=30,
                    autobinx=False,
                    xbins=dict(start=x_min, end=x_max, size=(x_max - x_min) / 20),
                    histnorm="probability density",
                )
            )

        if testdata is not None and len(testdata) > 0:
            fig.add_trace(
                go.Histogram(
                    x=testdata,
                    name="Testset Objects",
                    marker=dict(color=self.cs_diverging_testset[0]),
                    opacity=opacity,
                    nbinsx=30,
                    autobinx=False,
                    xbins=dict(start=x_min, end=x_max, size=(x_max - x_min) / 20),
                    histnorm="probability density",
                )
            )

        fig.update_layout(
            title=f"Histogram of {name}",
            xaxis_title=name,
            yaxis_title="Probability Density",
            template="plotly_white",
            height=200,
            width=450,
            margin=dict(l=20, r=20, t=30, b=20),
            barmode="overlay",
            showlegend=True,
        )
        fig.update_xaxes(range=[x_min, x_max])

        return fig

    def hex_hover_figures(
        self, hex_q, hex_r, metric, ss_style, metric_in_testset, usetestset
    ):
        """Calculate the figures for the tooltips for the hexbins."""
        lossfig = None
        metricfig = None

        # get texts
        conn = sqlite3.connect(self.data)
        query = f"""
            SELECT text
            FROM current_dp
            WHERE hex_q = {hex_q} AND hex_r = {hex_r} AND istestset = 0
        """
        texts = pd.read_sql_query(query, conn)["text"].tolist()
        placeholders = ",".join("?" for _ in texts)
        if usetestset:
            query = f"""
                SELECT text
                FROM current_dp
                WHERE hex_q = {hex_q} AND hex_r = {hex_r} AND istestset = 1
            """
            texts_t = pd.read_sql_query(query, conn)["text"].tolist()
            placeholders_t = ",".join("?" for _ in texts_t)

        # get loss table
        query = f"""
                        SELECT
                            iteration,
                            AVG(loss) AS mean,
                            MIN(loss) AS min,
                            MAX(loss) AS max
                        FROM trajectories
                        WHERE final_object = 1
                        AND text in ({placeholders})
                        GROUP BY iteration
                    """
        loss_df = pd.read_sql_query(query, conn, params=texts)

        # get reward data
        query = f"""
            SELECT total_reward
            FROM trajectories
            WHERE final_object = 1
            AND text in ({placeholders})
        """
        rewards_samples = (
            pd.read_sql_query(query, conn, params=texts).to_numpy().flatten()
        )
        if usetestset:
            query = f"SELECT total_reward FROM testset WHERE text in ({placeholders_t})"
            rewards_testset = (
                pd.read_sql_query(query, conn, params=texts_t).to_numpy().flatten()
            )

        # if metric custom metric is choosen, give dist as well
        if (
            ss_style == "Hex Obj. Metric"
            and metric != "total_reward"
            and metric != "loss"
        ):
            query = f"""
                SELECT {metric}
                FROM trajectories
                WHERE final_object = 1 AND text in ({placeholders})
            """
            metric_samples = (
                pd.read_sql_query(query, conn, params=texts).to_numpy().flatten()
            )
            if metric_in_testset and usetestset:
                query = f"SELECT {metric} FROM testset WHERE text in ({placeholders_t})"
                metric_testset = (
                    pd.read_sql_query(query, conn, params=texts_t).to_numpy().flatten()
                )

        # Use metric plot for correlation scatter
        if ss_style == "Hex Correlation":
            query = f"""
                SELECT id
                FROM current_dp
                WHERE hex_q = {hex_q} AND hex_r = {hex_r} AND istestset = 0
            """
            ids = pd.read_sql_query(query, conn)["id"].tolist()
            placeholders = ",".join("?" for _ in ids)
            query = f"""
                SELECT
                    SUM(total_reward) AS reward,
                    SUM(logprobs_forward) AS pf
                FROM trajectories
                WHERE final_id IN ({placeholders})
                GROUP BY final_id
            """
            corr_df = pd.read_sql_query(query, conn, params=ids)
            corr_df["reward"] = np.log(corr_df["reward"])
            # check if correlation is similar
            # print("corr", corr_df["reward"].corr(corr_df["pf"]))
        conn.close()

        # create loss figure
        if len(loss_df) != 0:
            lossfig = go.Figure()
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["max"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=self.cs_diverging_testset[-1]
                    .replace("rgb", "rgba")
                    .replace(")", ", 0.2)"),
                    opacity=0.2,
                    line=dict(width=0),
                    showlegend=True,
                    name="Range",
                )
            )
            lossfig.add_trace(
                go.Scatter(
                    x=loss_df["iteration"],
                    y=loss_df["mean"],
                    mode="lines" if len(loss_df) > 1 else "markers",
                    line=dict(color=self.cs_diverging_testset[-1], width=2),
                    marker=dict(size=8, color=self.cs_diverging_testset[-1]),
                    name="Mean",
                    showlegend=True,
                )
            )
            lossfig.update_layout(
                title="Average Loss per Iteration (Samples)",
                xaxis_title="Iteration",
                yaxis_title="Loss",
                template="plotly_white",
                height=200,
                width=450,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(nticks=4),
            )
            lossfig.update_xaxes(range=[0, loss_df["iteration"].max() * 1.05])
            lossfig.update_yaxes(range=[0, loss_df["max"].max() * 1.05])

        # create reward distribution
        if usetestset and len(rewards_testset) != 0:
            rewardfig = self.hex_distplot(
                rewards_samples, rewards_testset, "Total Reward"
            )
        else:
            rewardfig = self.hex_distplot(rewards_samples, None, "Total Reward")

        if (
            ss_style == "Hex Obj. Metric"
            and metric != "total_reward"
            and metric != "loss"
        ):
            if metric_in_testset and usetestset and len(metric_testset) != 0:
                metricfig = self.hex_distplot(metric_samples, metric_testset, metric)
            else:
                metricfig = self.hex_distplot(metric_samples, None, metric)
        if ss_style == "Hex Correlation":
            metricfig = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.15,
            )
            metricfig.add_trace(
                go.Scatter(
                    x=corr_df["reward"],
                    y=corr_df["pf"],
                    mode="markers",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            metricfig.add_trace(
                go.Scatter(
                    x=np.exp(corr_df["reward"]),
                    y=np.exp(corr_df["pf"]),
                    mode="markers",
                    marker=dict(color=self.cs_diverging_testset[-1]),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            metricfig.update_layout(
                title="(Log) Reward vs Forward (Log) Probabilities for Samples",
                template="plotly_white",
                height=200,
                width=550,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            metricfig.update_xaxes(row=1, col=1, title_text=" Log Reward")
            metricfig.update_yaxes(row=1, col=1, title_text="Logprobabilities")
            metricfig.update_xaxes(row=1, col=2, title_text="Reward")  # , range=[,])
            metricfig.update_yaxes(row=1, col=2, title_text="Probabilities")

        if usetestset:
            texts += texts_t
        return (lossfig, rewardfig, metricfig), texts

    def update_hex(self, df, selected_ids, ss_style, metric, usetestset, size=8.0):
        """Update the state space plot for hex style."""
        fig = go.Figure()

        # colorscales and titles
        colorscale = self.cs_main
        title = "State Space - "
        metric_min = df["metric"].min()
        metric_max = df["metric"].max()
        if ss_style == "Hex Ratio":
            if usetestset:
                colorscale = self.cs_diverging_testset
                metric_max = 1
                metric_min = -1
                legend_title = "Score"
                title += (
                    "Log Odds Ratio of Samples to Testset Objects (Scaled to [-1,1])"
                )
                title += (
                    "<br><sup>"
                    "1: Only Testset Objects. -1: Only Samples."
                    "0: Ratio Samples/Testset Objects Is the Same as the Global Ratio."
                    "</sup>"
                )
            else:
                legend_title = "N Samples"
                title += "Number of Sampled Objects"
        elif ss_style == "Hex Obj. Metric":
            legend_title = f"Mean {metric}"
            title += f"Mean {metric} of the Sampled Objects"
        elif ss_style == "Hex Correlation":
            metric_max = 1
            metric_min = 0
            legend_title = "R"
            title += (
                "Correlation Between the Sum of the Forward Logprobabilities and"
                "Reward for bins with > 10 Samples"
            )
        else:
            raise NotImplementedError("Unknown ss_style")
        title += "<br><sup>Select a hex or hover over it to see its details</sup>"

        def hex_corners(x, y, s):
            angles = np.deg2rad(np.arange(0, 360, 60) - 30)  # pointy-top
            return (
                x + s * np.cos(angles),
                y + s * np.sin(angles),
            )

        def map_color(v, vmin, vmax, cs, opacity):
            if v is None:
                return "black"
            if np.isnan(v) or v == float("nan"):
                return "grey"
            if v < vmin:
                idx = 0
            elif v > vmax:
                idx = -1
            else:
                norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                idx = int(norm * (len(cs) - 1))
            return cs[idx].replace("rgb", "rgba").replace(")", f", {opacity})")

        if selected_ids:
            # get texts first as ids in dp table might not be the same
            # (multiple final objects)
            conn = sqlite3.connect(self.data)
            placeholders = ",".join("?" for _ in selected_ids)
            query = f"""
                SELECT DISTINCT text
                FROM trajectories
                WHERE final_object = 1 AND final_id IN ({placeholders})
            """
            selected_texts = pd.read_sql_query(query, conn, params=selected_ids)[
                "text"
            ].tolist()
            if usetestset:
                query = (
                    f"SELECT DISTINCT text FROM testset WHERE id IN ({placeholders})"
                )
                selected_texts_t = pd.read_sql_query(query, conn, params=selected_ids)[
                    "text"
                ].tolist()
                selected_texts = selected_texts + selected_texts_t
            placeholders = ",".join("?" for _ in selected_texts)
            query = f"""
                SELECT id, text, iteration, {metric} AS metric, x, y, hex_q, hex_r
                FROM current_dp
                WHERE text in ({placeholders})
            """
            selected_df = pd.read_sql_query(query, conn, params=selected_texts)
            if ss_style == "Hex Ratio":
                selected_df["metric"] = np.where(selected_df["id"] >= 0, 1, -1)
            # check if all selected ids are in one hex:
            # in this case zoom in and disable hex hover
            zoom = (
                selected_df["hex_q"].nunique() == 1
                and selected_df["hex_r"].nunique() == 1
            )
            if zoom:
                cx = (
                    size
                    * np.sqrt(3)
                    * (selected_df["hex_q"][0] + selected_df["hex_r"][0] / 2)
                )
                cy = size * 1.5 * selected_df["hex_r"][0]
                zoom_corners = hex_corners(cx, cy, size)
                offset = size * 0.2
                zoom_xrange = [
                    zoom_corners[0].min() - offset,
                    zoom_corners[0].max() + offset,
                ]
                zoom_yrange = [
                    zoom_corners[1].min() - offset,
                    zoom_corners[1].max() + offset,
                ]

        hex_opacity = 0.3 if selected_ids else 1
        for _, row in df.iterrows():
            cx = size * np.sqrt(3) * (row["hex_q"] + row["hex_r"] / 2)
            cy = size * 1.5 * row["hex_r"]
            xs, ys = hex_corners(cx, cy, size)
            customdata = [
                "usehex",
                row["hex_q"],
                row["hex_r"],
                row["metric"],
                row["n_samples"],
                row["n_test"],
            ]

            fig.add_trace(
                go.Scatter(
                    x=np.append(xs, xs[0]),
                    y=np.append(ys, ys[0]),
                    mode="lines",
                    fill="toself",
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                    fillcolor=map_color(
                        row["metric"], metric_min, metric_max, colorscale, hex_opacity
                    ),
                    hoverinfo="none",
                    hovertemplate=None,
                    customdata=customdata,
                    showlegend=False,
                )
            )

        # selected ids scatter
        if selected_ids:
            selected_df_s = selected_df[selected_df["id"] >= 0]
            fig.add_trace(
                go.Scatter(
                    x=selected_df_s["x"],
                    y=selected_df_s["y"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=selected_df_s["metric"],
                        colorscale=colorscale,
                        line=dict(color="white", width=1),
                        showscale=False,
                        cmin=metric_min,
                        cmax=metric_max,
                    ),
                    customdata=selected_df_s[
                        ["id", "iteration", "metric", "text"]
                    ].values,
                    hoverinfo="none",
                    name="Selected Samples",
                )
            )
            if usetestset:
                selected_df_t = selected_df[selected_df["id"] < 0]
                selected_df_t["color"] = -1
                fig.add_trace(
                    go.Scatter(
                        x=selected_df_t["x"],
                        y=selected_df_t["y"],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=selected_df_t["color"],
                            colorscale=self.cs_diverging_testset,
                            line=dict(color="white", width=1),
                            showscale=False,
                            cmin=metric_min,
                            cmax=metric_max,
                        ),
                        customdata=selected_df_t[
                            ["id", "iteration", "metric", "text"]
                        ].values,
                        hoverinfo="none",
                        name="Testset",
                    )
                )

        # dummy trace for legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=0,
                    color=[metric_min, metric_max],
                    colorscale=colorscale,
                    cmin=metric_min,
                    cmax=metric_max,
                    showscale=True,
                    colorbar=dict(title=legend_title, len=0.9),
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True,
            template="plotly_dark",
            title=title,
            modebar_remove=["lasso2d", "select2d"],
        )

        if selected_ids and zoom:
            fig.update_xaxes(range=zoom_xrange)
            fig.update_yaxes(range=zoom_yrange)

        return fig

    def get_hexbin_data(self, conn, method, metric):
        """Get the data for the hex plots."""
        if method == "Hex Ratio":
            query = """
                SELECT
                    hex_r,
                    hex_q,
                    SUM(CASE WHEN istestset = 1 THEN 1 ELSE 0 END) AS n_test,
                    SUM(CASE WHEN istestset = 0 THEN 1 ELSE 0 END) AS n_samples
                FROM current_dp
                GROUP BY
                    hex_r,
                    hex_q;
                """
            df = pd.read_sql_query(query, conn)
            if df["n_test"].sum() == 0:  # no testset used, return frequency
                df["metric"] = df["n_samples"]
            else:
                # Log Odds ratio scaled to [-1,1]
                # Identical to tanh(log((samples/test)/(sum(samples)/sum(test))))
                eps = 0.0001
                ratio = (df["n_samples"].sum() + eps) / (df["n_test"].sum() + eps)
                df["metric"] = (df["n_samples"] + eps) / (df["n_test"] + eps)
                df["metric"] = (df["metric"] ** 2 - ratio**2) / (
                    df["metric"] ** 2 + ratio**2
                )
        elif method == "Hex Obj. Metric":
            query = f"""
                SELECT
                    hex_r,
                    hex_q,
                    SUM(CASE WHEN istestset = 1 THEN 1 ELSE 0 END) AS n_test,
                    SUM(CASE WHEN istestset = 0 THEN 1 ELSE 0 END) AS n_samples,
                    COALESCE(
                        AVG(CASE WHEN istestset = 0 THEN {metric} END),
                        0
                    ) AS metric
                FROM current_dp
                GROUP BY
                    hex_r,
                    hex_q;
            """
            df = pd.read_sql_query(query, conn)
        elif method == "Hex Correlation":
            query = """
                SELECT
                    hex_r,
                    hex_q,
                    SUM(CASE WHEN istestset = 1 THEN 1 ELSE 0 END) AS n_test,
                    SUM(CASE WHEN istestset = 0 THEN 1 ELSE 0 END) AS n_samples,
                    SUM(CASE WHEN istestset = 0 THEN LOG(total_reward) END) AS sum_a,
                    SUM(CASE WHEN istestset = 0 THEN pf END)                AS sum_b,
                    SUM(CASE WHEN
                            istestset = 0
                        THEN LOG(total_reward)*LOG(total_reward) END)       AS sum_a2,
                    SUM(CASE WHEN istestset = 0 THEN pf*pf END)             AS sum_b2,
                    SUM(CASE WHEN
                            istestset = 0
                        THEN LOG(total_reward)*pf END)                      AS sum_ab
                FROM current_dp
                GROUP BY
                    hex_r,
                    hex_q;
            """
            df = pd.read_sql_query(query, conn)
            n = df["n_samples"]
            num = n * df["sum_ab"] - df["sum_a"] * df["sum_b"]
            den = np.sqrt(
                (n * df["sum_a2"] - df["sum_a"] ** 2)
                * (n * df["sum_b2"] - df["sum_b"] ** 2)
            )
            df["metric"] = num / den
            df.loc[df["n_samples"] <= 10, "metric"] = np.nan
            df = df.drop(columns=["sum_a2", "sum_b2", "sum_a", "sum_b", "sum_ab"])

        else:
            raise NotImplementedError("Hexbin Data Method Not Implemented")
        return df

    @staticmethod
    def calculate_hexbins(df, size=8.0):
        """Assign pointy-top hex axial coordinates (q, r) to points in df.

        Parameters
        ----------
        df :
            Df with columns x and y.
        size :
            Center to corner hex radius. By default, 8.0.
        """

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        # Axial coordinates (fractional)
        q = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
        r = (2 / 3 * y) / size

        # Convert to cube coordinates
        cx = q
        cz = r
        cy = -cx - cz
        rx = np.round(cx)
        ry = np.round(cy)
        rz = np.round(cz)
        dx = np.abs(rx - cx)
        dy = np.abs(ry - cy)
        dz = np.abs(rz - cz)
        mask_x = (dx > dy) & (dx > dz)
        mask_y = (~mask_x) & (dy > dz)
        mask_z = ~(mask_x | mask_y)
        rx[mask_x] = -ry[mask_x] - rz[mask_x]
        ry[mask_y] = -rx[mask_y] - rz[mask_y]
        rz[mask_z] = -rx[mask_z] - ry[mask_z]

        # Axial coordinates are (q, r) = (x, z)
        return rx.astype(np.int64), rz.astype(np.int64)

    def create_dp_table(
        self,
        feature_cols,
        iteration,
        use_testset,
        feature_cols_testset=None,
        method="tsne",
        param_value=15,
        metric_lists=([], []),
    ):
        """Create dp table."""
        conn = sqlite3.connect(self.data)

        # Get logged data (unique texts, latest iteration)
        query = f"""
            SELECT
                final_id AS id,
                text,
                {", ".join(feature_cols)},
                {", ".join(metric_lists[0])},
                iteration,
                pf,
                features_valid
            FROM (
                SELECT
                    final_id,
                    text,
                    {", ".join(feature_cols)},
                    {", ".join(metric_lists[0])},
                    iteration,
                    final_object,
                    SUM(logprobs_forward) OVER (
                        PARTITION BY final_id
                    ) AS pf,
                    features_valid,
                    ROW_NUMBER() OVER (
                        PARTITION BY text
                        ORDER BY iteration DESC
                    ) AS rn
                FROM trajectories
                WHERE iteration BETWEEN ? AND ?
                AND final_object = 1
            )
            WHERE rn = 1
        """
        logged = pd.read_sql_query(query, conn, params=iteration)
        logged = logged[logged["features_valid"] == 1]
        query = """
            SELECT COUNT(*) AS count
            FROM trajectories
            WHERE final_object = 1
            AND iteration BETWEEN ? AND ?
            AND features_valid = 0
        """
        non_valid = pd.read_sql_query(query, conn, params=iteration)["count"][0]
        if non_valid:
            print(f"""
                {non_valid} objects of the logged data have not been
                downprojected due to invalid features.
                """)

        features = logged[feature_cols].to_numpy()
        df_dp = logged.drop(columns=feature_cols)

        # Get testset data
        testset_hasdata = pd.read_sql_query("SELECT COUNT(*) FROM testset", conn)
        testset_hasdata = testset_hasdata["COUNT(*)"].tolist()[0]
        if use_testset and testset_hasdata:
            query = f"""
                SELECT
                    id,
                    text,
                    {", ".join(feature_cols_testset)},
                    {", ".join(metric_lists[1])}
                FROM testset
                WHERE features_valid = 1
            """
            testset = pd.read_sql_query(query, conn)
            query = "SELECT COUNT(*) AS count FROM testset WHERE features_valid = 0"
            non_valid_t = pd.read_sql_query(query, conn)["count"][0]
            if non_valid_t:
                print(f"""
                    {non_valid_t} objects of the testset have not been
                    downprojected due to invalid features.
                """)

            # concat features
            features_t = testset[feature_cols_testset].to_numpy()
            testset = testset.drop(columns=feature_cols_testset)
            assert features.shape[1] == features_t.shape[1], f"""
                    Testset and Logged data have a different amout of features.\n
                    Testset: {feature_cols_testset}
                    Logged: {feature_cols}
                """
            features = np.concatenate((features, features_t), axis=0)
            df_dp = pd.concat([df_dp, testset], axis=0, ignore_index=True)

        # Downprojection
        if features.shape[1] <= 1:
            raise ValueError("Not enough features")
        elif features.shape[1] == 2:
            proj_s = features
        elif method == "tsne":
            proj_s = manifold.TSNE(
                perplexity=min(param_value, features.shape[0] - 1),
                init="pca",
                learning_rate="auto",
            ).fit_transform(features)

        elif method == "umap":
            reducer_s = UMAP(n_neighbors=min(param_value, features.shape[0] - 1))
            proj_s = reducer_s.fit_transform(features)

        else:
            raise NotImplementedError("Method not implemented")

        df_dp["x"] = proj_s[:, 0]
        df_dp["y"] = proj_s[:, 1]
        df_dp["istestset"] = df_dp["id"] < 0

        # calc hexbins
        hexbin_size = float(max(proj_s.max(axis=0) - proj_s.min(axis=0))) / 16
        df_dp["hex_q"], df_dp["hex_r"] = self.calculate_hexbins(df_dp, size=hexbin_size)

        # write
        df_dp.to_sql("current_dp", conn, if_exists="replace", index=False)
        conn.close()

        return df_dp, hexbin_size

    def update_DAG(
        self,
        iteration,
        direction="forward",
        metric="highest",
        max_freq=0,
        add_handlers=True,
        build_ids=[],
    ):
        """Updates the DAG based on the given metric and direction.

        Parameters
        ----------
        iteration :
            Iteration range.
        direction :
            Forward/backward. By default, "forward".
        metric :
            Lowest, highest, variance, frequency. By default, "highest".
        max_freq :
            Highest frequency in the data. By default, 0.
        add_handlers :
            Add handlers if in expanding mode, dont add in selection mode.
            By default, True.
        build_ids :
            States to build the dag from. By default, [].
        """

        conn = sqlite3.connect(self.data)
        placeholders = ",".join("?" for _ in build_ids)
        column = "logprobs_" + direction
        base_where = f"""
            source IN ({placeholders})
            AND target IN ({placeholders})
            AND iteration BETWEEN ? AND ?
        """
        params = build_ids + build_ids + [iteration[0], iteration[1]]

        if metric in ["highest", "lowest"]:
            query = f"""
                WITH edge_ids AS (
                    SELECT
                        source,
                        target,
                        GROUP_CONCAT(id, '-') AS id
                    FROM edges
                    WHERE {base_where}
                    GROUP BY source, target
                )
                SELECT
                    ei.id,
                    e.source,
                    e.target,
                    {"MAX" if metric == "highest" else "MIN"}(e.{column}) AS metric
                FROM edges e
                JOIN edge_ids ei
                  ON ei.source = e.source
                 AND ei.target = e.target
                WHERE
                    e.source IN ({placeholders})
                    AND e.target IN ({placeholders})
                    AND e.iteration BETWEEN ? AND ?
                GROUP BY e.source, e.target
            """
            params *= 2
        elif metric == "variance":
            query = f"""
            WITH edge_groups AS (
                SELECT
                    source,
                    target,
                    MAX(iteration) AS max_iteration,
                    AVG({column}) AS mean_metric
                FROM edges
                WHERE source IN ({placeholders})
                  AND target IN ({placeholders})
                  AND iteration BETWEEN ? AND ?
                GROUP BY source, target
            ),
            edge_ids AS (
                SELECT
                    source,
                    target,
                    GROUP_CONCAT(id, '-') AS id
                FROM edges
                WHERE source IN ({placeholders})
                  AND target IN ({placeholders})
                  AND iteration BETWEEN ? AND ?
                GROUP BY source, target
            )
            SELECT
                ei.id,
                e.source,
                e.target,
                eg.max_iteration AS iteration,
                e.{column} - eg.mean_metric AS metric
            FROM edge_groups eg
            JOIN edges e
              ON e.source = eg.source
             AND e.target = eg.target
             AND e.iteration = eg.max_iteration
            JOIN edge_ids ei
              ON ei.source = eg.source
             AND ei.target = eg.target
            WHERE e.source IN ({placeholders})
              AND e.target IN ({placeholders})
              AND e.iteration BETWEEN ? AND ?
            """
            params *= 3
        else:
            query = f"""
                WITH edge_ids AS (
                    SELECT
                        source,
                        target,
                        GROUP_CONCAT(id, '-') AS id
                    FROM edges
                    WHERE {base_where}
                    GROUP BY source, target
                )
                SELECT
                    ei.id,
                    e.source,
                    e.target,
                    COUNT(*) AS metric
                FROM edges e
                JOIN edge_ids ei
                  ON ei.source = e.source
                 AND ei.target = e.target
                WHERE
                    e.source IN ({placeholders})
                    AND e.target IN ({placeholders})
                    AND e.iteration BETWEEN ? AND ?
                GROUP BY e.source, e.target
            """
            params *= 2

        edges = pd.read_sql_query(query, conn, params=params)

        query = f"""
            SELECT *
            FROM nodes
            WHERE id IN ({placeholders})
            """
        nodes = pd.read_sql_query(query, conn, params=build_ids)

        nodes["image"] = nodes["id"].apply(self.image_fn)
        nodes["has_image"] = nodes["image"].apply(
            lambda x: "str" if isinstance(x, str) else "list"
        )
        nodes["image"] = nodes["image"].apply(
            lambda x: "\n".join(x) if isinstance(x, list) else x
        )

        if add_handlers:
            # get number of children
            query = f"""
                            SELECT
                                source,
                                COUNT(DISTINCT target) AS n_children
                            FROM edges
                            WHERE source IN ({placeholders})
                              AND target NOT IN ({placeholders})
                            GROUP BY source
                        """
            counts = pd.read_sql_query(query, conn, params=build_ids + build_ids)
            nodes = nodes.merge(counts, left_on="id", right_on="source", how="left")
            nodes["n_children"] = nodes["n_children"].fillna(0)

            # create handlers
            handler_nodes = (
                nodes[(nodes["node_type"] != "final") | (nodes["n_children"] > 0)]
                .copy()
                .drop(["reward"], axis=1)
            )
            handler_nodes["node_type"] = "handler"
            handler_nodes["id"] = "handler_" + handler_nodes["id"]
            handler_nodes["label"] = "Select children: " + handler_nodes[
                "n_children"
            ].astype(int).astype(str)
            handler_nodes["metric"] = metric
            handler_nodes["direction"] = direction
            handler_edges = (
                handler_nodes["id"].copy().to_frame().rename(columns={"id": "target"})
            )
            handler_edges["source"] = handler_edges["target"].str.removeprefix(
                "handler_"
            )

            nodes = pd.concat([nodes, handler_nodes], ignore_index=True)
            edges = pd.concat([edges, handler_edges], ignore_index=True)

        nodes["iteration0"] = iteration[0]
        nodes["iteration1"] = iteration[1]
        if direction == "backward":
            edges.rename(columns={"source": "target", "target": "source"}, inplace=True)

        if self.s0 != "#":
            nodes.loc[nodes["id"] == "#", "image"] = self.image_fn(self.s0)

        # convert to cytoscape structure
        nodes = [{"data": row} for row in nodes.to_dict(orient="records")]
        edges = [{"data": row} for row in edges.to_dict(orient="records")]

        conn.close()

        # Compute color scale
        if metric in ["highest", "lowest"]:
            vmin, vmax = -10, 0
            colorscale = self.cs_main
        elif metric == "variance":
            vmin, vmax = -3, 3
            colorscale = self.cs_diverging_edgechange
        elif metric == "frequency":
            vmin, vmax = 0, max_freq
            colorscale = self.cs_main
        else:
            vmin, vmax = -1, 1  # fallback
            colorscale = px.colors.sequential.Viridis

        # Create color mapping
        def get_color(value, vmin, vmax, colorscale):
            """Get color."""
            if np.isnan(value):
                return "#8e8e8e"
            if value < vmin:
                return colorscale[0]  # lowest color
            if value > vmax:
                return colorscale[-1]  # highest color
            if vmax == vmin:
                norm = 0.5
            else:
                norm = (value - vmin) / (vmax - vmin)
            idx = int(norm * (len(colorscale) - 1))
            return colorscale[idx]

        # Build stylesheet
        elements = nodes + edges

        stylesheet = [
            # Default node style
            {
                "selector": "node",
                "style": {
                    "background-color": "#fff",
                    "background-image": "data(image)",
                    "background-fit": "contain",
                    "background-clip": "none",
                    "shape": "round-rectangle",
                    "width": "50px",
                    "height": "40px",
                    "border-width": "1px",
                    "border-color": "#000000",
                },
            },
            # default if no image available
            {
                "selector": 'node[has_image = "list"]',
                "style": {
                    "background-image": "none",
                    "label": "data(image)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "text-wrap": "wrap",
                    "text-max-width": "110px",
                    "font-size": "8px",
                    "width": "150px",
                },
            },
            # START node default "#" (keep text label)
            {
                "selector": 'node[node_type = "start"]',
                "style": {
                    "background-color": "#BAEB9D",
                    "background-image": "none",  # No image for start node
                    "label": "data(id)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px",
                    "shape": "diamond",
                    "width": "40px",
                    "height": "40px",
                    "border-color": "#000000",
                    "border-width": "2px",
                    "font-weight": "bold",
                    "text-wrap": "wrap",
                    "text-max-width": "55px",
                },
            },
            # Final node
            {
                "selector": 'node[node_type = "final"]',
                "style": {
                    "background-color": "#fff",
                    "height": "60px",
                    "border-width": "3px",
                    "border-color": "#000000",
                },
            },
            # Handler node
            {
                "selector": 'node[node_type = "handler"]',
                "style": {
                    "background-color": "#fff",
                    "background-image": "none",
                    "label": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "10px",
                    "shape": "round-rectangle",
                    "width": "90px",
                    "height": "20px",
                    "border-width": "2px",
                    "border-color": "#000000",
                    # 'font-weight': 'bold',
                    "text-wrap": "wrap",
                    "text-max-width": "90px",
                },
            },
            # Default edge style
            {
                "selector": "edge",
                "style": {
                    "width": 3,
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "arrow-scale": 1.5,
                },
            },
        ]

        if self.s0 != "#":
            stylesheet.append(
                # START node custom (display image)
                {
                    "selector": 'node[node_type = "start"]',
                    "style": {
                        "label": "",
                        "background-color": "#fff",
                        "background-image": "data(image)",
                        "background-fit": "contain",
                        "background-clip": "none",
                        "shape": "round-rectangle",
                        "width": "60px",
                        "height": "45px",
                        "border-width": "5px",
                        "border-color": "#BAEB9D",
                    },
                },
            )

        # Add color styles for each edge
        for edge in edges:
            edge_id = edge["data"]["id"]
            edge_val = edge["data"].get("metric", 0)
            color = get_color(edge_val, vmin, vmax, colorscale)

            stylesheet.append(
                {
                    "selector": f'edge[id = "{edge_id}"]',
                    "style": {"line-color": color, "target-arrow-color": color},
                }
            )

        return {
            "elements": elements,
            "stylesheet": stylesheet,
        }

    def dag_remove_node_prune(self, nodelist, node_to_remove):
        """
        removes the node from the list and all unconnected children
        :return: new list
        """
        conn = sqlite3.connect(self.data)
        remaining = set(nodelist)
        if node_to_remove not in remaining:
            return remaining
        stack = [node_to_remove]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            query = "SELECT DISTINCT target FROM edges WHERE source = ?"
            children = pd.read_sql_query(query, conn, params=[node])["target"].tolist()
            for child in children:
                if child not in remaining:
                    continue
                query = "SELECT DISTINCT source FROM edges WHERE target = ?"
                parents = pd.read_sql_query(query, conn, params=[child])[
                    "source"
                ].tolist()
                if not any(parent in remaining for parent in parents):
                    stack.append(child)

        conn.close()
        return remaining

    def update_DAG_overview(self, direction, metric, iteration, page):
        """Update the edge heatmap

        Edge heatmap plot pagination is implemented with offset. This is the fast
        implementation and might break for bigger datasets. Maybe implement
        with keyset later on.
        """
        column = "logprobs_" + direction
        top_n = 150
        offset = top_n * page

        conn = sqlite3.connect(self.data)

        # Build query based on metric
        if metric == "highest":
            query = f"""
                    WITH edge_max AS (
                        SELECT source, target, MAX({column}) as max_val
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY max_val DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT
                        e.source,
                        e.target,
                        e.iteration,
                        e.{column} as value,
                        em.max_val as metric_val,
                        e.trajectory_id
                    FROM edges e
                    INNER JOIN edge_max em ON
                        e.source = em.source
                        AND e.target = em.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY em.max_val DESC, e.source, e.target, e.iteration
                """
        elif metric == "lowest":
            query = f"""
                    WITH edge_min AS (
                        SELECT source, target, MIN({column}) as min_val
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY min_val ASC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT
                        e.source,
                        e.target,
                        e.iteration,
                        e.{column} as value,
                        em.min_val as metric_val,
                        e.trajectory_id
                    FROM edges e
                    INNER JOIN edge_min em ON
                        e.source = em.source
                        AND e.target = em.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY em.min_val ASC, e.source, e.target, e.iteration
                """
        elif metric == "variance":
            query = f"""
                    WITH edge_stats AS (
                        SELECT
                            source,
                            target,
                            AVG({column}) as mean_val,
                            COUNT(*) as cnt
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        HAVING cnt > 1
                    ),
                    edge_variance AS (
                        SELECT
                            es.source,
                            es.target,
                            es.mean_val,
                            SUM(
                                (e.{column} - es.mean_val) *
                                (e.{column} - es.mean_val)
                            ) / es.cnt as variance
                        FROM edges e
                        INNER JOIN edge_stats es ON
                            e.source = es.source
                            AND e.target = es.target
                        WHERE e.iteration BETWEEN ? AND ?
                        GROUP BY es.source, es.target, es.mean_val, es.cnt
                        ORDER BY variance DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT
                        e.source,
                        e.target,
                        e.iteration,
                        e.{column} as value,
                        ev.mean_val, ev.variance as metric_val,
                        e.trajectory_id
                    FROM edges e
                    INNER JOIN edge_variance ev ON
                        e.source = ev.source
                        AND e.target = ev.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY ev.variance DESC, e.source, e.target, e.iteration
                """
        elif metric == "frequency":
            query = f"""
                    WITH edge_freq AS (
                        SELECT source, target, COUNT(*) as freq
                        FROM edges
                        WHERE iteration BETWEEN ? AND ?
                        GROUP BY source, target
                        ORDER BY freq DESC
                        LIMIT {top_n} OFFSET {offset}
                    )
                    SELECT
                        e.source,
                        e.target,
                        e.iteration,
                        e.{column} as value,
                        ef.freq as metric_val,
                        e.trajectory_id
                    FROM edges e
                    INNER JOIN edge_freq ef ON
                        e.source = ef.source
                        AND e.target = ef.target
                    WHERE e.iteration BETWEEN ? AND ?
                    ORDER BY ef.freq DESC, e.source, e.target, e.iteration
                """

        params = iteration * 3 if metric == "variance" else iteration * 2
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return go.Figure().add_annotation(text="No data found", showarrow=False)

        # Create edge identifier and preserve order
        df["edge_id"] = df["source"].astype(str) + "-" + df["target"].astype(str)

        # Get unique edges in order (already ordered by metric in SQL)
        unique_edges = df["edge_id"].unique()
        edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
        df["edge_idx"] = df["edge_id"].map(edge_to_idx)

        # For variance metric, adjust values to be zero-based (value - mean)
        # For frequency metric, use the frequency value itself for coloring
        if metric == "variance":
            edge_means = df.groupby("edge_id")["value"].transform("mean")
            df["plot_value"] = df["value"] - edge_means
        elif metric == "frequency":
            df["plot_value"] = df["metric_val"]  # Use frequency for coloring
        else:
            df["plot_value"] = df["value"]

        # create edge list and trajectory id list for hover and selection
        trajectory_id_list = df.groupby("edge_idx")["trajectory_id"].agg(list).tolist()
        edge_list = df[["source", "target", "edge_idx"]]
        edge_list = edge_list.drop_duplicates().reset_index(drop=True)
        edge_list = list(
            edge_list[["source", "target"]].itertuples(index=False, name=None)
        )

        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            index="iteration", columns="edge_idx", values="plot_value", aggfunc="first"
        ).sort_index()

        if metric == "variance":
            color_scale = self.cs_diverging_edgechange
            zmin, zmax, zmid = -3, 3, 0
            title = (
                f"Edge Heatmap<br><sup>Difference: {direction.capitalize()}"
                "Logprobability - Mean of Edge</sup>"
            )
        elif metric == "frequency":
            color_scale = self.cs_main
            zmin = 0
            zmax = df["metric_val"].max()
            zmid = None
            title = "Edge Heatmap<br><sup>Highest frequency</sup>"
        else:  # highest or lowest
            color_scale = self.cs_main
            zmin, zmax, zmid = -10, 0, None
            title = (
                f"Edge Heatmap<br><sup>{metric.capitalize()}"
                f"Value of {direction.capitalize()} Logprobabilities</sup>"
            )

        # heatmap_data = heatmap_data.sort_index(axis=1)

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=heatmap_data.columns,
                colorscale=color_scale,
                showscale=True,
                zmin=zmin,
                zmax=zmax,
                zmid=zmid,
                # customdata=customdata,
                # hovertemplate = "<<%{customdata}>><extra></extra>",
                colorbar=dict(
                    title=dict(text=None), orientation="h", y=1.01, yanchor="bottom"
                ),
            )
        )
        ticks = 10 * (np.arange(15) + 1) + (150 * page)
        fig.update_layout(
            autosize=True,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
            dragmode="select",
            title=dict(
                text=title,
            ),
            xaxis=dict(
                title="Iteration",
                ticks="outside",
                showticklabels=True,
                showline=False,
                zeroline=False,
                showgrid=False,
                showspikes=False,
            ),
            yaxis=dict(
                title="Edge Rank ",
                ticks="outside",
                tickvals=heatmap_data.columns[9::10],
                ticktext=[str(int(v)) for v in ticks],
                showticklabels=True,
                showgrid=False,
                showline=False,
                zeroline=False,
                showspikes=False,
            ),
        )
        fig.update_traces(hoverinfo="none", hovertemplate=None)
        fig.update_yaxes(autorange="reversed")

        if metric == "frequency":
            return fig, zmax, trajectory_id_list, edge_list
        return fig, None, trajectory_id_list, edge_list

    def edge_hover_fig(self, edge_data):
        """Edge hover fig."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=edge_data["iteration"],
                y=edge_data["logprobs_forward"],
                mode="lines+markers",
                name="forward",
                line=dict(color=self.cs_diverging_dir[-3]),
                marker=dict(color=self.cs_diverging_dir[-3]),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=edge_data["iteration"],
                y=edge_data["logprobs_backward"],
                mode="lines+markers",
                name="backward",
                line=dict(color=self.cs_diverging_dir[2]),
                marker=dict(color=self.cs_diverging_dir[2]),
            )
        )

        fig.update_layout(
            height=200,
            width=250,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                x=0.3, y=1.1, orientation="h", xanchor="center", yanchor="bottom"
            ),
            xaxis_title="Iteration",
            yaxis_title="Logprobs",
            template="plotly_white",
        )

        return fig

    def update_state_space(self, df, selected_ids=[], metric="total_reward"):
        """Updates the state space for final objects.

        Parameters
        ----------
        df :
            Dataframe with required columns text, x, y, metric, istetset, iteration.
        selected_ids :
            List of selected texts. By default, [].
        metric :
            Metric for coloring. By default, "total_reward".
        """
        # Normalize metric, scale to range 6-30px, set size=4 for
        # missing values (no metric in testset)
        m_min = df[metric].min()
        m_max = df[metric].max()
        df["metric_norm"] = 6 + (df[metric] - m_min) / (m_max - m_min) * (30 - 6)
        df["metric_norm"] = df["metric_norm"].fillna(4)

        if selected_ids:
            df["opacity"] = df["id"].isin(selected_ids) * 0.9 + 0.1
        else:
            df["opacity"] = 0.5

        # Separate test set and normal points
        df_test = df[df["istestset"] == 1]
        df_normal = df[df["istestset"] == 0]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_normal["x"],
                y=df_normal["y"],
                mode="markers",
                marker=dict(
                    size=df_normal["metric_norm"],
                    color=df_normal["iteration"],
                    colorscale=self.cs_iteration,
                    line=dict(color="black", width=1),
                    showscale=True,
                    colorbar=dict(title="Iteration", thickness=15, len=0.7),
                    opacity=df_normal["opacity"],
                ),
                customdata=df_normal[["id", "iteration", metric, "text"]].values,
                hoverinfo="none",
                name="Samples",
            )
        )

        # Test set points in red
        if not df_test.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_test["x"],
                    y=df_test["y"],
                    mode="markers",
                    marker=dict(
                        size=df_test["metric_norm"],
                        color=self.cs_diverging_testset[-1],
                        line=dict(color="black", width=1),
                        opacity=df_test["opacity"],
                    ),
                    customdata=df_test[["id", "iteration", metric, "text"]].values,
                    hoverinfo="none",
                    name="Test Set",
                )
            )

        fig.update_layout(
            autosize=True,
            title=(
                f"State Space of Final Objects<br><sup>Size shows {metric}"
                "for the latest iteration the object occured"
            ),
            template="plotly_dark",
            legend=dict(
                itemsizing="constant",  # ensures marker size is not scaled
            ),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        return fig

    def update_bump_all(self, df, n_top, selected_ids, fo_metric, order):
        """Optimized bump chart update for cumulative top-ranked objects where rewards
        are fixed but rank evolves as new objects appear.

        Parameters
        ----------
        fo_metric :
        df :
            Prepared dataframe (should NOT include testset rows).
        n_top :
            Number of top objects to display.
        selected_ids :
            List of final_ids to highlight.
        order :
            highest/lowest rank.
        """
        df["iteration"] = pd.Categorical(
            df["iteration"], categories=sorted(df["iteration"].unique()), ordered=True
        )

        iterations = df["iteration"].cat.categories
        records = []

        # Track objects seen so far and their fixed rewards
        seen_objects = {}

        for it in iterations:
            # Add objects from this iteration
            current_iter_data = df[df["iteration"] == it]
            for _, row in current_iter_data.iterrows():
                seen_objects[row["text"]] = row["metric"]

            # Compute ranks for all seen objects
            # Use text as tiebreaker to ensure stable ordering for equal rewards
            asc = [False, True] if order == "highest" else [True, True]
            tmp_rank = (
                pd.DataFrame(
                    {
                        "text": list(seen_objects.keys()),
                        "metric": list(seen_objects.values()),
                    }
                )
                .sort_values(["metric", "text"], ascending=asc)
                .head(n_top)
            )
            tmp_rank["rank"] = range(1, len(tmp_rank) + 1)
            tmp_rank["iteration"] = it

            records.append(tmp_rank[["iteration", "text", "rank", "metric"]])

        tmp = pd.concat(records, ignore_index=True)
        tmp = tmp.rename(columns={"rank": "value"})

        # Attach IDs
        tmp = tmp.merge(
            df[["final_id", "text"]].drop_duplicates(subset="text"),
            on="text",
            how="left",
        )

        # Precompute marker sizes: circle if object was sampled in that iteration
        tmp["sampled"] = tmp.apply(
            lambda r: (
                8
                if (
                    (df["text"] == r["text"]) & (df["iteration"] == r["iteration"])
                ).any()
                else 0
            ),
            axis=1,
        )

        # Sort objects by first appearance rank for consistent line ordering
        first_ranks = tmp.groupby("text")["value"].min().sort_values().index

        fig = go.Figure()

        first_iter = tmp.groupby("text")["iteration"].min()
        # Map iteration categories to numeric indices
        iter_to_idx = {it: i for i, it in enumerate(iterations)}
        first_iter_idx = first_iter.map(iter_to_idx)
        n_colors = len(self.cs_iteration)
        # Normalize first-iteration index → color
        obj_color = {
            text: self.cs_iteration[
                int(idx / max(1, len(iterations) - 1) * (n_colors - 1))
            ]
            for text, idx in first_iter_idx.items()
        }

        for obj in first_ranks:
            obj_df = tmp[tmp["text"] == obj].sort_values("iteration")

            selected_mask = obj_df["final_id"].isin(selected_ids)
            if not selected_ids:
                selected_mask[:] = True
            unselected_mask = ~selected_mask

            for mask, opacity in [(selected_mask, 1), (unselected_mask, 0.1)]:
                sub_df = obj_df[mask]
                if sub_df.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=sub_df["iteration"],
                        y=sub_df["value"],
                        mode="lines+markers",
                        marker=dict(
                            symbol="circle",
                            size=sub_df["sampled"],
                            color=obj_color[obj],
                            coloraxis="coloraxis",
                        ),
                        line=dict(width=1),
                        opacity=opacity,
                        # name="1",#obj if opacity == 1 else f"{obj} (faded)",
                        customdata=sub_df[
                            ["final_id", "value", "metric", "text"]
                        ].values,
                        showlegend=False,
                    )
                )

        # dummy legend trace
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=0,
                    color=[df["iteration"].min(), df["iteration"].max()],
                    colorscale=self.cs_iteration,
                    cmin=df["iteration"].min(),
                    cmax=df["iteration"].max(),
                    showscale=True,
                    colorbar=dict(
                        title="Iteration of<br>First Sample",
                    ),
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

        fig.update_traces(hoverinfo="none", hovertemplate=None)

        fig.update_layout(
            autosize=True,
            title=(
                f"Sampled Objects ranked by {fo_metric}, {order} object has "
                "highest rank<br><sup>"
                "For each Iteration the highest ranking objects so far are shown, "
                "objects from previous iterations persist as long as their rank is "
                "high enough. "
                "Markers show if an object was actually sampled in this iteration."
                "</sup>"
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            xaxis_title="Iteration",
            yaxis_title="Rank",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        fig.update_yaxes(autorange="reversed")

        return fig

    def update_bump_iter(self, df, selected_ids, fo_metric, order):
        """Optimized bump chart update for cumulative top-ranked objects where rewards
        are fixed but rank evolves as new objects appear.

        Parameters
        ----------
        order :
        fo_metric :
        df :
            Prepared dataframe (should NOT include testset rows).
        selected_ids :
            List of final_ids to highlight.
        """
        df = df.drop_duplicates(subset=["iteration", "text", "metric"])
        df = df.rename(columns={"rank": "oldrank"})
        df["rank"] = (
            df.groupby("iteration")["oldrank"]
            .rank(method="dense", ascending=True)
            .astype(int)
        )

        if selected_ids:
            df["opacity"] = df["final_id"].isin(selected_ids) * 0.9 + 0.1
        else:
            df["opacity"] = 1

        fig = go.Figure(
            go.Scatter(
                x=df["iteration"],
                y=df["rank"],
                mode="markers",
                marker=dict(
                    color=df["iteration"],
                    colorscale=self.cs_iteration,
                    colorbar=dict(title="Iteration of<br>First Sample"),
                    symbol="square",
                    line=dict(width=1, color="black"),
                    size=10,
                    opacity=df["opacity"],
                ),
                line=dict(width=1),
                customdata=df[["final_id", "rank", "metric", "text"]].values,
                showlegend=False,
            )
        )

        fig.update_traces(hoverinfo="none", hovertemplate=None)

        fig.update_layout(
            autosize=True,
            title=(
                f"Sampled Objects ranked by {fo_metric}, {order} object has "
                f"highest rank<br><sup>"
                "For each Iteration the highest ranking objects of this iteration are "
                "shown.</sup>"
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            xaxis_title="Iteration",
            yaxis_title="Rank",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        fig.update_yaxes(autorange="reversed")

        return fig
