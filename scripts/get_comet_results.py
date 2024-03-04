"""
Script to retrieve results from Comet.ml
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from comet_ml import Experiment
from comet_ml.api import API
from tqdm import tqdm


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments
    """
    # YAML config
    parser.add_argument(
        "--workspace",
        default="alexhernandezgarcia",
        type=str,
        help="Comet workspace",
    )
    parser.add_argument(
        "--project",
        default="gflownet-words",
        type=str,
        help="Comet project",
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        type=str,
        help="Root directory with results",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=str,
        help="Output CSV",
    )
    parser.add_argument("--save_each_csv", action="store_true")
    return parser


def metrics2df(metrics_dict):
    metric_names, steps = zip(*[(el["metricName"], el["step"]) for el in metrics_dict])
    metric_names = np.unique(metric_names).tolist()
    steps = np.asarray(steps)
    steps = np.unique(steps[steps != None]).tolist()
    columns = metric_names + ["step", "timestamp"]
    data_dict = {k: np.empty(np.max(steps) + 1) for k in metric_names}
    for d in metrics_dict:
        if d["step"] is not None:
            data_dict[d["metricName"]][d["step"]] = d["metricValue"]
    df = pd.DataFrame.from_dict(data_dict)
    df.index.name = "step"
    return df


def main(args):
    comet_api = API()
    comet_files = Path(args.root_dir).glob("**/comet.url")
    exp_dict = {}
    df_list = []
    df_keys = []
    for path in comet_files:
        print(f"Retrieving data from {path}")
        with open(path, "r") as f:
            url = f.readline()
        key = url.split("/")[-1].strip()
        exp_dict.update({key: {"path": path}})
        exp = comet_api.get_experiment(
            workspace=args.workspace, project_name=args.project, experiment=key
        )
        metrics = exp.get_metrics()
        df = metrics2df(metrics)
        df["path"] = path
        df_list.append(df)
        df_keys.append(path.parent)
        if args.save_each_csv:
            df.to_csv(path.parent / "comet.csv")
    df = pd.concat(df_list, keys=df_keys, names=["path", "step"])
    if args.output_csv:
        df.to_csv(args.output_csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))
    main(args)
