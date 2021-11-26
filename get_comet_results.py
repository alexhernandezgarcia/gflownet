"""
Script to retrieve results from Comet.ml
"""
from comet_ml import Experiment
from comet_ml.api import API
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
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
    return parser


def metrics2df(metrics):
    metric_names = np.unique([metric["metricName"] for metric in metrics])
    columns = metric_names + ["step"]


def main(args):
   comet_api = API() 
   comet_files = Path(args.root_dir).glob("**/comet.url")
   exp_dict = {}
   for path in comet_files:
       with open(path, 'r') as f:
           url = f.readline()
       key = url.split("/")[-1].strip()
       exp_dict.update({key: {"path": path}})
       exp = comet_api.get_experiment(workspace=args.workspace, project_name=args.project, experiment=key)
       metrics = exp.get_metrics()
       import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print(
        "Args:\n"
        + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()])
    )
    main(args)
