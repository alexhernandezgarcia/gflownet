"""
Double check that all initial data sets are equal.
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from oracle import numbers2letters


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument("--run_dir", default=None, type=str)
    parser.add_argument(
        "--local_path_pattern",
        default="episode0/datasets/nupack energy.npy",
        type=str,
    )
    parser.add_argument(
        "--k",
        default=[1, 10, 100],
        nargs="*",
        type=int,
        help="List of K, for Top-K",
    )
    parser.add_argument(
        "--queries_per_iter",
        default=100,
        type=int,
        help="Samples added to the data set per AL iteration",
    )
    parser.add_argument(
        "--max_rounds",
        default=20,
        type=int,
        help="Maximum rounds allowed to compute metrics",
    )
    parser.add_argument("--score_is_pos", action="store_true", default=False)
    return parser


def main(args):
    dataset = []
    data_path = Path(args.run_dir / Path(args.local_path_pattern))
    if not data_path.exists():
        return
    data_dict = np.load(data_path, allow_pickle=True).item()
    letters = numbers2letters(data_dict["samples"])
    scores = data_dict["scores"]
    if args.score_is_pos:
        scores_sorted = np.sort(scores)[::-1]
    else:
        scores_sorted = np.sort(scores)
    print(f"All rounds data set length: {len(letters)}")
    print("All rounds:")
    for k in args.k:
        mean_topk = np.mean(scores_sorted[:k])
        print(f"\tAverage score top-{k}: {mean_topk}")
    # Limit rounds
    letters_r = letters[: (args.max_rounds + 1) * args.queries_per_iter]
    scores_r = scores[: (args.max_rounds + 1) * args.queries_per_iter]
    scores_r_sorted = np.sort(scores_r)
    print(f"{args.max_rounds} rounds data set length: {len(letters_r)}")
    print(f"{args.max_rounds} rounds:")
    for k in args.k:
        mean_topk = np.mean(scores_r_sorted[:k])
        print(f"\tAverage score top-{k}: {mean_topk}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
