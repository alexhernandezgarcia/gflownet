import argparse
import math


def count_decision_trees(
    max_depth: int,
    num_features: int,
    num_thresholds: int = 1,
    allow_empty: bool = False,
) -> int:
    """
    Count the number of structurally distinct binary decision trees
    of depth at most `max_depth` over `num_features` features
    and `num_thresholds` thresholds per feature, where leaves are unlabeled.

    Recursion (with thresholds):
        N(D) = 1 + F * T * N(D-1)^2,  N(0) = 1

    If `allow_empty` is False, the leaf-only tree (no split at the root)
    is excluded from the final count, so the result is N(D) - 1.
    Subtrees within the tree may still be leaves — only the root is
    required to be a split.
    """
    if max_depth < 1:
        raise ValueError(
            "max_depth must be at least 1 when the empty tree is disallowed"
        )
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if num_thresholds < 1:
        raise ValueError("num_thresholds must be at least 1")

    label_choices = num_features * num_thresholds

    n = 1  # N(0): a single leaf
    for _ in range(max_depth):
        n = 1 + label_choices * n * n

    if not allow_empty:
        n -= 1  # remove the leaf-only tree

    return n


def main():
    parser = argparse.ArgumentParser(
        description="Count binary decision tree structures up to a given depth."
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        required=True,
        help="Maximum depth of the tree (D >= 1)",
    )
    parser.add_argument(
        "-F",
        "--num-features",
        type=int,
        required=True,
        help="Number of features (F >= 1)",
    )
    parser.add_argument(
        "-T",
        "--num-thresholds",
        type=int,
        default=1,
        help="Number of thresholds per feature (T >= 1, default: 1)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Include the leaf-only tree in the count (default: excluded)",
    )
    args = parser.parse_args()

    count = count_decision_trees(
        args.max_depth,
        args.num_features,
        args.num_thresholds,
        allow_empty=args.allow_empty,
    )
    log_count = math.log(count) if count > 0 else float("-inf")

    print(f"Max depth D        : {args.max_depth}")
    print(f"Number of features : {args.num_features}")
    print(f"Number of thresholds: {args.num_thresholds}")
    print(f"Leaf-only allowed  : {args.allow_empty}")
    print(f"Number of trees    : {count}")
    print(f"ln(count)          : {log_count:.6f}")


if __name__ == "__main__":
    main()
