"""Compare pattern first-appearance probabilities from saved analysis results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_first_occurrence(bidirectional_file, right_append_file, output_file):
    with Path(bidirectional_file).open() as f:
        bidirectional = json.load(f)
    with Path(right_append_file).open() as f:
        right_append = json.load(f)

    lengths = [row["length"] for row in bidirectional["pattern_first_occurrence"]]
    cases = {"Bidirectional": bidirectional, "Right append": right_append}
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (label, data) in enumerate(cases.items()):
        probabilities = {
            row["length"]: row["first_probability"]
            for row in data["pattern_first_occurrence"]
        }
        heights = [probabilities[length] for length in lengths]
        positions = [i + (index - 0.5) * width for i in range(len(lengths))]
        bars = ax.bar(positions, heights, width=width, label=label)
        ax.bar_label(bars, labels=[f"{p:.2%}" for p in heights], padding=4)

    ax.set_xticks(range(len(lengths)), lengths)
    ax.set_xlabel("Generated letters at first appearance")
    ax.set_ylabel("Probability over all trajectories")
    ax.set_title(f"First appearance of {bidirectional['pattern']}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    samples = Path(__file__).parent / "samples"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bidirectional", type=Path,
        default=samples / "Bidrectional" / "distribution_3_letters.json",
    )
    parser.add_argument(
        "--right_append", type=Path,
        default=samples / "RightAppend" / "distribution_3_letters.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=samples / "pattern_first_occurrence.png",
    )
    args = parser.parse_args()
    plot_first_occurrence(args.bidirectional, args.right_append, args.output)
