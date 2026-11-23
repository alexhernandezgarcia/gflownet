from gflownet.utils.common import load_gflownet_from_rundir
import argparse
import json
from pathlib import Path
from collections import Counter

debug = True

def load_model(run_dir):
    """Load the saved model without starting a WandB run."""
    gflownet, config = load_gflownet_from_rundir(run_dir, no_wandb=True)
    gflownet.forward_policy.model.eval()
    gflownet.backward_policy.model.eval()
    return gflownet, config

def sample_trajectories(gflownet, n_samples=10):
    """Sample trajectories given a model"""
    trajectories = []
    batch, times = gflownet.sample_batch(n_forward=n_samples)
    traj_ids = batch.get_unique_trajectory_indices()
    for traj_id in traj_ids:
        states = batch.get_states_of_trajectory(traj_id)
        trajectory = [gflownet.env.state2readable(state) for state in states]
        trajectories.append(trajectory)
        if debug:
            print(traj_id, trajectory)

    return trajectories

def analyze_pattern(trajectories, pattern):
    counts = Counter()

    for trajectory in trajectories:
        first_length = next((len(sequence) for sequence in trajectory if pattern in sequence), None)
        counts[first_length] += 1

    n_samples = len(trajectories)
    n_found = n_samples - counts[None]
    max_length = max(len(trajectory[-1]) for trajectory in trajectories)

    results = [
            {
                "length": length,
                "count": counts[length],
                "ratio_all": counts[length] / n_samples,
                "ratio_found": counts[length] / n_found if n_found > 0 else None,
                }
            for length in range(len(pattern), max_length + 1)
            ]

    return results, counts[None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--output", default="Bidrectional")
    parser.add_argument("--pattern", default="BC")
    args = parser.parse_args()

    gflownet, _ = load_model(args.run_dir)
    gflownet.random_action_prob = 0.0

    trajectories = sample_trajectories(gflownet, n_samples=args.n_samples)
    
    # Analyze the pattern in trajectories
    results, not_found = analyze_pattern(trajectories, pattern=args.pattern)
    for row in results:
        print(row)
    print(f"Pattern not found: {not_found}")

    output_dir = Path(__file__).parent/"samples"/args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trajectories.json"
    with output_file.open("w") as f:
        json.dump(
            {
                "trajectories": trajectories,
                "pattern": args.pattern,
                "results": results,
                "not_found": not_found,
            },
            f,
            indent=2,
        )
    print(f"Saved to {output_file}")


