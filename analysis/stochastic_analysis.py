import argparse

import torch

from analysis.analyze_sequence_trajectories import load_model

import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product

@torch.no_grad()
def inspect_actions(gflownet, state=None):
    env = gflownet.env.copy().reset()
    if state is not None:
        env.set_state(state, done=False)

    policy_outputs = gflownet.forward_policy(env.states2policy([env.state]))
    #logits = env._get_policy_outputs_of_meta_actions(policy_outputs)[0]
    #mask = env._meta_forward_mask(env.state, env.done)
    #mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    logits = policy_outputs[0]
    mask = env.get_mask_invalid_actions_forward()
    mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)

    probabilities = logits.masked_fill(mask, -torch.inf).softmax(dim=0)

    actions = env.action_space

    results = []
    source = env.state2readable()

    #for i in range(env.n_meta_actions):
    for i in range(len(actions)):
        if mask[i].item():
            continue
        action = actions[i]
        logit = logits[i].item()
        probability = probabilities[i].item()

        if action == env.eos:
            action_readable = "EOS"
            target = f"{source} [done]"
        else:
            #letter = env.envs_unique[action[1]].letters[0]
            letter = env.idx2token[action[1]]
            
            if action[0] == 0:
                action_readable = f"Insert {letter} left"
                target = f"{letter} {source}".strip()
            else:
                action_readable = f"Insert {letter} right"
                target =f"{source} {letter}".strip()

        results.append(
            {
                "action": action,
                "action_readable": action_readable,
                "target": target,
                "logit": logit,
                "probability": probability,
            }
        )
        print(
            f"{source or 'SOURCE'} -> {target} | {action_readable} | "
            f"logit={logit:.4f} p={probability:.4f}"
        )
    return results
#def get_all_states(right_only, left_only, tokens, max_length):
    
def transitions(gflownet, max_length):
    #states = gflownet.env.get_all_states()
    all_transitions = []
    #for state in states:
    for length in range(max_length + 1):
        for tokens in product(gflownet.env.tokens, repeat=length):
            state = gflownet.env.readable2state(" ".join(tokens))
            source = gflownet.env.state2readable(state)

            results = inspect_actions(gflownet, state)

            all_transitions.append({
                "source": source,
                "actions": results,
            })

    return all_transitions

def plot_distribution(
    all_transitions, source, output_file, two_letters=False,
    n_letters=None, pattern=None,
):
    if two_letters:
        n_letters = 2
    if n_letters is not None:
        distribution = sequence_distribution(all_transitions, n_letters)
        labels = sorted(distribution)
        probabilities = [distribution[label] for label in labels]
        title = f"Sequence distribution at length {n_letters}"
        if pattern is not None:
            pattern_probability = sum(
                probability for sequence, probability in distribution.items()
                if pattern in sequence
            )
            title += f" | P(contains {pattern}) = {pattern_probability:.2%}"
    else:
        record = next(row for row in all_transitions if row["source"] == source)
        labels = [result["action_readable"] for result in record["actions"]]
        probabilities = [result["probability"] for result in record["actions"]]
        title = f"Action distribution given {source or 'SOURCE'}"

    fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(labels)))
    colors = [
        "darkorange" if n_letters is not None and pattern and pattern in label
        else "steelblue"
        for label in labels
    ]
    bars = ax.barh(labels, probabilities, color=colors)

    ax.bar_label(
        bars,
        labels=[f"{p:.2%}" for p in probabilities],
        padding=4,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

def sequence_distribution(transitions, n_letters):
    """Propagate probability from SOURCE to the requested generated length."""
    actions_by_state = {
        row["source"]: row["actions"]
        for row in transitions
    }

    probabilities = {"": 1.0}
    for _ in range(n_letters):
        next_probabilities = defaultdict(float)
        for source, probability in probabilities.items():
            for result in actions_by_state[source]:
                if result["action"][1] == -1:
                    continue
                next_probabilities[result["target"]] += (
                    probability * result["probability"]
                )
        probabilities = dict(next_probabilities)

    return probabilities


def two_letter_distribution(transitions):
    return sequence_distribution(transitions, 2)


def pattern_distribution(transitions, pattern):
    """First appearance probabilities for a contiguous pattern at each length.

    In this fixed-length environment, end insertion preserves existing substrings,
    so the increase in cumulative probability gives the first appearance probability.
    """
    max_length = max(len(row["source"].split()) for row in transitions)
    previous = 0.0
    results = []
    for length in range(len(pattern.split()), max_length + 1):
        distribution = sequence_distribution(transitions, length)
        cumulative = sum(
            probability for sequence, probability in distribution.items()
            if pattern in sequence
        )
        results.append({
            "length": length,
            "first_probability": cumulative - previous,
            "cumulative_probability": cumulative,
        })
        previous = cumulative
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default="Bidrectional")
    parser.add_argument("--pattern", default="BC")
    parser.add_argument("--n_letters", default=None, type=int)
    parser.add_argument("--max_length_to_analyze", default=3, type=int)
    args = parser.parse_args()

    gflownet, _ = load_model(args.run_dir)

    #print("\nSource:")
    #inspect_actions(gflownet)

    #env = gflownet.env.copy().reset()
    #env.step((-1, 1, 0))  
    #env.step((1, 1, 0))   
    #env.step((1, -1, 0))  
    #env.step((-1, 1, 0))  
    #print("\nState B:")
    #inspect_actions(gflownet, env.state)

    all_transitions = transitions(gflownet, max_length=args.max_length_to_analyze)
    #print(f"all transitions: {all_transitions}")

    output_dir = Path(__file__).parent / "samples" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "transitions.json"
    with output_file.open("w") as f:
        json.dump(all_transitions, f, indent=2)
    print(f"Saved to {output_file}")

    plot_distribution(all_transitions, "B", output_dir / "distribution_B.png")
    plot_distribution(all_transitions, "", output_dir / "distribution_SOURCE.png")
    plot_distribution(all_transitions, "C", output_dir / "distribution_C.png")

    n_letters = len(args.pattern.split()) if args.n_letters is None else args.n_letters
    distribution = sequence_distribution(all_transitions, n_letters)
    pattern_results = pattern_distribution(all_transitions, args.pattern)
    print(f"Distribution at length {n_letters}: {distribution}")
    for result in pattern_results:
        print(result)

    with (output_dir / f"distribution_{n_letters}_letters.json").open("w") as f:
        json.dump({
            "pattern": args.pattern,
            "n_letters": n_letters,
            "distribution": distribution,
            "pattern_first_occurrence": pattern_results,
        }, f, indent=2)
    plot_distribution(
        all_transitions, "", output_dir / f"distribution_{n_letters}_letters.png",
        n_letters=n_letters, pattern=args.pattern,
    )




