import random
import torch
import pickle
from typing import List, Dict
import sys
import os
# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Print to verify
print("Project root:", project_root)
print("Current working directory:", os.getcwd())

# Change working directory to project root
os.chdir(project_root)
print("New working directory:", os.getcwd())


from gflownet.envs.iam.investment import InvestmentDiscrete, TECHS, AMOUNTS
from gflownet.envs.iam.plan import Plan
from gflownet.proxy.iam.iam_proxies import FAIRY


def generate_random_plan(plan_env: Plan, seed: int = None) -> List:
    """
    Generate a random investment plan by assigning random amounts to each technology
    in the order defined by the TECHS initialization list.

    Parameters
    ----------
    plan_env : Plan
        The Plan environment instance
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    List
        A list starting with n_techs, followed by investment dictionaries
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    token2idx_amounts = {
        token: idx for idx, token in plan_env.idx2token_amounts.items()
    }
    plan = [plan_env.n_techs-1]

    # Iterate through technologies in order
    for tech_idx in range(1, plan_env.n_techs + 1):
        tech_token = plan_env.idx2token_techs[tech_idx]

        # Get sector and tag from network structure
        sector_token = plan_env.network_structure["tech2sector"][tech_token]
        sector_idx = plan_env.token2idx_sectors[sector_token]

        tag_token = plan_env.network_structure["tech2tag"][tech_token]
        tag_idx = plan_env.token2idx_tags[tag_token]

        # Randomly select an amount
        amount_token = random.choice(AMOUNTS)
        amount_idx = token2idx_amounts[amount_token]

        # Create investment dictionary
        investment = {
            "SECTOR": sector_idx,
            "TAG": tag_idx,
            "TECH": tech_idx,
            "AMOUNT": amount_idx,
        }
        plan.append(investment)

    return plan


def generate_dataset(
        n_samples: int,
        output_path_base: str = "random_plans",
        device: str = "cpu",
        seed: int = None,
) -> Dict:
    """
    Generate a dataset of random investment plans and evaluate them with FAIRY proxy.

    Parameters
    ----------
    n_samples : int
        Number of random plans to generate
    output_path_base : str
        Base path to save file (without extension). Will save as {output_path_base}.pkl
    device : str
        Device to run FAIRY proxy on ('cpu', 'cuda', etc.)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary with keys "samples" (list of readable plans) and "energies" (list of floats)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Initializing Plan environment...")
    plan_env = Plan()

    print(f"Initializing FAIRY proxy on device: {device}...")
    fairy_proxy = FAIRY(device=device)

    print(f"Generating {n_samples} random plans...")
    plans_list = []
    readable_list = []

    for i in range(n_samples):
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Generated {i + 1}/{n_samples} plans...")

        # Generate random plan
        plan = generate_random_plan(plan_env, seed=seed + i if seed is not None else None)
        plans_list.append(plan)

        # Convert to readable format
        readable_list.append(plan_env.state2readable(plan))

    print(f"\nEvaluating plans with FAIRY proxy...")
    plans_for_proxy = plan_env.states2proxy(plans_list)
    # Batch evaluate all plans
    energies = fairy_proxy(plans_for_proxy)
    energies = energies.cpu().detach().numpy()

    # Create dictionary with samples and energies
    data_dict = {
        "samples": plans_list,
        #"readable": readable_list,
        #"energies": energies.tolist(),
    }

    print(f"Saving dataset...")
    # Save as pickle dictionary
    pkl_path = f"{output_path_base}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"  Pickle file: {pkl_path}")

    print(f"✓ Dataset created successfully!")
    print(f"  Total samples: {n_samples}")
    print(f"  Output file: {pkl_path}")
    print(f"  Energy statistics:")
    print(f"    Mean: {energies.mean():.6f}")
    print(f"    Std:  {energies.std():.6f}")
    print(f"    Min:  {energies.min():.6f}")
    print(f"    Max:  {energies.max():.6f}")

    return data_dict


if __name__ == "__main__":
    n_samples = 10000
    output_path = "data/iam/test_set_v0"

    # Set seed for reproducibility
    seed = 42

    # Generate dataset
    data = generate_dataset(
        n_samples=n_samples,
        output_path_base=output_path,
        device="cpu",  # Change to "cuda" if you have GPU available
        seed=seed,
    )

    # Display sample of results
    print("\nSample of generated dataset:")
    print(f"  Number of samples: {len(data['samples'])}")
    if len(data['samples']) > 0:
        print(f"  First sample readable:\n    {data['samples'][0][:150]}...")
#        print(f"  First energy: {data['energies'][0]}")