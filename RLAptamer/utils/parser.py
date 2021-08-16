import argparse
import os
import json

# TODO Change Default Paths
ckpt_path = "./logs/checkpoints"
data_path = "./data"


def get_arguments():
    """Parse all the arguments provided from the terminal.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Reinforced active learning for image segmentation"
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=ckpt_path,
        help="Path to store weights, logs and" " other experiment related files.",
    )
    parser.add_argument(
        "--data-path", type=str, default=data_path, help="Path where the datasets can be found."
    )
    parser.add_argument("--exp-name", type=str, default="", help="Experiment name")
    parser.add_argument(
        "--exp-name-toload", type=str, default="", help="Experiment name to load weights from"
    )
    parser.add_argument(
        "--exp-name-toload-rl", type=str, default="", help="Introduce an experiment name to load"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="last_jaccard_val.pth",
        help="Name of the checkpoints to load.",
    )
    parser.add_argument(
        "--checkpointer",
        action="store_true",
        help="Store weights after each epoch and reload " "them at the start.",
    )
    parser.add_argument(
        "--load-weights", action="store_true", help="Load weights from another file."
    )
    parser.add_argument(
        "--load-opt", action="store_true", help="Load optimizer weights from another file."
    )

    # Proxy Model Training Arguments
    parser.add_argument("--epoch-num", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.998,
        help="Coefficient to decrease learning rate with "
        "exponential scheduler."
        " lr = old_lr * (gamma**epochs)",
    )
    parser.add_argument(
        "--gamma-scheduler-dqn",
        type=float,
        default=0.97,
        help="Coefficient to decrease learning rate with "
        "exponential scheduler."
        " lr = old_lr * (gamma**epochs)",
    )
    parser.add_argument("--weight-decay", type=int, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95, help="Learning rate momentum.")
    parser.add_argument(
        "--patience",
        type=int,
        default=60,
        help="Early stopping patience. "
        "If validation jaccard has not reached a new best"
        " value in N epochs, the training stops.",
    )
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "RMSprop"])

    # Q-Network Training Parameters
    parser.add_argument("--lr-dqn", type=float, default=0.0001)
    parser.add_argument(
        "--rl-buffer", type=int, default=3200
    )  # Size of the Experience replay buffer
    parser.add_argument("--rl-episodes", type=int, default=50)  # Number of episodes
    parser.add_argument("--dqn-bs", type=int, default=16)
    parser.add_argument("--dqn-gamma", type=float, default=0.999)
    parser.add_argument("--dqn-epochs", type=int, default=1)
    parser.add_argument("--budget-labels", type=int, default=100)
    parser.add_argument("--acc-target", type=int, default=0.9)

    # Other
    parser.add_argument("--seed", type=int, default=26)  # Seed to control torch and numpy randoms

    return parser.parse_args()


def save_arguments(args):
    print_args = {}
    param_names = [elem for elem in filter(lambda aname: not aname.startswith("_"), dir(args))]
    for name in param_names:
        print_args.update({name: getattr(args, name)})
        print("[" + name + "]   " + str(getattr(args, name)))

    path = os.path.join(args.ckpt_path, args.exp_name, "args.json")

    with open(path, "w") as fp:
        json.dump(print_args, fp)
    print("Args saved in " + path)
