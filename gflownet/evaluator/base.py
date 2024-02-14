import os
from typing import Union

from gflownet.utils.common import load_gflow_net_from_run_path

_sentinel = object()


class GFlowNetEvaluator:
    def __init__(self, **kwargs):
        if kwargs.get("sentinel") is not _sentinel:
            raise NotImplementedError(
                "Base evaluator class should not be instantiated. Use "
                + "GFlowNetEvaluator.from_dir or GFlowNetEvaluator.from_agent methods."
            )
        self.gfn_agent = kwargs.get("gfn_agent")

    @staticmethod
    def from_dir(
        path: Union[str, os.PathLike],
        no_wandb: bool = True,
        print_config: bool = False,
        device: str = "cuda",
        load_final_ckpt: bool = True,
    ):
        gfn_agent = load_gflow_net_from_run_path(
            path,
            no_wandb=no_wandb,
            print_config=print_config,
            device=device,
            load_final_ckpt=load_final_ckpt,
        )
        return GFlowNetEvaluator.from_agent(gfn_agent)

    @staticmethod
    def from_agent(gfn_agent):
        return GFlowNetEvaluator(gfn_agent=gfn_agent, sentinel=_sentinel)

    def plot(self):
        print("Base evaluator plot method does not do anything.")

    def compute_metrics(self, metrics: list = []):
        print("Base evaluator compute_metrics method does not do anything.")

    def evaluate(self, n_episodes: int = 1):
        print("Base evaluator evaluate method does not do anything.")


if __name__ == "__main__":
    # dev test case, will move to tests
    gfn_run_dir = "/network/scratch/s/schmidtv/crystals/logs/icml24/crystalgfn/4074836/2024-01-27_20-54-55/5908fe41"
    gfne = GFlowNetEvaluator.from_dir(gfn_run_dir)
    gfne.plot()
    gfne.compute_metrics()
