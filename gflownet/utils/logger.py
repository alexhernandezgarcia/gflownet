import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from numpy import array
from omegaconf import OmegaConf
from torchtyping import TensorType


class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time

    Parameters
    ----------
    run_name : str
        Name of the run. By default it is None. If run_name is None and run_name_date
        and run_name_job are both False, then a random name will be assigned by wandb.
    run_name_date : bool
        Whether the date (and time) should be included in the run name. True by
        default.
    run_name_job : bool
        Whether the job ID should be included in the run name. True by default.
    progressbar : dict
        A dictionary of configuration parameters related to the progress bar, namely:
            - skip : bool
                If True, the progress bar is not displayed during training. False by
                default.
            - n_iters_mean : int
                The number of past iterations to take into account to compute averages
                of a metric, for example the loss. 100 by default.

    """

    def __init__(
        self,
        config: dict,
        do: dict,
        project_name: str,
        logdir: dict,
        lightweight: bool,
        debug: bool,
        run_name=None,
        run_name_date: bool = True,
        run_name_job: bool = True,
        run_id: str = None,
        tags: list = None,
        context: str = "0",
        notes: str = None,
        entity: str = None,
        progressbar: dict = {"skip": False, "n_iters_mean": 100},
        is_resumed: bool = False,
    ):
        self.config = config
        self.do = do
        self.do.times = self.do.times and self.do.online
        slurm_job_id = os.environ.get("SLURM_JOB_ID")

        # Determine run name
        if run_name is None:
            run_name = ""
        if run_name_job and slurm_job_id is not None:
            run_name = f"{run_name} {slurm_job_id}"
        if run_name_date:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = f"{run_name} {date_time}"

        if self.do.online:
            import wandb

            self.wandb = wandb
            wandb_config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            if slurm_job_id:
                wandb_config["slurm_job_id"] = slurm_job_id
            if run_id is not None:
                # Resume run
                self.run = self.wandb.init(
                    id=run_id,
                    project=project_name,
                    entity=entity,
                    resume="allow",
                )
            else:
                self.run = self.wandb.init(
                    config=wandb_config,
                    project=project_name,
                    name=run_name,
                    notes=notes,
                    entity=entity,
                    resume="allow",
                )
        else:
            self.wandb = None
            self.run = None
        self.add_tags(tags)
        self.context = context
        self.progressbar = progressbar
        self.loss_memory = []
        self.lightweight = lightweight
        self.debug = debug
        self.is_resumed = is_resumed
        # Log directory
        if "path" in logdir:
            self.logdir = Path(logdir.path)
        else:
            self.logdir = Path(logdir.root)
        if self.is_resumed:
            if self.debug:
                print(f"Run is resumed and will log into directory {self.logdir}")
        elif not self.logdir.exists() or logdir.overwrite:
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"logdir {logdir} already exists! - Ending run...")
            sys.exit(1)
        # Checkpoints directory
        self.ckpts_dir = self.logdir / logdir.ckpts
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)
        # Data directory
        self.datadir = self.logdir / "data"
        self.datadir.mkdir(parents=True, exist_ok=True)
        # Write wandb URL
        self.write_url_file()

    def write_url_file(self):
        if self.wandb is not None:
            self.url = self.wandb.run.get_url()
            if self.url:
                with open(self.logdir / "wandb.url", "w") as f:
                    f.write(self.url + "\n")

    def add_tags(self, tags: list):
        if not self.do.online:
            return
        self.run.tags = self.run.tags + tags

    def set_context(self, context: int):
        self.context = str(context)

    def progressbar_update(
        self, pbar, loss, rewards, jsd, use_context=True, n_mean=100
    ):
        if self.progressbar["skip"]:
            return
        if len(self.loss_memory) < self.progressbar["n_iters_mean"]:
            self.loss_memory.append(loss)
        else:
            self.loss_memory = self.loss_memory[1:] + [loss]
        description = "Loss: {:.4f} | Mean rewards: {:.2f} | JSD: {:.4f}".format(
            np.mean(self.loss_memory), np.mean(rewards), jsd
        )
        pbar.update(1)
        pbar.set_description(description)

    def log_histogram(self, key, value, step, use_context=True):
        if not self.do.online:
            return
        if use_context:
            key = self.context + "/" + key
        fig = plt.figure()
        plt.hist(value)
        plt.title(key)
        plt.ylabel("Frequency")
        plt.xlabel(key)
        fig = self.wandb.Image(fig)
        self.wandb.log({key: fig}, step)

    def log_plots(self, figs: Union[dict, list], step, use_context=True):
        if not self.do.online:
            self.close_figs(figs)
            return
        if isinstance(figs, dict):
            keys = figs.keys()
            figs = list(figs.values())
        else:
            assert isinstance(figs, list), "figs must be a list or a dict"
            keys = [f"Figure {i} at step {step}" for i in range(len(figs))]

        for key, fig in zip(keys, figs):
            if use_context:  # fixme
                context = self.context + "/" + key
            if fig is not None:
                figimg = self.wandb.Image(fig)
                self.wandb.log({key: figimg}, step)

        self.close_figs(figs)

    def close_figs(self, figs: list):
        for fig in figs:
            if fig is not None:
                plt.close(fig)

    def log_rewards_and_scores(
        self,
        rewards: TensorType["n_samples"],
        logrewards: TensorType["n_samples"],
        scores: TensorType["n_samples"],
        step: int,
        prefix: str,
        use_context: bool = True,
    ):
        """
        Logs the rewards, log-rewards and proxy scores passed as arguments.

        Parameters
        ----------
        rewards : tensor
            Rewards of a batch of states.
        logrewards : tensor
            Log-rewards of a batch of states.
        scores : tensor
            Proxy scores of a batch of states.
        step : int
            The training iteration number.
        prefix : str
            Prefix to be added to the metric names.
        use_context : bool
            If True, prepend self.context + / to the key of the metric.
        """
        if not self.do.online:
            return

        metrics = {
            f"{prefix} rewards mean": rewards.mean(),
            f"{prefix} rewards max": rewards.max(),
            f"{prefix} logrewards mean": logrewards.mean(),
            f"{prefix} logrewards max": logrewards.max(),
        }
        if scores is not None:
            metrics.update(
                {
                    f"{prefix} scores mean": scores.mean(),
                    f"{prefix} scores min": scores.min(),
                    f"{prefix} scores max": scores.max(),
                }
            )

        self.log_metrics(metrics, step=step, use_context=use_context)

    def log_min_max_mean(
        self,
        scores: Union[npt.NDArray, TensorType["n_samples"]],
        step: int,
        prefix: str,
        use_context: bool = True,
    ):
        """
        Logs the minimum, maximum and mean of the values passed as arguments.

        Parameters
        ----------
        values : tensor
            Values corresponding to batch of states.
        step : int
            The training iteration number.
        prefix : str
            Prefix to be added to the metric names.
        use_context : bool
            If True, prepend self.context + / to the key of the metric.
        """
        if not self.do.online:
            return

        metrics = {
            f"{prefix} min": rewards.min(),
            f"{prefix} max": rewards.max(),
            f"{prefix} mean": rewards.mean(),
        }
        self.log_metrics(metrics, step=step, use_context=use_context)

    def log_metrics(
        self,
        metrics: dict,
        step: int,
        use_context: bool = True,
    ):
        """
        Logs metrics to wandb.

        Parameters
        ----------
        metrics : dict
            A dictionary of metrics to be logged to wandb.
        step : int
            The training iteration number.
        use_context : bool
            If True, prepend self.context + / to the key of the metric.
        """
        if not self.do.online:
            return

        for key, value in metrics.items():
            # Append context
            if use_context:
                key = self.context + "/" + key

            # Ignore if value is None
            if value is None:
                continue

            self.wandb.log({key: value}, step=step)

    def log_summary(self, summary: dict):
        if not self.do.online:
            return
        self.run.summary.update(summary)

    def save_checkpoint(
        self,
        forward_policy,
        backward_policy,
        state_flow,
        logZ,
        optimizer,
        buffer,
        step: int,
        final: bool = False,
    ):
        if final:
            ckpt_id = "final"
            if self.debug:
                print("Saving final checkpoint in:")
        else:
            ckpt_id = "iter_{:06d}".format(step)
            if self.debug:
                print(f"Saving checkpoint of step {step} in:")

        ckpt_path = self.ckpts_dir / (ckpt_id + ".ckpt")
        if self.debug:
            print(f"\t{ckpt_path}")

        # Forward model
        if forward_policy.is_model:
            forward_ckpt = forward_policy.model.state_dict()
        else:
            forward_ckpt = None

        # Backward model
        if backward_policy and backward_policy.is_model:
            backward_ckpt = backward_policy.model.state_dict()
        else:
            backward_ckpt = None

        # State flow model
        if state_flow:
            state_flow_ckpt = state_flow.model.state_dict()
        else:
            state_flow_ckpt = None

        # LogZ
        if isinstance(logZ, torch.nn.Parameter) and logZ.requires_grad:
            logZ_ckpt = logZ.detach().cpu()
        else:
            logZ_ckpt = None

        # Buffer
        buffer_ckpt = {
            "train": None,
            "test": None,
            "replay": None,
        }
        if hasattr(buffer, "train") and buffer.train is not None:
            if hasattr(buffer.train_config, "pkl") and buffer.train_config.pkl:
                buffer_ckpt["train"] = str(buffer.train_config.pkl)
        if hasattr(buffer, "test") and buffer.test is not None:
            if hasattr(buffer.test_config, "pkl") and buffer.test_config.pkl:
                buffer_ckpt["test"] = str(buffer.test_config.pkl)
        if hasattr(buffer, "replay") and len(buffer.replay) > 0:
            if hasattr(buffer, "replay_csv") and buffer.replay_csv is not None:
                buffer_ckpt["replay"] = str(buffer.replay_csv)

        # WandB run ID
        if self.do.online:
            run_id_ckpt = self.run.id
        else:
            run_id_ckpt = None

        checkpoint = {
            "step": step,
            "forward": forward_ckpt,
            "backward": backward_ckpt,
            "state_flow": state_flow_ckpt,
            "logZ": logZ_ckpt,
            "optimizer": optimizer.state_dict(),
            "buffer": buffer_ckpt,
            "run_id": run_id_ckpt,
        }
        torch.save(checkpoint, ckpt_path)

    def log_time(self, times: dict, use_context: bool):
        if self.do.times:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, use_context=use_context)

    def end(self):
        if not self.do.online:
            return
        self.wandb.finish()
