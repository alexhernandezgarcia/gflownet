import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from numpy import array


class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config: dict,
        project_name: str,
        logdir: str,
        overwrite_logdir: bool,
        sampler: dict,
        run_name=None,
        tags: list = None,
        record_time: bool = False,
    ):
        self.config = config
        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(
                date_time,
            )
        self.run = wandb.init(config=config, project=project_name, name=run_name)
        self.add_tags(tags)
        self.sampler = sampler
        self.context = "0"
        self.record_time = record_time
        # Log directory
        self.logdir = Path(logdir)
        # TODO: Update
        if self.logdir.exists():
            with open(self.logdir / "comet.url", "w") as f:
                f.write(self.comet.url + "\n")
        # rewrite prettily
        if sampler.test.period in [None, -1]:
            self.test_period = np.inf
        else:
            self.test_period = sampler.test.period
        if sampler.policy.period in [None, -1]:
            self.policy_period = np.inf
        else:
            self.policy_period = sampler.policy.period
        if sampler.train.period in [None, -1]:
            self.train_period = np.inf
        else:
            self.train_period = sampler.train.period
        if sampler.oracle.period in [None, -1]:
            self.oracle_period = np.inf
        else:
            self.oracle_period = sampler.oracle.period

    def add_tags(self, tags: list):
        self.run.tags = self.run.tags + tags

    def set_context(self, context: int):
        self.context = str(context)

    def log_metric(self, key: str, value, use_context=True):
        if use_context:
            key = self.context + "/" + key
        wandb.log({key: value})

    def log_histogram(self, key, value, use_context=True):
        # need this condition for when we are training gfn without active learning and context = ""
        # we can't make use_context=False because then when the same gfn is used with AL, context won't be recorded (undesirable)
        if use_context:
            key = self.context + "/" + key
        fig = plt.figure()
        plt.hist(value)
        plt.title(key)
        plt.ylabel("Frequency")
        plt.xlabel(key)
        fig = wandb.Image(fig)
        wandb.log({key: fig})

    def log_metrics(self, metrics: dict, step: int, use_context: bool = True):
        if use_context:
            for key, _ in metrics.items():
                key = self.context + "/" + key
        wandb.log(metrics, step)

    def log_sampler_train(
        self,
        rewards: list,
        proxy_vals,
        states_term: list,
        data: list,
        it: int,
        use_context: bool,
    ):
        if not it % self.train_period:
            train_metrics = dict(
                zip(
                    [
                        "mean_reward",
                        "max_reward",
                        "mean_proxy",
                        "min_proxy",
                        "max_proxy",
                        "mean_seq_length",
                        "batch_size",
                    ],
                    [
                        np.mean(rewards),
                        np.max(rewards),
                        np.mean(proxy_vals),
                        np.min(proxy_vals),
                        np.max(proxy_vals),
                        np.mean([len(state) for state in states_term]),
                        len(data),
                    ],
                )
            )
            self.log_metrics(
                train_metrics,
                use_context=use_context,
                step=it,
            )

    def log_sampler_test(self, corr, data_logq: list, it: int, use_context: bool):
        if not it % self.test_period:
            test_metrics = dict(
                zip(
                    [
                        "test_corr_logq_score",
                        "test_mean_log",
                    ],
                    [
                        corr[0, 1],
                        np.mean(data_logq),
                    ],
                )
            )
            self.log_metrics(
                test_metrics,
                use_context=use_context,
                step=it,
            )

    def log_sampler_oracle(self, energies, it, use_context):
        if not it % self.oracle_period:
            energies_sorted = np.sort(energies)
            dict_topk = {}
            for k in self.sampler.oracle.k:
                mean_topk = np.mean(energies_sorted[:k])
                dict_topk.update({"oracle_mean_top{}".format(k): mean_topk})
            self.log_metrics(dict_topk, use_context=use_context, step=it)

    def save_model(self, policy_path, model_path, model, it, iter=False):
        if not it % self.policy_period:
            path = policy_path.parent / Path(
                model_path.stem
                + self.context
                + "_iter{:06d}".format(it)
                + policy_path.suffix
            )
            torch.save(model.state_dict(), path)
            if iter == False:
                torch.save(model.state_dict(), policy_path)

    def log_time(self, times, it, use_context):
        if self.record_time:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, step=it, use_contxt=use_context)

    def end(self):
        wandb.finish()
