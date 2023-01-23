from datetime import datetime
import numpy as np
import torch
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
        do: dict,
        project_name: str,
        logdir: dict,
        train: dict,
        test: dict,
        oracle: dict,
        checkpoints: dict,
        progress: bool,
        lightweight: bool,
        run_name=None,
        tags: list = None,
    ):
        self.config = config
        self.do = do
        self.do.times = self.do.times and self.do.online
        self.train = train
        self.test = test
        self.oracle = oracle
        self.checkpoints = checkpoints
        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(
                date_time,
            )
        if self.do.online:
            import wandb
            import matplotlib.pyplot as plt

            self.wandb = wandb
            self.plt = plt
            self.run = self.wandb.init(
                config=config, project=project_name, name=run_name
            )
        self.add_tags(tags)
        self.context = "0"
        self.progress = progress
        self.lightweight = lightweight
        # Log directory
        self.logdir = Path(logdir.root)
        if self.logdir.exists() or logdir.overwrite:
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            # TODO: this message seems contradictory with the logic
            print(f"logdir {logdir} already exists! - Ending run...")
        self.ckpts_dir = self.logdir / logdir.ckpts
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)

    def do_train(self, step):
        if self.train.period is None or self.train.period < 0:
            return False
        else:
            return not step % self.train.period

    def do_test(self, step):
        if self.test.period is None or self.test.period < 0:
            return False
        else:
            return not step % self.test.period

    def do_oracle(self, step):
        if self.oracle.period is None or self.oracle.period < 0:
            return False
        else:
            return not step % self.oracle.period

    def do_checkpoints(self, step):
        if self.checkpoints.period is None or self.checkpoints.period < 0:
            return False
        else:
            return not step % self.checkpoints.period

    def add_tags(self, tags: list):
        if not self.do.online:
            return
        self.run.tags = self.run.tags + tags

    def set_context(self, context: int):
        self.context = str(context)

    def set_forward_policy_ckpt_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.pf_ckpt_path = None
        else:
            self.pf_ckpt_path = self.ckpts_dir / f"_{ckpt_id}"

    def set_backward_policy_ckpt_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.pb_ckpt_path = None
        else:
            self.pb_ckpt_path = self.ckpts_dir / f"_{ckpt_id}"

    def log_metric(self, key: str, value, step, use_context=True):
        if not self.do.online:
            return
        if use_context:
            key = self.context + "/" + key
        self.wandb.log({key: value}, step)

    def log_histogram(self, key, value, step, use_context=True):
        if not self.do.online:
            return
        if use_context:
            key = self.context + "/" + key
        fig = self.plt.figure()
        self.plt.hist(value)
        self.plt.title(key)
        self.plt.ylabel("Frequency")
        self.plt.xlabel(key)
        fig = self.wandb.Image(fig)
        self.wandb.log({key: fig}, step)

    def log_metrics(self, metrics: dict, step: int, use_context: bool = True):
        if not self.do.online:
            return
        if use_context:
            for key, _ in metrics.items():
                key = self.context + "/" + key
        self.wandb.log(metrics, step)

    def log_sampler_train(
        self,
        rewards: list,
        proxy_vals: array,
        states_term: list,
        data: list,
        step: int,
        use_context: bool,
    ):
        if not self.do.online:
            return
        if self.do_train(step):
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
                step=step,
            )

    def log_sampler_test(
        self, corr: array, data_logq: list, step: int, use_context: bool
    ):
        if not self.do.online:
            return
        if self.do_test(step):
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
                step=step,
            )

    def log_sampler_oracle(self, energies: array, step: int, use_context: bool):
        if not self.do.online:
            return
        if step.do_oracle(step):
            energies_sorted = np.sort(energies)
            dict_topk = {}
            for k in self.oracle.k:
                mean_topk = np.mean(energies_sorted[:k])
                dict_topk.update({"oracle_mean_top{}".format(k): mean_topk})
            self.log_metrics(dict_topk, use_context=use_context, step=step)

    def log_sampler_loss(
        self,
        losses: list,
        l1_error: float,
        kl_div: float,
        jsd: float,
        step: int,
        use_context: bool,
    ):
        if not self.do.online:
            return
        loss_metrics = dict(
            zip(
                ["loss", "term_loss", "flow_loss", "l1", "kl", "jsd"],
                [loss.item() for loss in losses] + [l1_error, kl_div, jsd],
            )
        )
        self.log_metrics(
            loss_metrics,
            use_context=use_context,
            step=step,
        )

    def save_models(
        self, forward_policy, backward_policy, step: int = 1e9, final=False
    ):
        if self.do_checkpoints(step) or final:
            if final:
                ckpt_id = "final"
            else:
                ckpt_id = "_iter{:06d}".format(step)
            if forward_policy.is_model and self.pf_ckpt_path is not None:
                import ipdb

                ipdb.set_trace()
                stem = self.pf_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                path = self.pf_ckpt_path.parent + stem
                torch.save(forward_policy.model.state_dict(), path)
            if backward_policy.is_model and self.pf_ckpt_path is not None:
                stem = self.pb_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                path = self.pb_ckpt_path.parent + stem
                torch.save(backward_policy.model.state_dict(), path)

    def log_time(self, times: dict, step: int, use_context: bool):
        if self.do.times:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, step=step, use_contxt=use_context)

    def end(self):
        if not self.do.online:
            return
        self.wandb.finish()
