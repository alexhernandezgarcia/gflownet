import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import array
from omegaconf import OmegaConf


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
        debug: bool,
        run_name=None,
        tags: list = None,
        context: str = "0",
        notes: str = None,
    ):
        self.config = config
        self.do = do
        self.do.times = self.do.times and self.do.online
        self.train = train
        self.test = test
        self.oracle = oracle
        self.checkpoints = checkpoints
        slurm_job_id = os.environ.get("SLURM_JOB_ID")

        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(
                date_time,
            )
            if slurm_job_id is not None:
                run_name = slurm_job_id + " - " + run_name

        if self.do.online:
            import wandb

            self.wandb = wandb
            wandb_config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            if slurm_job_id:
                wandb_config["slurm_job_id"] = slurm_job_id
            self.run = self.wandb.init(
                config=wandb_config, project=project_name, name=run_name, notes=notes
            )
        else:
            self.wandb = None
            self.run = None
        self.add_tags(tags)
        self.context = context
        self.progress = progress
        self.lightweight = lightweight
        self.debug = debug
        # Log directory
        self.logdir = Path(logdir.root)
        if self.logdir.exists() or logdir.overwrite:
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            # TODO: this message seems contradictory with the logic
            print(f"logdir {logdir} already exists! - Ending run...")
        self.ckpts_dir = self.logdir / logdir.ckpts
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)
        # Write wandb URL
        self.write_url_file()

    def do_train(self, step):
        if self.train.period is None or self.train.period < 0:
            return False
        else:
            return not step % self.train.period

    def do_test(self, step):
        if self.test.period is None or self.test.period < 0:
            return False
        elif step == 1 and self.test.first_it:
            return True
        else:
            return not step % self.test.period

    def do_top_k(self, step):
        if self.test.top_k is None or self.test.top_k < 0:
            return False

        if self.test.top_k_period is None or self.test.top_k_period < 0:
            return False

        return step == 2 or step % self.test.top_k_period == 0

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

    def set_forward_policy_ckpt_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.pf_ckpt_path = None
        else:
            self.pf_ckpt_path = self.ckpts_dir / f"{ckpt_id}_"

    def set_backward_policy_ckpt_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.pb_ckpt_path = None
        else:
            self.pb_ckpt_path = self.ckpts_dir / f"{ckpt_id}_"

    def set_state_flow_ckpt_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.sf_ckpt_path = None
        else:
            self.sf_ckpt_path = self.ckpts_dir / f"{ckpt_id}_"

    def progressbar_update(
        self, pbar, losses, rewards, jsd, step, use_context=True, n_mean=100
    ):
        if self.progress:
            mean_main_loss = np.mean(np.array(losses)[-n_mean:, 0], axis=0)
            description = "Loss: {:.4f} | Mean rewards: {:.2f} | JSD: {:.4f}".format(
                mean_main_loss, np.mean(rewards), jsd
            )
            pbar.set_description(description)

    def log_metric(self, key: str, value, step, use_context=True):
        if not self.do.online:
            return
        if use_context:
            key = self.context + "/" + key
        self.wandb.log({key: value}, step=step)

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

    def log_plots(self, figs: list, step, fig_names=None, use_context=True):
        if not self.do.online:
            self.close_figs(figs)
            return
        keys = fig_names or [f"Figure {i} at step {step}" for i in range(len(figs))]
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

    def log_metrics(self, metrics: dict, step: int, use_context: bool = True):
        if not self.do.online:
            return
        for key, value in metrics.items():
            self.log_metric(key, value, step=step, use_context=use_context)

    def log_summary(self, summary: dict):
        if not self.do.online:
            return
        self.run.summary.update(summary)

    def log_train(
        self,
        losses,
        rewards: list,
        proxy_vals: array,
        states_term: list,
        batch_size: int,
        logz,
        learning_rates: list,  # [lr, lr_logZ]
        step: int,
        use_context: bool,
    ):
        if not self.do.online or not self.do_train(step):
            return
        if logz is None:
            logz = 0.0
        else:
            logz = logz.sum()
        if len(learning_rates) == 1:
            learning_rates += [-1.0]
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
                    "logZ",
                    "lr",
                    "lr_logZ",
                    "step",
                ],
                [
                    np.mean(rewards),
                    np.max(rewards),
                    np.mean(proxy_vals),
                    np.min(proxy_vals),
                    np.max(proxy_vals),
                    np.mean([len(state) for state in states_term]),
                    batch_size,
                    logz,
                    learning_rates[0],
                    learning_rates[1],
                    step,
                ],
            )
        )
        self.log_metrics(
            train_metrics,
            use_context=use_context,
            step=step,
        )
        loss_metrics = dict(
            zip(
                ["Loss", "Loss (terminating)", "Loss (non-term.)"],
                [loss.item() for loss in losses],
            )
        )
        self.log_metrics(
            loss_metrics,
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
            self.log_metrics(dict_topk, use_context=use_context)

    def log_losses(
        self,
        losses: list,
        step: int,
        use_context: bool,
    ):
        if not self.do.online:
            return
        loss_metrics = dict(
            zip(
                ["loss", "term_loss", "flow_loss"],
                [loss.item() for loss in losses],
            )
        )
        self.log_metrics(
            loss_metrics,
            use_context=use_context,
        )

    def log_test_metrics(
        self,
        l1: float,
        kl: float,
        jsd: float,
        corr_prob_traj_rewards: float,
        var_logrewards_logp: float,
        nll_tt: float,
        mean_logprobs_std: float,
        mean_probs_std: float,
        logprobs_std_nll_ratio: float,
        step: int,
        use_context: bool,
    ):
        if not self.do.online:
            return
        metrics = dict(
            zip(
                [
                    "L1 error",
                    "KL Div.",
                    "Jensen Shannon Div.",
                    "Corr. (test probs., rewards)",
                    "Var(logR - logp) test",
                    "NLL of test data",
                    "Mean BS Std(logp)",
                    "Mean BS Std(p)",
                    "BS Std(logp) / NLL",
                ],
                [
                    l1,
                    kl,
                    jsd,
                    corr_prob_traj_rewards,
                    var_logrewards_logp,
                    nll_tt,
                    mean_logprobs_std,
                    mean_probs_std,
                    logprobs_std_nll_ratio,
                ],
            )
        )
        self.log_metrics(
            metrics,
            use_context=use_context,
            step=step,
        )

    def save_models(
        self, forward_policy, backward_policy, state_flow, step: int = 1e9, final=False
    ):
        if self.do_checkpoints(step) or final:
            if final:
                ckpt_id = "final"
            else:
                ckpt_id = "_iter{:06d}".format(step)
            if forward_policy.is_model and self.pf_ckpt_path is not None:
                stem = self.pf_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                path = self.pf_ckpt_path.parent / stem
                torch.save(forward_policy.model.state_dict(), path)
            if (
                backward_policy
                and backward_policy.is_model
                and self.pb_ckpt_path is not None
            ):
                stem = self.pb_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                path = self.pb_ckpt_path.parent / stem
                torch.save(backward_policy.model.state_dict(), path)

            if state_flow is not None and self.sf_ckpt_path is not None:
                stem = self.sf_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                path = self.sf_ckpt_path.parent / stem
                torch.save(state_flow.model.state_dict(), path)

    def log_time(self, times: dict, use_context: bool):
        if self.do.times:
            times = {"time_{}".format(k): v for k, v in times.items()}
            self.log_metrics(times, use_context=use_context)

    def end(self):
        if not self.do.online:
            return
        self.wandb.finish()
