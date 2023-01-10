import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class wandb_logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(self, config, project_name, run_name=None):
        self.config = config
        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(
                date_time,
            )
        self.run = wandb.init(config=config, project=project_name, name=run_name)
        self.add_tags(config.log.tags)
        self.context = "0"

    def add_tags(self, tags):
        self.run.tags = self.run.tags + tags

    def set_context(self, context):
        self.context = str(context)

    def log_metric(self, key, value, use_context=True):
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

    def log_metrics(self, metrics, step, use_context=True):
        if use_context:
            for key, _ in metrics.items():
                key = self.context + "/" + key
        wandb.log(metrics, step)

    def end(self):
        wandb.finish()
