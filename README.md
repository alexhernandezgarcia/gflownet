# GFlowNet

This repository implements GFlowNets, generative flow networks for probabilistic modelling, on PyTorch. A design guideline behind this implementation is the separation of the logic of the GFlowNet agent and the environments on which the agent can be trained on. In other words, this implementation should allow its extension with new environments without major or any changes to to the agent. Another design guideline is flexibility and modularity. The configuration is handled via the use of [Hydra](https://hydra.cc/docs/intro/).

## Installation

### pip

```bash
python -m pip install --upgrade https://github.com/alexhernandezgarcia/gflownet/archive/main.zip
```

## How to train a GFlowNet model

To train a GFlowNet model with the default configuration, simply run

```bash
python main.py user.logdir.root=<path/to/log/files/>
```

Alternatively, you can create a user configuration file in `config/user/<username>.yaml` specifying a `logdir.root` and run

```bash
python main.py user=<username>
```

Using Hydra, you can easily specify any variable of the configuration in the command line. For example, to train GFlowNet with the trajectory balance loss, on the continuous torus (`ctorus`) environment and the corresponding proxy:

```bash
python main.py gflownet=trajectorybalance env=ctorus proxy=torus
```

The above command will overwrite the `env` and `proxy` default configuration with the configuration files in `config/env/ctorus.yaml` and `config/proxy/torus.yaml` respectively.

Hydra configuration is hierarchical. For instance, a handy variable to change while debugging our code is to avoid logging to wandb. You can do this by setting `logger.do.online=False`.
