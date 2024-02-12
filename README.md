# GFlowNet

This repository implements GFlowNets, generative flow networks for probabilistic modelling, on PyTorch. A design guideline behind this implementation is the separation of the logic of the GFlowNet agent and the environments on which the agent can be trained on. In other words, this implementation facilitates the extension with new environments for new applications. The configuration is handled via the use of [Hydra](https://hydra.cc/docs/intro/).

## Contributors

Many wonderful scientists and developers have contributed to this repository: [Alex Hernandez-Garcia](https://github.com/alexhernandezgarcia), [Nikita Saxena](https://github.com/nikita-0209), [Alexandra Volokhova](https://github.com/AlexandraVolokhova), [Michał Koziarski](https://github.com/michalkoziarski), [Divya Sharma](https://github.com/sh-divya), [Pierre Luc Carrier](https://github.com/carriepl) and [Victor Schmidt](https://github.com/vict0rsch). The GFlowNet implementation was initially part of [github.com/InfluenceFunctional/ActiveLearningPipeline](https://github.com/InfluenceFunctional/ActiveLearningPipeline).

## Research

This repository has been used in at least the following research articles:

- Lahlou et al. [A theory of continuous generative flow networks](https://proceedings.mlr.press/v202/lahlou23a/lahlou23a.pdf). ICML, 2023.
- Hernandez-Garcia, Saxena et al. [Multi-fidelity active learning with GFlowNets](https://arxiv.org/abs/2306.11715). RealML at NeurIPS 2023.
- Mila AI4Science et al. [Crystal-GFN: sampling crystals with desirable properties and constraints](https://arxiv.org/abs/2310.04925). AI4Mat at NeurIPS 2023 (spotlight).
- Volokhova, Koziarski et al. [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782). AI4Mat at NeurIPS 2023.

## Installation

+ This project **requires** `python 3.10` and `cuda 11.8`.
+ To initalize your environment, we have provided `prereq_ubuntu.sh`. This handles the prerequisites for installing this package.
+ In the case that you want to use the `molecules` environment, you should also run `prereq_geometric.sh`.
+ After these prerequisites are satified, you can simply pip install this package:

```bash
cd /path/to/gflownet
pip install -e .[all]
```

TODO: Include information about tags here.

### pip

TODO: Remove? It looks like we used to depend on conda? I'm confused.

```bash
conda install xtb-python -c conda-forge -y
conda install pyshtools  -c conda-forge -y

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

## GFlowNet loss functions

Currently, the implementation includes the following GFlowNet losses:

- [Flow-matching (FM)](https://arxiv.org/abs/2106.04399): `gflownet=flowmatch`
- [Trajectory balance (TB)](https://arxiv.org/abs/2201.13259): `gflownet=trajectorybalance`
- [Detailed balance (DB)](https://arxiv.org/abs/2201.13259): `gflownet=detailedbalance`
- [Forward-looking (FL)](https://arxiv.org/abs/2302.01687): `gflownet=forwardlooking`

## Logging to wandb

The repository supports logging of train and evaluation metrics to [wandb.ai](https://wandb.ai), but it is disabled by default. In order to enable it, set the configuration variable `logger.do.online` to `True`.
