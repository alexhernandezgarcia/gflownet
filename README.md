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

**Quickstart: If you simply want to install everything, run `setup_all.sh`.**

+ This project **requires** `python 3.10` and `cuda 11.8`.
+ Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.
+ The recommend installation is as follows:

```bash
python3.10 -m venv ~/envs/gflownet  # Initalize your virtual env.
source ~/envs/gflownet/bin/activate  # Activate your environment.
./prereq_ubuntu.sh  # Installs some packages required by dependencies.
./prereq_python.sh  # Installs python packages with specific wheels.
./prereq_geometric.sh  # OPTIONAL - for the molecule environment.
pip install .[all]  # Install the remaining elements of this package.
```

Aside from the base packages, you can optionally install `dev` tools using this tag, `materials` dependencies using this tag, or `molecules` packages using this tag. The simplest option is to use the `all` tag, as above, which installs all dependencies.

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

## Cite

Bibtex Format

```txt
@misc{hernandez-garcia2024,
  author = {Hernandez-Garcia, Alex and Saxena, Nikita and Volokhova, Alexandra and Koziarski, Michał and Sharma, Divya and Viviano, Joseph D and Carrier, Pierre Luc and Schmidt, Victor},
  title  = {gflownet},
  url    = {https://github.com/alexhernandezgarcia/gflownet},
  year   = {2024},
}
```

Or [CFF file](./CITATION.cff)
