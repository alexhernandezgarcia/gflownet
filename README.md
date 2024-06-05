# GFlowNet

This repository implements GFlowNets, generative flow networks for probabilistic modelling, on PyTorch. A design guideline behind this implementation is the separation of the logic of the GFlowNet agent and the environments on which the agent can be trained on. In other words, this implementation facilitates the extension with new environments for new applications. The configuration is handled via the use of [Hydra](https://hydra.cc/docs/intro/).

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

## Main Components of the GFlowNet Library

The GFlowNet library has 5 core components, each playing a crucial role in the network's operation. Understanding these components is essential for effectively using and extending the library for your own task. These 5 component are the Logger, Proxy, Environment,  Policies (Forward and Backward), and GFlowNet Agent. 

### Logger
The purpose of the Logger is to manage all logging activities during the training and evaluation of the network. It captures and stores logs to track the model's performance and debugging information. For instance, it logs details such as training progress, performance metrics, and any potential errors or warnings that occur. It also integrates to WandB providing a cloud-based platform for visualizing and comparing experiments. 

### Environment
The Environment is main and most important component of GFlowNet Library. To understand the Environment, let us consider a simple enviroment currently implemented in the library, the Scramble enviroment. 

The Scramble environment simulates a simple letter arrangement game where sequences are constructed by adding one letter at a time, up to a maximum sequence length (in our case 7). Each enviroment would have State Representation and Actions. For instance, for scramble enviroment, Each ``State`` within the environment is a list of indices corresponding to letters. These indices start from 1 and are padded with index 0 to denote unused slots up to the maximum length. For example, if our sequence length is 7, and our constructed word is `Alex`, we will have `[1, 11, 4, 23, 0, 0, 0]`. In the library, we already have helper functions that automatically format and convert the states to human readable and vice versa. 

``Actions`` in the Scramble environment are single-element tuples containing the index of the letter to be added to the sequence. For instance, the end of the sequence (EOS) action is denoted by (-1,). It is tuple because you could have certain enviroments where actions could be more than single action. 

In GFlowNet library, we make it easy adding new enviroments for your own task, in the documentation, we show how to do this seamlessly. 

### Proxy

The Proxy plays a crucial role in computing rewards for the actions taken within an environment. In other words, In the context of GFlowNets, the proxy can be thought of as a transformation function `R(x) = g(e(x))`, where `e(x)` represents an encoding or transformation or computes the score of the generated output x, and g translates this into a reward (i.e. `R(x)`). For instance, let us say we sample the word `Alex` in our Scramble game's enviroment, if the word is valid in our vocabulary, then we will have a maximum score (e.g. `39`), and if g is the identity function then our reward would be equal to the proxy directly. For certain enviroments, the proxy is just scorer but to have a common name for more complex enviroments (e.g. in molecule generation where it could be energy function) we call it Proxy in the GFlowNet library.

### Policies (Forward and Backward)

The policies are neural networks that model the probability distributions of possible actions in a given state. They are key to deciding the next state given previous state in the network's exploration of the environment. Both forward and backward policies receive the current state as input and output a flow distribution over possible actions. We call it flow, because the idea of GFlowNet is to flow a sequence of intermediate steps before generating the final object `x` (e.g. we could have `s_1 -> s_2 -> s_3 -> ... -> x`). Particularly, the forward policy determines the next state, while the backward policy determines the previous state.

### GFlowNet Agent

The GFlowNet Agent is central component that ties all others together. It orchestrates the interaction between the environment, policies, and proxy to conduct the training and generation tasks. The agent handles the training process, action sampling, and trajectory generation, leveraging the loss functions to optimize performance. Some of the Key Features and Functionalities are initializing and configuring the environment and proxy to ensure they are ready for training and evaluation. Manages both forward and backward policies to determine the next actions based on the current state. Does the training by utilizing  different types of loss functions like flow matching, trajectory balance, and detailed balance, Action Sampling and Metrics and Logging. 

To understand better the above components, let us play with the scramble enviroment below:

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

## Contributors

Many wonderful scientists and developers have contributed to this repository: [Alex Hernandez-Garcia](https://github.com/alexhernandezgarcia), [Nikita Saxena](https://github.com/nikita-0209), [Alexandra Volokhova](https://github.com/AlexandraVolokhova), [Michał Koziarski](https://github.com/michalkoziarski), [Divya Sharma](https://github.com/sh-divya), [Pierre Luc Carrier](https://github.com/carriepl) and [Victor Schmidt](https://github.com/vict0rsch). The GFlowNet implementation was initially part of [github.com/InfluenceFunctional/ActiveLearningPipeline](https://github.com/InfluenceFunctional/ActiveLearningPipeline).

## Research

This repository has been used in at least the following research articles:

- Lahlou et al. [A theory of continuous generative flow networks](https://proceedings.mlr.press/v202/lahlou23a/lahlou23a.pdf). ICML, 2023.
- Hernandez-Garcia, Saxena et al. [Multi-fidelity active learning with GFlowNets](https://arxiv.org/abs/2306.11715). RealML at NeurIPS 2023.
- Mila AI4Science et al. [Crystal-GFN: sampling crystals with desirable properties and constraints](https://arxiv.org/abs/2310.04925). AI4Mat at NeurIPS 2023 (spotlight).
- Volokhova, Koziarski et al. [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782). AI4Mat at NeurIPS 2023.

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
