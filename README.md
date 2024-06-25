# gflownet

gflownet is a library built upon [PyTorch](https://pytorch.org/) to easily train and extend [GFlowNets](https://arxiv.org/abs/2106.04399), also known as GFN or generative flow networks. GFlowNets are a machine learning framework for probabilistic and generative modelling, with a wide range of applications, especially in [scientific discovery](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00002h) problems.

In a nutshell, GFlowNets can be regarded as a generative model designed to sample objects $x \in \mathcal{X}$ proportionally to a reward function $R(x)$. This results in the potential of sampling diverse objects with high rewards. For example, given the reward landscape depicted below, defined over a two-dimensional space, a well-trained GFlowNet will be able to sample from the four high-reward corners with high probability.

<p align="center">
  <img width="400" src="docs/images/reward_landscape.png" />
</p>

GFlowNets rely on the principle of **compositionality** to generate samples. A meaningful decomposition of samples $x$ into multiple intermediate states $s_0\rightarrow s_1 \rightarrow \dots \rightarrow x$ can yield generalisable patterns. These patterns can then be learned by neural networks trained to model the value of transitions $F_{\theta}(s_t \rightarrow s_{t+1})$.

Consider the problem of generating [Tetris](https://en.wikipedia.org/wiki/Tetris)-like boards. A natural decomposition of the sample generation process would be to add one piece at a time, starting from an empty board. For any state representing a board with pieces, we could identify its valid parents and children, as illustrated in the figure below.

<p align="center">
  <img width="400" src="docs/images/tetris_flows.png" />
</p>

We could define a reward function $R(x)$ as the number of cells occupied by pieces, for instance. The goal of training a GFlowNet on this task would be to discover (sample) diverse solutions (boards with pieces) with high rewards. This represents an intuitive yet complex problem where GFlowNets can be used, which is [implemented in this library](https://github.com/alexhernandezgarcia/gflownet/blob/ahg/313-update-readme-components/gflownet/envs/tetris.py). Many problems in scientific discoveries, such as the inverse design of proteins, molecules, or crystals share similarties with this intuitive task.

## Main Components of the GFlowNet Library

The GFlowNet library comprises four core components: environment, proxy, policy models (forward and backward), and GFlowNet agent.

### Environment

The environment defines the state space $\mathcal{S}$ and action space $\mathbb{A}$ of a particular problem, for example the Tetris task. To illustrate the environment, let's consider an even simpler environment currently implemented in the library: the [Scrabble](https://en.wikipedia.org/wiki/Scrabble) environment, inspired by the popular board game.

The Scrabble environment simulates a simple letter arrangement game where words are constructed by adding one letter at a time, up to a maximum sequence length (typically 7). Therefore, the action space is the set of all English letters plus a special end-of-sequence (EOS) action; and the state space is the set of all possible words with up to 7 letters. We can represent each `state` as a list of indices corresponding to the letters, padded with zeroes to the maximum length. For example, the state for the word "CAT" would be represented as `[3, 1, 20, 0, 0, 0, 0]`. Actions in the Scrabble environment are single-element tuples containing the index of the letter, plus the end-of-sequence (EOS) action `(-1,)`.

Using the gflownet library for a new task will typically require implementing your own environment. The library is particularly designed to make such extensions as easy as possible. In the documentation, we show how to do it step by step. You can also watch [this live-coding tutorial](https://www.youtube.com/watch?v=tMVJnzFqa6w&t=5h22m35s) on how to code the Scrabble environment.

### Proxy

We use the term "[proxy](gflownet/proxy/base.py)" to refer to the function or model that provides the rewards for the states of an environment. In other words, In the context of GFlowNets, the proxy can be thought of as a function $E(x)$ from which the reward is derived: $R(x) = g(E(x))$, where $g$ is a function that transforms the proxy values into non-zero rewards, that is "the higher the reward the better". For example, we can implement a proxy that simulates the scores of a word in the Scrabble game. That is, the [`ScrabbleScorer`](gflownet/proxy/scrabble.py) proxy computes the sum of the score of each letter of a word. For the word "CAT" that is $E(x) = 3 + 1 + 1 = 5$. While in many environments the proxy functions is a simple scorer, more complex settings like molecule or [crystal generation](gflownet/proxy/crystals/dave.py) may be use proxies that represent the energy or a property predicted by a pre-trained machine learning model.

Adapting the gflownet library for a new task will also likely require implementing your own proxy, which is usually fairly simple, as illustrated in the documentation.

### Policy models

The policy models are neural networks that model the forward and backward transitions between states, $F_{F_{\theta}}(s_t \rightarrow s_{t+1})$ (forward) and $F_{B_{\theta}}(s_{t+1} \rightarrow s_t)$ (backward). These models take a state as input and output a distribution over the actions in the action space. For continuous environments, the outputs are the parameters of a probability distribution to sample continuous-valued actions. For many tasks, simple multi-layer perceptrons with a few layers do the job, but technically any architecture could be used as policy model. 

### GFlowNet Agent

The GFlowNet Agent is the central component that ties all others together. It orchestrates the interaction between the environment, policies, and proxy, as well as other auxiliary components such as the Evaluator and the Logger. The GFlowNet can construct training batches by sampling trajectories, optimise the policy models via gradient descent, compute evaluation metrics, log data to [Weights & Biases](https://wandb.ai/), etc. The agent can be configured to optimise any of the following loss functions implemented in the library: flow matching (FM), trajectory balance (TB), and detailed balance (TB) and forward-looking (FL). 

## Installation

**If you simply want to install everything, clone the repo and run `setup_all.sh`:**

```bash
git clone git@github.com:alexhernandezgarcia/gflownet.git
cd gflownet
./setup_all.sh
```

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

## Quickstart: How to train a GFlowNet model

The gflownet library uses [Hydra](https://hydra.cc/docs/intro/) to handle configuration files. This allows, for instance, to easily train a GFlowNet with the configuration of a specific YAML file. For example, to train a GFlowNet with a 10x10 Grid environment and the corners proxy, with the configuration from `./config/experiments/grid/corners.yaml`, we can simply run:

```bash
python main.py +experiments=grid/corners
```

Alternatively, we can explicitly indicate the environment and the proxy as follows:

```bash
python main.py env=grid proxy=box/corners
```

The above command will train a GFlowNet with the default configuration, except for the environment, which will use `./config/env/grid.yaml`; and the proxy, which will use `./config/proxy/box/corners.yaml`.

A typical use case of the gflownet library is to extend it with a new environment and a new proxy to fit your purposes. In that case, you could create their respective configuration files `./config/env/myenv.yaml` and `./config/proxy/myproxy.yaml` and run

```bash
python main.py env=myenv proxy=myproxy
```

All other configurable options are handled similarly. For example, we recommend creating a user configuration file in `./config/user/myusername.yaml` specifying the directory for the log files in `logdir.root`. Then, it can be included in the command with `user=myusername` or `user=$USER` if the name of the YAML file matches our system username.

As another example, you may also want to configure the functionality of the Logger, the class which helps manage logging to [Weights & Biases](https://wandb.ai/) during the training and evaluation of the model. Logging to WandB is disabled by default. In order to enable it, make sure to set up your WandB API key and set the configuration variable `logger.do.online` to `True` in your experiment config file or via the command line:

```bash
python main.py +experiments=grid/corners logger.do.online=True
```

Finally, also note that by default, PyTorch will operate on the CPU because we have not observed performance improvements by running on the GPU. You may run on GPU with `device=cuda`.

## Exploring the Scrabble Environment

To better understand the GFlowNet components, let us explore the Scrabble environment in more detail below.

When initializing any GFlowNet agent, it's useful to explore the properties of the environment. The library offers various functionalities for this purpose. Below are some detailed examples, among others:

1. Checking the Initial State 

You can observe the initial state of the environment. For Scrabble environment, this would be an empty board or sequence:

```python
env.state
>>> [0, 0, 0, 0, 0, 0, 0]
```

2. Exploring the Action Space
```python
env.get_action_space()
>>> [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (-1,)]
```
For Scrabble environment, the action space is all english alphabet letters indexed from 1 to 26. The action (-1,) represents the end-of-sequence (EOS) action, indicating the termination of word formation.

3. Taking a Random Step

```python
new_state, action_taken, valid = env.step_random()
print("New State:", new_state)
print("Action Taken:", action_taken)
print("Action Valid:", valid)

>>> New State: [24, 0, 0, 0, 0, 0, 0]
>>> Action Taken: (24,)
>>> Action Valid: True
```

This function randomly selects a valid action (adding a letter or ending the sequence) and applies it to the environment. The output shows the new state, the action taken, and whether the action was valid.

4. Performing a Specific Action

```python
action = (1,)  # Action to add 'A'
new_state, performed_action, is_valid = env.step(action)
print("Updated State:", new_state)
print("Performed Action:", performed_action)
print("Was the Action Valid:", is_valid)
>>> Updated State: [24, 1, 0, 0, 0, 0, 0]
>>> Performed Action: (1,)
>>> Was the Action Valid: True
```

5. Displaying the State as a human readable

```python
env.state2readable(env.state)
>>> 'X A'
```

6. Interpreting Actions as a human readable

```python
print("Action Meaning:", env.idx2token[action[0]])
>>> Action Meaning: A
```

7. Sampling a Random Trajectory

```python
new_state, action_sequence = env.trajectory_random()
print("New State:", new_state)
print("Action Sequence:" action_sequence)

>>> New State: [16, 16, 17, 20, 11, 16, 0]
>>> Action Sequence: [(16,), (16,), (17,), (20,), (11,), (16,), (-1,)]
```

8. Reset enviroment 

```python
env.reset()
env.state
>>> [0, 0, 0, 0, 0, 0, 0]
```

So far, we've discussed how to manually set actions or use random actions in the GFlowNet environment. This approach is useful for testing or understanding the basic mechanics of the environment. However, in practice, the goal of a GFlowNet agent is to learn from its experiences to take increasingly effective actions that are driven by a learned policy. 

As the agent interacts with the environment, it collects data about the outcomes of its actions. This data is used to train a policy network, which models the probability distribution of possible actions given the current state. Over time, the policy network learns to favor actions that lead to more successful outcomes with higher reward, optimizing the agent's performance.

9. Sample a batch of trajectories from a trained agent 

```python 
batch, _ = gflownet.sample_batch(n_forward=3,  train=False)
batch.states
>>> [[20, 20, 21, 3, 0, 0, 0], [12, 16, 8, 6, 14, 11, 20], [17, 17, 16, 23, 20, 16, 24]]
```

We can convert the first state to human readable:

```python
env.state2readable(batch.states[0])
>>> 'T T U C'
```

We can also compute the rewards and the proxy for all states or single state.

```python
proxy(env.states2proxy(batch.states))
>>> tensor([ 6., 19., 39.])
```
Or single state

```python
proxy(env.state2proxy(batch.states[0]))
>>> tensor([6.])
```

The `state2proxy` and `states2proxy` are helper functions that transform the input to appropriate format. For example to tensor. 

We can also compute the rewards, and since our transformation function `g` is the identity, the rewards should be equal to the proxy directly. 

```python
proxy.rewards(env.states2proxy(batch.states))
>>> tensor([ 6., 19., 39.])
```

## GFlowNet loss functions

Currently, the implementation includes the following GFlowNet losses:

- [Flow-matching (FM)](https://arxiv.org/abs/2106.04399): `gflownet=flowmatch`
- [Trajectory balance (TB)](https://arxiv.org/abs/2201.13259): `gflownet=trajectorybalance`
- [Detailed balance (DB)](https://arxiv.org/abs/2201.13259): `gflownet=detailedbalance`
- [Forward-looking (FL)](https://arxiv.org/abs/2302.01687): `gflownet=forwardlooking`

## Contributors

Many wonderful scientists and developers have contributed to this repository: [Alex Hernandez-Garcia](https://github.com/alexhernandezgarcia), [Nikita Saxena](https://github.com/nikita-0209), [Alexandra Volokhova](https://github.com/AlexandraVolokhova), [Michał Koziarski](https://github.com/michalkoziarski), [Divya Sharma](https://github.com/sh-divya), [Pierre Luc Carrier](https://github.com/carriepl) and [Victor Schmidt](https://github.com/vict0rsch).

## Research

This repository has been used in at least the following research articles:

- Lahlou et al. [A theory of continuous generative flow networks](https://proceedings.mlr.press/v202/lahlou23a/lahlou23a.pdf). ICML, 2023.
- Hernandez-Garcia, Saxena et al. [Multi-fidelity active learning with GFlowNets](https://arxiv.org/abs/2306.11715). RealML at NeurIPS 2023.
- Mila AI4Science et al. [Crystal-GFN: sampling crystals with desirable properties and constraints](https://arxiv.org/abs/2310.04925). AI4Mat at NeurIPS 2023 (spotlight).
- Volokhova, Koziarski et al. [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782). AI4Mat at NeurIPS 2023.

## Cite

Bibtex Format

```text
@misc{hernandez-garcia2024,
  author = {Hernandez-Garcia, Alex and Saxena, Nikita and Volokhova, Alexandra and Koziarski, Michał and Sharma, Divya and Viviano, Joseph D and Carrier, Pierre Luc and Schmidt, Victor},
  title  = {gflownet},
  url    = {https://github.com/alexhernandezgarcia/gflownet},
  year   = {2024},
}
```

Or [CFF file](./CITATION.cff)
