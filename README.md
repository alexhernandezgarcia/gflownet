# GFlowNet

This repository implements GFlowNets, generative flow networks for probabilistic modelling, on PyTorch. A design guideline behind this implementation is the separation of the logic of the GFlowNet agent and the environments on which the agent can be trained on. In other words, this implementation facilitates the extension with new environments for new applications. 

<div style="text-align: center;">
    <img src="docs/images/image.png" alt="Tetris Environment" width="600" height="600"/>
    <br>
    <em>Figure 1: The Tetris environment</em>
</div>

Figure 1 illustrates the Tetris environment implemented in our library. This environment is a simplified version of Tetris, where the action space includes choosing different Tetris pieces, rotating them, and deciding where to drop them on the game board. Each action affects the game state, demonstrating the potential of GFlowNets to manage complex, dynamic environments. The Tetris environment provides a familiar yet complex example of applying GFlowNets to problem spaces that are both spatial and temporal.

For more details on how to configure and interact with the Tetris environment using our GFlowNet library, refer to our [detailed documentation](link-to-detailed-docs) or check out [this example](link-to-example) which walks through setting up and training a GFlowNet in this environment.

## Main Components of the GFlowNet Library

The GFlowNet library comprises four core components, each playing a crucial role in the network's operation. Understanding these components is essential for effectively using and extending the library for your tasks. These components are the Environment, Proxy, Policies (Forward and Backward), and the GFlowNet Agent.

### Environment

The Environment is the main and most important component of the GFlowNet Library. To illustrate this, consider a simple environment currently implemented in the library: the Scrabble environment. 

The Scrabble environment simulates a simple letter arrangement game where sequences are constructed by adding one letter at a time, up to a maximum sequence length (in our case 7). Each environment has State Representations and Actions. For instance, in the Scrabble enviroment, Each `State` is a list of indices corresponding to letters. These indices start from 1 and are padded with index 0 to denote unused slots up to the maximum length. For example, if our sequence length is 7, and our constructed word is `Alex`, it would be represented as `[1, 11, 4, 23, 0, 0, 0]`. The library includes helper functions that automatically format and convert states to and from a human-readable format. 

``Actions`` in the Scrabble environment are single-element tuples containing the index of the letter to be added to the sequence. For instance, the end of the sequence (EOS) action is denoted by (-1,). The tuple format allows us to represent more than single action, because certain enviroments could have multiple actions. 

In the library, we make it easy adding new enviroments for your own task. In the documentation, we show how to do this seamlessly. You can also watch a live coding tutorial on how to add your custom enviroment [here](https://www.youtube.com/watch?v=tMVJnzFqa6w&t=5h22m35s)

### Proxy

The Proxy plays a crucial role in computing rewards for the actions taken within an environment. In other words, In the context of GFlowNets, the proxy can be thought of as a transformation function `R(x) = g(e(x))`, where `e(x)` represents an encoding or transformation or computes the score of the generated output `x`, and `g` translates this into a reward (i.e. `R(x)`). For example, if the word `Alex` is sampled in our Scrabble environment and is valid in our vocabulary, it might receive a score of 39. If `g` is the identity function, then our reward would directly be equal to the proxy score (i.e. `e(x)`). While in many environments the proxy functions is a simple scorer, in more complex settings (like molecule generation where it could be an energy function), we consistently refer to it as the Proxy in the GFlowNet library.

### Policies (Forward and Backward)

The policies are neural networks that model the probability distributions of possible actions given a current state. They are key to deciding the next state given previous state in the network's exploration of the environment. Both forward and backward policies receive the current state as input and output a flow distribution over possible actions. We use the term "flow" here, because the idea of GFlowNet is to flow a sequence of intermediate steps before generating the final object `x` (e.g. to generate `x` we might take the steps `s_1 -> s_2 -> s_3 -> ... -> x`). Particularly, the forward policy determines the next state, while the backward policy determines the previous state (i.e. helps retrace steps to a previous state).

### GFlowNet Agent

The GFlowNet Agent is the central component that ties all others together. It orchestrates the interaction between the environment, policies, and proxy to conduct training and generation tasks. The agent manages the training setup, action sampling, trajectory generation, and metrics logging. Some of the features and functionalities of the agent are initializing and configuring the environment and proxy to ensure they are ready for training and evaluation. The agent also manages both forward and backward policies to determine the next actions based on the current state. The agent can utilize the various types of loss functions implemented in the library, such as flow matching, trajectory balance, and detailed balance to optimize model's performance during training. 

#### Exploring the Scrabble Environment

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

The configuration is handled via the use of [Hydra](https://hydra.cc/docs/intro/). To train a GFlowNet model with the default configuration, simply run

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

Hydra configuration is hierarchical. For instance, You can seamlessly modify exisiting flag or variable in the configuration by setting `logger.do.online=False`. For more, feel free to read the [Hydra documentation](https://hydra.cc/docs/intro/). 

Note that by default, PyTorch will operate on the CPU because we have not observed performance improvements by running on the GPU. You may run on GPU with `device=cuda`.

## GFlowNet loss functions

Currently, the implementation includes the following GFlowNet losses:

- [Flow-matching (FM)](https://arxiv.org/abs/2106.04399): `gflownet=flowmatch`
- [Trajectory balance (TB)](https://arxiv.org/abs/2201.13259): `gflownet=trajectorybalance`
- [Detailed balance (DB)](https://arxiv.org/abs/2201.13259): `gflownet=detailedbalance`
- [Forward-looking (FL)](https://arxiv.org/abs/2302.01687): `gflownet=forwardlooking`

## Logger 

The library also has Logger class which helps to manage all logging activities during the training and evaluation of the network. It captures and stores logs to track the model's performance and debugging information. For instance, it logs details such as training progress, performance metrics, and any potential errors or warnings that occur. It also integrates to [wandb.ai](https://wandb.ai) providing a cloud-based platform for logging the train and evaluation metrics to [wandb.ai](https://wandb.ai). The WandB is disabled by default. In order to enable it, set the configuration variable `logger.do.online` to `True`.

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

```text
@misc{hernandez-garcia2024,
  author = {Hernandez-Garcia, Alex and Saxena, Nikita and Volokhova, Alexandra and Koziarski, Michał and Sharma, Divya and Viviano, Joseph D and Carrier, Pierre Luc and Schmidt, Victor},
  title  = {gflownet},
  url    = {https://github.com/alexhernandezgarcia/gflownet},
  year   = {2024},
}
```

Or [CFF file](./CITATION.cff)
