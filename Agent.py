# This code is the modified version of code from
# ksenia-konyushkova/intelligent_annotation_dialogs/exp1_IAD_RL.ipynb

from scipy.sparse.construct import rand
from RLmodels import QueryNetworkDQN, ParameterUpdateDQN
import numpy as np
import os
import random
import torch
import math
import torch.functional as F
from utils import *
from replay_buffer import QuerySelectionReplayMemory, ParameterUpdateReplayMemory
from oracle import Oracle


class DQN:
    """The DQN class that learns a RL policy.


    Attributes:
        policy_net: An object of class QueryNetworkDQN that is used for q-value prediction
        target_net: An object of class QueryNetworkDQN that is a lagging copy of estimator

    """

    def __init__(self, config):
        """Inits the DQN object.

        Args:
            experiment_dir: A string with parth to the folder where to save the agent and training data.
            lr: A float with a learning rate for Adam optimiser.
            batch_size: An integer indicating the size of a batch to be sampled from replay buffer for estimator update.
            target_copy_factor: A float used for updates of target_estimator,
                with a rule (1-target_copy_factor)*target_estimator weights
                + target_copy_factor*estimator

        """

        torch.manual_seed(config.seeds.model)
        self.config = config
        self.exp_name = 'learned_'
        self.load = False if config.querier.model_ckpt is None else True
        self.action_state_length = 5 # [energy, variance, 3 distance metrics]
        self.singleton_state_variables = 5 # [test loss, test std, n proxy models, cluster cutoff and elapsed time]
        self.state_dataset_size = int(config.querier.model_state_size * self.action_state_length + self.singleton_state_variables) # This depends on size of dataset V
        self.model_state_latent_dimension = config.querier.latent_space_width # latent dim of model state
        self.device = config.device
        self.epsilon = 0.1

        # Magic Hyperparameters for Greedy Sampling in Action Selection
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 10
        self.rl_pool = 100  # Size of Unlabelled Dataset aka Number of actions

        self.optimizer_param = {
            "opt_choice": config.querier.opt,
            "momentum": config.querier.momentum,
            "ckpt_path": "./ckpts/",
            "exp_name_toload": config.querier.model_ckpt,
            "exp_name": self.exp_name,
            "snapshot": 0,
            "load_opt": self.load,
        }

        self.opt_choice = self.optimizer_param["opt_choice"]
        self.momentum = self.optimizer_param["momentum"]
        if self.load:
            self._load_models()
        else:
            self._create_models()

        self._create_and_load_optimizer(**self.optimizer_param)

    def _load_models(self):
        """Load Policy model weights. Needs to know episode and exp_name.
        """
        # TODO write loader
        # TODO write saver
        if os.path.exists("ckpts/" + exp_name):  # reload model
            policy_checkpoint = torch.load(f"ckpts/policy_{episode}")
            target_checkpoint = torch.load(f"ckpts/target_{episode}")

        self.model.load_state_dict(policy_checkpoint["model_state_dict"])
        self.model.load_state_dict(target_checkpoint["model_state_dict"])

    def save_models(self, episode):
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"ckpts/policy_{episode}",
        )
        torch.save({"model_state_dict": self.target.state_dict()}, f"ckpts/target_{episode}")

    def count_parameters(
        net: torch.nn.Module,
    ) -> int:  # TODO - delete - MK didn't work for whatever reason
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def _create_and_load_optimizer(
        self,
        opt_choice,
        momentum,
        ckpt_path,
        exp_name_toload,
        exp_name,
        snapshot,
        load_opt,
        wd=0.001,
        lr_dqn=0.0001,
    ):
        opt_kwargs = {"lr": lr_dqn, "weight_decay": wd, "momentum": momentum}

        if opt_choice == "SGD":
            self.optimizer = torch.optim.SGD(
                params=filter(lambda p: p.requires_grad, self.policy_net.parameters()), **opt_kwargs
            )
        elif opt_choice == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                params=filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr_dqn
            )

        name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
        opt_policy_path = os.path.join(ckpt_path, name, "opt_policy_" + str(snapshot))

        if load_opt:
            print("(Opt load) Loading policy optimizer")
            self.optimizer.load_state_dict(torch.load(opt_policy_path))

        print("Policy optimizer created")

    def updateModelState(self, model_state, model):
        """
        update the model state and store it for later sampling
        :param model_state:
        :return:
        """
        model_state_dict = model_state

        # things to put into the model state
        # test loss and standard deviation between models
        self.model_state = torch.stack(
            (
                torch.tensor(model_state_dict["test loss"]),
                torch.tensor(model_state_dict["test std"]),
            )
        )

        # sample energies
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best cluster energies"]))
        )

        # sample uncertainties
        self.model_state = torch.cat(
            (self.model_state, torch.Tensor(model_state_dict["best cluster deviations"]))
        )

        # internal dist, dataset dist, random set dist
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best clusters internal diff"]))
        )
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best clusters dataset diff"]))
        )
        self.model_state = torch.cat(
            (self.model_state, torch.tensor(model_state_dict["best clusters random set diff"]))
        )

        # n proxy models,         # clustering cutoff,         # progress fraction
        singletons = torch.stack(
            (
                torch.tensor(model_state_dict["n proxy models"]),
                torch.tensor(model_state_dict["clustering cutoff"]),
                torch.tensor(model_state_dict["iter"] / model_state_dict["budget"]),
            )
        )

        self.model_state = torch.cat((self.model_state, singletons))
        self.model_state = self.model_state.to(self.device)

        self.proxyModel = model  # this should already be on correct device - passed directly from the main program

        # get data to compute distances
        # model state samples
        self.modelStateSamples = model_state_dict["best cluster samples"]
        # training dataset
        self.trainingSamples = np.load('datasets/' + self.config.dataset.oracle + '.npy', allow_pickle=True).item()
        self.trainingSamples = self.trainingSamples['samples']
        # large random sample
        numSamples = min(int(1e4), self.config.dataset.dict_size ** self.config.dataset.max_length // 100) # either 1e4, or 1% of the sample space, whichever is smaller
        dataoracle = Oracle(self.config)
        self.randomSamples = dataoracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        self.randomSamples = self.randomSamples['samples']


    def evaluate(self, sample, output="Average"):  # just evaluate the proxy
        return self.proxyModel.evaluate(sample, output=output)

class QuerySelectionAgent(DQN):
    def __init__(self, config):
        super().__init__(config)
        self.memory = QuerySelectionReplayMemory(self.config.buffer_size)

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        # TODO add model state single variables to DQN net
        self.policy_net = QueryNetworkDQN(
            model_state_length=self.state_dataset_size,
            action_state_length=self.action_state_length,
            model_state_latent_dimension=self.model_state_latent_dimension,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
        ).to(self.device)
        self.target_net = QueryNetworkDQN(
            model_state_length=self.state_dataset_size,
            action_state_length=self.action_state_length,
            model_state_latent_dimension=self.model_state_latent_dimension,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper. #MK what are we biasing
        ).to(self.device)

        printRecord("Policy network has " + str(get_n_params(self.policy_net)) + " parameters.")

        # print("DQN Models created!")

    def evaluateQ(
        self, sample: np.array,
    ):
        """ get the q-value for a particular sample, given its 'action state'

        :param model_state: (torch.Variable) Torch tensor containing the model state representation.
        :param action_state: (torch.Variable) Torch tensor containing the action state representations.
        :param steps_done: (int) Number of aptamers labeled so far.
        :param test: (bool) Whether we are testing the DQN or training it. Disables greedy-epsilon when True.

        :return: Action (index of Sequence to Label)
        """
        action_state = self.getActionState(sample)

        self.policy_net.eval()
        with torch.no_grad():
            q_val = self.policy_net(action_state)

        return q_val

    def getActionState(self, sample):
        """
        get the proxy model predictions and sample distances
        :param sample:
        :return:
        """
        energies, uncertainties = self.proxyModel.evaluate(sample, output="Both")
        internalDist = binaryDistance(
            np.concatenate((sample, self.modelStateSamples)),
            pairwise=False,
            extractInds=len(sample),
        )
        datasetDist = binaryDistance(
            np.concatenate((sample, self.trainingSamples)), pairwise=False, extractInds=len(sample)
        )
        randomDist = binaryDistance(
            np.concatenate((sample, self.randomSamples)), pairwise=False, extractInds=len(sample)
        )

        actionState = []
        for i in range(len(sample)):
            actionState.append(
                [energies[i], uncertainties[i], internalDist[i], datasetDist[i], randomDist[i]]
            )
        self.actionState = torch.Tensor(actionState).to(self.device)
        return self.actionState  # return action state

    def train(self, memory_batch, BATCH_SIZE=32, GAMMA=0.999, dqn_epochs=1):
        """Train a q-function estimator on a minibatch.

        Train estimator on minibatch, partially copy
        optimised parameters to target_estimator.
        We use double DQN that means that estimator is
        used to select the best action, but target_estimator
        predicts the q-value.

        :(ReplayMemory) memory: Experience replay buffer
        :param Transition: definition of the experience replay tuple
        :param BATCH_SIZE: (int) Batch size to sample from the experience replay
        :param GAMMA: (float) Discount factor
        :param dqn_epochs: (int) Number of epochs to train the DQN
        """
        # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        if len(memory_batch) < BATCH_SIZE:
            return
        print("Optimize model...")
        print(len(memory_batch))
        self.policy_net.train()
        loss_item = 0
        for ep in range(dqn_epochs):
            self.optimizer.zero_grad()
            transitions = memory_batch.sample(BATCH_SIZE)
            for transition in transitions:
                # Get Target q-value function value for the action at s_t+1
                # that yields the highest discounted return
                with torch.no_grad():
                    # Get Q-values for every action
                    q_val_ = [
                        self.target_net(
                            transition.next_model_state.detach(), action_i_state.detach()
                        )
                        for action_i_state in transition.next_action_state
                    ]

                    action_i = torch.argmax(torch.stack(q_val_))

                max_next_q_value = self.policy_net(
                    transition.next_model_state.detach(),
                    transition.next_action_state[action_i].detach(),
                )

                # Get Predicted Q-values at s_t
                online_q_values = self.policy_net(
                    transition.model_state.detach(), transition.action_state[action_taken].detach()
                )
                # Compute the Target Q values (No future return if terminal).
                # Use Bellman Equation which essentially states that sum of r_t+1 and the max_q_value at time t+1
                # is the target/expected value of the Q-function at time t.
                target_q_values = (
                    (max_next_q_value * GAMMA) + transition.reward
                    if transition.terminal
                    else transition.reward
                )

                # Compute MSE loss Comparing Q(s) obtained from Online Policy to
                # target Q value (Q'(s)) obtained from Target Network + Bellman Equation
                loss = F.mse_loss(online_q_values, target_q_values)
                loss_item += loss.item()
                loss.backward()
            self.optimizer.step()

            del loss
            del transitions


class ParameterUpdateAgent(DQN):
    def __init__(self, config):
        super().__init__(config)
        self.memory = ParameterUpdateReplayMemory(self.config.al.buffer_size)

    def _create_models(self):
        """Creates the Online and Target DQNs

        """
        # Query network (and target network for DQN)
        self.policy_net = ParameterUpdateDQN(
            model_state_length=self.state_dataset_size,
            model_state_latent_dimension=self.model_state_latent_dimension,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
        ).to(self.device)
        self.target_net = ParameterUpdateDQN(
            model_state_length=self.state_dataset_size,
            model_state_latent_dimension=self.model_state_latent_dimension,
            bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper. #MK what are we biasing
        ).to(self.device)

        printRecord("Policy network has " + str(get_n_params(self.policy_net)) + " parameters.")

        # print("DQN Models created!")

    #TODO sample within train funciton self.memory_buffer.sample(self.config.q_batch_size)
    def train(self, BATCH_SIZE=32, GAMMA=0.999, dqn_epochs=1):
        """Train a q-function estimator on a minibatch.

        Train estimator on minibatch, partially copy
        optimised parameters to target_estimator.
        We use double DQN that means that estimator is
        used to select the best action, but target_estimator
        predicts the q-value.

        :(ReplayMemory) memory: Experience replay buffer
        :param Transition: definition of the experience replay tuple
        :param BATCH_SIZE: (int) Batch size to sample from the experience replay
        :param GAMMA: (float) Discount factor
        :param dqn_epochs: (int) Number of epochs to train the DQN
        """
        # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        memory_batch = self.memory.sample(self.config.al.q_batch_size)
        if len(memory_batch) < BATCH_SIZE:
            return
        print("Optimize model...")
        print(len(memory_batch))
        self.policy_net.train()
        loss_item = 0
        for ep in range(dqn_epochs):
            self.optimizer.zero_grad()
            transitions = memory_batch.sample(BATCH_SIZE)
            for transition in transitions:
                # Get Target q-value function value for the action at s_t+1
                # that yields the highest discounted return
                with torch.no_grad():
                    # Get Q-values for every action
                    q_val_ = self.target_net(transition.next_model_state.detach())

                    action_i = torch.argmax(q_val_)

                max_next_q_value = self.policy_net(transition.next_model_state.detach())[action_i]

                # Get Predicted Q-values at s_t
                online_q_value = self.policy_net(transition.model_state.detach())[
                    transition.action
                ]
                # Compute the Target Q values (No future return if terminal).
                # Use Bellman Equation which essentially states that sum of r_t+1 and the max_q_value at time t+1
                # is the target/expected value of the Q-function at time t.
                target_q_value = (
                    (max_next_q_value * GAMMA) + transition.reward
                    if transition.terminal
                    else transition.reward
                )

                # Compute MSE loss Comparing Q(s) obtained from Online Policy to
                # target Q value (Q'(s)) obtained from Target Network + Bellman Equation
                loss = F.mse_loss(online_q_value, target_q_value)
                loss_item += loss.item()
                loss.backward()
            self.optimizer.step()

            del loss
            del transitions

    def evaluateQ(self):
        """ get the q-value for a particular sample, given its 'action state'

        :param model_state: (torch.Variable) Torch tensor containing the model state representation.
        :param steps_done: (int) Number of aptamers labeled so far.
        :param test: (bool) Whether we are testing the DQN or training it. Disables greedy-epsilon when True.

        :return: Action (index of Sequence to Label)
        """

        self.policy_net.eval()
        with torch.no_grad():
            q_val = self.policy_net(self.model_state)

        return q_val

    def getAction(self):
        action = np.zeros(self.model_state_latent_dimension)
        if random.random() > self.epsilon:
            q_values = self.evaluateQ()
            action_id = torch.argmax(q_values)

        else:
            action_id = int(random.random() * self.model_state_latent_dimension)

        action[action_id] = 1
        return action

    def push_to_buffer(
        self, model_state, action, next_model_state, reward, terminal
    ):
        """Saves a transition."""
        self.memory.push(model_state, action, next_model_state, reward, terminal)

