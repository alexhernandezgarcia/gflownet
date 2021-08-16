import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Replace with Desired Network Architecture. Currently
class QueryNetworkDQN(nn.Module):
    def __init__(
        self, model_state_length, action_state_length, bias_average,
    ):
        """Initialises the Query Network. A Network that computes Q-values starting from model state and action state.

        :param: model_state_length: An integer indicating the number of features in model state.
        :param: action_state_length: An integer indicating the number of features in action state.
        :param: bias_average: A float that is used to initialize the bias in the last layer.
        """
        self.model_state_length = model_state_length
        self.action_state_length = action_state_length

        # A fully connected layers with model_state as input
        self.fc1 = nn.Sigmoid(
            nn.Linear(self.model_state_length, 10)
        )  # not trainable if not is_target_dqn

        # Concatenate the output of first fully connected layer with action_state

        # A fully connected layer with fc2concat as input
        self.fc3 = nn.Sigmoid(
            nn.Linear(self.action_state_length + 10, 5)
        )  # not trainable if not is_target_dqn

        # The last linear fully connected layer
        # The bias on the last layer is initialized to some value
        # normally it is the - average episode duriation / 2
        # like this NN find optimum better even as the mean is not 0
        self.predictions = nn.Linear(
            self.action_state_length + 10, 5
        )  # not trainable if not is_target_dqn
        nn.init.constant_(self.predictions.weight, bias_average)

    def forward(self, x, action_input):

        out = self.fc1(x)
        fc2concat = torch.concat([out, action_input], 1)
        out = self.fc3(fc2concat)
        out = self.predictions(out)
