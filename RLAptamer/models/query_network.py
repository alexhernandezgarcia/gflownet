import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Replace with Desired Network Architecture. Currently
class QueryNetworkDQN(nn.Module):
    def __init__(self, model_state_length, action_state_length, model_state_latent_dimension, bias_average,):
        """Initialises the Query Network. A Network that computes Q-values starting from model state and action state.
        :param: model_state_length: An integer indicating the number of features in model state.
        :param: action_state_length: An integer indicating the number of features in action state.
        :param: bias_average: A float that is used to initialize the bias in the last layer.
        """
        super(QueryNetworkDQN, self).__init__()
        self.model_state_length = model_state_length
        self.action_state_length = action_state_length
        self.model_state_latent_dimension = model_state_latent_dimension


        # A fully connected layers with model_state as input
        self.fc1 = nn.Linear(self.model_state_length, self.model_state_latent_dimension)
        # not trainable if not is_target_dqn

        # Concatenate the output of first fully connected layer with action_state

        # A fully connected layer with fc2concat as input
        self.fc3 = nn.Linear(self.action_state_length + self.model_state_latent_dimension, self.model_state_latent_dimension)  # not trainable if not is_target_dqn

        # The last linear fully connected layer
        # The bias on the last layer is initialized to some value
        # normally it is the - average episode duriation / 2
        # like this NN find optimum better even as the mean is not 0
        self.predictions = nn.Linear(self.model_state_latent_dimension, 1)  # not trainable if not is_target_dqn
        nn.init.constant_(self.predictions.weight, bias_average)


    def storeLatent(self, model_state):
        self.model_latent = torch.sigmoid(self.fc1(model_state.unsqueeze(0).float()))


    def forward(self, action_input):
        if action_input.ndim < 2:
            action_input = action_input.unsqueeze(0)
        fc2concat = torch.cat((self.model_latent.repeat(len(action_input),1), action_input), 1)
        out = torch.sigmoid(self.fc3(fc2concat))
        return self.predictions(out)
