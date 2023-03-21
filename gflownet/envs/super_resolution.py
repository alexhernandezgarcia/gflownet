import torch
from torch.nn import functional as F

from gflownet.envs.base_conditional import GFlowNetCondEnv

class SuperResolutionEnv(GFlowNetCondEnv):
    def __init__(self, image_shape, length_traj, n_comp, path_to_data, **kwardgs):
        super().__init__(**kwargs)
        self.path_to_data = path_to_data
        self.condition = None
        self.image_shape = image_shape # (C, H, W)
        self.state_shape = (image_shape[0] + 1, *image_shape[1:]) # +1 channel for the step
        self.length_traj = length_traj
        self.n_comp = n_comp
        # define proxy, make proxy raise an error if true_image is not set

    def reset_state(self):
        step_channel = torch.zeros(self.image_shape[1:]).to(self.condition)
        self.state = torch.cat([self.downscale(self.condition), step_channel])

    def reset_condition(self, condition=None):
        if condition in not None:
            self.condition = condition
        else:
            print("Condition is not reset")
        # set true image to proxy?

    def downscale(self, true_image):
        weight = torch.zeros([3,3,2,2])
        weight[[0,1,2], [0,1,2]] = torch.ones([3, 2,2])*0.25
        weight = weight.to(true_image)
        downscaled = F.conv2d(true_image.unsqueeze(0), weight, stride=2, padding='valid')
        result = F.interpolate(downscaled, size=true_image.shape[-2:], mode='nearest')
        return result[0]

    def get_condition_dataloader(self, batch_size, train=True):
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.CIFAR10(root=self.path_to_data, train=train,
                                        download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=train, num_workers=2)
        return dataloader

    def statebatch2policy(self, states: List[TensorType], conditions: List[TensorType]) -> TensorType["batch", "policy_input_dim"]:
        return torch.stack(states, dim=0)

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"], conditions: TensorType["batch", "condition_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        return states

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"],  conditions: TensorType["batch", "condition_dim"]
    ) -> tuple[TensorType["batch", "state_proxy_dim"], TensorType["batch", "condition_proxy_dim"]]:
        return (states[:, :-1], conditions)
        
    def get_actions_space(self):
        """Actions are increments to the image, which are torch float tensors of self.image_shape
        Stop astion is a tensor of torch.inf's of image shape"""
        actions = [torch.zeros(self.image_shape, dtype=torch.float32)]
        # stop action
        actions += [torch.inf*torch.ones(self.image_shape, dtype=torch.float32)]
        return actions

    def get_policy_output(self, params: dict):
        """Returns parameters of GMM: 
         - each pixel in each channel is independend GMM
         - shape of parameners vector is [C, H, W, 3*K], where K is number of components
         - 3*K: weights, means, scales"""
        policy_output = torch.ones(list(self.image_shape) + [3 * self.n_comp])
        policy_output[..., self.n_comp : 2 * self.n_comp] = params.gmm_mean
        policy_output[..., 2 * self.n_comp : 3 * self.n_comp] = params.gmm_scale
        return policy_output

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns [True] if the only possible action is eos, [False] otherwise.
        """
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        if done:
            return [True]
        elif state[-1, 0, 0] >= self.length_traj:
            return [True]
        else:
            return [False]

    

        
    


    # problem: I need true_image only for computing proxy, it would be nice to condition it within the env 
    # and call it fromm a specific env, not a generic one

