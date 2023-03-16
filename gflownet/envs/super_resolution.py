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

    # problem: I need true_image only for computing proxy, it would be nice to condition it within the env 
    # and call it fromm a specific env, not a generic one

