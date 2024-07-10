from gflownet.policy.base import Policy


class UniformPolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_model(self):
        """
        Instantiates the policy model as a fixed tensor of ones, to define a uniform
        distribution over the action space.

        Returns
        -------
        model : torch.tensor
            A tensor of `self.output_dim` ones.
        is_model : bool
            False, because the policy is not a model.
        """
        return (
            torch.ones(
                (len(states), self.output_dim), dtype=self.float, device=self.device
            ),
            False,
        )
