from gflownet.policy.base import Policy


class RandomPolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_model(self):
        """
        Instantiates the policy model as a fixed tensor with the values of
        `self.random_output` defined by the environment.

        Returns
        -------
        model : torch.tensor
            The tensor `self.random_output` defined by the environment.
        is_model : bool
            False, because the policy is not a model.
        """
        return (
            torch.tile(self.random_output, (len(states), 1)).to(
                dtype=self.float, device=self.device
            ),
            False,
        )
